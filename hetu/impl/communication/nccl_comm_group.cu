#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/stream/TaskQueue.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include <numeric>
#include <mutex>

namespace hetu {
namespace impl {
namespace comm {

using hetu::operator<<;

DECLARE_HT_EXCEPTION(nccl_error);

#define NCCL_CALL(f)                                                           \
  for (auto result = (f); result != ncclSuccess; result = ncclSuccess)         \
  __HT_FATAL_SILENT(hetu::impl::comm::nccl_error)                              \
    << "NCCL call " << #f << " failed: " << ncclGetErrorString(result)

namespace {

static std::once_flag nccl_init_flag;
static std::mutex nccl_create_group_mutex;
static std::vector<
  std::vector<std::map<std::vector<int>, NCCLCommunicationGroup>>>
  nccl_comm_groups(
    (HT_NUM_STREAMS_PER_DEVICE) + 1,
    std::vector<std::map<std::vector<int>, NCCLCommunicationGroup>>(
      HT_MAX_GPUS_RUN_TIME));
// The worldwide groups would be excessively accessed. Cache them here.
static std::vector<std::vector<NCCLCommunicationGroup>>
  worldwide_nccl_comm_groups(
    (HT_NUM_STREAMS_PER_DEVICE) + 1,
    std::vector<NCCLCommunicationGroup>(HT_MAX_GPUS_RUN_TIME));

inline ncclRedOp_t to_NCCL_Op(ReductionType red_type) {
  switch (red_type) {
    case kSUM: return ncclSum;
    case kPROD: return ncclProd;
    case kMAX: return ncclMax;
    case kMIN: return ncclMin;
    case kAVG:
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 10))
      return ncclAvg;
#else
      HT_NOT_IMPLEMENTED << "ncclAvg requires NCCL 2.10+";
      return ncclSum; // shoud not reach here
#endif
    default:
      HT_NOT_IMPLEMENTED << "Reduction type " << red_type
                         << " is not supported for NCCL.";
      return ncclSum; // shoud not reach here
  }
}

inline ncclDataType_t to_NCCL_Datatype(DataType dtype) {
  switch (dtype) {
    case kUInt8: return ncclUint8;
    case kInt8: return ncclInt8;
    case kInt32: return ncclInt32;
    case kInt64: return ncclInt64;
    case kFloat16: return ncclFloat16;
    case kFloat32: return ncclFloat32;
    case kFloat64: return ncclFloat64;
    default:
      HT_NOT_IMPLEMENTED << "Data type " << dtype
                         << " is not supported for NCCL.";
      return ncclFloat32; // shoud not reach here
  }
}

inline int to_num_bytes(DataType dtype) {
  switch (dtype) {
    case kUInt8: return 1;
    case kInt8: return 1;
    case kInt32: return 4;
    case kInt64: return 8;
    case kFloat16: return 2;
    case kFloat32: return 4;
    case kFloat64: return 8;
    default:
      HT_NOT_IMPLEMENTED << "Data type " << dtype
                         << " is not supported for NCCL.";
      __builtin_unreachable();
  }
}

static void NCCL_Init_Once() {
  std::call_once(nccl_init_flag, []() {
    // register exit handler
    HT_ASSERT(std::atexit([]() {
                std::lock_guard<std::mutex> lock(nccl_create_group_mutex);
                nccl_comm_groups.clear();
                worldwide_nccl_comm_groups.clear();
              }) == 0)
      << "Failed to register the exit function for NCCL.";
  });
}

} // namespace

struct NCCLGroupGuard {
  NCCLGroupGuard(bool group = false) : is_group_call(group) {
    if (is_group_call)
      NCCL_CALL(ncclGroupStart());
  }
  ~NCCLGroupGuard() {
    if (is_group_call)
      NCCL_CALL(ncclGroupEnd());
  }
  bool is_group_call;
};

NCCLCommunicationGroupDef::NCCLCommunicationGroupDef(
  const std::vector<int>& world_ranks, const Stream& stream)
: CommunicationGroupDef(world_ranks, stream) {
  HT_ASSERT(_stream.device().is_cuda())
    << "NCCL communication group must be initialized with "
    << "a stream related with CUDA. Got " << _stream << ".";
  int world_size = GetWorldSize();
  HT_ASSERT(_world_ranks.back() < world_size)
    << "Invalid ranks " << _world_ranks << " for world size " << world_size
    << ".";

  if (_world_ranks.size() == static_cast<size_t>(world_size)) {
    _rank = GetWorldRank();
    _size = world_size;
  } else {
    _rank = GetGroupRank(_world_ranks);
    _size = _world_ranks.size();
    HT_ASSERT(_rank != -1) << "The current rank " << GetWorldRank()
                           << " is not included in the group " << _world_ranks
                           << ".";
  }
  HT_ASSERT(_rank >= 0 && _rank < _size)
    << "Failed to get rank and/or size. "
    << "(Got rank " << _rank << " and size " << _size << ".)";

  CreateNCCLUniqueId(_world_ranks, _unique_id);
  {
    NCCLGroupGuard();
    hetu::cuda::CUDADeviceGuard guard(_stream.device_index());
    NCCL_CALL(ncclCommInitRank(&_comm, _size, _unique_id, _rank));
  }

  _barrier_arr = NDArray::empty({1}, _stream.device(), kInt8);

  HT_LOG_DEBUG << "Initialized NCCL comm group for " << _world_ranks
               << " with stream " << _stream << ".";
}

NCCLCommunicationGroupDef::~NCCLCommunicationGroupDef() {
  Sync();
  NCCL_CALL(ncclCommDestroy(_comm));
}

void NCCLCommunicationGroupDef::Broadcast(NDArray& data, int broadcaster) {
  HT_ASSERT_CUDA_DEVICE(data);
  void* buf = data->raw_data_ptr();
  auto numel = data->numel();
  auto nccl_dtype = to_NCCL_Datatype(data->dtype());
  int root = world_to_group_rank(broadcaster);
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard();
    NCCL_CALL(ncclBcast(buf, numel, nccl_dtype, root, _comm, cuda_stream));
  }
}

void NCCLCommunicationGroupDef::AllReduce(const NDArray& input, NDArray& output,
                                          ReductionType red_type) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_EXCHANGABLE(input, output);
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  auto numel = input->numel();
  auto nccl_dtype = to_NCCL_Datatype(input->dtype());
  auto nccl_red_op = to_NCCL_Op(red_type);
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard();
    NCCL_CALL(ncclAllReduce(send_buf, recv_buf, numel, nccl_dtype, nccl_red_op,
                            _comm, cuda_stream));
  }
}

void NCCLCommunicationGroupDef::AllReduceCoalesce(const NDArrayList& inputs,
                                                  NDArrayList& outputs,
                                                  NDArray contiguous_buffers,
                                                  ReductionType red_type) {
  for (size_t i = 0; i < inputs.size(); i++) {
    HT_ASSERT_CUDA_DEVICE(inputs[i]);
    HT_ASSERT_CUDA_DEVICE(outputs[i]);
    HT_ASSERT_EXCHANGABLE(inputs[i], outputs[i]);
  }
  auto nccl_dtype = to_NCCL_Datatype(inputs[0]->dtype());
  auto nccl_red_op = to_NCCL_Op(red_type);

  if (contiguous_buffers->numel() > 0) {
    void* buffer_ptr = contiguous_buffers->raw_data_ptr();
    int offset = 0;
    {
      CUDAStream cuda_stream(_stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
      NCCLGroupGuard();
      // D2D Copy
      for (size_t i = 0; i < inputs.size(); i++) {
        void* send_buf = inputs[i]->raw_data_ptr();
        int num_bytes = inputs[i]->numel() * to_num_bytes(inputs[i]->dtype());
        CUDA_CALL(cudaMemcpyAsync(buffer_ptr + offset, send_buf, num_bytes,
                                  cudaMemcpyDeviceToDevice, cuda_stream));
        offset += num_bytes;
      }
      NCCL_CALL(ncclAllReduce(buffer_ptr, buffer_ptr,
                              contiguous_buffers->numel(), nccl_dtype,
                              nccl_red_op, _comm, cuda_stream));
      // D2D Copy Back
      offset = 0;
      for (size_t i = 0; i < outputs.size(); i++) {
        void* recv_buf = outputs[i]->raw_data_ptr();
        int num_bytes = outputs[i]->numel() * to_num_bytes(outputs[i]->dtype());
        CUDA_CALL(cudaMemcpyAsync(recv_buf, buffer_ptr + offset, num_bytes,
                                  cudaMemcpyDeviceToDevice, cuda_stream));
        offset += num_bytes;
      }
    }
  } else {
    {
      CUDAStream cuda_stream(_stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
      NCCLGroupGuard();
      for (size_t i = 0; i < inputs.size(); i++) {
        void* send_buf = inputs[i]->raw_data_ptr();
        void* recv_buf = outputs[i]->raw_data_ptr();
        auto numel = inputs[i]->numel();
        NCCL_CALL(ncclAllReduce(send_buf, recv_buf, numel, nccl_dtype,
                                nccl_red_op, _comm, cuda_stream));
      }
    }
  }
}

void NCCLCommunicationGroupDef::AlltoAll(const NDArray& input,
                                         NDArray& output) {
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 7))
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_EXCHANGABLE(input, output);
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  int numel = input->numel() / _size;
  auto nccl_dtype = to_NCCL_Datatype(input->dtype());
  int bytes_per_element = to_num_bytes(input->dtype());
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    // TODO: NCCLGroupGuard(true);
    NCCL_CALL(ncclGroupStart());
    for (int i = 0; i < _size; i++) {
      NCCL_CALL(ncclSend(send_buf + i * numel * bytes_per_element, numel,
                         nccl_dtype, i, _comm, cuda_stream));
      NCCL_CALL(ncclRecv(recv_buf + i * numel * bytes_per_element, numel,
                         nccl_dtype, i, _comm, cuda_stream));
    }
    NCCL_CALL(ncclGroupEnd());
  }
#else
  HT_NOT_IMPLEMENTED << "P2P communication requires NCCL 2.7+";
#endif
}

void NCCLCommunicationGroupDef::Reduce(const NDArray& input, NDArray& output,
                                       int reducer, ReductionType red_type) {
  HT_ASSERT_CUDA_DEVICE(input);
  int root = world_to_group_rank(reducer);
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = nullptr;
  if (_rank == root) {
    HT_ASSERT_CUDA_DEVICE(output);
    HT_ASSERT_EXCHANGABLE(input, output);
    recv_buf = output->raw_data_ptr();
  }
  auto numel = input->numel();
  auto nccl_dtype = to_NCCL_Datatype(input->dtype());
  auto nccl_red_op = to_NCCL_Op(red_type);
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard();
    NCCL_CALL(ncclReduce(send_buf, recv_buf, numel, nccl_dtype, nccl_red_op,
                         root, _comm, cuda_stream));
  }
}

void NCCLCommunicationGroupDef::AllGather(const NDArray& input,
                                          NDArray& output) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_SAME_DTYPE(input, output);
  size_t input_size = input->numel();
  size_t output_size = output->numel();
  HT_ASSERT(input->shape(0) * _size == output->shape(0) &&
            input_size * _size == output_size)
    << "Invalid shapes for AllGather: "
    << "(send) " << input->shape() << " vs. "
    << "(recv) " << output->shape() << ".";
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  auto nccl_dtype = to_NCCL_Datatype(input->dtype());
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard();
    NCCL_CALL(ncclAllGather(send_buf, recv_buf, input_size, nccl_dtype, _comm,
                            cuda_stream));
  }
}

void NCCLCommunicationGroupDef::ReduceScatter(const NDArray& input,
                                              NDArray& output,
                                              ReductionType red_type) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_SAME_DTYPE(input, output);
  size_t input_size = input->numel();
  size_t output_size = output->numel();
  HT_ASSERT(input->shape(0) == output->shape(0) * _size &&
            input_size == output_size * _size)
    << "Invalid shapes for ReduceScatter: "
    << "(send) " << input->shape() << " vs. "
    << "(recv) " << output->shape() << ".";
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  auto nccl_dtype = to_NCCL_Datatype(input->dtype());
  auto nccl_red_op = to_NCCL_Op(red_type);
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard();
    NCCL_CALL(ncclReduceScatter(send_buf, recv_buf, output_size, nccl_dtype,
                                nccl_red_op, _comm, cuda_stream));
  }
}

void NCCLCommunicationGroupDef::Gather(const NDArray& input, NDArray& output,
                                       int gatherer) {
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 7))
  HT_ASSERT_CUDA_DEVICE(input);
  int root = world_to_group_rank(gatherer);
  size_t input_size = input->numel();
  if (_rank == root) {
    HT_ASSERT_CUDA_DEVICE(output);
    HT_ASSERT_SAME_DTYPE(input, output);
    HT_ASSERT(input->shape(0) * _size == output->shape(0) &&
              input_size * _size == output->numel())
      << "Invalid shapes for Gather: "
      << "(send) " << input->shape() << " vs. "
      << "(recv) " << output->shape() << ".";
  }
  auto nccl_dtype = to_NCCL_Datatype(input->dtype());
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard(true);
    if (_rank == root) {
      // splitting on axis zero is in place
      NDArrayList fragments = NDArray::split(output, _size, 0);
      NDArray::copy(input, cuda_stream.stream_id(), fragments[_rank]);
      for (int src = 0; src < _size; src++) {
        if (src != _rank) {
          NCCL_CALL(ncclRecv(fragments[src]->raw_data_ptr(), input_size,
                             nccl_dtype, src, _comm, cuda_stream));
        }
      }
    } else {
      NCCL_CALL(ncclSend(input->raw_data_ptr(), input_size, nccl_dtype, root,
                         _comm, cuda_stream));
    }
  }
#else
  HT_NOT_IMPLEMENTED << "P2P communication requires NCCL 2.7+";
#endif
}

void NCCLCommunicationGroupDef::Scatter(const NDArray& input, NDArray& output,
                                        int scatterer) {
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 7))
  HT_ASSERT_CUDA_DEVICE(output);
  int root = world_to_group_rank(scatterer);
  size_t output_size = output->numel();
  if (_rank == root) {
    HT_ASSERT_CUDA_DEVICE(input);
    HT_ASSERT_SAME_DTYPE(input, output);
    HT_ASSERT(input->shape(0) == output->shape(0) * _size &&
              input->numel() == output_size * _size)
      << "Invalid shapes for Scatter: "
      << "(send) " << input->shape() << " vs. "
      << "(recv) " << output->shape() << ".";
  }
  auto nccl_dtype = to_NCCL_Datatype(output->dtype());
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard(true);
    if (_rank == root) {
      // splitting on axis zero is in place
      NDArrayList fragments = NDArray::split(input, _size, 0);
      for (int dst = 0; dst < _size; dst++) {
        if (dst != _rank) {
          NCCL_CALL(ncclSend(fragments[dst]->raw_data_ptr(), output_size,
                             nccl_dtype, dst, _comm, cuda_stream));
        }
      }
      NDArray::copy(fragments[_rank], cuda_stream.stream_id(), output);
    } else {
      NCCL_CALL(ncclRecv(output->raw_data_ptr(), output_size, nccl_dtype, root,
                         _comm, cuda_stream));
    }
  }
#else
  HT_NOT_IMPLEMENTED << "P2P communication requires NCCL 2.7+";
#endif
}

void NCCLCommunicationGroupDef::Send(const NDArray& data, int receiver) {
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 7))
  int dst = world_to_group_rank(receiver);
  HT_ASSERT(dst != _rank) << "Cannot send to self.";
  size_t size = data->numel();
  if (size == 0)
    return;
  void* send_buf = data->raw_data_ptr();
  auto nccl_dtype = to_NCCL_Datatype(data->dtype());
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard();
    NCCL_CALL(ncclSend(send_buf, size, nccl_dtype, dst, _comm, cuda_stream));
  }
#else
  HT_NOT_IMPLEMENTED << "P2P communication requires NCCL 2.7+";
#endif
}

void NCCLCommunicationGroupDef::Recv(NDArray& data, int sender) {
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 7))
  int src = world_to_group_rank(sender);
  HT_ASSERT(src != _rank) << "Cannot receive from self.";
  size_t size = data->numel();
  if (size == 0)
    return;
  void* recv_buf = data->raw_data_ptr();
  auto nccl_dtype = to_NCCL_Datatype(data->dtype());
  {
    CUDAStream cuda_stream(_stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    NCCLGroupGuard();
    NCCL_CALL(ncclRecv(recv_buf, size, nccl_dtype, src, _comm, cuda_stream));
  }
#else
  HT_NOT_IMPLEMENTED << "P2P communication requires NCCL 2.7+";
#endif
}

Task NCCLCommunicationGroupDef::ISend(const NDArray& data, int receiver) {
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 7))
  int dst = world_to_group_rank(receiver);
  HT_ASSERT(dst != _rank) << "Cannot send to self.";
  size_t size = data->numel();
  if (size == 0)
    return Task();
  void* send_buf = data->raw_data_ptr();
  auto nccl_dtype = to_NCCL_Datatype(data->dtype());
  auto task = Task([send_buf, size, nccl_dtype, dst, this]() {
    {
      CUDAStream cuda_stream(this->_stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
      NCCL_CALL(ncclSend(send_buf, size, nccl_dtype, dst, _comm, cuda_stream));
    }
  });
#else
  HT_NOT_IMPLEMENTED << "P2P communication requires NCCL 2.7+";
  auto task = Task();
#endif
  return task;
}

Task NCCLCommunicationGroupDef::IRecv(NDArray& data, int sender) {
#if defined(NCCL_MAJOR) &&                                                     \
  ((NCCL_MAJOR > 2) ||                                                         \
   (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 7))
  int src = world_to_group_rank(sender);
  HT_ASSERT(src != _rank) << "Cannot receive from self.";
  size_t size = data->numel();
  if (size == 0)
    return Task();
  void* recv_buf = data->raw_data_ptr();
  auto nccl_dtype = to_NCCL_Datatype(data->dtype());

  auto task = Task([recv_buf, size, nccl_dtype, src, this]() {
    {
      CUDAStream cuda_stream(this->_stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
      NCCL_CALL(ncclRecv(recv_buf, size, nccl_dtype, src, _comm, cuda_stream));
    }
  });
#else
  HT_NOT_IMPLEMENTED << "P2P communication requires NCCL 2.7+";
  auto task = Task();
#endif
  return task;
}

void NCCLCommunicationGroupDef::BatchedISendIRecv(
  const std::vector<Task>& tasks) {
  {
    // NCCLGroupGuard(true);
    NCCL_CALL(ncclGroupStart());
    for (auto& task : tasks) {
      task();
    }
    NCCL_CALL(ncclGroupEnd());
  }
}

void NCCLCommunicationGroupDef::Barrier(bool sync) {
  // simulate Barrier via a small-scale AllReduce
  AllReduce(_barrier_arr, _barrier_arr);
  if (sync)
    Sync();
}

void NCCLCommunicationGroupDef::Sync() {
  CUDAStream(_stream).Sync();
}

void NCCLCommunicationGroupDef::CreateNCCLUniqueId(
  const std::vector<int>& world_ranks, ncclUniqueId& id) {
  // Currently we rely on MPI to synchronize the ncclUniqueId.
  // It may be replaced with a distributed memcache in the future.
  auto mpi_comm_group = MPICommunicationGroup::GetOrCreate(world_ranks);
  // Walkaround: communication groups handle ndarrays only
  NDArray id_arr = NDArray::empty({sizeof(ncclUniqueId)}, Device(kCPU), kUInt8);
  int broadcaster = mpi_comm_group->group_to_world_rank(0);
  if (mpi_comm_group->rank() == 0) {
    NCCL_CALL(ncclGetUniqueId(&id));
    memcpy(id_arr->raw_data_ptr(), &id, sizeof(ncclUniqueId));
    mpi_comm_group->Broadcast(id_arr, broadcaster);
    mpi_comm_group->Sync();
  } else {
    mpi_comm_group->Broadcast(id_arr, broadcaster);
    mpi_comm_group->Sync();
    memcpy(&id, id_arr->raw_data_ptr(), sizeof(ncclUniqueId));
  }
}

NCCLCommunicationGroup&
NCCLCommunicationGroup::GetOrCreate(const std::vector<int>& world_ranks,
                                    const Stream& stream) {
  HT_ASSERT(stream.device().is_cuda())
    << "The argument \"stream\" for "
    << "MPICommunicationGroup::GetOrCreate "
    << "must be a CUDA stream. Got " << stream << ".";
  // Note: stream id could be -1, we shall shift it by one when accessing
  int stream_id = static_cast<int>(stream.stream_index());
  int device_id = static_cast<int>(stream.device_index());

  NCCL_Init_Once();

  HT_ASSERT(world_ranks.empty() ||
            CommunicationGroupDef::IsRanksValid(world_ranks))
    << "Invalid world ranks: " << world_ranks;
  auto world_size = static_cast<size_t>(GetWorldSize());

  if (world_ranks.empty() ||
      static_cast<int>(world_ranks.size()) == world_size) {
    if (!worldwide_nccl_comm_groups[stream_id + 1][device_id].is_defined()) {
      std::unique_lock<std::mutex> lock(nccl_create_group_mutex);
      // double check for thread-safety
      if (!worldwide_nccl_comm_groups[stream_id + 1][device_id].is_defined()) {
        std::vector<int> all_world_ranks(world_size);
        std::iota(all_world_ranks.begin(), all_world_ranks.end(), 0);
        worldwide_nccl_comm_groups[stream_id + 1][device_id] =
          NCCLCommunicationGroup(all_world_ranks, stream);
        nccl_comm_groups[stream_id + 1][device_id].insert(
          {all_world_ranks,
           worldwide_nccl_comm_groups[stream_id + 1][device_id]});
      }
    }
    return worldwide_nccl_comm_groups[stream_id + 1][device_id];
  } else {
    HT_ASSERT(GetGroupRank(world_ranks) != -1)
      << "Cannot get comm group " << world_ranks << " on rank "
      << GetWorldRank() << ".";
    auto it = nccl_comm_groups[stream_id + 1][device_id].find(world_ranks);
    if (it == nccl_comm_groups[stream_id + 1][device_id].end()) {
      std::unique_lock<std::mutex> lock(nccl_create_group_mutex);
      // double check for thread-safety
      it = nccl_comm_groups[stream_id + 1][device_id].find(world_ranks);
      if (it == nccl_comm_groups[stream_id + 1][device_id].end()) {
        NCCLCommunicationGroup comm_group(world_ranks, stream);
        auto insertion = nccl_comm_groups[stream_id + 1][device_id].insert(
          {comm_group->world_ranks(), comm_group});
        HT_ASSERT(insertion.second)
          << "Failed to insert NCCLCommunicationGroup for ranks "
          << comm_group->world_ranks() << ".";
        it = insertion.first;
      }
    }
    return it->second;
  }
}

NCCLCommunicationGroup&
NCCLCommunicationGroup::GetOrCreateWorldwide(const Stream& stream) {
  HT_ASSERT(stream.device().is_cuda())
    << "The argument \"stream\" for "
    << "MPICommunicationGroup::GetOrCreate "
    << "must be a CUDA stream. Got " << stream << ".";
  // Note: stream id could be -1, we shall shift it by one when accessing
  int stream_id = static_cast<int>(stream.stream_index());
  int device_id = static_cast<int>(stream.device_index());

  if (worldwide_nccl_comm_groups[stream_id + 1][device_id].is_defined())
    return worldwide_nccl_comm_groups[stream_id + 1][device_id];
  else
    return GetOrCreate({}, stream);
}

} // namespace comm
} // namespace impl
} // namespace hetu
