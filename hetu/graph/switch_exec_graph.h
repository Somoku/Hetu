#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/dtype.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/impl/communication/comm_group.h"
#include <nccl.h>

namespace hetu {
namespace graph {

using ExecGraphPair = std::pair<std::shared_ptr<ExecutableGraph>, std::shared_ptr<ExecutableGraph>>;
using DevicePair2Val = std::unordered_map<std::pair<Device, Device>, size_t>;
using Device2DTListPairMap = std::unordered_map<Device, std::pair<std::vector<Device>, std::vector<Tensor>>>;

std::ostream& operator<<(std::ostream& os, const SwitchExecGraph& switcher);

enum class SWITCH_ALGORITHM_LEVEL : int8_t {
  FCFS = 0,
  ROUND_ROBIN,
  GREEDY,
};

enum class SWITCH_PROFILE_LEVEL : int8_t {
  TRACE = 0,
  NVLINK,
  TIME,
  INFO,
};

class ParamBuffer {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamBuffer(const TensorList& tensor_list = {}) {
      _tensor_list = tensor_list;
      for (const auto& tensor : tensor_list) {
        if (_dtype == DataType::UNDETERMINED) {
          _dtype = tensor->dtype();
        } else {
          HT_ASSERT(_dtype == tensor->dtype())
            << "ParamBuffer dtype should be consistent";
        }
        HT_ASSERT(_tensor_offset_mapping.find(tensor->id()) == _tensor_offset_mapping.end())
          << "tensor should be unique in the ParamBuffer"
          << ", but there are multiple " << tensor;
        _tensor_offset_mapping[tensor->id()] = _buffer_size;
        _buffer_size += tensor->numel() * DataType2Size(tensor->dtype());
      }
    }

    bool HasTensor(const Tensor& tensor) {
      return _tensor_offset_mapping.find(tensor->id()) != _tensor_offset_mapping.end();
    }

    void AddTensor(const Tensor& tensor) {
      HT_ASSERT(_tensor_offset_mapping.find(tensor->id()) == _tensor_offset_mapping.end())
        << "tensor should be unique in the ParamBuffer"
        << ", but there are multiple " << tensor;
      if (_dtype == DataType::UNDETERMINED) {
          _dtype = tensor->dtype();
      } else {
        HT_ASSERT(_dtype == tensor->dtype())
          << "ParamBuffer dtype should be consistent";
      }
      _tensor_list.push_back(tensor);
      _tensor_offset_mapping[tensor->id()] = _buffer_size;
      _buffer_size += tensor->numel() * DataType2Size(tensor->dtype());
    }

    bool IsAllocated() const {
      return _is_allocated;
    }

    DataType dtype() const {
      return _dtype;
    }

    Stream stream() const {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer in advance";
      return _stream;
    }

    void* AsRawPtr() {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer in advance";
      return _raw_ptr;
    }

    std::shared_ptr<NDArrayStorage> AsStorage() {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer in advance";
      return _storage;
    }

    NDArray AsNDArray() {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer in advance";
      // 设置成一个一维的NDArray
      auto meta = NDArrayMeta().set_dtype(_dtype)
                               .set_device(_stream.device())
                               .set_shape({_buffer_size / DataType2Size(_dtype)});
      return NDArray(meta, _storage);
    }

    const size_t GetByteOffest(const Tensor& tensor) const {
      auto it = _tensor_offset_mapping.find(tensor->id());
      HT_ASSERT(it != _tensor_offset_mapping.end())
        << "Can't find tensor " << tensor << " in the ParamBuffer";
      return it->second;
    }

    const size_t GetElementOffest(const Tensor& tensor) const {
      auto it = _tensor_offset_mapping.find(tensor->id());
      HT_ASSERT(it != _tensor_offset_mapping.end())
        << "Can't find tensor " << tensor << " in the ParamBuffer";
      return it->second / DataType2Size(_dtype);
    }

    const size_t GetTensorOffest(const Tensor& tensor) const {
      auto len = _tensor_list.size();
      for (size_t i = 0; i < len; ++i) {
        if (_tensor_list[i]->id() == tensor->id()) {
          return i;
        }
      }
      HT_RUNTIME_ERROR << "Can't find tensor " << tensor << " in the ParamBuffer";
    }

    void Alloc(const Stream& stream);
    void Free();

  protected:
    TensorList _tensor_list;
    bool _is_allocated{false};
    DataType _dtype{DataType::UNDETERMINED};
    std::unordered_map<TensorId, size_t> _tensor_offset_mapping; // 这里的offset是字节数
    Stream _stream; // 记录在哪个stream上alloc的
    std::shared_ptr<NDArrayStorage> _storage; // high-level cuda mempool api
    void* _raw_ptr; // low-level nccl mem api
    size_t _buffer_size{0};
};

class ParamSlice {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamSlice(const TensorName& block_name, 
               const std::vector<int32_t>& slice_num,
               SwitchExecGraph* switcher): 
      _block_name(block_name),
      _slice_num(slice_num),
      _switcher(switcher) {
    }

    const std::string name() const {
      std::string suffix = "_slice";
      for (const auto& x : _slice_num) {
        suffix += "_" + std::to_string(x);
      }
      return _block_name + suffix;
    }

    const Tensor& OwnedSliceInst(size_t idx) const {
      HT_ASSERT(idx < _owned_slice_instances.size())
        << "idx out of range";
      return _owned_slice_instances[0];
    }

    const Tensor& NeededSliceInst(size_t idx) const {
      HT_ASSERT(idx < _needed_slice_instances.size())
        << "idx out of range";
      return _needed_slice_instances[0];
    }

    void AddOwnedSliceInst(const Device& device, const Tensor& tensor);

    void AddNeededSliceInst(const Device& device, const Tensor& tensor);

    void ParamSliceComm(Device2DTListPairMap& send_mapping, Device2DTListPairMap& recv_mapping);

  protected:
    TensorName _block_name;
    // 在一个block中的slice编号
    // 例如block有3*2*5个slice
    // 那么一个合法的_slice_num就是{2,1,3}
    std::vector<int32_t> _slice_num; 
    SwitchExecGraph* _switcher;

    TensorList _owned_slice_instances;
    TensorList _needed_slice_instances;
    std::vector<Device> _owned_devices;
    std::vector<Device> _needed_devices;

    // level 0, round-robin alg
    size_t _round_robin = 0;
};

class ParamBlock {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamBlock(const TensorName& block_name, 
               const std::vector<int32_t>& block_shape,
               SwitchExecGraph* switcher):
      _block_name(block_name), 
      _block_shape(block_shape),
      _switcher(switcher) {
    }

    const std::string name() const {
      return _block_name;
    }

    const std::vector<int32_t>& BlockShape() const {
      return _block_shape;
    }

    std::vector<std::shared_ptr<ParamSlice>>& GetParamSlices() {
      return _param_slices;
    }

    std::shared_ptr<ParamSlice>& GetParamSlice(const std::vector<int32_t>& slice_num) {
      size_t size = slice_num.size();
      HT_ASSERT(size == _block_shape.size() && size > 0) 
        << "size should be equal to block shape size and non-zero";
      size_t cnt = 1, sum = 0;
      for (int32_t i = size - 1; i >= 0; --i) {
        HT_ASSERT(slice_num[i] < _block_shape[i]) 
          << "slice_num dim " << i << " is out of range"
          << ", slice_num = " << slice_num << " and block_shape = " << _block_shape;
        sum += slice_num[i] * cnt;
        cnt *= _block_shape[i];
      }
      HT_ASSERT(sum < _param_slices.size()) 
        << "slice is out of range";
      return _param_slices[sum];
    }

    void ParamBlockComm(Device2DTListPairMap& send_mapping, Device2DTListPairMap& recv_mapping);

  protected:
    std::vector<int32_t> _block_shape;
    TensorName _block_name;
    SwitchExecGraph* _switcher;

    std::vector<std::shared_ptr<ParamSlice>> _param_slices; 
};

class SwitchExecGraph {
  protected:
    friend class DefineAndRunGraph;
    friend class ExecutableGraph;
    friend class ParamBlock;
    friend class ParamSlice;

  public:
    SwitchExecGraph(DefineAndRunGraph* define_graph, size_t plan_before, size_t plan_after) {
      _define_graph = define_graph;
      _define_graph_params = define_graph->params();
      _switch_plan_pair = std::make_pair(plan_before, plan_after);
      _switch_graph_pair = std::make_pair(define_graph->GetPlan(plan_before).exec_graph, 
                                          define_graph->GetPlan(plan_after).exec_graph);
      /*
      _switch_graph_params_pair = std::make_pair(plan_before.exec_graph->params()),
                                                 plan_after.exec_graph->params());
      */
      char* algorithm_env = std::getenv("HETU_SWITCH_ALGORITHM");
      if (algorithm_env != nullptr) {
        std::string algorithm_level = algorithm_env;
        std::transform(algorithm_level.begin(), algorithm_level.end(), algorithm_level.begin(), ::toupper);
        if (algorithm_level == "GREEDY") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::GREEDY;
        } else if (algorithm_level == "ROUND_ROBIN") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::ROUND_ROBIN;
        } else if (algorithm_level == "FCFS") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::FCFS;
        } else {
          HT_RUNTIME_ERROR << "NotImplementedError";
        }
      }
      char* profile_env = std::getenv("HETU_SWITCH_PROFILE");
      if (profile_env != nullptr) {
        std::string profile_level = profile_env;
        std::transform(profile_level.begin(), profile_level.end(), profile_level.begin(), ::toupper);
        if (profile_level == "INFO") {
          _profile_level = SWITCH_PROFILE_LEVEL::INFO;
        } else if (profile_level == "TIME") {
          _profile_level = SWITCH_PROFILE_LEVEL::TIME;
        } else if (profile_level == "NVLINK") {
          _profile_level = SWITCH_PROFILE_LEVEL::NVLINK;
        } else if (profile_level == "TRACE") {
          _profile_level = SWITCH_PROFILE_LEVEL::TRACE;
        } else {
          HT_RUNTIME_ERROR << "NotImplementedError";
        }
      }
    }

    void RecordTensorInfo(const Tensor& tensor, const std::string& info) {
      if (_info_mapping.find(tensor->id()) != _info_mapping.end()) {
        HT_ASSERT(_info_mapping[tensor->id()] == info)
          << "tensor " << tensor << " is already existed in the info mapping of the switcher"
          << ", the new info is conflicted with the existed info";
      }
      _info_mapping[tensor->id()] = info;
    }

    const ExecGraphPair& SwitchGraphPair() const {
      return _switch_graph_pair;
    }
    
    void SwitchParams();

  protected:
    void CreateParamBlock(ParamBlock& block,
                          std::vector<int32_t>& slice_num, 
                          const TensorName& block_name,
                          int32_t dim);

    void MakeAllParamSlices(const Tensor& param, ParamBlock& block, 
                            const Device& device, const DeviceGroup& group,
                            std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                            const std::unordered_map<int32_t, int32_t>& state,
                            const std::vector<int32_t>& multiple, int32_t dim);

    Tensor MergeAllParamSlices(const Tensor& param, ParamBlock& block, 
                               const Device& device, const DeviceGroup& group,
                               std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                               const std::unordered_map<int32_t, int32_t>& state,
                               const std::vector<int32_t>& multiple, int32_t dim);
    
    void MakeCommGraph();

    void BufferBatchedIsendIrecvExec(const Operator& op);
    
    void BufferBatchedIsendIrecv(const Operator& op,
                                 Tensor2NDArrayMap& tensor2data,
                                 Tensor2IntMap& tensor2degrees);

    void SwitchParam(const DistributedStates& src_ds, const DeviceGroup& src_group,
                     const DistributedStates& dst_ds, const DeviceGroup& dst_group,
                     const Tensor& comm_input, const Tensor& after_param);

    void ProfileRunningDetails();
    void ProfileNvlinkStart();
    void ProfileNvlinkEnd();

  protected:
    // basic attributes
    DefineAndRunGraph* _define_graph; // 定义图
    TensorCRefList _define_graph_params; // 定义图的params tensor
    std::pair<size_t, size_t> _switch_plan_pair; // 需要切换的两个exec graph plan的编号
    ExecGraphPair _switch_graph_pair; // 需要切换的两个exec graph的指针

    // comm graph related
    std::shared_ptr<ExecutableGraph> _comm_graph; // 为了应对切换过程中的复杂通信情况而建立的执行图 
    std::unordered_set<Device> _comm_set; // 参与通信图的所有devices
    OpRefList _comm_topo; // 该图的local_topo
    Tensor2ShapeMap _comm_shape_plan; // 该图所有tensor的运行时的shape
    FeedDict _comm_feed_dict; // 该图的输入
    Tensor2TensorMap _comm_feed_dict_mapping; // 该图的输入到before graph的映射
    TensorList _comm_results; // 该图通信的结果，与_define_graph_params一一对应
    Tensor2TensorMap _comm_results_mapping; // 该图的输出到after graph的映射
    TensorList _dummy_links; // 只有send没有recv时BatchedISendIRecvOp的输出dummy tensor需要被记录并在之后fetch

    // comm plan related
    SWITCH_ALGORITHM_LEVEL _algorithm_level = SWITCH_ALGORITHM_LEVEL::GREEDY; // 采用的算法
    DevicePair2Val _p2p_val_mapping; // 记录了每两个device之间的p2p通信通路的总value（目前value是指次数）
    Device2DTListPairMap _send_mapping; // 记录了每个device要send的(device, tensor)的pair
    Device2DTListPairMap _recv_mapping; // 记录了每个device要recv的(device, placeholder的tensor（之后会替换）)的pair
    std::vector<std::shared_ptr<ParamBlock>> _param_blocks; // 记录了graph所包含的所有的抽象ParamBlock

    // memory optimization related
    std::unordered_map<Device, std::shared_ptr<ParamBuffer>> _send_buffers; // 记录通信时给每个device聚合发送时所用的buffer
    std::unordered_map<Device, std::shared_ptr<ParamBuffer>> _recv_buffers; // 记录通信时从每个device聚合接收时所用的buffer
    std::vector<std::unique_ptr<hetu::impl::CUDAEvent>> _buffer_transfer_events; // 记录将原先的param构成一长条buffer的events
    bool _use_concat_buffer = true; // 是否对concat算子的输出做一个buffer
    std::shared_ptr<ParamBuffer> _concat_buffer; // 通信后从_recv_buffers里concat形成的所有param构成的buffer

    // profile related
    SWITCH_PROFILE_LEVEL _profile_level = SWITCH_PROFILE_LEVEL::INFO; // profile的粒度（开启后会进行同步，因此端到端速度可能会变慢）
    Tensor2StringMap _info_mapping; // 记录tensor到相应param slice名称的映射（只针对BatchedISendIRecvOp的send部分的tensor）
    unsigned int _device_count = 0; // 有多少个GPU
    std::vector<unsigned int> _nvlink_counts; // 每个GPU有多少条NVLink
    std::vector<std::vector<unsigned long long>> _nvlink_txs; // 记录执行通信代码片段前每个GPU每条NVLink的Raw Tx
    std::vector<std::vector<unsigned long long>> _nvlink_rxs; // 记录执行通信代码片段前每个GPU每条NVLink的Raw Rx
};

} // namespace graph
} // namespace hetu