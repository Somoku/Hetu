#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void relu_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = (input[idx] < 0) ? 0 : input[idx];
  }
}

template <typename spec_t>
void relu_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                       size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = (input[idx] < 0) ? 0 : output_grad[idx];
  }
}

void ReluCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ReluCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
          auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
          auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

          auto Relu_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                              dnnl::algorithm::eltwise_relu, mat_md, mat_md, float(0.0), float(0.0));
          auto Relu = dnnl::eltwise_forward(Relu_pd);

          std::unordered_map<int, dnnl::memory> relu_args;
          relu_args.insert({DNNL_ARG_SRC, src_mem});
          relu_args.insert({DNNL_ARG_DST, dst_mem});     

          dnnl::stream engine_stream(eng);
          Relu.execute(engine_stream, relu_args);
          engine_stream.wait();
        },"Relu");
      //cpu_stream.Sync();
    });
}

void ReluGradientCpu(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());
  dnnl::stream engine_stream(eng);
  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ReluGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output_grad, input_grad]() {
        dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
          auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
          auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
          auto g_dst_mem = dnnl::memory(mat_md, eng, output_grad->data_ptr<spec_t>());
          auto g_src_mem = dnnl::memory(mat_md, eng, input_grad->data_ptr<spec_t>());

          auto Relu_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                              dnnl::algorithm::eltwise_relu, mat_md, mat_md, float(0.0), float(0.0));

          auto Relu_bwd_pd = dnnl::eltwise_backward::primitive_desc(eng,
                dnnl::algorithm::eltwise_relu, mat_md, mat_md,
                mat_md, float(0.0), float(0.0), Relu_pd);
          
          auto Relu_bwd = dnnl::eltwise_backward(Relu_bwd_pd);

          std::unordered_map<int, dnnl::memory> relu_args;
          relu_args.insert({DNNL_ARG_SRC, src_mem});
          relu_args.insert({DNNL_ARG_DIFF_DST, g_dst_mem});      
          relu_args.insert({DNNL_ARG_DIFF_SRC, g_src_mem});  
        
          dnnl::stream engine_stream(eng);
          Relu_bwd.execute(engine_stream, relu_args);
          engine_stream.wait();
        },"ReluGradient");
      //cpu_stream.Sync();

    });
}

} // namespace impl
} // namespace hetu
