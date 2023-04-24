#include "hetu/autograd/ops/EmbeddingLookup.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void EmbeddingLookupOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::EmbeddingLookup, inputs.at(0),
                                  inputs.at(1), outputs.at(0), stream());
}

TensorList EmbeddingLookupOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    EmbeddingLookupGradientOp(grad_outputs.at(0), _inputs[1], _outputs[0],
                              grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input, Tensor()};
}

HTShapeList
EmbeddingLookupOpDef::DoInferShape(const HTShapeList& input_shapes) {
  set_grad_embed(input_shapes[0]);
  HTShape output_shape = input_shapes[1];
  output_shape.emplace_back(input_shapes[0][1]);
  return {output_shape};
}

void EmbeddingLookupOpDef::DeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  DistributedStates ds_id = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_id.is_valid() && 
            ds_input.get_device_num() == ds_id.get_device_num()) 
    << "EmbeddingLookupOpDef: distributed states for input and id must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_id.get_dim(-2) == 1) 
    << "Tensor input and id shouldn't be partial";
  HT_ASSERT(ds_input.check_pure_duplicate())
    << "Tensor input(embedding table) cannot be splited!";
  _outputs[0]->set_distributed_states(ds_id);
}

void EmbeddingLookupGradientOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::EmbeddingLookupGradient,
    inputs.at(0), inputs.at(1), outputs.at(0), stream());
}

HTShapeList
EmbeddingLookupGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  EmbeddingLookupOp& input_ptr =
    reinterpret_cast<EmbeddingLookupOp&>(_inputs[2]->producer());
  if (input_ptr) {
    set_embed_shape(input_ptr->get_grad_embed());
  }
  return {get_embed_shape()};
}

void EmbeddingLookupGradientOpDef::DeduceStates() {
  DistributedStates ds_grad_output = _inputs[0]->get_distributed_states();
  DistributedStates ds_id = _inputs[1]->get_distributed_states();
  DistributedStates ds_ori_output = _inputs[1]->get_distributed_states();
  int32_t device_num = ds_grad_output.get_device_num();
  HT_ASSERT(ds_grad_output.is_valid() && ds_id.is_valid() && ds_ori_output.is_valid()) 
    << "EmbeddingLookupGradientOpDef: distributed states for grad_output and id and ori_output must be valid!";
  HT_ASSERT(ds_grad_output.get_dim(-2) == 1 && ds_id.get_dim(-2) == 1 && ds_ori_output.get_dim(-2) == 1) 
    << "Tensor grad_output and id and ori_output shouldn't be partial";
  HT_ASSERT(ds_grad_output.check_equal(ds_id) && ds_id.check_equal(ds_ori_output))
    << "Distributed states for tensor grad_output and id and ori_output must be equal!";
  
  _outputs[0]->set_distributed_states({device_num, {{-2, device_num}, {-1, 1}}, {-2}}); // pure partial
} 

} // namespace autograd
} // namespace hetu
