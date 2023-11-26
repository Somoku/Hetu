#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/core/symbol.h"

namespace hetu {
namespace graph {

class SliceOpImpl;
class SliceOp;
class SliceGradientOpImpl;
class SliceGradientOp;

class SliceOpImpl : public OpInterface {
 public:
  // symbolic shape constructor, _ori_shape is crucial
  SliceOpImpl(const SyShape& begin_pos, const SyShape& output_shape, const int64_t& padding_axis = -1, bool inplace = false)
  : OpInterface(quote(SliceOp)),
    _begin_pos(begin_pos),
    _output_shape(output_shape),
    _padding_axis(padding_axis),
    _inplace(inplace),
    _symbolic(true) {
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << " splice op begin_pos = " << get_begin_pos();
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << " splice op begin_pos = " << get_output_shape();
  }
  // fixed shape constructor
  SliceOpImpl(const HTShape& begin_pos, const HTShape& output_shape, const int64_t& padding_axis = -1, bool inplace = false)
  : OpInterface(quote(SliceOp)),
    _begin_pos(begin_pos.begin(), begin_pos.end()),
    _output_shape(output_shape.begin(), output_shape.end()),
    _padding_axis(padding_axis),
    _inplace(inplace),
    _symbolic(false) {
  }
  
  uint64_t op_indicator() const noexcept override {
    return SLICE_OP;
  }

  HTShape get_begin_pos() const {
    return get_HTShape_from_SyShape(_begin_pos);
  }

  HTShape get_output_shape() const {
    return get_HTShape_from_SyShape(_output_shape);
  }

  const SyShape& get_symbolic_begin_pos() const {
    return _begin_pos;
  }

  const SyShape& get_symbolic_output_shape() const {
    return _output_shape;
  }

  int64_t get_padding_axis() const {
    return _padding_axis;
  }

  bool inplace() const {
    return _inplace;
  }

  bool symbolic() const {
    return _symbolic;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT(inputs[0]->ndim() == _begin_pos.size() && 
              inputs[0]->ndim() == _output_shape.size());
    int len = _begin_pos.size();
    for (int i = 0; i < len; ++i) {
      HT_ASSERT(_begin_pos[i]->get_val() >= 0 && (_output_shape[i]->get_val() > 0 || _output_shape[i]->get_val() == -1));
      HT_ASSERT(_begin_pos[i]->get_val() + _output_shape[i]->get_val() <= inputs[0]->shape(i));
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(get_output_shape())
                                           .set_device(inputs[0]->device());
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << " splice op DoInferMeta() finished";
    return {output_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  // deprecated: only used in gpt inference, before symbolic shape is realized
  HTShapeList DoInferDynamicShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  SyShape _begin_pos;
  SyShape _output_shape;
  bool _symbolic;

  int64_t _padding_axis;

  bool _inplace;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SliceOpImpl&>(rhs);
      return (get_begin_pos() == rhs_.get_begin_pos()
              && get_output_shape() == rhs_.get_output_shape()
              && inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeSliceOp(Tensor input, const HTShape& begin_pos, const HTShape& output_shape,
                   OpMeta op_meta = OpMeta());
Tensor MakeSliceInplaceOp(Tensor input, const HTShape& begin_pos, const HTShape& output_shape,
                          OpMeta op_meta = OpMeta());

class SliceGradientOpImpl : public OpInterface {

 public:
  // symbolic shape constructor
  SliceGradientOpImpl(const SyShape& begin_pos,
                      const SyShape& output_shape)
  : OpInterface(quote(SliceGradientOp)),
    _begin_pos(begin_pos),
    _output_shape(output_shape),
    _symbolic(true) {
  }
  // fixed shape constructor
  SliceGradientOpImpl(const HTShape& begin_pos,
                      const HTShape& output_shape)
  : OpInterface(quote(SliceGradientOp)),
    _begin_pos(begin_pos.begin(), begin_pos.end()),
    _output_shape(output_shape.begin(), output_shape.end()),
    _symbolic(false) {
  }

  HTShape get_begin_pos() const {
    return get_HTShape_from_SyShape(_begin_pos);
  }

  HTShape get_output_shape() const {
    return get_HTShape_from_SyShape(_output_shape);
  }

  const SyShape& get_symbolic_begin_pos() const {
    return _begin_pos;
  }

  const SyShape& get_symbolic_output_shape() const {
    return _output_shape;
  }

  bool symbolic() const {
    return _symbolic;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[2]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  SyShape _begin_pos;
  SyShape _output_shape;
  bool _symbolic;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SliceOpImpl&>(rhs);
      return (get_begin_pos() == rhs_.get_begin_pos()
              && get_output_shape() == rhs_.get_output_shape());
    }
    return false;
  }
};

Tensor MakeSliceGradientOp(Tensor grad_output, Tensor ori_output, Tensor ori_input,
                           const HTShape& begin_pos, const HTShape& output_shape,
                           OpMeta op_meta = OpMeta());

Tensor MakeSliceGradientOp(Tensor grad_output, Tensor ori_output, Tensor ori_input,
                           const SyShape& begin_pos, const SyShape& output_shape,
                           OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
