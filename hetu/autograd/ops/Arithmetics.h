#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class AddElewiseOpDef;
class AddElewiseOp;
class AddByConstOpDef;
class AddByConstOp;

class SubElewiseOpDef;
class SubElewiseOp;
class SubByConstOpDef;
class SubByConstOp;
class SubFromConstOpDef;
class SubFromConstOp;

class OppositeOpDef;
class OppositeOp;

class MulElewiseOpDef;
class MulElewiseOp;
class MulByConstOpDef;
class MulByConstOp;

class DivElewiseOpDef;
class DivElewiseOp;
class DivByConstOpDef;
class DivByConstOp;
class DivFromConstOpDef;
class DivFromConstOp;

class ReciprocalOpDef;
class ReciprocalOp;

class AddElewiseOpDef : public OperatorDef {
 private:
  friend class AddElewiseOp;
  struct constrcutor_access_key {};

 public:
  AddElewiseOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AddElewiseOp), {a, b}, op_meta) {
    HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class AddElewiseOp final : public OpWrapper<AddElewiseOpDef> {
 public:
  AddElewiseOp(Tensor a, Tensor b, const OpMeta& op_meta = OpMeta())
  : OpWrapper<AddElewiseOpDef>(make_ptr<AddElewiseOpDef>(
      AddElewiseOpDef::constrcutor_access_key(), a, b, op_meta)) {}
};

class AddByConstOpDef : public OperatorDef {
 private:
  friend class AddByConstOp;
  struct constrcutor_access_key {};

 public:
  AddByConstOpDef(const constrcutor_access_key&, Tensor input, double value,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AddByConstOp), {input}, op_meta), _value(value) {
    AddOutput(input->meta());
  }

  inline double const_value() const {
    return _value;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _value;
};

class AddByConstOp final : public OpWrapper<AddByConstOpDef> {
 public:
  AddByConstOp(Tensor input, double value, const OpMeta& op_meta = OpMeta())
  : OpWrapper<AddByConstOpDef>(make_ptr<AddByConstOpDef>(
      AddByConstOpDef::constrcutor_access_key(), input, value, op_meta)) {}
  
  AddByConstOp(double value, Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<AddByConstOpDef>(make_ptr<AddByConstOpDef>(
      AddByConstOpDef::constrcutor_access_key(), input, value, op_meta)) {}
};

class SubElewiseOpDef : public OperatorDef {
 private:
  friend class SubElewiseOp;
  struct constrcutor_access_key {};

 public:
  SubElewiseOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SubElewiseOp), {a, b}, op_meta) {
    HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class SubElewiseOp final : public OpWrapper<SubElewiseOpDef> {
 public:
  SubElewiseOp(Tensor a, Tensor b, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SubElewiseOpDef>(make_ptr<SubElewiseOpDef>(
      SubElewiseOpDef::constrcutor_access_key(), a, b, op_meta)) {}
};

class SubByConstOpDef : public OperatorDef {
 private:
  friend class SubByConstOp;
  struct constrcutor_access_key {};

 public:
  SubByConstOpDef(const constrcutor_access_key&, Tensor input, double value,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SubByConstOp), {input}, op_meta), _value(value) {
    AddOutput(input->meta());
  }

  inline double const_value() const {
    return _value;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _value;
};

class SubByConstOp final : public OpWrapper<SubByConstOpDef> {
 public:
  SubByConstOp(Tensor input, double value, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SubByConstOpDef>(make_ptr<SubByConstOpDef>(
      SubByConstOpDef::constrcutor_access_key(), input, value, op_meta)) {}
};

class SubFromConstOpDef : public OperatorDef {
 private:
  friend class SubFromConstOp;
  struct constrcutor_access_key {};

 public:
  SubFromConstOpDef(const constrcutor_access_key&, double value, Tensor input,
                    const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SubFromConstOp), {input}, op_meta), _value(value) {
    AddOutput(input->meta());
  }

  inline double const_value() const {
    return _value;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _value;
};

class SubFromConstOp final : public OpWrapper<SubFromConstOpDef> {
 public:
  SubFromConstOp(double value, Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SubFromConstOpDef>(make_ptr<SubFromConstOpDef>(
      SubFromConstOpDef::constrcutor_access_key(), value, input, op_meta)) {}
};

class NegateOpDef : public OperatorDef {
 private:
  friend class NegateOp;
  struct constrcutor_access_key {};

 public:
  NegateOpDef(const constrcutor_access_key&, Tensor input,
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(NegateOp), {input}, op_meta) {
    AddOutput(input->meta());
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class NegateOp final : public OpWrapper<NegateOpDef> {
 public:
  NegateOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<NegateOpDef>(make_ptr<NegateOpDef>(
      NegateOpDef::constrcutor_access_key(), input, op_meta)) {}
};

class MulElewiseOpDef : public OperatorDef {
 private:
  friend class MulElewiseOp;
  struct constrcutor_access_key {};

 public:
  MulElewiseOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MulElewiseOp), {a, b}, op_meta) {
    HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class MulElewiseOp final : public OpWrapper<MulElewiseOpDef> {
 public:
  MulElewiseOp(Tensor a, Tensor b, const OpMeta& op_meta = OpMeta())
  : OpWrapper<MulElewiseOpDef>(make_ptr<MulElewiseOpDef>(
      MulElewiseOpDef::constrcutor_access_key(), a, b, op_meta)) {}
};

class MulByConstOpDef : public OperatorDef {
 private:
  friend class MulByConstOp;
  struct constrcutor_access_key {};

 public:
  MulByConstOpDef(const constrcutor_access_key&, Tensor input, double value,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MulByConstOp), {input}, op_meta), _value(value) {
    AddOutput(input->meta());
  }

  inline double const_value() const {
    return _value;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _value;
};

class MulByConstOp final : public OpWrapper<MulByConstOpDef> {
 public:
  MulByConstOp(Tensor input, double value, const OpMeta& op_meta = OpMeta())
  : OpWrapper<MulByConstOpDef>(make_ptr<MulByConstOpDef>(
      MulByConstOpDef::constrcutor_access_key(), input, value, op_meta)) {}
  
  MulByConstOp(double value, Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<MulByConstOpDef>(make_ptr<MulByConstOpDef>(
      MulByConstOpDef::constrcutor_access_key(), input, value, op_meta)) {}
};

class DivElewiseOpDef : public OperatorDef {
 private:
  friend class DivElewiseOp;
  struct constrcutor_access_key {};

 public:
  DivElewiseOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(DivElewiseOp), {a, b}, op_meta) {
    HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class DivElewiseOp final : public OpWrapper<DivElewiseOpDef> {
 public:
  DivElewiseOp(Tensor a, Tensor b, const OpMeta& op_meta = OpMeta())
  : OpWrapper<DivElewiseOpDef>(make_ptr<DivElewiseOpDef>(
      DivElewiseOpDef::constrcutor_access_key(), a, b, op_meta)) {}
};

class DivByConstOpDef : public OperatorDef {
 private:
  friend class DivByConstOp;
  struct constrcutor_access_key {};

 public:
  DivByConstOpDef(const constrcutor_access_key&, Tensor input, double value,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(DivByConstOp), {input}, op_meta), _value(value) {
    AddOutput(input->meta());
  }

  inline double const_value() const {
    return _value;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _value;
};

class DivByConstOp final : public OpWrapper<DivByConstOpDef> {
 public:
  DivByConstOp(Tensor input, double value, const OpMeta& op_meta = OpMeta())
  : OpWrapper<DivByConstOpDef>(make_ptr<DivByConstOpDef>(
      DivByConstOpDef::constrcutor_access_key(), input, value, op_meta)) {}
};

class DivFromConstOpDef : public OperatorDef {
 private:
  friend class DivFromConstOp;
  struct constrcutor_access_key {};

 public:
  DivFromConstOpDef(const constrcutor_access_key&, double value, Tensor input,
                    const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(DivFromConstOp), {input}, op_meta), _value(value) {
    AddOutput(input->meta());
  }

  inline double const_value() const {
    return _value;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _value;
};

class DivFromConstOp final : public OpWrapper<DivFromConstOpDef> {
 public:
  DivFromConstOp(double value, Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<DivFromConstOpDef>(make_ptr<DivFromConstOpDef>(
      DivFromConstOpDef::constrcutor_access_key(), value, input, op_meta)) {}
};

class ReciprocalOpDef : public OperatorDef {
 private:
  friend class ReciprocalOp;
  struct constrcutor_access_key {};

 public:
  ReciprocalOpDef(const constrcutor_access_key&, Tensor input,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReciprocalOp), {input}, op_meta) {
    AddOutput(input->meta());
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class ReciprocalOp final : public OpWrapper<ReciprocalOpDef> {
 public:
  ReciprocalOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReciprocalOpDef>(make_ptr<ReciprocalOpDef>(
      ReciprocalOpDef::constrcutor_access_key(), input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu