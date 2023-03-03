#pragma once

#include "hetu/core/stream.h"

namespace hetu {
namespace impl {

/******************************************************
 * External declaration of kernels
 ******************************************************/

#define DECLARE_KERNEL(KERNEL, DEVICE, ...)                                    \
  extern void KERNEL##DEVICE(__VA_ARGS__)

#define DECLARE_KERNEL_CPU(KERNEL, ...) DECLARE_KERNEL(KERNEL, Cpu, __VA_ARGS__)
#define DECLARE_KERNEL_CUDA(KERNEL, ...)                                       \
  DECLARE_KERNEL(KERNEL, Cuda, __VA_ARGS__)

#define DECLARE_KERNEL_CPU_AND_CUDA(KERNEL, ...)                               \
  DECLARE_KERNEL_CPU(KERNEL, __VA_ARGS__);                                     \
  DECLARE_KERNEL_CUDA(KERNEL, __VA_ARGS__)

DECLARE_KERNEL_CPU_AND_CUDA(Abs, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ArraySet, NDArray&, double, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AddConst, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AddElewise, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AvgPool, const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AvgPoolGradient, const NDArray&, const NDArray&,
                            const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CUDA(BatchMatMul, const NDArray& a, bool trans_a,
                    const NDArray& b, bool trans_b, NDArray& output,
                    const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BatchNorm, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, double, double, NDArray&,
                            NDArray&, NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BatchNormGradient, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, NDArray&, NDArray&,
                            double, NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BinaryCrossEntropy, const NDArray& pred,
                            const NDArray& label, NDArray& loss, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BinaryCrossEntropyGradient, const NDArray& pred,
                            const NDArray& label, const NDArray& grad_loss,
                            NDArray& output, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Broadcast, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BroadcastGradient, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BroadcastShape, const NDArray&, NDArray&,
                            const HTShape&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BroadcastShapeMul, const NDArray&, double, NDArray&,
                            const HTShape&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Ceil, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Concat, const NDArray&, const NDArray&, NDArray&,
                            size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ConcatGradient, const NDArray&, NDArray&, size_t,
                            size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Concatenate, const NDArray&, NDArray&, size_t,
                            size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ConcatenateGradient, const NDArray&, NDArray&,
                            size_t, size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2d, const NDArray&, const NDArray&, NDArray&,
                            const int, const int, const int, const int,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dGradientofFilter, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dGradientofData, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dAddBias, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dBroadcast, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dReduceSum, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DataTransfer, const NDArray& from, NDArray& to,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Diagonal, const NDArray&, NDArray&, int, int, int,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DiagonalGradient, const NDArray&, NDArray&, int,
                            int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DivConst, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DivElewise, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Dropout, const NDArray&, double, uint64_t, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DropoutGradientWithRecomputation, const NDArray&,
                            double, uint64_t, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DropoutGradient, const NDArray&, const NDArray&,
                            double, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Dropout2d, const NDArray&, double, uint64_t,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Dropout2dGradientWithRecomputation, const NDArray&,
                            double, uint64_t, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(EmbeddingLookup, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(EmbeddingLookupGradient, const NDArray&,
                            const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Exp, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Floor, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(IndexAdd, const NDArray&, const NDArray&, NDArray&,
                            size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(InstanceNorm, const NDArray&, NDArray&, NDArray&,
                            NDArray&, float, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(InstanceNormGradient, const NDArray&,
                            const NDArray&, NDArray&, const NDArray&,
                            const NDArray&, float, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LayerNorm, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, NDArray&, NDArray&, float,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LayerNormGradient, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, NDArray&, NDArray&,
                            const NDArray&, const NDArray&, float,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LeakyRelu, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LeakyReluGradient, const NDArray&, const NDArray&,
                            double, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Linear, const NDArray& a, bool trans_a, const NDArray& b,
                    bool trans_b, const NDArray& bias, NDArray& output,
                    const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Log, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MatDot, const NDArray&, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CUDA(MatMul, const NDArray& a, bool trans_a, const NDArray& b,
                    bool trans_b, NDArray& output, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MaxPool, const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MaxPoolGradient, const NDArray&, const NDArray&,
                            const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MulConst, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MulElewise, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Onehot, const NDArray&, size_t, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Opposite, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Pad, const NDArray&, NDArray&, const HTShape&,
                            const Stream&, size_t, double);
DECLARE_KERNEL_CPU_AND_CUDA(PadGradient, const NDArray&, NDArray&,
                            const HTShape&, const Stream&, size_t);
DECLARE_KERNEL_CPU_AND_CUDA(Pow, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Reduce, const NDArray&, NDArray&, const HTAxes&,
                            ReductionType, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceMax, const NDArray&, NDArray&, const int64_t*,
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceMean, const NDArray&, NDArray&,
                            const int64_t*, int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceMin, const NDArray&, NDArray&, const int64_t*,
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceSum, const NDArray&, NDArray&, const int64_t*,
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Relu, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReluGradient, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Reshape, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Round, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SGDUpdate, const NDArray&, NDArray&, NDArray&,
                            float, float, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Sigmoid, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Slice, const NDArray&, NDArray&, int64_t*,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SliceGradient, const NDArray&, NDArray&, int64_t*,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Softmax, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxGradient, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxCrossEntropy, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxCrossEntropyGradient, const NDArray&,
                            const NDArray&, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Sqrt, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReciprocalSqrt, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Reciprocal, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Tanh, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(TanhGradient, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Transpose, const NDArray&, NDArray&, int64_t*,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Where, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(NormalInits, NDArray&, double, double, uint64_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(UniformInits, NDArray&, double, double, uint64_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(TruncatedNormalInits, NDArray&, double, double,
                            double, double, uint64_t, const Stream&);

// Communication kernels
DECLARE_KERNEL_CPU_AND_CUDA(BroadcastComm, const NDArray&, NDArray&, int,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AllReduce, const NDArray&, NDArray&,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceComm, const NDArray&, NDArray&, int,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AllGather, const NDArray&, NDArray&,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceScatter, const NDArray&, NDArray&,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Gather, const NDArray&, NDArray&, int,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Scatter, const NDArray&, NDArray&, int,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(P2PSend, const NDArray&, const Device&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(P2PRecv, NDArray&, const Device&, const Stream&);

/******************************************************
 * Dispatching kernels for operations
 ******************************************************/

#define HT_DISPATCH_KERNEL_CASE_CPU(KERNEL, ...)                               \
  case kCPU: {                                                                 \
    KERNEL##Cpu(__VA_ARGS__);                                                  \
    break;                                                                     \
  }

#define HT_DISPATCH_KERNEL_CASE_CUDA(KERNEL, ...)                              \
  case kCUDA: {                                                                \
    KERNEL##Cuda(__VA_ARGS__);                                                 \
    break;                                                                     \
  }

#define HT_DISPATCH_KERNEL_SWITCH(DEVICE_TYPE, OP_TYPE, ...)                   \
  do {                                                                         \
    const auto& _device_type = DEVICE_TYPE;                                    \
    switch (_device_type) {                                                    \
      __VA_ARGS__                                                              \
      default:                                                                 \
        HT_NOT_IMPLEMENTED << "\"" << OP_TYPE << "\" is not implemented for "  \
                           << "\"" << DeviceType2Str(_device_type) << "\"";    \
    }                                                                          \
  } while (0)

#define HT_DISPATCH_KERNEL_CPU_AND_CUDA(DEVICE_TYPE, OP_TYPE, KERNEL, ...)     \
  HT_DISPATCH_KERNEL_SWITCH(                                                   \
    DEVICE_TYPE, OP_TYPE,                                                      \
    HT_DISPATCH_KERNEL_CASE_CPU(KERNEL, __VA_ARGS__)                           \
      HT_DISPATCH_KERNEL_CASE_CUDA(KERNEL, __VA_ARGS__))

#define HT_DISPATCH_KERNEL_CPU_ONLY(DEVICE_TYPE, OP_TYPE, KERNEL, ...)         \
  HT_DISPATCH_KERNEL_SWITCH(DEVICE_TYPE, OP_TYPE,                              \
                            HT_DISPATCH_KERNEL_CASE_CPU(KERNEL, __VA_ARGS__))

#define HT_DISPATCH_KERNEL_CUDA_ONLY(DEVICE_TYPE, OP_TYPE, KERNEL, ...)        \
  HT_DISPATCH_KERNEL_SWITCH(DEVICE_TYPE, OP_TYPE,                              \
                            HT_DISPATCH_KERNEL_CASE_CUDA(KERNEL, __VA_ARGS__))

} // namespace impl
} // namespace hetu
