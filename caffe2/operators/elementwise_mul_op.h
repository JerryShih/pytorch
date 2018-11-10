#ifndef CAFFE2_OPERATORS_ELEMENTWISE_MUL_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_MUL_OP_H_

#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct MulFunctor {
  template <typename TIn, typename TOut>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A,
      const TIn* B,
      TOut* C,
      Context* context) const {
    math::Mul(
        A_dims.size(),
        A_dims.data(),
        B_dims.size(),
        B_dims.data(),
        A,
        B,
        C,
        context);
    return true;
  }

  template <typename TGrad, typename TIn, typename TOut>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC_data,
      const TIn* A_data,
      const TIn* B_data,
      const TOut* C_data,
      TGrad* dA_data,
      TGrad* dB_data,
      Context* context) const;
};

template <class Context>
struct MulFunctorLP {
  template <typename TIn, typename TOut>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A,
      const TIn* B,
      TOut* C,
      Context* context,
      OperatorBase* op) const {
    printf(
        "bignose test mul int8 node %s\n",
        op->debug_def().op_calibration_param().name().c_str());

    Tensor input_lp(Context::GetDeviceType());
    const auto& def = op->debug_def();
    if (op->GetInputCalibrationParam(def.input(0))) {
      input_lp.CopyFrom(op->template Input<Tensor>(0, Context::GetDeviceType()));
      float input_threshold = op->GetInputCalibrationParam(def.input(0))
                                  ->blob_param(0)
                                  .threshold_y();
      input_lp.Quantize<TIn, int8_t>(128.0f, input_threshold, 0);
    }

    math::Mul(
        A_dims.size(),
        A_dims.data(),
        B_dims.size(),
        B_dims.data(),
        input_lp.template data<TIn>(),
        B,
        C,
        context);
    //// Node: skip all int8 op for mul's output.
    //// The mul op now comes witn add op(we use them for TGScale op). This mul-lp op
    //// is not a regular mul op.
    // Tensor* output = op->template Output<Tensor>(0, Context::GetDeviceType());
    // if (op->GetOpCalibrationParam()) {
    //   float output_threshold =
    //       op->GetOpCalibrationParam()->blob_param(0).threshold_y();
    //   float right_shift = op->GetOpCalibrationParam()->right_shift_width();
    //   output->RightShift<TOut>(right_shift);
    //   output->Saturate<TOut, int8_t>();
    //   output->DequantizeInt8<TOut>(output_threshold);
    // }

    return true;
  }

  template <typename TGrad, typename TIn, typename TOut>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC_data,
      const TIn* A_data,
      const TIn* B_data,
      const TOut* C_data,
      TGrad* dA_data,
      TGrad* dB_data,
      Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_MUL_OP_H_
