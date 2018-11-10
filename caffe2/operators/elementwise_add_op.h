#ifndef CAFFE2_OPERATORS_ELEMENTWISE_ADD_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_ADD_OP_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct AddFunctor {
  template <typename TIn, typename TOut>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A,
      const TIn* B,
      TOut* C,
      Context* context) const {
    math::Add(
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
      const TGrad* dC,
      const TIn* /* A */,
      const TIn* /* B */,
      const TOut* /* C */,
      TGrad* dA,
      TGrad* dB,
      Context* context) const {
    const std::vector<int> C_dims =
        elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            A_dims, B_dims);
    std::vector<int> A_axes;
    std::vector<int> B_axes;
    elementwise_ops_utils::ComputeBinaryBroadcastBackwardAxes(
        A_dims, B_dims, &A_axes, &B_axes);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        A_axes.size(),
        A_axes.data(),
        TGrad(1),
        dC,
        dA,
        context);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        B_axes.size(),
        B_axes.data(),
        TGrad(1),
        dC,
        dB,
        context);
    return true;
  }
};

template <class Context>
struct AddFunctorLP {
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
        "bignose test add int8 node %s\n",
        op->debug_def().op_calibration_param().name().c_str());
    //// Node: skip all int8 op for add's input.
    //// The add op now comes witn mul op(we use them for TGScale op). This add-lp op
    //// is not a regular add op.
    // Tensor input_lp(Context::GetDeviceType());
    // const auto& def = op->debug_def();
    // if (op->GetInputCalibrationParam(def.input(0)) &&
    //     op->GetOpCalibrationParam()) {
    //   input_lp.CopyFrom(op->template Input<Tensor>(0, Context::GetDeviceType()));
    //   float input_threshold = op->GetInputCalibrationParam(def.input(0))
    //                               ->blob_param(0)
    //                               .threshold_y();
    //   input_lp.Quantize<TIn, int8_t>(128.0f, input_threshold, 0);
    //   input_lp.LeftShift<TIn>(op->GetOpCalibrationParam()->right_shift_width());
    // }

    math::Add(
        A_dims.size(),
        A_dims.data(),
        B_dims.size(),
        B_dims.data(),
        //input_lp.template data<TIn>(),
        A,
        B,
        C,
        context);

    Tensor* output = op->template Output<Tensor>(0, Context::GetDeviceType());
    if (op->GetOpCalibrationParam()) {
      float output_threshold =
          op->GetOpCalibrationParam()->blob_param(0).threshold_y();
      float right_shift = op->GetOpCalibrationParam()->right_shift_width();
      output->RightShift<TOut>(right_shift);
      output->Saturate<TOut, int8_t>();
      output->DequantizeInt8<TOut>(output_threshold);
    }

    return true;
  }

  template <typename TGrad, typename TIn, typename TOut>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC,
      const TIn* /* A */,
      const TIn* /* B */,
      const TOut* /* C */,
      TGrad* dA,
      TGrad* dB,
      Context* context) const {
    const std::vector<int> C_dims =
        elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            A_dims, B_dims);
    std::vector<int> A_axes;
    std::vector<int> B_axes;
    elementwise_ops_utils::ComputeBinaryBroadcastBackwardAxes(
        A_dims, B_dims, &A_axes, &B_axes);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        A_axes.size(),
        A_axes.data(),
        TGrad(1),
        dC,
        dA,
        context);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        B_axes.size(),
        B_axes.data(),
        TGrad(1),
        dC,
        dB,
        context);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_ADD_OP_H_
