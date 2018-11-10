#include "caffe2/operators/leaky_relu_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool LeakyReluOp<float, CPUContext, false>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  ConstEigenVectorMap<float> Xvec(X.template data<float>(), X.numel());
  EigenVectorMap<float> Yvec(Y->template mutable_data<float>(), Y->numel());
  Yvec = Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * alpha_;
  return true;
}

template <>
bool LeakyReluOp<float, CPUContext, true>::RunOnDevice() {
  printf(
      "bignose test LeakyReluOp%s\n",
      this->debug_def().op_calibration_param().name().c_str());

  Tensor X{CPUContext::GetDeviceType()};
  auto* Y = Output(0);
  X.CopyFrom(Input(0));
  Y->ResizeLike(X);

  float input_threshold =
      this->GetInputCalibrationParam(this->debug_def().input(0))
          ->blob_param(0)
          .threshold_y();
  X.Quantize<float, int8_t>(128.0f, input_threshold, 0);

  float gt_scale = this->GetOpCalibrationParam()->leakyrelu_param().gt_scale();
  float le_scale = this->GetOpCalibrationParam()->leakyrelu_param().le_scale();
  int gt_rshift =
      this->GetOpCalibrationParam()->leakyrelu_param().gt_right_shift_width();
  int le_rshift =
      this->GetOpCalibrationParam()->leakyrelu_param().le_right_shift_width();
  CAFFE_ENFORCE(gt_rshift >= 0, "leakyRelu gt_rshift");
  gt_scale *= 1.0f / (1 << gt_rshift);
  CAFFE_ENFORCE(le_rshift >= 0, "leakyRelu le_rshift");
  le_scale *= 1.0f / (1 << le_rshift);

  ConstEigenVectorMap<float> Xvec(X.template data<float>(), X.size());
  EigenVectorMap<float> Yvec(Y->template mutable_data<float>(), Y->size());
  Yvec = Xvec.cwiseMax(0.f) * gt_scale + Xvec.cwiseMin(0.f) * le_scale;

  float output_threshold =
      this->GetOpCalibrationParam()->blob_param(0).threshold_y();

  Y->Saturate<float, int8_t>();
  Y->DequantizeInt8<float>(output_threshold);

  return true;
}

template <>
bool LeakyReluGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& Y = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(Y);
  CAFFE_ENFORCE_EQ(Y.numel(), dY.numel());
  ConstEigenVectorMap<float> Yvec(Y.template data<float>(), Y.numel());
  ConstEigenVectorMap<float> dYvec(dY.template data<float>(), dY.numel());
  EigenVectorMap<float> dXvec(dX->template mutable_data<float>(), dX->numel());
  Eigen::VectorXf gtZero = (Yvec.array() >= 0.0f).cast<float>();
  dXvec = dYvec.array() * gtZero.array() -
      dYvec.array() * (gtZero.array() - 1.0f) * alpha_;
  return true;
}

REGISTER_CPU_OPERATOR(LeakyRelu, LeakyReluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LeakyReluLP, LeakyReluOp<float, CPUContext, true>);
REGISTER_CPU_OPERATOR(
    LeakyReluGradient,
    LeakyReluGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LeakyRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("alpha", "*(type: float; default: 0.01)* Coefficient of leakage.")
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<2>)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
The *LeakyRelu* op takes one input tensor $X$ and an argument $alpha$, and produces one output tensor $Y$ of the same shape as $X.$ The op performs the element wise leaky relu operation, defined as

$$y=LeakyRelu(x) =\begin{cases}\alpha x & x < 0\\x & otherwise\end{cases}$$

The default value of *alpha* is 0.01.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LeakyRelu",
    ["X"],
    ["Y"],
    alpha=0.01
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[-0.91060215  0.09374836  2.1429708 ]
 [-0.748983    0.19164062 -1.5130422 ]
 [-0.29539835 -0.8530696   0.7673204 ]]

Y:
 [[-0.00910602  0.09374836  2.1429708 ]
 [-0.00748983  0.19164062 -0.01513042]
 [-0.00295398 -0.0085307   0.7673204 ]]

```

</details>


)DOC")
    .Input(0, "X", "Input tensor of data to be operated on.")
    .Output(0, "Y", "Output tensor, calculated as described above.");

OPERATOR_SCHEMA(LeakyReluLp)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("alpha", "*(type: float; default: 0.01)* Coefficient of leakage.")
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<2>)
    .IdenticalTypeAndShape();

OPERATOR_SCHEMA(LeakyReluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .Arg("alpha", "Coefficient of leakage")
    .InheritOnnxSchema();

class GetLeakyReluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LeakyReluGradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(LeakyRelu, GetLeakyReluGradient);

} // namespace caffe2
