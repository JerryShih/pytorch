#include "caffe2/operators/clip_op.h"
#include "caffe2/utils/eigen_utils.h"
#include <cmath>
#include <limits>

namespace caffe2 {

template <>
bool ClipOp<float, CPUContext, false>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  EigenVectorMap<float>(Y->template mutable_data<float>(), Y->numel()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.numel())
          .cwiseMax(min_)
          .cwiseMin(max_);
  return true;
}

template <>
bool ClipOp<float, CPUContext, true>::RunOnDevice() {
  printf("bignose test FCLP %s\n",
      this->debug_def().op_calibration_param().name().c_str());

  Tensor input_lp(CPUContext::GetDeviceType());
  input_lp.CopyFrom(Input(0));
  auto* Y = Output(0);
  float max = max_;
  float min = min_;

  const auto& def = this->debug_def();
  if (this->GetInputCalibrationParam(def.input(0))) {
    float input_threshold = this->GetInputCalibrationParam(def.input(0))
                                ->blob_param(0)
                                .threshold_y();
    input_lp.Quantize<float, int8_t>(128.0f, input_threshold, 0);

    max = static_cast<int>(max / input_threshold * 128.0f);
    min = static_cast<int>(min / input_threshold * 128.0f);

    // Saturate the clip max and min value to s8 type.
    max = std::fmax(max, std::numeric_limits<int8_t>::min());
    max = std::fmin(max, std::numeric_limits<int8_t>::max());
    min = std::fmax(min, std::numeric_limits<int8_t>::min());
    min = std::fmin(min, std::numeric_limits<int8_t>::max());
  }

  Y->ResizeLike(input_lp);
  EigenVectorMap<float>(Y->template mutable_data<float>(), Y->size()) =
      ConstEigenVectorMap<float>(input_lp.data<float>(), input_lp.size())
          .cwiseMax(min)
          .cwiseMin(max);

  if (this->GetOpCalibrationParam()) {
    float output_threshold =
        this->GetOpCalibrationParam()->blob_param(0).threshold_y();
    Y->Saturate<float, int8_t>();
    Y->DequantizeInt8<float>(output_threshold);
  }

  return true;
}

template <>
bool ClipGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.numel(), 0);
  CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
  dX->ResizeLike(Y);
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  for (int i = 0; i < Y.numel(); ++i) {
    dXdata[i] = dYdata[i] * (Ydata[i] > min_ && Ydata[i] < max_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(Clip, ClipOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ClipLP, ClipOp<float, CPUContext, true>);
REGISTER_CPU_GRADIENT_OPERATOR(ClipGradient, ClipGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Clip)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
This operator limits the given input within an interval. The interval is
specified by the `min` and `max` arguments. They default to
*numeric_limits::lowest()* and *numeric_limits::max()* respectively. The
clipping operation can be done in an in-place fashion by using the same output
blob as the input blob.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/clip_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Clip",
    ["X"],
    ["Y"],
    min=20.0,
    max=60.0

)

workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```
X: [[45. 16. 59. 99. 48.]
 [12. 44. 46. 82. 28.]
 [ 1. 91. 18.  9. 71.]
 [24. 37. 61. 12. 81.]
 [36. 38. 30. 84. 40.]]
Y: [[45. 20. 59. 60. 48.]
 [20. 44. 46. 60. 28.]
 [20. 60. 20. 20. 60.]
 [24. 37. 60. 20. 60.]
 [36. 38. 30. 60. 40.]]
```

</details>

)DOC")
    .Arg(
        "min",
        "*(type: float)* Minimum value, under which element is "
        "replaced by min (default=*numeric_limits::lowest()*).")
    .Arg(
        "max",
        "*(type: float)* Maximum value, under which element is "
        "replaced by max (default=*numeric_limits::max()*).")
    .Input(
        0,
        "X",
        "*(Tensor`<float>`)* Input tensor within range "
        "[*numeric_limits::lowest()*, *numeric_limits::max()*].")
    .Output(
        0,
        "Y",
        "*(Tensor`<float>`)* Output tensor clipped within range [`min`, `max`].")
    .InheritOnnxSchema();

OPERATOR_SCHEMA(ClipLP)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .InheritOnnxSchema();

GRADIENT_OPERATOR_SCHEMA(ClipGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});

class GetClipGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ClipGradient", "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Clip, GetClipGradient);
}  // namespace caffe2
