#include "caffe2/operators/elementwise_mul_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Mul,
    BinaryElementwiseOp<NumericTypes, CPUContext, MulFunctor<CPUContext>>);

REGISTER_CPU_OPERATOR(
    MulLP,
    BinaryElementwiseOpLP<
        NumericTypes,
        CPUContext,
        MulFunctorLP<CPUContext>,
        SameTypeAsInput>);

} // namespace caffe2
