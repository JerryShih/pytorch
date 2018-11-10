#include "caffe2/operators/elementwise_add_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Add,
    BinaryElementwiseOp<NumericTypes, CPUContext, AddFunctor<CPUContext>>);

REGISTER_CPU_OPERATOR(
    AddLP,
    BinaryElementwiseOpLP<
        NumericTypes,
        CPUContext,
        AddFunctorLP<CPUContext>,
        SameTypeAsInput>);

} // namespace caffe2
