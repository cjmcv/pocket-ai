#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_ADD_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_ADD_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

// For Add, Sub, Mul ops.  (binary)
typedef struct {
    uint32_t op_id;

    bool requires_broadcast;

    // float activation params.
    float float_activation_min;
    float float_activation_max;

    Tensor *input_tensor[2];
    Tensor *output_tensor;
} AddParams;

// ref: tensorflow\lite\kernels\internal\reference\add.h: Add
inline void Add(const AddParams& params) {

    PAI_DCHECK_EQ(params.input_tensor[0]->type, kPaiInferFloat32);
    const Shape& input1_shape = params.input_tensor[0]->shape;
    const float* input1_data = (float*)params.input_tensor[0]->data;

    PAI_DCHECK_EQ(params.input_tensor[1]->type, kPaiInferFloat32);
    const Shape& input2_shape = params.input_tensor[1]->shape;
    const float* input2_data = (float*)params.input_tensor[1]->data;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    const Shape& output_shape = params.output_tensor->shape;
    float* output_data = (float*)params.output_tensor->data;

    float activation_min = params.float_activation_min;
    float activation_max = params.float_activation_max;

    PAI_DCHECK_EQ(params.requires_broadcast, false);
    int flat_size = GetShapeFlatSize(output_shape);
    PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input1_shape));
    PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input2_shape));

    for (int i = 0; i < flat_size; ++i) {
        output_data[i] = ActivationFunctionWithMinMax(
            input1_data[i] + input2_data[i], activation_min, activation_max);
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_ADD_HPP_