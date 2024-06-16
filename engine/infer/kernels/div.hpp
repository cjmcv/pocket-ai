#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_DIV_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_DIV_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"
#include "engine/infer/gemmlowp_common.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;

    bool requires_broadcast;

    // float activation params.
    float float_activation_min;
    float float_activation_max;

    Tensor *input_tensor[2];
    Tensor *output_tensor;
} DivParams;

// ref: tensorflow\lite\kernels\internal\reference\div.h：#226 Div
inline void Div(const DivParams& params) {
    PAI_DCHECK_EQ(params.input_tensor[0]->type, kPaiInferInt8);
    const int8_t* input1_data = (int8_t*)params.input_tensor[0]->data;
    const Shape& input1_shape = params.input_tensor[0]->shape;

    PAI_DCHECK_EQ(params.input_tensor[1]->type, kPaiInferInt8);
    const int8_t* input2_data = (int8_t*)params.input_tensor[1]->data;
    const Shape& input2_shape = params.input_tensor[1]->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    int8_t* output_data = (int8_t*)params.output_tensor->data;
    const Shape& output_shape = params.output_tensor->shape;

    PAI_DCHECK_EQ(params.requires_broadcast, false); // 只支持维度一致不广播的情况
    int flat_size = GetShapeFlatSize(output_shape);
    PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input1_shape));
    PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input2_shape));

    float output_activation_min = params.float_activation_min;
    float output_activation_max = params.float_activation_max;

    for (int i = 0; i < flat_size; ++i) {
        output_data[i] = ActivationFunctionWithMinMax(
            input1_data[i] / input2_data[i], output_activation_min,
            output_activation_max);
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_DIV_HPP_