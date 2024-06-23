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
    PAI_DCHECK_EQ(params.input_tensor[0]->type, kPaiInferFloat32);
    const float* input1_data = (float*)params.input_tensor[0]->data;
    const Shape& input1_shape = params.input_tensor[0]->shape;

    PAI_DCHECK_EQ(params.input_tensor[1]->type, kPaiInferFloat32);
    const float* input2_data = (float*)params.input_tensor[1]->data;
    const Shape& input2_shape = params.input_tensor[1]->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    float* output_data = (float*)params.output_tensor->data;
    const Shape& output_shape = params.output_tensor->shape;

    float output_activation_min = params.float_activation_min;
    float output_activation_max = params.float_activation_max;

    int flat_size = GetShapeFlatSize(output_shape);
    if (params.requires_broadcast == false) {
        PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input1_shape));
        PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input2_shape));
        for (int i = 0; i < flat_size; ++i) {
            output_data[i] = ActivationFunctionWithMinMax(
                input1_data[i] / input2_data[i], output_activation_min,
                output_activation_max);
        }
    }
    else {
        PAI_DCHECK_EQ(1, GetShapeFlatSize(input2_shape)); // 广播仅支持2号输入维度为1的情况
        for (int i = 0; i < flat_size; ++i) {
            output_data[i] = ActivationFunctionWithMinMax(
                input1_data[i] / input2_data[0], output_activation_min,
                output_activation_max);
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_DIV_HPP_