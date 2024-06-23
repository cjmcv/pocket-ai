#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_MUL_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_MUL_QUANT_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;

    bool requires_broadcast;
    // uint8_t inference params.
    int32_t input1_offset;
    int32_t input2_offset;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;

    // TODO(b/158622529): Union the following activation params.
    // uint8_t, etc, activation params.
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;

    Tensor *input_tensor[2];
    Tensor *output_tensor;
} MulQuantParams;

// Broadcasting is not supported for now.
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting
// ref: tensorflow\lite\kernels\internal\reference\integer_ops\mul.h: Mul -> MulElementwise 
inline void MulQuant(const MulQuantParams& params) {

    PAI_DCHECK_EQ(params.input_tensor[0]->type, kPaiInferInt8);
    const int8_t* input1_data = (int8_t*)params.input_tensor[0]->data;
    const Shape& input1_shape = params.input_tensor[0]->shape;

    PAI_DCHECK_EQ(params.input_tensor[1]->type, kPaiInferInt8);
    const int8_t* input2_data = (int8_t*)params.input_tensor[1]->data;
    const Shape& input2_shape = params.input_tensor[1]->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    int8_t* output_data = (int8_t*)params.output_tensor->data;
    const Shape& output_shape = params.output_tensor->shape;

    int flat_size = GetShapeFlatSize(output_shape);
    if (params.requires_broadcast == false) {
        PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input1_shape));
        PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input2_shape));
        for (int i = 0; i < flat_size; ++i) {
            const int32_t input1_val = params.input1_offset + input1_data[i];
            const int32_t input2_val = params.input2_offset + input2_data[i];
            const int32_t unclamped_result =
                params.output_offset +
                MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                            params.output_multiplier,
                                            params.output_shift);
            const int32_t clamped_output =
                std::min(params.quantized_activation_max,
                        std::max(params.quantized_activation_min, unclamped_result));
            output_data[i] = static_cast<int8_t>(clamped_output);
        }
    }
    else {
        PAI_DCHECK_EQ(1, GetShapeFlatSize(input2_shape)); // 广播仅支持2号输入维度为1的情况
        for (int i = 0; i < flat_size; ++i) {
            const int32_t input1_val = params.input1_offset + input1_data[i];
            const int32_t input2_val = params.input2_offset + input2_data[0];
            const int32_t unclamped_result =
                params.output_offset +
                MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                            params.output_multiplier,
                                            params.output_shift);
            const int32_t clamped_output =
                std::min(params.quantized_activation_max,
                        std::max(params.quantized_activation_min, unclamped_result));
            output_data[i] = static_cast<int8_t>(clamped_output);
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_QUANT_HPP_