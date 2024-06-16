#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_ADD_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_ADD_QUANT_HPP_

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
    // uint8_t inference params.
    int32_t input1_offset;
    int32_t input2_offset;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    // Add / Sub, not Mul, uint8_t inference params.
    int32_t left_shift;
    int32_t input1_multiplier;
    int32_t input1_shift;
    int32_t input2_multiplier;
    int32_t input2_shift;

    // TODO(b/158622529): Union the following activation params.
    // uint8_t, etc, activation params.
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;

    Tensor *input_tensor[2];
    Tensor *output_tensor;
} AddQuantParams;

// ref: tensorflow\lite\kernels\internal\reference\integer_ops\add.h: AddFunc
inline int8_t AddFunc(int8_t x, int8_t y, const AddQuantParams& params) {
    const int32_t input1_val = params.input1_offset + x;
    const int32_t input2_val = params.input2_offset + y;
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                std::max(params.quantized_activation_min, raw_output));
    return static_cast<int8_t>(clamped_output);
}

// Broadcasting is not supported for now.
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting
// ref: tensorflow\lite\kernels\internal\reference\integer_ops\add.h
inline void AddQuant(const AddQuantParams& params) {

    PAI_DCHECK_EQ(params.input_tensor[0]->type, kPaiInferInt8);
    const int8_t* input1_data = (int8_t*)params.input_tensor[0]->data;
    const Shape& input1_shape = params.input_tensor[0]->shape;

    PAI_DCHECK_EQ(params.input_tensor[1]->type, kPaiInferInt8);
    const int8_t* input2_data = (int8_t*)params.input_tensor[1]->data;
    const Shape& input2_shape = params.input_tensor[1]->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    int8_t* output_data = (int8_t*)params.output_tensor->data;
    const Shape& output_shape = params.output_tensor->shape;

    PAI_DCHECK_EQ(params.requires_broadcast, false);
    int flat_size = GetShapeFlatSize(output_shape);
    PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input1_shape));
    PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input2_shape));

    for (int i = 0; i < flat_size; ++i) {
        output_data[i] = AddFunc(input1_data[i], input2_data[i], params);
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_QUANT_HPP_