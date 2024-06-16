#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_DIV_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_DIV_QUANT_HPP_

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
} DivQuantParams;

// ref: tensorflow\lite\kernels\internal\reference\div.h： Div->DivElementwise
// Element-wise div that can often be used for inner loop of broadcast Div as
// well as the non-broadcast Div.
inline void DivQuant(const DivQuantParams& params) {

    // DivCheckAddParams<int8_t>(params);
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
        int32_t input1_val = params.input1_offset + input1_data[i];
        int32_t input2_val = params.input2_offset + input2_data[i];
        PAI_DCHECK_NE(input2_val, 0);
        if (input2_val < 0) {
            // Invert signs to avoid a negative input2_val as input2_inv needs to be
            // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
            input1_val = -input1_val;
            input2_val = -input2_val;
        }
        int recip_shift;
        const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
        const int headroom = CountLeadingSignBits(input1_val);
        const int32_t unscaled_quotient =
            MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                        headroom);
        const int total_shift = params.output_shift - recip_shift - headroom;
        const int32_t unclamped_result =
            params.output_offset +
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                unscaled_quotient, params.output_multiplier, total_shift);
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                    std::max(params.quantized_activation_min, unclamped_result));
        output_data[i] = static_cast<int8_t>(clamped_output);
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_DIV_QUANT_HPP_