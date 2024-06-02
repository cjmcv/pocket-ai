#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_SOFTMAX_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_SOFTMAX_QUANT_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

/////////////////////////////
#include "gemmlowp/fixedpoint.h"
// tensorflow\lite\kernels\internal\common.h: GetReciprocal
inline int32_t GetReciprocal(int32_t x, int x_integer_digits,
                             int* num_bits_over_unit) {
    int headroom_plus_one = pai::infer::CountLeadingZeros(static_cast<uint32_t>(x));
    // This is the number of bits to the left of the binary point above 1.0.
    // Consider x=1.25.  In that case shifted_scale=0.8 and
    // no later adjustment will be needed.
    *num_bits_over_unit = x_integer_digits - headroom_plus_one;
    const int32_t shifted_sum_minus_one =
        static_cast<int32_t>((static_cast<uint32_t>(x) << headroom_plus_one) -
                            (static_cast<uint32_t>(1) << 31));

    gemmlowp::FixedPoint<int32_t, 0> shifted_scale =
        gemmlowp::one_over_one_plus_x_for_x_in_0_1(
            gemmlowp::FixedPoint<int32_t, 0>::FromRaw(shifted_sum_minus_one));
    return shifted_scale.raw();
}
////////////////////////////

namespace pai {
namespace infer {


typedef struct {
    uint32_t op_id;

    int32_t input_multiplier;
    int32_t input_left_shift;
    int diff_min;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
} SoftmaxQuantParams;

// ref: tensorflow\lite\kernels\internal\reference\softmax.h: Softmax
// Quantized softmax with int8_t/uint8_t input and int8_t/uint8_t/int16_t output.
inline void SoftmaxQuant(const SoftmaxQuantParams& params) {
    const int32_t input_beta_multiplier = params.input_multiplier;
    const int32_t input_beta_left_shift = params.input_left_shift;
    const int diff_min = params.diff_min;

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
    Shape &input_shape = params.input_tensor->shape;
    int8_t* input_data = (int8_t*)params.input_tensor->data;
    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    Shape &output_shape = params.output_tensor->shape;
    int8_t* output_data = (int8_t*)params.output_tensor->data;
    // The representation chosen for the input to the exp() function is Q5.26.
    // We need to leave extra space since values that we skip might be as large as
    // -32 before multiplying by input_beta_multiplier, and therefore as large as
    // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
    // accumulation, but exp(-16) definitely is.
    static const int kScaledDiffIntegerBits = 5;
    static const int kAccumulationIntegerBits = 12;
    using FixedPointScaledDiff =
        gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
    using FixedPointAccum =
        gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
    using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

    const int trailing_dim = input_shape.dims_count - 1;
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

    for (int i = 0; i < outer_size; ++i) {
        int8_t max_in_row = std::numeric_limits<int8_t>::min();
        for (int c = 0; c < depth; ++c) {
            max_in_row = std::max(max_in_row, input_data[i * depth + c]);
        }

        FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
        for (int c = 0; c < depth; ++c) {
            int32_t input_diff =
            static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
            if (input_diff >= diff_min) {
                const int32_t input_diff_rescaled =
                    MultiplyByQuantizedMultiplierGreaterThanOne(
                        input_diff, input_beta_multiplier, input_beta_left_shift);
                const FixedPointScaledDiff scaled_diff_f8 =
                    FixedPointScaledDiff::FromRaw(input_diff_rescaled);
                sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                                exp_on_negative_values(scaled_diff_f8));
            }
        }

        int num_bits_over_unit;
        FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
            sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

        for (int c = 0; c < depth; ++c) {
            int32_t input_diff =
                static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
            if (input_diff >= diff_min) {
                const int32_t input_diff_rescaled =
                    MultiplyByQuantizedMultiplierGreaterThanOne(
                        input_diff, input_beta_multiplier, input_beta_left_shift);
                const FixedPointScaledDiff scaled_diff_f8 =
                    FixedPointScaledDiff::FromRaw(input_diff_rescaled);

                FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
                int32_t unsat_output = RoundingDivideByPOT(
                    (shifted_scale * exp_in_0).raw(),
                    num_bits_over_unit + 31 - (sizeof(int8_t) * 8));

                const int32_t shifted_output =
                    unsat_output +
                    static_cast<int32_t>(std::numeric_limits<int8_t>::min());

                output_data[i * depth + c] = static_cast<int8_t>(std::max(
                    std::min(shifted_output,
                            static_cast<int32_t>(std::numeric_limits<int8_t>::max())),
                    static_cast<int32_t>(std::numeric_limits<int8_t>::min())));
            } else {
                output_data[i * depth + c] = std::numeric_limits<int8_t>::min();
            }
        }
    }
}


} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_SOFTMAX_QUANT_HPP_