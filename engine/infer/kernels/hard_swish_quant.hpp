#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_HARD_SWISH_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_HARD_SWISH_QUANT_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"
#include "gemmlowp/fixedpoint.h"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    
    // zero_point of the input activations.
    int16_t input_zero_point;
    // zero_point of the output activations.
    int16_t output_zero_point;
    // 16bit fixed-point component of the multiplier to apply to go from the
    // "high-res input scale", which is the input scale multiplied by 2^7, to the
    // "relu-ish scale", which 3.0/32768.
    // See the implementation of HardSwishPrepare.
    int16_t reluish_multiplier_fixedpoint_int16;
    // exponent/bit-shift component of the aforementioned multiplier.
    int reluish_multiplier_exponent;
    // 16bit fixed-point component of the multiplier to apply to go from the
    // "high-res input scale", which is the input scale multiplied by 2^7, to the
    // output scale.
    // See the implementation of HardSwishPrepare.
    int16_t output_multiplier_fixedpoint_int16;
    // exponent/bit-shift component of the aforementioned multiplier.
    int output_multiplier_exponent;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;

} HardSwishQuantParams;

inline int16_t SaturatingLeftShift(int16_t value, int amount) {
    int64_t result = static_cast<int64_t>(value) * (1 << amount);
    result = std::min<int64_t>(result, std::numeric_limits<int16_t>::max());
    result = std::max<int64_t>(result, std::numeric_limits<int16_t>::min());
    return result;
}

// Similar to ARM instruction SQDMULH.
// Similar to gemmlowp::SaturatingRoundingDoublingHighMul except
// rounding to zero instead of to nearest (SQRDMULH).
inline std::int16_t SaturatingDoublingHighMul(std::int16_t a, std::int16_t b) {
    bool overflow = a == b && a == std::numeric_limits<std::int16_t>::min();
    std::int32_t a_32(a);
    std::int32_t b_32(b);
    std::int32_t ab_32 = a_32 * b_32;
    std::int16_t ab_x2_high16 = static_cast<std::int16_t>((ab_32) / (1 << 15));
    return overflow ? std::numeric_limits<std::int16_t>::max() : ab_x2_high16;
}

// ref: tensorflow\lite\kernels\internal\reference\hard_swish.h: HardSwish
inline void HardSwishQuant(const HardSwishQuantParams& params) {

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
    const int8_t* input_data = (int8_t*)params.input_tensor->data;
    const Shape &input_shape = params.input_tensor->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    int8_t* output_data = (int8_t*)params.output_tensor->data;
    const Shape &output_shape = params.output_tensor->shape;
    PAI_DCHECK_EQ(output_shape.dims_count, 2);

    const int flat_size = MatchingFlatSize(input_shape, output_shape);

    for (int i = 0; i < flat_size; i++) {
        const int16_t input_value = input_data[i] - params.input_zero_point;
        // Left-shift as much as we can without overflow/saturation to put
        // significant bits in the high bits of our 16-bit fixedpoint values, so
        // that fixed-point approximate computations below are as accurate as
        // possible.
        const int16_t input_value_on_hires_input_scale = input_value * (1 << 7);
        // Compute the input value on essentially the output scale, just not
        // right-shifted yet. This is the value that we'll use in the (x >= +3)
        // case, and that in the general case we'll multiply against the "relu-ish"
        // fixed-point multiplier in [0, 1].
        const int16_t input_value_on_preshift_output_scale =
            gemmlowp::SaturatingRoundingDoublingHighMul(
                input_value_on_hires_input_scale,
                params.output_multiplier_fixedpoint_int16);
        // Now compute the "relu-ish multiplier". In the (-3 <= x <= +3) case, that
        // is just an affine rescaling of x from [-3, 3] to [0, 1]. In the general
        // case, it is just that plus saturation at the boundaries of [-3, 3].
        // First, we rescale from [-3, 3] to [-1, 1], saturating.
        // That is done by rescaling the input value with a fixed-point multiplier
        // (reluish_multiplier_fixedpoint) and bit-shift such that we represent
        // that input value on the scale where the real value 3.0f is represented
        // by the quantized value 32768.  (+32768 is actually not representable as
        // int16_t, so this saturates at +32767, and that is seen empirically to be
        // a negligible contribution to numerical error/bias).
        //
        // This code is careful to correctly implement any magnitude of multiplier,
        // involving either a right shift or a left shift, with correct saturation
        // behavior in the left-shift case. This forces this code to be more
        // complicated, but is necessary for real applications: a partially
        // trained quantized MobileNet v3-small model that motivated this code
        // exhibits some large [min, max] range boundaries, of the order of
        // magnitude of 10 or 100 depending on layers.
        //
        // The next few lines are basically just an ordinary
        // MultiplyByQuantizedMultiplier, except that we are more careful here
        // about the fine details of saturation when left-shifting, because here
        // overflow in left-shift is a common case, not an anomaly as
        // MultiplyByQuantizedMultiplier assumes.
        int16_t reluish_value = input_value_on_hires_input_scale;
        // Shift left, saturating, as much as we can while ensuring that this
        // saturation will not contribute to the result. That is, left shift amount
        // reduced by 1.
        if (params.reluish_multiplier_exponent > 0) {
            reluish_value = SaturatingLeftShift(
                reluish_value, params.reluish_multiplier_exponent - 1);
        }
        // Apply the fixed-point multiplier, dividing the value by a divisor
        // ranging in [1, 2].
        reluish_value = gemmlowp::SaturatingRoundingDoublingHighMul(
            reluish_value, params.reluish_multiplier_fixedpoint_int16);
        // Apply the last bit of left-shift. Thus, in the left-shifting case, if
        // any saturation affects the result, it is happening here --- any
        // saturation having occurred above is overwritten here, not affecting the
        // result.
        if (params.reluish_multiplier_exponent > 0) {
            reluish_value = SaturatingLeftShift(reluish_value, 1);
        }
        // Shift right, in the right-shifting case.
        if (params.reluish_multiplier_exponent < 0) {
            reluish_value = gemmlowp::RoundingDivideByPOT(
                reluish_value, -params.reluish_multiplier_exponent);
        }
        // At this point we have rescaled the value into a 16bit fixedpoint
        // reluish_value in [-1, 1].
        // We now convert that to a 16bit fixedpoint value in [0, 1].
        reluish_value = (reluish_value + (1 << 15)) >> 1;
        // Use of SaturatingDoublingHighMul here is important to cancel the biases
        // from the above SaturatingRoundingDoublingHighMul.
        //
        // On a partially trained MobileNet-v3-small,
        //
        //                                       | bias on    |  ImageNet
        //                                       | quantized  |  Top-1
        // Operation used here                   | values     |  accuracy (50k)
        // --------------------------------------+------------+-----------
        // SaturatingDoublingHighMul             | -0.0024    |  58.920
        // SaturatingRoundingDoublingHighMul     | -0.0067    |  58.064
        //
        // In activations_test, this is covered by this testcase:
        //     QuantizedActivationsOpTest.HardSwishBias
        //
        const int16_t preshift_output_value = SaturatingDoublingHighMul(
            reluish_value, input_value_on_preshift_output_scale);
        // We were so far operating on the pre-shift output scale. Now we finally
        // apply that output shift, arriving at the final output scale.
        int16_t output_value = gemmlowp::RoundingDivideByPOT(
            preshift_output_value, -params.output_multiplier_exponent);
        output_value += params.output_zero_point;
        output_value =
            std::min<int16_t>(output_value, std::numeric_limits<int8_t>::max());
        output_value =
            std::max<int16_t>(output_value, std::numeric_limits<int8_t>::min());
        output_data[i] = output_value;
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_HARD_SWISH_QUANT_HPP_