#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_QUANT_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    
    int32_t zero_point;
    double scale;

    Tensor *input_tensor;
    Tensor *output_tensor;
} AffineQuantizationParams;

// tensorflow\lite\kernels\internal\reference\quantize.h: AffineQuantize
// From float to int
inline void AffineQuantize(const AffineQuantizationParams& params) {
    const int32_t zero_point = params.zero_point;
    const double scale = params.scale;
    const int flat_size = MatchingFlatSize(params.input_tensor->shape, params.output_tensor->shape);
    static constexpr int32_t min_val = std::numeric_limits<int8_t>::min();
    static constexpr int32_t max_val = std::numeric_limits<int8_t>::max();

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
    const float *input_data = (float*)params.input_tensor->data;
    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    float *output_data = (float*)params.output_tensor->data;

    for (int i = 0; i < flat_size; i++) {
        const float val = input_data[i];
        int32_t unclamped =
            static_cast<int32_t>(std::round(val / static_cast<float>(scale))) +
            zero_point;
        int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
        output_data[i] = clamped;
    }
}

// // From int to int, designed for fixed-point processor
// // tensorflow\lite\kernels\internal\reference\requantize.h: Requantize
// template <typename input_type, typename output_type>
// inline void Requantize(const input_type* input_data, int32_t size,
//                        int32_t effective_scale_multiplier,
//                        int32_t effective_scale_shift, int32_t input_zeropoint,
//                        int32_t output_zeropoint, output_type* output_data) {
//   // ruy::profiler::ScopeLabel label("Requantize");
//   const bool same_scale =
//       (effective_scale_multiplier == 1 << 30 && effective_scale_shift == 1);
//   if (same_scale) {
//     const bool mixed_type_int8_uint8 =
//         std::is_same<input_type, int8_t>::value &&
//         std::is_same<output_type, uint8_t>::value;
//     const bool mixed_type_uint8_int8 =
//         std::is_same<input_type, uint8_t>::value &&
//         std::is_same<output_type, int8_t>::value;
//     const int32_t zero_point_diff = input_zeropoint - output_zeropoint;
//     // Fast path to do requantization for the case when just a shift of 128 is
//     // needed.
//     if ((mixed_type_int8_uint8 && zero_point_diff == -128) ||
//         (mixed_type_uint8_int8 && zero_point_diff == 128)) {
//       for (int i = 0; i < size; ++i) {
//         output_data[i] = input_data[i] ^ 0x80;
//       }
//       return;
//     }
//   }
//   static constexpr int32_t kMinOutput = std::numeric_limits<output_type>::min();
//   static constexpr int32_t kMaxOutput = std::numeric_limits<output_type>::max();
//   for (int i = 0; i < size; ++i) {
//     const int32_t input = input_data[i] - input_zeropoint;
//     const int32_t output =
//         MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
//                                       effective_scale_shift) +
//         output_zeropoint;
//     const int32_t clamped_output =
//         std::max(std::min(output, kMaxOutput), kMinOutput);
//     output_data[i] = static_cast<output_type>(clamped_output);
//   }
// }

// // tensorflow\lite\micro\kernels\quantize_common.cc: PrepareQuantizeReference
//     double effective_scale = static_cast<double>(input->params.scale) /
//                              static_cast<double>(output->params.scale);
//
//     QuantizeMultiplier(effective_scale, &data->requantize_output_multiplier,
//                        &data->requantize_output_shift);

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_QUANT_HPP_