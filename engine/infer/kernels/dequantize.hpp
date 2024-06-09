#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_DEQUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_DEQUANT_HPP_

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
} DequantizationParams;

// ref: tensorflow\lite\kernels\internal\reference\dequantize.h: Dequantize
// From int to float
// Dequantizes into a float without rounding.
template <typename InputT, typename OutputT>
inline void Dequantize(const DequantizationParams& params) {

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
    const int8_t* input_data = (int8_t*)params.input_tensor->data;
    const Shape &input_shape = params.input_tensor->shape;                     
    
    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    float* output_data = (float*)params.output_tensor->data;
    const Shape &output_shape = params.output_tensor->shape;

    int32_t zero_point = params.zero_point;
    const double scale = params.scale;
    const int flat_size = MatchingFlatSize(input_shape, output_shape);

    for (int i = 0; i < flat_size; i++) {
        const int32_t val = input_data[i];
        const float result = static_cast<float>(scale * (val - zero_point));
        output_data[i] = result;
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_QUANT_HPP_