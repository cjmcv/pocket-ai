#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_HARD_SWISH_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_HARD_SWISH_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    
    Tensor *input_tensor;
    Tensor *output_tensor;
} HardSwishParams;

// ref: tensorflow\lite\kernels\internal\reference\hard_swish.h: HardSwish
inline void HardSwish(const HardSwishParams& params) {
    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
    const float* input_data = (float*)params.input_tensor->data;
    const Shape &input_shape = params.input_tensor->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    float* output_data = (float*)params.output_tensor->data;
    const Shape &output_shape = params.output_tensor->shape;

    auto matching_size = MatchingFlatSize(input_shape, output_shape);
    const float* in_end = input_data + matching_size;
    for (; input_data < in_end; input_data++, output_data++) {
        const float in = *input_data;
        *output_data =
            in * std::min(static_cast<float>(6), std::max(static_cast<float>(0), in + 3)) /
            6;
    }
}


} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_HARD_SWISH_QUANT_HPP_