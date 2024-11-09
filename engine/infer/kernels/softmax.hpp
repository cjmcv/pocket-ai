#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_SOFTMAX_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_SOFTMAX_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"
#include "engine/infer/gemmlowp_common.hpp"

namespace pai {
namespace infer {
typedef struct {
    uint32_t op_id;

    double beta;
    Tensor *input_tensor;
    Tensor *output_tensor;
} SoftmaxParams;

// ref: tensorflow\lite\kernels\internal\reference\softmax.h: Softmax
inline void Softmax(const SoftmaxParams& params) {
    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
    Shape &input_shape = params.input_tensor->shape;
    float* input_data = (float*)params.input_tensor->data;
    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    Shape &output_shape = params.output_tensor->shape;
    float* output_data = (float*)params.output_tensor->data;

    const int trailing_dim = input_shape.dims_count - 1;
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

    for (int i = 0; i < outer_size; ++i) {
        // Find max element value which we'll use to ensure numerical stability
        // taking advantage of the following equality:
        // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
        float max = std::numeric_limits<float>::lowest();
        for (int c = 0; c < depth; ++c) {
            max = std::max(max, input_data[i * depth + c]);
        }

        // Compute sum.
        float sum = 0.f;
        for (int c = 0; c < depth; ++c) {
            const float exp_c = std::exp((input_data[i * depth + c] - max) *
                                        static_cast<float>(params.beta));
            output_data[i * depth + c] = exp_c;
            sum += exp_c;
        }

        // Compute result.
        for (int c = 0; c < depth; ++c) {
            output_data[i * depth + c] = output_data[i * depth + c] / sum;
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_SOFTMAX_HPP_