#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_MAX_POOLING_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_MAX_POOLING_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;

    PaddingValues padding_values;
    int stride_height;
    int stride_width;
    int filter_height;
    int filter_width;
    // float activation params.
    float float_activation_min;
    float float_activation_max;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
} PoolParams;

// ref: tensorflow\lite\kernels\internal\reference\pooling.h: MaxPool
inline void MaxPool(const PoolParams& params) {
    
    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
    const float* input_data = (float*)params.input_tensor->data;
    Shape &input_shape = params.input_tensor->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    float* output_data = (float*)params.output_tensor->data;
    Shape &output_shape = params.output_tensor->shape;
    
    PAI_DCHECK_EQ(input_shape.dims_count, 4);
    PAI_DCHECK_EQ(output_shape.dims_count, 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.dims[1];
    const int input_width = input_shape.dims[2];
    const int output_height = output_shape.dims[1];
    const int output_width = output_shape.dims[2];
    const int stride_height = params.stride_height;
    const int stride_width = params.stride_width;
    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                const int in_x_origin =
                    (out_x * stride_width) - params.padding_values.width;
                const int in_y_origin =
                    (out_y * stride_height) - params.padding_values.height;
                // Compute the boundaries of the filter region clamped so as to
                // ensure that the filter window fits in the input array.
                const int filter_x_start = std::max(0, -in_x_origin);
                const int filter_x_end =
                    std::min(params.filter_width, input_width - in_x_origin);
                const int filter_y_start = std::max(0, -in_y_origin);
                const int filter_y_end =
                    std::min(params.filter_height, input_height - in_y_origin);
                float max = std::numeric_limits<float>::lowest();
                for (int filter_y = filter_y_start; filter_y < filter_y_end;
                    ++filter_y) {
                    for (int filter_x = filter_x_start; filter_x < filter_x_end;
                        ++filter_x) {
                    const int in_x = in_x_origin + filter_x;
                    const int in_y = in_y_origin + filter_y;
                    max = std::max(
                        max,
                        input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
                    }
                }
                output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
                    ActivationFunctionWithMinMax(max, params.float_activation_min,
                                                params.float_activation_max);
                }
            }
        }
    }
}


} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_MAX_POOLING_HPP_