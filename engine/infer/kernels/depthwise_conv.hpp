#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_DEPTHWISE_CONV_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_DEPTHWISE_CONV_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;

    PaddingValues padding_values;
    int16_t stride_height;
    int16_t stride_width;
    int16_t dilation_height_factor;
    int16_t dilation_width_factor;
    int16_t depth_multiplier;
    
    int32_t* output_multiplier;
    int32_t* output_shift;
    // uint8_t, etc, activation params.
    int32_t float_activation_min;
    int32_t float_activation_max;
    //
    Tensor filter_tensor;
    Tensor bias_tensor;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
} DepthwiseParams;

// ref: tensorflow\lite\kernels\internal\reference\depthwiseconv_float.h: DepthwiseConv
inline void DepthwiseConv(const DepthwiseParams& params) {
    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
    const Shape& input_shape = params.input_tensor->shape;
    const int8_t* input_data = (int8_t*)params.input_tensor->data;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    const Shape& output_shape = params.output_tensor->shape;
    int8_t* output_data = (int8_t*)params.output_tensor->data;

    PAI_DCHECK_EQ(params.filter_tensor.type, kPaiInferInt8);
    const Shape& filter_shape = params.filter_tensor.shape;
    const int8_t* filter_data = (int8_t*)params.filter_tensor.data;

    PAI_DCHECK_EQ(params.bias_tensor.type, kPaiInferInt32);
    const Shape& bias_shape = params.bias_tensor.shape;
    const int32_t* bias_data = (int32_t*)params.bias_tensor.data;

    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const float output_activation_min = params.float_activation_min;
    const float output_activation_max = params.float_activation_max;
    PAI_DCHECK_EQ(input_shape.dims_count, 4);
    PAI_DCHECK_EQ(filter_shape.dims_count, 4);
    PAI_DCHECK_EQ(output_shape.dims_count, 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.dims[1];
    const int input_width = input_shape.dims[2];
    const int input_depth = input_shape.dims[3];
    const int filter_height = filter_shape.dims[1];
    const int filter_width = filter_shape.dims[2];
    const int output_height = output_shape.dims[1];
    const int output_width = output_shape.dims[2];
    PAI_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    PAI_DCHECK_EQ(GetShapeFlatSize(bias_shape), output_depth);

    for (int b = 0; b < batches; ++b) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int ic = 0; ic < input_depth; ++ic) {
                    for (int m = 0; m < depth_multiplier; m++) {
                        const int oc = m + ic * depth_multiplier;
                        const int in_x_origin = (out_x * stride_width) - pad_width;
                        const int in_y_origin = (out_y * stride_height) - pad_height;
                        float total = 0.f;
                        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int in_y =
                                    in_y_origin + dilation_height_factor * filter_y;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                float input_value =
                                    input_data[Offset(input_shape, b, in_y, in_x, ic)];
                                float filter_value = filter_data[Offset(
                                    filter_shape, 0, filter_y, filter_x, oc)];
                                total += (input_value * filter_value);
                                }
                            }
                        }
                        float bias_value = 0.0f;
                        if (bias_data) {
                            bias_value = bias_data[oc];
                        }
                        output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                            ActivationFunctionWithMinMax(total + bias_value,
                                                        output_activation_min,
                                                        output_activation_max);
                    }
                }
            }
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_DEPTHWISE_CONV_PER_CHANNEL_HPP_