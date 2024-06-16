#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_DEPTHWISE_CONV_PER_CHANNEL_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_DEPTHWISE_CONV_PER_CHANNEL_HPP_

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
    // uint8_t inference params.
    // TODO(b/65838351): Use smaller types if appropriate.
    int32_t input_offset;
    int32_t weights_offset;
    int32_t output_offset;
    
    int32_t* output_multiplier;
    int32_t* output_shift;
    // uint8_t, etc, activation params.
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
    //
    Tensor filter_tensor;
    Tensor bias_tensor;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
} DepthwisePerChannelParams;

// ref: tensorflow\lite\kernels\internal\reference\integer_ops\depthwise_conv.h: DepthwiseConvPerChannel
inline void DepthwiseConvPerChannel(const DepthwisePerChannelParams& params) {

    const int32_t* output_multiplier = params.output_multiplier;
    const int32_t* output_shift = params.output_shift;

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
    
    // Get parameters.
    // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t input_offset = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    // Check dimensions of the tensors.
    PAI_DCHECK_EQ(input_shape.dims_count, 4);
    PAI_DCHECK_EQ(filter_shape.dims_count, 4);
    PAI_DCHECK_EQ(output_shape.dims_count, 4);

    PAI_DCHECK_LE(output_activation_min, output_activation_max);
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

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    for (int m = 0; m < depth_multiplier; ++m) {
                        const int output_channel = m + in_channel * depth_multiplier;
                        const int in_x_origin = (out_x * stride_width) - pad_width;
                        const int in_y_origin = (out_y * stride_height) - pad_height;
                        int32_t acc = 0;
                        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int in_y =
                                    in_y_origin + dilation_height_factor * filter_y;
                                // Zero padding by omitting the areas outside the image.
                                const bool is_point_inside_image =
                                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height);
                                if (is_point_inside_image) {
                                    int32_t input_val = input_data[Offset(
                                        input_shape, batch, in_y, in_x, in_channel)];
                                    int32_t filter_val = filter_data[Offset(
                                        filter_shape, 0, filter_y, filter_x, output_channel)];
                                    // Accumulate with 32 bits accumulator.
                                    // In the nudging process during model quantization, we force
                                    // real value of 0.0 be represented by a quantized value. This
                                    // guarantees that the input_offset is a int8_t, even though
                                    // it is represented using int32_t. int32_t += int8_t *
                                    // (int8_t - int8_t) so the highest value we can get from each
                                    // accumulation is [-127, 127] * ([-128, 127] -
                                    // [-128, 127]), which is [-32512, 32512]. log2(32512)
                                    // = 14.98, which means we can accumulate at least 2^16
                                    // multiplications without overflow. The accumulator is
                                    // applied to a filter so the accumulation logic will hold as
                                    // long as the filter size (filter_y * filter_x * in_channel)
                                    // does not exceed 2^16, which is the case in all the models
                                    // we have seen so far.
                                    // TODO(b/174275578): Add a check to make sure the
                                    // accumulator depth is smaller than 2^16.
                                    acc += filter_val * (input_val + input_offset);
                                }
                            }
                        }
                        if (bias_data) {
                            acc += bias_data[output_channel];
                        }
                        acc = MultiplyByQuantizedMultiplier(
                            acc, output_multiplier[output_channel],
                            output_shift[output_channel]);
                        acc += output_offset;
                        acc = std::max(acc, output_activation_min);
                        acc = std::min(acc, output_activation_max);
                        output_data[Offset(output_shape, batch, out_y, out_x,
                                        output_channel)] = static_cast<int8_t>(acc);
                    }
                }
            }
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_DEPTHWISE_CONV_PER_CHANNEL_HPP_