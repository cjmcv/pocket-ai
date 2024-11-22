#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_CONV_PER_CHANNEL_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_CONV_PER_CHANNEL_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"
#include "engine/infer/kernels/common.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    // common
    PaddingValues padding_values;
    int16_t stride_height;
    int16_t stride_width;
    int16_t dilation_height_factor;
    int16_t dilation_width_factor;
    // uint8_t inference params.
    int32_t input_offset;
    int32_t weights_offset;
    int32_t output_offset;
    int32_t *output_multiplier;
    int *output_shift;
    // uint8_t, etc, activation params.
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
    //
    Tensor filter_tensor;
    Tensor bias_tensor;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
    void *temp_buffer;
} ConvPerChannelParams;

// ref: tflite_micro\tensorflow\lite\kernels\internal\reference\integer_ops: ConvPerChannel
// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(const ConvPerChannelParams& params) {
    // Get parameters.
    const int32_t input_offset = params.input_offset;  // r = s(q - Z)
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int32_t output_offset = params.output_offset;

    PAI_DCHECK_EQ(params.filter_tensor.type, kPaiInferInt8);
    const Shape &filter_shape = params.filter_tensor.shape;
    const int8_t* filter_data = (int8_t*)params.filter_tensor.data;

    PAI_DCHECK_EQ(params.bias_tensor.type, kPaiInferInt32);
    const Shape &bias_shape = params.bias_tensor.shape;
    const int32_t* bias_data = (int32_t*)params.bias_tensor.data;

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
    const Shape &input_shape = params.input_tensor->shape;
    const int8_t* input_data = (int8_t*)params.input_tensor->data;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    const Shape &output_shape = params.output_tensor->shape;
    int8_t* output_data = (int8_t*)params.output_tensor->data;

    // Set min and max value of the output.
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    // printf("%d, (%d, %d), (%d, %d), (%d, %d), %d, (%d, %d).\n",
    //        input_offset, stride_width, stride_height,
    //        dilation_width_factor, dilation_height_factor,
    //        pad_width, pad_height,
    //        output_offset,
    //        output_activation_min, output_activation_max);
    // Consistency check.
    PAI_DCHECK_EQ(input_shape.dims_count, 4);
    PAI_DCHECK_EQ(filter_shape.dims_count, 4);
    PAI_DCHECK_EQ(output_shape.dims_count, 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.dims[3];
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3); // 权重的 num 是卷积核个数，等于输出层的 channel 个数，per channel是针对每个卷积核和每个输出channel
    if (bias_data) {
        int buffer_size = 1;
        for (int i = 0; i < bias_shape.dims_count; i++) {
            buffer_size *= bias_shape.dims[i];
        }
        PAI_DCHECK_EQ(buffer_size, output_depth);
    }

    // Check dimensions of the tensors.
    const int input_height = input_shape.dims[1];
    const int input_width = input_shape.dims[2];
    const int filter_height = filter_shape.dims[1];
    const int filter_width = filter_shape.dims[2];
    const int filter_input_depth = filter_shape.dims[3];
    const int groups = input_depth / filter_input_depth;
    PAI_DCHECK_NE(groups, 0);
    PAI_DCHECK_EQ(input_depth % filter_input_depth, 0);
    const int filters_per_group = output_depth / groups;
    PAI_DCHECK_NE(filters_per_group, 0);
    const int output_height = output_shape.dims[1];
    const int output_width = output_shape.dims[2];

#if 1
    // !!!The output result is not completely consistent with the original version!!!
    // For groups == 0
    // output_height * output_width * input_channel * kernel_height * kernel_width
    // printf("pad: %d, %d.\n", pad_height, pad_width);
    int M = output_height * output_width;
    int N = output_depth;
    int K = input_depth * filter_height * filter_width;
    int8_t *scratch_buffer_2col = (int8_t *)params.temp_buffer; // M*K*sizeof(int8_t)
    int32_t *scratch_buffer_gemm = (int32_t *)((int8_t*)params.temp_buffer + M*K* sizeof(int8_t));
    for (int batch = 0; batch < batches; ++batch) {
        const int8_t* in = input_data + batch * input_height * input_width * input_depth;
        int8_t* out = output_data + batch * output_height * output_width * output_depth;
        Im2Col(scratch_buffer_2col, in, input_depth, output_height, output_width, input_height, input_width, filter_height, filter_width,
               pad_height, /* pad_top*/ pad_height, /* pad_bottom*/ pad_width, /* pad_left */ pad_width, /*pad_right*/
               stride_height, stride_width, dilation_height_factor, dilation_width_factor);

        GemmTransBQuant(M, N, K, scratch_buffer_2col, filter_data, scratch_buffer_gemm, bias_data, input_offset);

        for (int i = 0; i < output_height * output_width; i++) {
            for (int oc = 0; oc < output_depth; ++oc) {
                int32_t idx = i*output_depth+oc;
                int32_t acc_scaled = MultiplyByQuantizedMultiplier(scratch_buffer_gemm[idx], params.output_multiplier[oc], params.output_shift[oc]); // Equal to out[i] * bias_scale[oc] / output_scale[oc];
                acc_scaled += output_offset;
                acc_scaled = std::max(acc_scaled, output_activation_min);
                acc_scaled = std::min(acc_scaled, output_activation_max);
                out[idx] = static_cast<int8_t>(acc_scaled);
            }
        }
    }
#else
    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    auto group = out_channel / filters_per_group;
                    // printf("%d, %d.\n", out_channel, filters_per_group);
                    int32_t acc = 0;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;

                            // Zero padding by omitting the areas outside the image.
                            const bool is_point_inside_image =
                                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                (in_y < input_height);

                            if (!is_point_inside_image) {
                                continue;
                            }

                            for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                                int32_t input_val =
                                    input_data[Offset(input_shape, batch, in_y, in_x,
                                                      in_channel + group * filter_input_depth)];
                                int32_t filter_val = filter_data[Offset(
                                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
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
                        acc += bias_data[out_channel];
                    }
                    acc = MultiplyByQuantizedMultiplier(
                        acc, params.output_multiplier[out_channel], params.output_shift[out_channel]);
                    acc += output_offset;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        static_cast<int8_t>(acc);
                }
            }
        }
    }
#endif
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_CONV_PER_CHANNEL_HPP_