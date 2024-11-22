#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_TRANSPOSE_CONV_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_TRANSPOSE_CONV_HPP_

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
    int32_t *output_shift;
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
} TransposeConvPerChannelParams;

// ref: tensorflow/lite/kernels/internal/reference/transpose_conv.h#27 -> TransposeConv
inline void TransposeConvPerChannel(const TransposeConvPerChannelParams& params) {

    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int32_t input_offset = params.input_offset;  // r = s(q - Z)
    const int32_t filter_offset = params.weights_offset; // fixed to 0
    const int32_t output_offset = params.output_offset;
    const int32_t *output_multiplier = params.output_multiplier; // scale
    const int32_t *output_shift = params.output_shift;

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

    PAI_DCHECK_EQ(input_shape.dims_count, 4);
    PAI_DCHECK_EQ(filter_shape.dims_count, 4);
    PAI_DCHECK_EQ(output_shape.dims_count, 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    const int input_height = input_shape.dims[1];
    const int input_width = input_shape.dims[2];
    const int filter_height = filter_shape.dims[1];
    const int filter_width = filter_shape.dims[2];
    const int output_height = output_shape.dims[1];
    const int output_width = output_shape.dims[2];
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    if (bias_data) {
        int size = GetShapeFlatSize(bias_shape);
        PAI_DCHECK_EQ(size, output_depth);
    }

#if 1
    int M = input_height * input_width;
    int N = output_depth * filter_height * filter_width;
    int K = input_depth;

    int32_t *scratch_buffer_gemm = (int32_t*)params.temp_buffer; // M*N = (input_height * input_width) * (output_depth * filter_height * filter_width) * sizeof(int32_t)
    int32_t *scratch_buffer_2im = (int32_t*)((int8_t*)params.temp_buffer + M*N * sizeof(int32_t)); // output_height * output_width * output_depth * sizeof(int32_t) 
    for (int b = 0; b < batches; ++b) {
        const int8_t *in = input_data + b * input_height * input_width * input_depth;
        int8_t *out = output_data + b * output_height * output_width * output_depth;

        GemmTransBQuant(M, N, K, in, filter_data, scratch_buffer_gemm, NULL, input_offset);
        Col2Im(scratch_buffer_2im, scratch_buffer_gemm, output_depth, output_height, output_width, filter_height, filter_width,
               pad_height, /* pad_top*/ pad_height, /* pad_bottom*/ pad_width, /* pad_left */ pad_width, /*pad_right*/
               stride_height, stride_width, params.dilation_height_factor, params.dilation_width_factor);

        for (int i = 0; i < output_height * output_width; ++i) {
            for (int c = 0; c < output_depth; ++c) {
                int idx = i * output_depth + c;
                scratch_buffer_2im[idx] += bias_data[c];

                int32_t acc = MultiplyByQuantizedMultiplier(scratch_buffer_2im[idx], 
                                                            output_multiplier[c], output_shift[c]);
                acc += output_offset;
                acc = std::max(acc, output_activation_min);
                acc = std::min(acc, output_activation_max);
                out[idx] = static_cast<int8_t>(acc);
            }
        }
    }
#else
    // Although transpose convolution simplifies to convolution with transposed
    // weights for strides of 1, non-unitary striding complicates matters. To
    // keep this reference implementation as clear as possible, we use a
    // "scatter" access pattern, where we loop through all the input elements,
    // computing their influence on the output, rather than looping through the
    // output elements in the typical "gather" access pattern of a conv. We
    // therefore must initialize the output array to zero.
    const int num_elements = GetShapeFlatSize(output_shape);
    // We need to initialize scratch_buffer to all 0s, as we apply the same
    // 'scatter' based trick as in float version.
    int32_t *scratch_buffer = (int32_t*)params.temp_buffer;
    memset(scratch_buffer, 0, num_elements * sizeof(int32_t));

    // Loop through input elements one at a time.
    for (int batch = 0; batch < batches; ++batch) {
        for (int in_y = 0; in_y < input_height; ++in_y) {
            for (int in_x = 0; in_x < input_width; ++in_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    // Loop through the output elements it will influence.
                    const int out_x_origin = (in_x * stride_width) - pad_width;
                    const int out_y_origin = (in_y * stride_height) - pad_height;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                                // Compute output element location.
                                const int out_x = out_x_origin + filter_x;
                                const int out_y = out_y_origin + filter_y;
                                // We cannot accumulate out of bounds.
                                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) && (out_y < output_height)) {
                                    const int8_t input_value = input_data[Offset(
                                        input_shape, batch, in_y, in_x, in_channel)];
                                    const int8_t filter_value =
                                        filter_data[Offset(filter_shape, out_channel, filter_y,
                                                          filter_x, in_channel)];
                                    scratch_buffer[Offset(output_shape, batch, out_y, out_x,
                                                          out_channel)] +=
                                        (input_value + input_offset) * filter_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                  int32_t acc = scratch_buffer[Offset(output_shape, batch, out_y, out_x,
                                                      out_channel)];
                  if (bias_data) {
                      acc += bias_data[out_channel];
                  }
                  acc = MultiplyByQuantizedMultiplier(
                      acc, output_multiplier[out_channel], output_shift[out_channel]);
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

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_CONV_HPP_