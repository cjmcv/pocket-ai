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
    // float
    float float_activation_min;
    float float_activation_max;
    //
    Tensor filter_tensor;
    Tensor bias_tensor;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
    void *temp_buffer;
} TransposeConvParams;

// ref: tensorflow/lite/kernels/internal/reference/transpose_conv.h#27 -> TransposeConv
inline void TransposeConv(const TransposeConvParams& params) {

    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;

    PAI_DCHECK_EQ(params.filter_tensor.type, kPaiInferFloat32);
    const Shape &filter_shape = params.filter_tensor.shape;
    const float* filter_data = (float*)params.filter_tensor.data;

    PAI_DCHECK_EQ(params.bias_tensor.type, kPaiInferFloat32);
    const Shape &bias_shape = params.bias_tensor.shape;
    const float* bias_data = (float*)params.bias_tensor.data;

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
    const Shape &input_shape = params.input_tensor->shape;
    const float* input_data = (float*)params.input_tensor->data;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    const Shape &output_shape = params.output_tensor->shape;
    float* output_data = (float*)params.output_tensor->data;

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
    const float output_activation_min = params.float_activation_min;
    const float output_activation_max = params.float_activation_max;
    if (bias_data) {
        int size = GetShapeFlatSize(bias_shape);
        PAI_DCHECK_EQ(size, output_depth);
    }

#if 1
    float *gemm_buffer = (float *)params.temp_buffer; // (input_height * input_width) * (output_depth * filter_height * filter_width) * 4
    for (int i = 0; i < batches; ++i) {
        const float* input_ptr  = input_data + i * input_height * input_width * input_depth;
        float* output_ptr = output_data + i * output_height * output_width * output_depth;
        int M = input_height * input_width;
        int N = output_depth * filter_height * filter_width;
        int K = input_depth;
        GemmTransB(M, N, K, input_ptr, filter_data, gemm_buffer, nullptr);
        Col2Im(output_ptr, gemm_buffer, output_depth, output_height, output_width, filter_height, filter_width,
                pad_height, /* pad_top*/ pad_height, /* pad_bottom*/ pad_width, /* pad_left */ pad_width, /*pad_right*/ 
                stride_height, stride_width, params.dilation_height_factor, params.dilation_width_factor);

        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    float acc = output_data[Offset(output_shape, i, out_y, out_x,
                                                    out_channel)];
                    if (bias_data) acc += bias_data[out_channel];

                    output_data[Offset(output_shape, i, out_y, out_x, out_channel)] =
                        ActivationFunctionWithMinMax(acc, output_activation_min,
                                                    output_activation_max);
                }
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
    for (int i = 0; i < num_elements; i++) {
        output_data[i] = 0.0f;
    }

    // Loop through input elements one at a time.
    for (int batch = 0; batch < batches; ++batch) {
        for (int in_y = 0; in_y < input_height; ++in_y) {
            for (int in_x = 0; in_x < input_width; ++in_x) {
                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    // Loop through the output elements it will influence
                    const int out_x_origin = (in_x * stride_width) - pad_width;
                    const int out_y_origin = (in_y * stride_height) - pad_height;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int out_channel = 0; out_channel < output_depth;
                                ++out_channel) {
                                // Compute output element location
                                const int out_x = out_x_origin + filter_x;
                                const int out_y = out_y_origin + filter_y;
                                // We cannot accumulate out of bounds
                                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                                    (out_y < output_height)) {
                                float input_value = input_data[Offset(
                                    input_shape, batch, in_y, in_x, in_channel)];
                                float filter_value =
                                    filter_data[Offset(filter_shape, out_channel, filter_y,
                                                        filter_x, in_channel)];
                                output_data[Offset(output_shape, batch, out_y, out_x,
                                                    out_channel)] +=
                                    input_value * filter_value;
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
                    float acc = output_data[Offset(output_shape, batch, out_y, out_x,
                                                    out_channel)];
                    if (bias_data) acc += bias_data[out_channel];

                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        ActivationFunctionWithMinMax(acc, output_activation_min,
                                                    output_activation_max);
                }
            }
        }
    }
#endif
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_CONV_HPP_