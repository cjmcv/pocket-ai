#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_CONV_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_CONV_HPP_

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
} ConvParams;

// ref: tflite_micro\tensorflow\lite\kernels\internal\reference\integer_ops: ConvPerChannel
// Fixed-point per-channel-quantization convolution reference kernel.
inline void Conv(const ConvParams& params) {
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const float output_activation_min = params.float_activation_min;
    const float output_activation_max = params.float_activation_max;

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

    // (void)im2col_data;   // only used in optimized code.
    // (void)im2col_shape;  // only used in optimized code.
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.dims[3];
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data) {
        int size = GetShapeFlatSize(bias_shape);
        PAI_DCHECK_EQ(size, output_depth);
    }
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
    // For groups == 0
    int M = output_height * output_width;
    int N = output_depth;
    int K = input_depth * filter_height * filter_width;
    float *scratch_buffer_2col = (float *)params.temp_buffer; // M*K*sizeof(float)
    for (int batch = 0; batch < batches; ++batch) {
        const float* in  = input_data + batch * input_height * input_width * input_depth;
        float* out = output_data + batch * output_height * output_width * output_depth;
        Im2Col(scratch_buffer_2col, in, input_depth, output_height, output_width, input_height, input_width, filter_height, filter_width,
                pad_height, /* pad_top*/ pad_height, /* pad_bottom*/ pad_width, /* pad_left */ pad_width, /*pad_right*/
                stride_height, stride_width, dilation_height_factor, dilation_width_factor);

        GemmTransB(M, N, K, scratch_buffer_2col, filter_data, out, bias_data);

        for (int i = 0; i < output_height * output_width; i++) {
            for (int oc = 0; oc < output_depth; ++oc) {
                out[i*output_depth+oc] = ActivationFunctionWithMinMax(out[i*output_depth+oc],
                                                                      output_activation_min,
                                                                      output_activation_max);
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
                    float total = 0.f;
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
                            for (int in_channel = 0; in_channel < filter_input_depth;
                                ++in_channel) {
                                float input_value =
                                    input_data[Offset(input_shape, batch, in_y, in_x,
                                                    in_channel + group * filter_input_depth)];
                                float filter_value = filter_data[Offset(
                                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                                total += (input_value * filter_value);
                            }
                        }
                    }
                    float bias_value = 0.0f;
                    if (bias_data) {
                        bias_value = bias_data[out_channel];
                    }
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        ActivationFunctionWithMinMax(total + bias_value,
                                                    output_activation_min,
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