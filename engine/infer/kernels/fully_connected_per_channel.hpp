#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_PER_CHANNEL_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_PER_CHANNEL_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    
    int32_t input_offset;
    int32_t weights_offset;
    int32_t output_offset;
    int32_t* output_multiplier;
    int* output_shift;
    // uint8_t, etc, activation params.
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
    //
    Tensor filter_tensor;
    Tensor bias_tensor;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;

} FullyConnectedPerChannelParams;

// ref: tensorflow\lite\kernels\internal\reference\integer_ops\fully_connected.h: FullyConnectedPerChannel
// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.
void FullyConnectedPerChannel(const FullyConnectedPerChannelParams& params) {
    const int32_t input_offset = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    PAI_DCHECK_LE(output_activation_min, output_activation_max);

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
    const int8_t* input_data = (int8_t*)params.input_tensor->data;
    const Shape &input_shape = params.input_tensor->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    int8_t* output_data = (int8_t*)params.output_tensor->data;
    const Shape &output_shape = params.output_tensor->shape;
    PAI_DCHECK_EQ(output_shape.dims_count, 2);

    PAI_DCHECK_EQ(params.filter_tensor.type, kPaiInferInt8);
    const int8_t* filter_data = (int8_t*)params.filter_tensor.data;
    const Shape &filter_shape = params.filter_tensor.shape;
    PAI_DCHECK_GE(filter_shape.dims_count, 2);

    PAI_DCHECK_EQ(params.bias_tensor.type, kPaiInferInt32);
    const int32_t* bias_data = (int32_t*)params.bias_tensor.data;
    const Shape &bias_shape = params.bias_tensor.shape;

    const int filter_dim_count = filter_shape.dims_count;
    const int batches = output_shape.dims[0];
    const int output_depth = output_shape.dims[1];
    PAI_DCHECK_LE(output_depth, filter_shape.dims[filter_dim_count - 2]);
    const int accum_depth = filter_shape.dims[filter_dim_count - 1];
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            int32_t acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
                int32_t input_val = input_data[b * accum_depth + d];
                int32_t filter_val = filter_data[out_c * accum_depth + d];
                acc += filter_val * (input_val + input_offset);
            }
            if (bias_data) {
                acc += bias_data[out_c];
            }
            int32_t acc_scaled = MultiplyByQuantizedMultiplier(
                acc, params.output_multiplier[out_c], params.output_shift[out_c]);
            acc_scaled += output_offset;
            acc_scaled = std::max(acc_scaled, output_activation_min);
            acc_scaled = std::min(acc_scaled, output_activation_max);
            output_data[out_c + output_depth * b] =
                static_cast<int8_t>(acc_scaled);
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_PER_CHANNEL_HPP_