#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_QUANT_HPP_

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
    int32_t output_multiplier;
    int output_shift;
    // uint8_t, etc, activation params.
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
    //
    Tensor filter_tensor;
    Tensor bias_tensor;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;

} FullyConnectedQuantParams;

// ref: tensorflow\lite\kernels\internal\reference\integer_ops\fully_connected.h: FullyConnected
void FullyConnectedQuant(const FullyConnectedQuantParams& params) {
    const int32_t input_offset = params.input_offset;
    const int32_t filter_offset = params.weights_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_multiplier = params.output_multiplier;
    const int output_shift = params.output_shift;
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

    PAI_DCHECK_GE(filter_shape.dims_count, 2);
    PAI_DCHECK_GE(output_shape.dims_count, 1);

    const int filter_dim_count = filter_shape.dims_count;
    const int output_dim_count = output_shape.dims_count;
    const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1); // Expect for the last dimension, the size of all dimensions is combined as batch.
    const int output_depth = output_shape.dims[output_dim_count - 1]; // The last one is output_depth
    PAI_DCHECK_LE(output_depth, filter_shape.dims[filter_dim_count - 2]); // output_depth also equal to the second to last dims.
    const int accum_depth = filter_shape.dims[filter_dim_count - 1]; // The last dim of filter is accum_depth

    // printf("fully_connected: %d, %d, %d, %d", filter_dim_count, batches, output_depth, accum_depth);
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            int32_t acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
                int32_t input_val = input_data[b * accum_depth + d];
                int32_t filter_val = filter_data[out_c * accum_depth + d];
                acc += (filter_val + filter_offset) * (input_val + input_offset);
            }
            if (bias_data) {
                acc += bias_data[out_c];
            }
            int32_t acc_scaled =
                MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
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

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_QUANT_HPP_