#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"
#include "engine/infer/kernels/common.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    
    float float_activation_min;
    float float_activation_max;
    //
    Tensor filter_tensor;
    Tensor bias_tensor;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
    void *temp_buffer;
} FullyConnectedParams;

// ref: tensorflow\lite\kernels\internal\reference\fully_connected.h: FullyConnected
inline void FullyConnected(const FullyConnectedParams& params) {

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
    const float* input_data = (float*)params.input_tensor->data;
    const Shape &input_shape = params.input_tensor->shape;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    float* output_data = (float*)params.output_tensor->data;
    const Shape &output_shape = params.output_tensor->shape;
    PAI_DCHECK_EQ(output_shape.dims_count, 2);

    PAI_DCHECK_EQ(params.filter_tensor.type, kPaiInferFloat32);
    const float* filter_data = (float*)params.filter_tensor.data;
    const Shape &filter_shape = params.filter_tensor.shape;
    PAI_DCHECK_GE(filter_shape.dims_count, 2);

    PAI_DCHECK_EQ(params.bias_tensor.type, kPaiInferFloat32);
    const float* bias_data = (float*)params.bias_tensor.data;
    const Shape &bias_shape = params.bias_tensor.shape;

    const float output_activation_min = params.float_activation_min;
    const float output_activation_max = params.float_activation_max;
    // TODO(b/62193649): This really should be:
    //     const int batches = ArraySize(output_dims, 1);
    // but the current --variable_batch hack consists in overwriting the 3rd
    // dimension with the runtime batch size, as we don't keep track for each
    // array of which dimension is the batch dimension in it.
    const int output_dims_count = output_shape.dims_count;
    const int weights_dims_count = filter_shape.dims_count;
    const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1); // Except the last one.
    const int output_depth = MatchingDim(filter_shape, weights_dims_count - 2,
                                        output_shape, output_dims_count - 1);
    const int accum_depth = filter_shape.dims[weights_dims_count - 1];


#if 1
    int32_t *out = (int32_t*)params.temp_buffer;
    int M = batches;
    int N = output_depth; // Total number of input dimensions except for batches
    int K = accum_depth;  // Total number of output dimensions except for batches
    GemmTransB(M, N, K, input_data, filter_data, output_data, bias_data);

    for (int i = 0; i < batches; i++) {
        for (int oc = 0; oc < output_depth; ++oc) {
            output_data[i*output_depth+oc] = ActivationFunctionWithMinMax(
                    output_data[i*output_depth+oc], output_activation_min, output_activation_max);
        }
    }
#else
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            float total = 0.f;
            for (int d = 0; d < accum_depth; ++d) {
                total += input_data[b * accum_depth + d] *
                        filter_data[out_c * accum_depth + d];
            }
            float bias_value = 0.0f;
            if (bias_data) {
                bias_value = bias_data[out_c];
            }
            output_data[out_c + output_depth * b] = ActivationFunctionWithMinMax(
                total + bias_value, output_activation_min, output_activation_max);
        }
    }
#endif
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_FULLY_CONNECTED_HPP_