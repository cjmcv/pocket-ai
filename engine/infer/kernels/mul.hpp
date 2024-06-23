#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_MUL_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_MUL_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

// For Add, Sub, Mul ops.  (binary)
typedef struct {
    uint32_t op_id;

    bool requires_broadcast;

    // float activation params.
    float float_activation_min;
    float float_activation_max;

    Tensor *input_tensor[2];
    Tensor *output_tensor;
} MulParams;

// data2 broadcase to data1
inline void MulBroadcast(const float* data1, const Shape &shape1, const float* data2, const Shape &shape2, 
                        float activation_min, float activation_max, uint32_t flat_size, float *output_data) {
    if (GetShapeFlatSize(shape2) == 1) {
        for (int i = 0; i < flat_size; ++i) {
            output_data[i] = ActivationFunctionWithMinMax(
                data1[i] * data2[0], activation_min, activation_max);
        }
    }
    else if (shape2.dims[1] == 1 && shape2.dims[2] == 1) {
        PAI_DCHECK_EQ(shape1.dims[3], shape2.dims[3]);
        int channels = shape1.dims[3];
        for (int i = 0; i < flat_size / channels; ++i) {
            for (int ch = 0; ch < channels; ch++) {
                output_data[i * channels + ch] = ActivationFunctionWithMinMax(
                    data1[i * channels + ch] * data2[ch], activation_min, activation_max);
            }
        }
    }
    else {
        // not supported;
        PAI_DCHECK(0);
    }
}

// ref: tensorflow\lite\kernels\internal\reference\add.h: Add
inline void Mul(const MulParams& params) {

    PAI_DCHECK_EQ(params.input_tensor[0]->type, kPaiInferFloat32);
    const Shape& input1_shape = params.input_tensor[0]->shape;
    const float* input1_data = (float*)params.input_tensor[0]->data;

    PAI_DCHECK_EQ(params.input_tensor[1]->type, kPaiInferFloat32);
    const Shape& input2_shape = params.input_tensor[1]->shape;
    const float* input2_data = (float*)params.input_tensor[1]->data;

    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    const Shape& output_shape = params.output_tensor->shape;
    float* output_data = (float*)params.output_tensor->data;

    float activation_min = params.float_activation_min;
    float activation_max = params.float_activation_max;

    int flat_size = GetShapeFlatSize(output_shape);
    if (params.requires_broadcast == false) {
        PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input1_shape));
        PAI_DCHECK_EQ(flat_size, GetShapeFlatSize(input2_shape));
        for (int i = 0; i < flat_size; ++i) {
            output_data[i] = ActivationFunctionWithMinMax(
                input1_data[i] * input2_data[i], activation_min, activation_max);
        }
    }
    else {
        if (GetShapeFlatSize(input1_shape) > GetShapeFlatSize(input2_shape)) {
            MulBroadcast(input1_data, input1_shape, input2_data, input2_shape, 
                        activation_min, activation_max, flat_size, output_data);
        }
        else {
            MulBroadcast(input2_data, input2_shape, input1_data, input1_shape, 
                        activation_min, activation_max, flat_size, output_data);
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_MUL_HPP_