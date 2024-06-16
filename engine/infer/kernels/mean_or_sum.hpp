#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_SUM_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_SUM_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

#include "binary_common.hpp"

namespace pai {
namespace infer {

// ref: tensorflow\lite\kernels\internal\reference\reduce.h: Mean
// ref: tensorflow\lite\kernels\internal\reference\reduce.h: ReduceGeneric
// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis.
inline bool MeanOrSum(const MeanOrSumParams &params) {

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
    const float* input_data = (float*)params.input_tensor->data;
    
    Shape &input_shape = params.input_tensor->shape;
    const int* input_dims = input_shape.dims;
    const int input_num_dims = input_shape.dims_count;
    
    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
    float* output_data = (float*)params.output_tensor->data;
    
    Shape &output_shape = params.output_tensor->shape;
    const int32_t* output_dims = output_shape.dims;
    const int32_t output_num_dims = output_shape.dims_count;

    const int32_t* axis = params.axis;
    const int num_axis_dimensions = params.num_axis;
    float* temp_sum = (float*)(*params.temp_buffer);

    // Reset output data.
    size_t num_outputs = 1;
    for (int idx = 0; idx < output_num_dims; ++idx) {
        size_t current = static_cast<size_t>(output_dims[idx]);
        // Overflow prevention.
        if (num_outputs > std::numeric_limits<size_t>::max() / current) {
            return false;
        }
        num_outputs *= current;
    }
    for (size_t idx = 0; idx < num_outputs; ++idx) {
        output_data[idx] = float();
        temp_sum[idx] = float();
    }

    // Return early when input shape has zero dim. This is done after initializing
    // data for output tensor because there are cases that the input tensor is
    // empty but output tensor is not. In that case, output tensor should be
    // filled with init_value.
    if (params.is_compute_sum) {
        for (int i = 0; i < input_num_dims; ++i) {
            if (input_dims[i] == 0) return true;
        }
    }

    // Resolve axis.
    int temp_index[kMaxNumberOfAxis];
    int resolved_axis[kMaxNumberOfReducedAxis];
    int num_resolved_axis = 0;
    if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                    &num_resolved_axis)) {
        return false;
    }

    if (params.is_compute_sum) {
        return ReduceSumImpl<float, float>(input_data, input_dims, output_dims, input_num_dims,
                                            output_num_dims, resolved_axis, num_resolved_axis,
                                            temp_index, output_data);
    }
            
    if (!ReduceSumImpl<float, float>(input_data, input_dims, output_dims, input_num_dims,
                            output_num_dims, resolved_axis, num_resolved_axis,
                            temp_index, temp_sum)) {
        return false;
    }

    // Calculate mean by dividing output_data by num of aggregated element.
    size_t num_elements_in_axis = 1;
    for (int idx = 0; idx < num_resolved_axis; ++idx) {
        size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
        // Overflow prevention.
        if (current > (std::numeric_limits<size_t>::max() / num_elements_in_axis)) {
            return false;
        }
        num_elements_in_axis *= current;
    }

    if (num_elements_in_axis > 0) {
        for (size_t idx = 0; idx < num_outputs; ++idx) {
            output_data[idx] =
                static_cast<float>(temp_sum[idx] / static_cast<float>(num_elements_in_axis));
        }
    }
    return true;
}

// // Computes the generic value (i.e., sum/max/min/prod) of elements across
// // dimensions given in axis. It needs to pass in init_value and reducer.
// inline bool ReduceGeneric(const float* input_data, const int* input_dims,
//                           const int input_num_dims, float* output_data,
//                           const int* output_dims, const int output_num_dims,
//                           const int* axis, const int64_t num_axis_dimensions) {

//     // Reset output data.
//     size_t num_elements = 1;
//     for (int idx = 0; idx < output_num_dims; ++idx) {
//         size_t current = static_cast<size_t>(output_dims[idx]);
//         // Overflow prevention.
//         if (current > 0 &&
//             num_elements > std::numeric_limits<size_t>::max() / current) {
//             return false;
//         }
//         num_elements *= current;
//     }
//     for (size_t idx = 0; idx < num_elements; ++idx) {
//         output_data[idx] = 0.0f;
//     }

//     // Return early when input shape has zero dim. This is done after initializing
//     // data for output tensor because there are cases that the input tensor is
//     // empty but output tensor is not. In that case, output tensor should be
//     // filled with init_value.
//     for (int i = 0; i < input_num_dims; ++i) {
//         if (input_dims[i] == 0) return true;
//     }

//     // Resolve axis.
//     int temp_index[kMaxNumberOfAxis];
//     int resolved_axis[kMaxNumberOfReducedAxis];
//     int num_resolved_axis = 0;
//     if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
//                     &num_resolved_axis)) {
//         return false;
//     }

//     return ReduceSumImpl<float, float>(input_data, input_dims, output_dims, input_num_dims,
//                         output_num_dims, resolved_axis, num_resolved_axis,
//                         temp_index, output_data);
// }


} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_SUM_HPP_