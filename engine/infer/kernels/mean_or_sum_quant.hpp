#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_QUANT_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

#include "binary_common.hpp"

namespace pai {
namespace infer {

// ref: tensorflow\lite\kernels\internal\reference\reduce.h: EvalMeanHelper -> EvalIntegerMean -> QuantizedMeanOrSum
// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis for quantized values.
inline bool MeanOrSumQuant(const MeanOrSumQuantParams &params) {

    PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
    const int8_t* input_data = (int8_t*)params.input_tensor->data;
    int32_t input_zero_point = params.input_zero_point;
    
    Shape &input_shape = params.input_tensor->shape;
    const int* input_dims = input_shape.dims;
    const int input_num_dims = input_shape.dims_count;
    
    PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
    int8_t* output_data = (int8_t*)params.output_tensor->data;
    int32_t output_multiplier = params.multiplier;
    int32_t output_shift = params.shift;
    int32_t output_zero_point = params.output_zero_point;
    
    Shape &output_shape = params.output_tensor->shape;
    const int32_t* output_dims = output_shape.dims;
    const int32_t output_num_dims = output_shape.dims_count;

    const int32_t* axis = params.axis;
    const int num_axis_dimensions = params.num_axis;
    int32_t* temp_sum = (int32_t*)(params.temp_buffer);

    int temp_index[kMaxNumberOfAxis];
    int resolved_axis[kMaxNumberOfReducedAxis];

    const int32_t kMinValue = std::numeric_limits<int8_t>::min();
    const int32_t kMaxValue = std::numeric_limits<int8_t>::max();

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
        output_data[idx] = int8_t();
        temp_sum[idx] = int32_t();
    }

    // Return early when input shape has zero dim. This is done after initializing
    // data for output tensor because there are cases that the input tensor is
    // empty but output tensor is not. In that case, output tensor should be
    // filled with init_value.
    for (int i = 0; i < input_num_dims; ++i) {
        if (input_dims[i] == 0) return true;
    }

    // Resolve axis.
    int num_resolved_axis = 0;
    if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                    &num_resolved_axis)) {
        return false;
    }

    if (!ReduceSumImpl<int8_t, int32_t>(input_data, input_dims, output_dims, input_num_dims,
                            output_num_dims, resolved_axis, num_resolved_axis,
                            temp_index, temp_sum)) {
        return false;
    }

    // Calculate mean by dividing output_data by num of aggregated element.
    int64_t num_elements_in_axis = 1;
    for (int idx = 0; idx < num_resolved_axis; ++idx) {
        size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
        // Overflow prevention.
        if (current > static_cast<size_t>(std::numeric_limits<int64_t>::max() /
                                        num_elements_in_axis)) {
            return false;
        }
        num_elements_in_axis *= current;
    }

    if (num_elements_in_axis == 0) {
        return true;
    }

    // Readapt output rescaling when calculating the mean to integrate a
    // 1/num_elements_in_axis multiplier.
    if (!params.is_compute_sum) {
        PAI_DCHECK_GE(num_elements_in_axis, 0);
        int shift =
            63 - CountLeadingZeros(static_cast<uint64_t>(num_elements_in_axis));
        // To avoid any overflow risk 'shift' should be <= 32 and to satisfy
        // 'MultiplyByQuantizedMultiplier' pre-conditions 'output_shift - shift'
        // should be >= -31. Clamp the value at the price of some precision loss.
        shift = std::min(shift, 32);
        shift = std::min(shift, 31 + output_shift);
        output_multiplier = static_cast<int32_t>(
            (static_cast<int64_t>(output_multiplier) << shift) /
            num_elements_in_axis);
        output_shift = output_shift - shift;
    }

    for (size_t idx = 0; idx < num_outputs; ++idx) {
        const int32_t shifted_sum =
            static_cast<int32_t>(temp_sum[idx] - input_zero_point * num_elements_in_axis);
        int32_t output = MultiplyByQuantizedMultiplier(
                            shifted_sum, output_multiplier, output_shift) +
                        output_zero_point;
        output = std::min(std::max(output, kMinValue), kMaxValue);
        output_data[idx] = static_cast<int8_t>(output);
    }
    return true;
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_QUANT_HPP_