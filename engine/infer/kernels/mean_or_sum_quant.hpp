#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_QUANT_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_MEAN_QUANT_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

const int kMaxNumberOfAxis = 5;
const int kMaxNumberOfReducedAxis = 2;

// For Mean, Sum, Max ops. (reduce)
// ref: OpDataReduce
typedef struct {
    uint32_t op_id;

    bool is_compute_sum;

    int32_t multiplier;
    int32_t shift;
    int32_t input_zero_point;
    int32_t output_zero_point;

    int32_t num_output_elements;
    int32_t num_axis;
    int32_t axis[kMaxNumberOfAxis];
    
    void* temp_buffer;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
} MeanOrSumQuantParams;

// Gets offset of index if reducing on axis. When reducing, the flattened offset
// will not change, if the input index changes on the given axis. For example,
// if you have a 3D tensor and you are reducing to 2D by eliminating axis 0,
// then index (0, 1, 2) and index (1, 1, 2) will map to the same flattened
// offset.
// TODO(kanlig): uses Dims to represent dimensions.
inline size_t ReducedOutputOffset(const int num_dims, const int* dims,
                                  const int* index, const int num_axis,
                                  const int* axis) {
    if (num_dims == 0) {
        return 0;
    }
    PAI_DCHECK(dims != nullptr);
    PAI_DCHECK(index != nullptr);
    size_t offset = 0;
    for (int idx = 0; idx < num_dims; ++idx) {
        // if we need to skip this axis
        bool is_axis = false;
        if (axis != nullptr) {
            for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
                if (idx == axis[axis_idx]) {
                    is_axis = true;
                    break;
                }
            }
        }
        if (!is_axis) {
            offset = offset * static_cast<size_t>(dims[idx]) +
                    static_cast<size_t>(index[idx]);
        }
    }
    return offset;
}


// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis.
template <typename In, typename Out>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int* output_dims, const int input_num_dims,
                   const int output_num_dims, const int* axis,
                   const int num_axis, int* input_iter,
                   Out reducer(Out current, const In in), Out* output_data) {
    // Reset input iterator.
    for (int idx = 0; idx < input_num_dims; ++idx) {
        input_iter[idx] = 0;
    }
    // Iterate through input_data.
    do {
        size_t input_offset =
            ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
        size_t output_offset = ReducedOutputOffset(input_num_dims, input_dims,
                                                input_iter, num_axis, axis);
        output_data[output_offset] =
            reducer(output_data[output_offset], input_data[input_offset]);
    } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

// This method expects that output_data has been initialized.
template <typename In, typename Out>
inline bool ReduceSumImpl(const In* input_data, const int* input_dims,
                          const int* output_dims, const int input_num_dims,
                          const int output_num_dims, const int* axis,
                          const int num_axis, int* input_iter,
                          Out* output_data) {
    auto reducer = [](const Out current, const In in) -> Out {
        const Out actual_in = static_cast<Out>(in);
        return current + actual_in;
    };
    return Reduce<In, Out>(input_data, input_dims, output_dims, input_num_dims,
                            output_num_dims, axis, num_axis, input_iter, reducer,
                            output_data);
}

// This method parses the input 'axis' to remove duplicates and handle negative
// values, and returns a valid 'out_axis'
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis) {
    *out_num_axis = 0;  // Just in case.
    // Short-circuit axis resolution for scalars; the axis will go unused.
    if (num_dims == 0) {
        return true;
    }
    // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
    for (int64_t idx = 0; idx < num_axis; ++idx) {
        // Handle negative index. A positive index 'p_idx' can be represented as a
        // negative index 'n_idx' as: n_idx = p_idx-num_dims
        // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]  */
        int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
        PAI_DCHECK(current >= 0 && current < num_dims);
        if (current < 0 || current >= num_dims) {
            return false;
        }
        bool is_dup = false;
        for (int j = 0; j < *out_num_axis; ++j) {
            if (out_axis[j] == current) {
                is_dup = true;
                break;
            }
        }
        if (!is_dup) {
            out_axis[*out_num_axis] = current;
            *out_num_axis += 1;
        }
    }
    return true;
}

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
    int32_t* temp_sum = (int32_t*)params.temp_buffer;

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