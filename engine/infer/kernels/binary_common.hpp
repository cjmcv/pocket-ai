#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_BINARY_COMMON_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_BINARY_COMMON_HPP_

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
    
    void *temp_buffer;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
} MeanOrSumQuantParams;

// For Mean, Sum, Max ops. (reduce)
// ref: OpDataReduce
typedef struct {
    uint32_t op_id;

    bool is_compute_sum;

    int32_t num_output_elements;
    int32_t num_axis;
    int32_t axis[kMaxNumberOfAxis];
    
    void *temp_buffer;
    //
    Tensor *input_tensor;
    Tensor *output_tensor;
} MeanOrSumParams;

// ref: tensorflow\lite\kernels\internal\types.h: NextIndex
// Gets next index to iterate through a multidimensional array.
inline bool NextIndex(const int num_dims, const int* dims, int* current) {
    if (num_dims == 0) {
        return false;
    }
    PAI_DCHECK(dims != nullptr);
    PAI_DCHECK(current != nullptr);
    int carry = 1;
    for (int idx = num_dims - 1; idx >= 0; --idx) {
        int current_val = current[idx] + carry;
        PAI_DCHECK_GE(dims[idx], current_val);
        if (dims[idx] == current_val) {
            current[idx] = 0;
        } else {
            current[idx] = current_val;
            carry = 0;
            break;
        }
    }
    return (carry == 0);
}

// ref: tensorflow\lite\kernels\internal\types.h: ReducedOutputOffset
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

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_BINARY_COMMON_HPP_