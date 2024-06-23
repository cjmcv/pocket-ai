#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_TILE_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_TILE_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    int32_t *multipliers;

    Tensor *input_tensor;
    Tensor *output_tensor;
} TileParams;

template <typename T>
void CopyMultipleTimes(const T* in_data, int32_t in_size, int32_t multiplier,
                       T* out_data) {
    for (int32_t i = 0; i < multiplier; ++i) {
        const T* in_end = in_data + in_size;
        T* new_out_data = std::copy(in_data, in_end, out_data);
        in_data = out_data;
        out_data = new_out_data;
    }
}

// ref: (in tensorflow)tensorflow\lite\kernels\tile.cc：#112 TileOneDimension
// template <typename T, typename M>
// std::pair<int, int> TileOneDimension(const TfLiteIntArray& in_dimensions,
//                                      const T* in_data, const M* multipliers,
//                                      T* out_data, int dimension) {
template <typename T>
std::pair<int, int> TileOneDimension(const int32_t *in_dimensions, const int32_t dims_count, 
                                    const T* in_data, const int32_t* multipliers,
                                    T* out_data, int dimension) {
        
    if (dims_count == 0) {
        // If input tensor is a scalar, then just copy it to output (no need to
        // multiply).
        *out_data = *in_data;
        return std::make_pair(0, 0);
    }

    const int dimension_size = in_dimensions[dimension];
    if (dimension == dims_count - 1) {
        CopyMultipleTimes(in_data, dimension_size, multipliers[dimension],
                        out_data);
        return std::make_pair(
            dimension_size,
            dimension_size * static_cast<int>(multipliers[dimension]));
    }
    int total_stride_size = 0, total_tiled_stride_size = 0;
    const T* copy_from_data = in_data;
    T* copy_to_data = out_data;
    for (int i = 0; i < dimension_size; ++i) {
        int stride_size = 0, tiled_stride_size = 0;
        std::tie(stride_size, tiled_stride_size) =
            TileOneDimension(in_dimensions, dims_count, 
                            copy_from_data, multipliers,
                            copy_to_data, dimension + 1);
        copy_from_data += stride_size;
        copy_to_data += tiled_stride_size;
        total_stride_size += stride_size;
        total_tiled_stride_size += tiled_stride_size;
    }
    CopyMultipleTimes(out_data, total_tiled_stride_size,
                        multipliers[dimension] - 1,
                        out_data + total_tiled_stride_size);
    return std::make_pair(
        total_stride_size,
        static_cast<int>(total_tiled_stride_size * multipliers[dimension]));
}

template <typename T>
inline void Tile(const TileParams& params) {
    const Shape& input_shape = params.input_tensor->shape;
    const T* input_data = (T*)params.input_tensor->data;

    const Shape& output_shape = params.output_tensor->shape;
    T* output_data = (T*)params.output_tensor->data;

    TileOneDimension<T>(input_shape.dims, input_shape.dims_count, input_data, params.multipliers, output_data, 0);
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_TILE_HPP_