#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_PAD_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_PAD_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef union {
    int8_t int8_value;
    int32_t int32_value;
    float fp32_value;
} UnionType;

typedef struct {
    uint32_t op_id;
    
    int8_t left_padding_count;
    int32_t left_padding[5];
    int8_t right_padding_count;
    int32_t right_padding[5];
    UnionType pad_value;

    Tensor *input_tensor;
    Tensor *output_tensor;
} PadParams;

// ref: tensorflow\lite\kernels\internal\reference\pad.h: PadImpl

// Pad supports activation tensors with up to 5 dimensions.
constexpr int PadKernelMaxDimensionCount() { return 5; }

// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32_t is considered a
// specialization distinct from P=int32_t.
template <typename T>
inline void Pad(const PadParams& params) {

    const Shape &output_shape = params.output_tensor->shape;

    Shape ext_output_shape;
    uint32_t ext_step = PadKernelMaxDimensionCount() - output_shape.dims_count;
    for (uint32_t i=0; i<ext_step; i++)
        ext_output_shape.dims[i] = 1;
    for (uint32_t i=ext_step; i<PadKernelMaxDimensionCount(); i++)
        ext_output_shape.dims[i] = output_shape.dims[i-ext_step];

    PAI_DCHECK_LE(params.left_padding_count, PadKernelMaxDimensionCount());
    PAI_DCHECK_LE(params.right_padding_count, PadKernelMaxDimensionCount());

    // Runtime calls are currently fixed at 5 dimensions. Copy inputs so we can
    // pad them to 5 dims (yes, we are "padding the padding").
    int left_padding_copy[PadKernelMaxDimensionCount()];
    for (int i = 0; i < PadKernelMaxDimensionCount(); i++) {
        left_padding_copy[i] = 0;
    }
    for (int i = 0; i < params.left_padding_count; ++i) {
        left_padding_copy[i + PadKernelMaxDimensionCount() -
                        params.left_padding_count] = params.left_padding[i];
    }
    int right_padding_copy[PadKernelMaxDimensionCount()];
    for (int i = 0; i < PadKernelMaxDimensionCount(); i++) {
        right_padding_copy[i] = 0;
    }
    for (int i = 0; i < params.right_padding_count; ++i) {
        right_padding_copy[i + PadKernelMaxDimensionCount() -
                        params.right_padding_count] =
            params.right_padding[i];
    }

//   static int fcnt = 0;
//   fcnt++;

//   if (fcnt == 2) {
//     printf("end\n");
//     std::abort();    
//   }

  
//   printf("PadImplabc: %ld.\n", sizeof(T));
//   printf("%d, %d.\n", params.left_padding_count, params.right_padding_count);
//   for (int i=0; i<params.left_padding_count; i++)
//     printf("%d, ", params.left_padding[i]);
//   printf("\n");
//   for (int i=0; i<params.right_padding_count; i++)
//     printf("%d, ", params.right_padding[i]);

    const int output_batch = ext_output_shape.dims[0];
    const int output_plane = ext_output_shape.dims[1];
    const int output_height = ext_output_shape.dims[2];
    const int output_width = ext_output_shape.dims[3];
    const int output_depth = ext_output_shape.dims[4];

    const int left_b_padding = left_padding_copy[0];
    const int left_p_padding = left_padding_copy[1];
    const int left_h_padding = left_padding_copy[2];
    const int left_w_padding = left_padding_copy[3];
    const int left_d_padding = left_padding_copy[4];

    const int right_b_padding = right_padding_copy[0];
    const int right_p_padding = right_padding_copy[1];
    const int right_h_padding = right_padding_copy[2];
    const int right_w_padding = right_padding_copy[3];
    const int right_d_padding = right_padding_copy[4];

    // printf("a<%d, %d, %d, %d, %d>, ", output_batch, output_plane, output_height, output_width, output_depth);
    // printf("b<%d, %d, %d, %d, %d>, ", left_b_padding, left_p_padding, left_h_padding, left_w_padding, left_d_padding);
    // printf("c<%d, %d, %d, %d, %d>, ", right_b_padding, right_p_padding, right_h_padding, right_w_padding, right_d_padding);

    T pad_value;
    if (sizeof(T) == 1) {
        PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferInt8);
        PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferInt8);
        pad_value = params.pad_value.int8_value;
    }
    else {
        PAI_DCHECK_EQ(params.input_tensor->type, kPaiInferFloat32);
        PAI_DCHECK_EQ(params.output_tensor->type, kPaiInferFloat32);
        pad_value = params.pad_value.fp32_value;
    }

    const T* in_ptr = (T *)params.input_tensor->data;
    T* out_ptr = (T *)params.output_tensor->data;
    for (int out_b = 0; out_b < output_batch; ++out_b) {
        for (int out_p = 0; out_p < output_plane; ++out_p) {
            for (int out_h = 0; out_h < output_height; ++out_h) {
                for (int out_w = 0; out_w < output_width; ++out_w) {
                    for (int out_d = 0; out_d < output_depth; ++out_d) {
                        if (out_b < left_b_padding ||
                            out_b >= output_batch - right_b_padding ||
                            out_p < left_p_padding ||
                            out_p >= output_plane - right_p_padding ||
                            out_h < left_h_padding ||
                            out_h >= output_height - right_h_padding ||
                            out_w < left_w_padding ||
                            out_w >= output_width - right_w_padding ||
                            out_d < left_d_padding ||
                            out_d >= output_depth - right_d_padding) {
                            *out_ptr++ = pad_value;
                        } else {
                            *out_ptr++ = *in_ptr++;
                        }
                    }
                }
            }
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_PAD_HPP_