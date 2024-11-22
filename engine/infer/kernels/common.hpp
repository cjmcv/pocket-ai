#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_COMMON_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_COMMON_HPP_

#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <string.h>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

template <typename DType>
void Im2Col(DType* dst, const DType* src, const int channels, 
            const int output_h, const int output_w, 
            const int input_h, const int input_w,
            const int kernel_h, const int kernel_w,
            const int pad_top, const int pad_bottom, 
            const int pad_left, const int pad_right,
            const int stride_h, const int stride_w, 
            const int dilation_h, const int dilation_w) {
    size_t vl;
    const int output_plane_size = output_h * output_w * kernel_h * kernel_w;
    memset(dst, 0, output_plane_size * channels * sizeof(DType));
    for (int ky = 0, h_offset = 0; ky < kernel_h; ky++, h_offset += dilation_h) {
        int dst_ky_offset = ky * kernel_w * channels;
        for (int kx = 0, w_offset = 0; kx < kernel_w; kx++, w_offset += dilation_w) {
            int dst_kx_offset = kx * channels;
            int oh_begin = std::max(((pad_top - h_offset + stride_h - 1) / stride_h), 0);
            int oh_end   = std::min(((input_h + pad_bottom - h_offset + stride_h - 1) / stride_h), output_h);
            oh_end       = std::max(oh_begin, oh_end);
            int ow_begin = std::max(((pad_left - w_offset + stride_w - 1) / stride_w), 0);
            int ow_end   = std::min(((input_w + pad_right - w_offset + stride_w - 1) / stride_w), output_w);
            ow_end       = std::max(ow_begin, ow_end);
            int ih = oh_begin * stride_h - pad_top + h_offset;
            for (int oh = oh_begin; oh < oh_end; ++oh, ih += stride_h) {
                int iw = ow_begin * stride_w - pad_left + w_offset;
                for (int ow = ow_begin; ow < ow_end; ++ow, iw += stride_w) {
                    const DType* src_ptr = src + (ih * input_w + iw) * channels;
                    DType* dst_ptr = dst + (oh * output_w + ow) * kernel_h * kernel_w * channels + dst_ky_offset + dst_kx_offset;
                    memcpy(dst_ptr, src_ptr, sizeof(DType) * channels);
                }
            }
        }
    }
}

template <typename DType>
void Col2Im(DType* dst, const DType* src, const int channels,
                const int output_h, const int output_w, 
                const int kernel_h, const int kernel_w,
                const int pad_h0, const int pad_h1, 
                const int pad_w0, const int pad_w1,
                const int stride_h, const int stride_w, 
                const int dilation_h, const int dilation_w) {
    memset(dst, 0, output_h * output_w * channels * sizeof(DType));
    const int col_height = (output_h + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int col_width  = (output_w + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int plane_size = channels * kernel_h * kernel_w;
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int im_h = -pad_h0 + kh * dilation_h;
            for (int col_h = 0; col_h < col_height; col_h++) {
                int im_w = -pad_w0 + kw * dilation_w;
                for (int col_w = 0; col_w < col_width; col_w++) {
                    if (im_h >= 0 && im_h < output_h && im_w >= 0 && im_w < output_w) {
                        for (int c = 0; c < channels; c++) {
                            int im_offset = (im_h * output_w + im_w) * channels + c;
                            int col_offset = (col_h * col_width + col_w) * plane_size + 
                                    c * kernel_h * kernel_w + kh * kernel_w + kw;
                            dst[im_offset] += src[col_offset];
                        }
                    }
                    im_w += stride_w;
                }
                im_h += stride_h;
            }
        } 
    }
}

void GemmTransB(const int M, const int N, const int K, 
                const float *A, const float *B, float *C, const float *bias) {
    if (bias) {
        for (int i=0; i<M; i++)
            memcpy(C + i*N, bias, N * sizeof(float));
    }
    else
        memset(C, 0, M * N * sizeof(float));

    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<K; k++) {
                C[i*N+j] += A[i*K+k] * B[j*K+k];
            }
        }
    }
}

void GemmTransBQuant(const int M, const int N, const int K, 
                            const int8_t *A, const int8_t *B, int32_t *C, 
                            const int32_t *bias, const int32_t input_offset) {
    if (bias) {
        for (int i=0; i<M; i++)
            memcpy(C + i*N, bias, N * sizeof(int32_t));
    }
    else
        memset(C, 0, M * N * sizeof(int32_t));

    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<K; k++) {
                C[i*N+j] += (A[i*K+k] + input_offset) * B[j*K+k];
            }
        }
    }
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_COMMON_HPP_