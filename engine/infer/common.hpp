#ifndef POCKET_AI_ENGINE_INFERENCE_COMMON_HPP_
#define POCKET_AI_ENGINE_INFERENCE_COMMON_HPP_

#include <iostream>
#include "types.hpp"
#include "util/logger.hpp"

namespace pai {
namespace infer {

#define ENABLE_PAI_INFER_DEBUG

#ifdef ENABLE_PAI_INFER_DEBUG
#define PAI_INFER_ASSERT_FALSE abort()
#else
#define PAI_INFER_ASSERT_FALSE (static_cast<void>(0))
#endif

#define PAI_DCHECK(condition) (condition) ? (void)0 : PAI_INFER_ASSERT_FALSE
#define PAI_DCHECK_EQ(x, y) ((x) == (y)) ? (void)0 : PAI_INFER_ASSERT_FALSE
#define PAI_DCHECK_NE(x, y) ((x) != (y)) ? (void)0 : PAI_INFER_ASSERT_FALSE
#define PAI_DCHECK_GE(x, y) ((x) >= (y)) ? (void)0 : PAI_INFER_ASSERT_FALSE
#define PAI_DCHECK_GT(x, y) ((x) > (y)) ? (void)0 : PAI_INFER_ASSERT_FALSE
#define PAI_DCHECK_LE(x, y) ((x) <= (y)) ? (void)0 : PAI_INFER_ASSERT_FALSE

inline int Offset(const Shape& shape, int i0, int i1, int i2, int i3) {
    PAI_DCHECK_EQ(shape.dims_count, 4);
    return ((i0 * shape.dims[1] + i1) * shape.dims[2] + i2) * shape.dims[3] + i3;
}

inline int Offset(const Shape& shape, int i0, int i1, int i2, int i3, int i4) {
    PAI_DCHECK_EQ(shape.dims_count, 5);
    return (((i0 * shape.dims[1] + i1) * shape.dims[2] + i2) * shape.dims[3] + i3) *
              shape.dims[4] + i4;
}

// Get common shape dim, DCHECKing that they all agree.
inline int MatchingDim(const Shape& shape1, int index1,
                       const Shape& shape2, int index2) {
    PAI_DCHECK_EQ(shape1.dims[index1], shape2.dims[index2]);
    return std::min(shape1.dims[index1], shape2.dims[index2]);
}

//////////////
// This function implements the same computation as the ARMv7 NEON VQRDMULH
// instruction.
inline int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
    bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
    int64_t a_64(a);
    int64_t b_64(b);
    int64_t ab_64 = a_64 * b_64;
    int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
    return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

inline int32_t MaskIfLessThan(int32_t a, int32_t b) {
    return a < b ? ~0 : 0;
}

inline int32_t MaskIfGreaterThan(int32_t a, int32_t b) {
    return a > b ? ~0 : 0;
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
inline int32_t RoundingDivideByPOT(int32_t x, int exponent) {
    // assert(exponent >= 0);
    // assert(exponent <= 31);
    const int32_t mask = (1ll << exponent) - 1;
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + (MaskIfLessThan(x, 0) & 1);
    return (x >> exponent) + (MaskIfGreaterThan(remainder, threshold) & 1);
}

// ref: tensorflow\lite\kernels\internal\common.cc: MultiplyByQuantizedMultiplier
inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    int32_t xx = SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier);

    return RoundingDivideByPOT(xx, right_shift);
}

inline int32_t MultiplyByQuantizedMultiplierGreaterThanOne(
    int32_t x, int32_t quantized_multiplier, int left_shift) {
    return SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                           quantized_multiplier);
}

// Data is required to be contiguous, and so many operators can use either the
// full array flat size or the flat size with one dimension skipped (commonly
// the depth).
inline int FlatSizeSkipDim(const Shape& shape, int skip_dim) {
    const int dims_count = shape.dims_count;
    PAI_DCHECK(skip_dim >= 0 && skip_dim < dims_count);
    const auto* dims_data = shape.dims;
    int flat_size = 1;
    for (int i = 0; i < dims_count; ++i) {
        flat_size *= (i == skip_dim) ? 1 : dims_data[i];
    }
    return flat_size;
}


// A combination of MatchingFlatSize() and FlatSizeSkipDim().
inline int MatchingFlatSizeSkipDim(const Shape& shape, int skip_dim,
                                   const Shape& check_shape_0) {
    const int dims_count = shape.dims_count;
    for (int i = 0; i < dims_count; ++i) {
        if (i != skip_dim) {
            PAI_DCHECK_EQ(shape.dims[i], check_shape_0.dims[i]);
        }
    }
  return FlatSizeSkipDim(shape, skip_dim);
}

// tensorflow\lite\kernels\internal\common.h: CountLeadingZeros
template <typename T>
int CountLeadingZeros(T integer_input) {
    static_assert(std::is_unsigned<T>::value, "Only unsigned integer types handled.");
    if (integer_input == 0) {
        return std::numeric_limits<T>::digits;
    }
    #if defined(__GNUC__)
    if (std::is_same<T, uint32_t>::value) {
        return __builtin_clz(integer_input);
    } else if (std::is_same<T, uint64_t>::value) {
        return __builtin_clzll(integer_input);
    }
    #endif
    const T one_in_leading_positive = static_cast<T>(1)
                                        << (std::numeric_limits<T>::digits - 1);
    int leading_zeros = 0;
    while (integer_input < one_in_leading_positive) {
        integer_input <<= 1;
        ++leading_zeros;
    }
    return leading_zeros;
}

//////////////
// Debug
inline bool CheckTensr(Tensor &tensor, void *ref_data = nullptr) {
    bool check_pass = true;

#ifdef ENABLE_PAI_INFER_DEBUG
    printf("Tensor id: %d", tensor.id);
    printf("    shape: [");
    uint32_t num = 1;
    for (uint32_t i=0; i<tensor.shape.dims_count; i++) {
        num *= tensor.shape.dims[i];
        printf("%d ", tensor.shape.dims[i]);
    }
    printf("]\n");
    printf("     data: ");
    if (tensor.type == kPaiInferFloat32) {
        float *data = (float *)tensor.data;
        float *ref = (float *)ref_data;
        for (uint32_t i=0; i<num; i++) {
            printf("%f, ", data[i]);
            if (data[i] > ref[i] + 0.00001f || data[i] < ref[i] - 0.00001f) {
                check_pass = false;
                printf("%f[%f], ", data[i], ref[i]);            
            }
            else {
                printf("%f, ", data[i]);
            }
        }
    }
    else if (tensor.type == kPaiInferInt8 || tensor.type == kPaiInferUInt8) {
        int8_t *data = (int8_t *)tensor.data;
        int8_t *ref = (int8_t *)ref_data;
        for (uint32_t i=0; i<num; i++) {
            if (data[i] != ref[i]) {
                check_pass = false;
                printf("%d[%d], ", data[i], ref[i]);                
            }
            else {
                printf("%d, ", data[i]);
            }
        }        
    }
    printf("\n\n");
#endif // #ifdef ENABLE_PAI_INFER_DEBUG

    return check_pass;
}

} // namespace infer
} // namespace pai

#endif //POCKET_AI_ENGINE_INFERENCE_COMMON_HPP_