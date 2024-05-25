#ifndef POCKET_AI_ENGINE_INFERENCE_COMMON_HPP_
#define POCKET_AI_ENGINE_INFERENCE_COMMON_HPP_

#include <iostream>
#include "util/logger.hpp"

namespace pai {
namespace infer {

#define PAI_INFER_ASSERT_FALSE abort()
// #define PAI_INFER_ASSERT_FALSE (static_cast<void>(0))

#define PAI_DCHECK_EQ(x, y) ((x) == (y)) ? (void)0 : PAI_INFER_ASSERT_FALSE
#define PAI_DCHECK_NE(x, y) ((x) != (y)) ? (void)0 : PAI_INFER_ASSERT_FALSE

// Supports up to 5 dimensions.
static constexpr int kMaxShapeDims = 5;

typedef struct {
    int32_t dims_count;
    int32_t dims[kMaxShapeDims];
} Shape;

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

inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier,
                                      int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  int32_t xx = SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier);

  return RoundingDivideByPOT(xx, right_shift);
}
//////////////


} // namespace infer
} // namespace pai

#endif //POCKET_AI_ENGINE_INFERENCE_COMMON_HPP_