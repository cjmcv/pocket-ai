/*!
* \brief Util.
*/

#ifndef POCKET_AI_UTIL_TYPE_SWITCH_HPP_
#define POCKET_AI_UTIL_TYPE_SWITCH_HPP_

#include <iostream>
#include <math.h>
#include <cstring>

#include "logger.hpp"

#ifdef __F16C__
#include "immintrin.h"
#endif

namespace pai {
namespace util {

enum Type {
    FP32 = 0,
    FP16 = 1,
    BF16,
    INT32,
    UINT32,
    INT16,
    UINT16,
    INT8,
    UINT8,
    // Used to mark the total number of elements in Type.
    TYPES_NUM
};

enum MemoryLoc {
    ON_HOST = 0,
    ON_DEVICE, 
};

// Get type from type flag.
#define TYPE_SWITCH(type, DType, ...)                 \
    switch (type) {                                   \
    case pai::util::Type::FP32:                       \
        {                                             \
            typedef float DType;                      \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case pai::util::Type::INT32:                      \
        {                                             \
            typedef int32_t DType;                    \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case pai::util::Type::UINT32:                     \
        {                                             \
            typedef uint32_t DType;                   \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case pai::util::Type::INT16:                      \
        {                                             \
            typedef int16_t DType;                    \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case pai::util::Type::FP16:                       \
    case pai::util::Type::BF16:                       \
    case pai::util::Type::UINT16:                     \
        {                                             \
            typedef uint16_t DType;                   \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case pai::util::Type::INT8:                       \
        {                                             \
            typedef int8_t DType;                     \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case pai::util::Type::UINT8:                      \
        {                                             \
            typedef uint8_t DType;                    \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    default:                                          \
        PAI_LOGE("Unknown type enum %d \n", type);    \
    }

// Get type flag from type.
template<typename DType>
struct DataType;
template<>
struct DataType<float> {
    static const int kFlag = pai::util::Type::FP32;
};
template<>
struct DataType<int32_t> {
    static const int kFlag = pai::util::Type::INT32;
};
template<>
struct DataType<uint32_t> {
    static const int kFlag = pai::util::Type::UINT32;
};
template<>
struct DataType<int16_t> {
    static const int kFlag = pai::util::Type::INT16;
};
template<>
struct DataType<uint16_t> {
    static const int kFlag = pai::util::Type::UINT16;
};
template<>
struct DataType<int8_t> {
    static const int kFlag = pai::util::Type::INT8;
};
template<>
struct DataType<uint8_t> {
    static const int kFlag = pai::util::Type::INT8;
};


/////////////////////////////////////////////////////////////
// FP16 <-> FP32
// ref: https://github.com/NVIDIA/cutlass/include/cutlass/half.h
//      Sign bit + Exponent bit + Mantissa bit
// fp32: 1+8+23
// bf16: 1+8+7
// fp16: 1+5+10

typedef uint16_t half_t;

static half_t ConvertFp32ToHalf(float flt) {
#ifdef __F16C__
    return _cvtss_sh(flt, 0);
#else
    uint32_t s = *reinterpret_cast<uint32_t*>(&flt);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
        // sign-preserving zero
        return sign;
    }

    if (exp > 15) {
        if (exp == 128 && mantissa) {
            // not a number
            u = 0x7fff;
        } else {
            // overflow to infinity
            u = sign | 0x7c00;
        }
        return u;
    }

    int sticky_bit = 0;

    if (exp >= -14) {
        // normal fp32 to normal fp16
        exp = uint16_t(exp + uint16_t(15));
        u = uint16_t(((exp & 0x1f) << 10));
        u = uint16_t(u | (mantissa >> 13));
    } else {
        // normal single-precision to subnormal half_t-precision representation
        int rshift = (-14 - exp);
        if (rshift < 32) {
            mantissa |= (1 << 23);

            sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

            mantissa = (mantissa >> rshift);
            u = (uint16_t(mantissa >> 13) & 0x3ff);
        } else {
            mantissa = 0;
            u = 0;
        }
    }

    // round to nearest even
    int round_bit = ((mantissa >> 12) & 1);
    sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
        u = uint16_t(u + 1);
    }

    u |= sign;

    return u;
#endif
}

/// Converts a half-precision value stored as a uint16_t to a float
static float ConvertHalfToFp32(half_t h) {

#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    uint32_t sign = ((h >> 15) & 1);
    uint32_t exp = ((h >> 10) & 0x1f);
    uint32_t mantissa = (h & 0x3ff);
    unsigned f = 0;

    if (exp > 0 && exp < 31) {
        // normal
        exp += 112;
        f = (sign << 31) | (exp << 23) | (mantissa << 13);
    }
    else if (exp == 0) {
        if (mantissa) {
            // subnormal
            exp += 113;
            while ((mantissa & (1 << 10)) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3ff;
            f = (sign << 31) | (exp << 23) | (mantissa << 13);
        }
        else {
            // sign-preserving zero
            f = (sign << 31);
        }
    }
    else if (exp == 31) {
        if (mantissa) {
            f = 0x7fffffff; // not a number
        }
        else {
            f = (0xff << 23) | (sign << 31); //  inf
        }
    }
    float flt;
    std::memcpy(&flt, &f, sizeof(flt));
    return flt;
#endif
}

class HalfToFp32Table {
public:
    HalfToFp32Table() {
        table_f32_f16_ = new float[1 << 16]; // 65536 x 4 = 256 KB
    }
    ~HalfToFp32Table() {
        delete[] table_f32_f16_;
    }
    void Create() {
        for (int i = 0; i < (1 << 16); ++i) {
            table_f32_f16_[i] = ConvertHalfToFp32(i);
        }
    }
    float LoopupTable(half_t h) {
        return table_f32_f16_[h];
    }
private:
    float *table_f32_f16_;
};

/////////////////////////////////////////////////////////////
// BF16 <-> FP32
// ref: https://github.com/NVIDIA/cutlass/include/cutlass/bfloat16.h
typedef uint16_t bfloat16_t;

static bfloat16_t ConvertFp32ToBf16(float flt) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&flt);
    if ((bits & 0x7f800000) != 0x7f800000) {
        bool mantissa_bit = ((bits & (1 << 16)) != 0);
        bool round_bit = ((bits & (1 << 15)) != 0);
        bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);

        if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
            bits += uint32_t(1 << 16);
        }
    }
    else if (bits & ~0xff800000) {
        bits = 0x7fffffff;
    }
    return uint16_t((bits >> 16) & 0xffff);
}

static float ConvertBf16ToFp32(bfloat16_t bflt) {
    uint32_t bits = static_cast<uint32_t>(bflt) << 16;
    return *reinterpret_cast<float*>(&bits);
}

} // util
} // pai.
#endif //POCKET_AI_UTIL_TYPE_SWITCH_HPP_
