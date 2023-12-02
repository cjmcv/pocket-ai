/*!
* \brief Util.
*/

#ifndef PTK_UTIL_TYPE_SWITCH_HPP_
#define PTK_UTIL_TYPE_SWITCH_HPP_

#include <iostream>
#include "logger.hpp"
// #include "../common.hpp"

namespace ptk {
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
    case ptk::util::Type::FP32:                       \
        {                                             \
            typedef float DType;                      \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case ptk::util::Type::INT32:                      \
        {                                             \
            typedef int32_t DType;                    \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case ptk::util::Type::UINT32:                     \
        {                                             \
            typedef uint32_t DType;                   \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case ptk::util::Type::INT16:                      \
    case ptk::util::Type::FP16:                       \
    case ptk::util::Type::BF16:                       \
        {                                             \
            typedef int16_t DType;                    \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case ptk::util::Type::UINT16:                     \
        {                                             \
            typedef uint16_t DType;                   \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case ptk::util::Type::INT8:                       \
        {                                             \
            typedef int8_t DType;                     \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    case ptk::util::Type::UINT8:                      \
        {                                             \
            typedef uint8_t DType;                    \
            {__VA_ARGS__}                             \
        }                                             \
        break;                                        \
    default:                                          \
        PTK_LOGE("Unknown type enum %d \n", type);    \
    }

// Get type flag from type.
template<typename DType>
struct DataType;
template<>
struct DataType<float> {
    static const int kFlag = ptk::util::Type::FP32;
};
template<>
struct DataType<int32_t> {
    static const int kFlag = ptk::util::Type::INT32;
};
template<>
struct DataType<uint32_t> {
    static const int kFlag = ptk::util::Type::INT32;
};
template<>
struct DataType<int16_t> {
    static const int kFlag = ptk::util::Type::INT16;
};
template<>
struct DataType<uint16_t> {
    static const int kFlag = ptk::util::Type::UINT16;
};
template<>
struct DataType<int8_t> {
    static const int kFlag = ptk::util::Type::INT8;
};
template<>
struct DataType<uint8_t> {
    static const int kFlag = ptk::util::Type::INT8;
};

} // util
} // ptk.
#endif //PTK_UTIL_TYPE_SWITCH_HPP_
