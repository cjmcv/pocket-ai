#ifndef POCKET_AI_ENGINE_INFERENCE_TYPES_HPP_
#define POCKET_AI_ENGINE_INFERENCE_TYPES_HPP_

#include <iostream>
#include <stdint.h>
#include "util/logger.hpp"

namespace pai {
namespace infer {

typedef enum {
    kPaiInferNoType = 0,
    kPaiInferFloat32,
    kPaiInferInt32,
    kPaiInferUInt8,
    kPaiInferInt64,
    kPaiInferBool,
    kPaiInferInt16,
    kPaiInferInt8,
    kPaiInferFloat16,
    kPaiInferFloat64,
    kPaiInferUInt64,
    kPaiInferUInt32,
    kPaiInferUInt16,
    kPaiInferInt4,
} PaiInferType;

typedef struct {
    uint32_t id;
    PaiInferType type;
    Shape shape;
    void* data;    
} Tensor;

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_TYPES_HPP_