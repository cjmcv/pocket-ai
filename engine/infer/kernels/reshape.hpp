#ifndef POCKET_AI_ENGINE_INFERENCE_KERNELS_RESHAPE_HPP_
#define POCKET_AI_ENGINE_INFERENCE_KERNELS_RESHAPE_HPP_

#include <stdint.h>
#include <algorithm>

#include "engine/infer/common.hpp"
#include "engine/infer/types.hpp"

namespace pai {
namespace infer {

typedef struct {
    uint32_t op_id;
    Tensor *input_tensor;
    Tensor *output_tensor;
} ReshapeParams;

inline void Reshape(const ReshapeParams& params) {
    PAI_DCHECK_EQ(params.input_tensor->data, params.output_tensor->data);
    // Do nothing.
}

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFERENCE_KERNELS_RESHAPE_HPP_