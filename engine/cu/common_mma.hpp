// %%cuda_group_save --group shared --name "common_mma.h"

#ifndef POCKET_AI_ENGINE_CUDA_COMMON_MMA_HPP_
#define POCKET_AI_ENGINE_CUDA_COMMON_MMA_HPP_

#include "pocket-ai/engine/cu/common.hpp"
//#include "common.h"

#include <mma.h>
using namespace nvcuda;

namespace pai {
namespace cu {

#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

} // namespace cu
} // namespace pai

#endif //POCKET_AI_ENGINE_CUDA_COMMON_MMA_HPP_