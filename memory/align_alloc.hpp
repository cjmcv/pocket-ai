/*!
* \brief 
*/

#ifndef POCKET_AI_MEMORY_ALIGN_ALLOC_UTIL_HPP_
#define POCKET_AI_MEMORY_ALIGN_ALLOC_UTIL_HPP_

#include <stdlib.h>
#include "util/logger.hpp"

namespace pai {
namespace memory {
    
static inline void **AlignPtr(void **ptr, size_t alignment) {
    return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
}

static inline size_t AlignSize(size_t sz, int n) {
    return (sz + n - 1) & -n;
}

static inline void *AlignMalloc(size_t size, size_t alignment = 16) {
    void **origin = (void **)malloc(size + sizeof(void *) + alignment);
    if (!origin)
        return NULL;

    void **aligned = AlignPtr(origin + 1, alignment);
    aligned[-1]    = origin;
    return aligned;
}

static inline void AlignFree(void *aligned) {
    if (aligned) {
        void *origin = ((void **)aligned)[-1];
        free(origin);
    }
}

} // memory.
} // pai.
#endif // POCKET_AI_MEMORY_ALIGN_ALLOC_UTIL_HPP_
