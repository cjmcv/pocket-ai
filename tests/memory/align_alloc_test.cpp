/*!
* \brief . 
*/

#include "memory/align_alloc.hpp"

#include "gtest/gtest.h"

namespace {

using namespace ptk::memory;

void AlignAllocTest() {
    EXPECT_EQ(AlignSize(100, 2), 100);
    EXPECT_EQ(AlignSize(100, 4), 100);
    EXPECT_EQ(AlignSize(100, 6), 104); // 102
    EXPECT_EQ(AlignSize(100, 8), 104);
    EXPECT_EQ(AlignSize(100, 16), 112);
    EXPECT_EQ(AlignSize(100, 32), 128);
    EXPECT_EQ(AlignSize(100, 64), 128);

    for (uint32_t size = 1000; size < 2000; size += 220) {
        for (uint32_t alignment = 2; alignment <= 128; alignment *= 2) {
            char *p = (char *)AlignMalloc(1000, alignment);
            EXPECT_EQ((uint64_t)p % alignment, 0);
            AlignFree(p);   
        }        
    }

}

TEST(MemoryTest, AlignAlloc) {
    AlignAllocTest();
}

}  // end of namespace.