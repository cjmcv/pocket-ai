/*!
* \brief . 
*/

#include "memory/allocator.hpp"

#include "gtest/gtest.h"

namespace {

using namespace ptk::memory;

void AllocatorScratchBufferTest() {
    Allocator allocator;
    std::vector<void *> ptrs;
    for (uint32_t i=0; i<20; i++) {
        ptrs.push_back(allocator.FastMalloc(1000 + i*200));
        allocator.FastFree(ptrs[i]);
    }

    std::vector<void *> ptrs2;
    for (uint32_t i=0; i<20; i++) {
        ptrs2.push_back(allocator.FastMalloc(1000 + i*200));
        EXPECT_EQ(ptrs2[i], ptrs[i]);
        allocator.FastFree(ptrs2[i]);
    }
    allocator.Clear();
}

void AllocatorScratchBufferTest2() {
    Allocator allocator;
    std::vector<void *> ptrs;
    for (uint32_t i=0; i<20; i++) {
        ptrs.push_back(allocator.FastMalloc(1000 + i*200));
        allocator.FastFree(ptrs[i]);
    }

    std::vector<void *> ptrs2;
    for (uint32_t i=0; i<20; i++) {
        ptrs2.push_back(allocator.FastMalloc(1000 + i*100));
        // allocator.Show();
        bool is_find = false;
        for (uint32_t j=0; j<20; j++) {
            if (ptrs2[i] == ptrs[j]) {
                is_find = true;
            }
        }
        EXPECT_EQ(is_find, true);
        allocator.FastFree(ptrs2[i]);
    }
    allocator.Clear();
}

// 
void AllocatorManuallySpecifyTest() {
    Allocator allocator;
    allocator.SetCompareRatio(0.9f);
    std::vector<void *> ptrs, ptrs2;
    for (uint32_t i=0; i<20; i++) {
        ptrs.push_back(allocator.FastMalloc(1000 + i*200));
        allocator.FastFree(ptrs[i]);
    }
    // allocator.Show();

    ptrs.clear();
    ptrs2.clear();
    {
        // reuse 2600
        void *p0 = allocator.FastMalloc(2600);
        // allocator.Show();
        // reuse 2800 => 2800 * 0.9 = 2520 < 2600 && 2600 < 2800
        void *p1 = allocator.FastMalloc(2600);
        // allocator.Show();
        // new one 2600 at the end of storage
        void *p2 = allocator.FastMalloc(2600);
        // allocator.Show();
        allocator.FastFree(p0);
        allocator.FastFree(p1);
        allocator.FastFree(p2);

        // allocator.Show();

        // Reuse the first one 2600
        ptrs2.push_back(allocator.FastMalloc(2600));
        // allocator.Show();
        allocator.FastFree(ptrs2.back());

        // Reuse the first one 2600, and set group id to 1
        ptrs.push_back(allocator.FastMalloc(2600, 1, MEM_BIND_HEAD));
        // allocator.Show();
        allocator.FastFree(ptrs.back());
        
        // Reuse the 2800
        ptrs2.push_back(allocator.FastMalloc(2600));
        // allocator.Show();
        allocator.FastFree(ptrs2.back());

        // Reuse the first one 2600 with group 1
        ptrs.push_back(allocator.FastMalloc(2600, 1, MEM_BIND_BODY));
        // allocator.Show();
        allocator.FastFree(ptrs.back());

        // Reuse the first one 2600 with group 1
        ptrs.push_back(allocator.FastMalloc(2600, 1, MEM_BIND_BODY));
        // allocator.Show();
        allocator.FastFree(ptrs.back());

        // Reuse the 2800
        ptrs2.push_back(allocator.FastMalloc(2600));
        // allocator.Show();
        allocator.FastFree(ptrs2.back());

        // Reuse the first one 2600 with group 1 and reset the group id to 0
        ptrs.push_back(allocator.FastMalloc(2600, 1, MEM_BIND_TAIL));
        // allocator.Show();
        allocator.FastFree(ptrs.back());
        // allocator.Show();
    }
    // 关联的必须使用同一个。指针相同
    for (uint32_t i=1; i<ptrs.size(); i++) {
        EXPECT_EQ(ptrs[0], ptrs[i]);
    }
    for (uint32_t i=1; i<ptrs2.size(); i++) {
        EXPECT_NE(ptrs[0], ptrs2[i]);
    }
    allocator.Clear();
}

void AllocatorSubBlockTest() {
    Allocator allocator;
    std::vector<void *> ptrs;
    for (uint32_t i=0; i<5; i++) {
        ptrs.push_back(allocator.FastMalloc(i*200));
        allocator.FastFree(ptrs[i]);
    }
    allocator.Show();
    
    std::vector<void *> ptrs2;
    for (uint32_t i=0; i<15; i++) {
        ptrs2.push_back(allocator.FastMalloc(i*40, true));
        allocator.Show();
        allocator.FastFree(ptrs2[i], true);
        allocator.Show();
    }
    allocator.Clear();
}


TEST(MemoryTest, Allocator) {
    AllocatorScratchBufferTest();
    AllocatorScratchBufferTest2();
    AllocatorManuallySpecifyTest();
    AllocatorSubBlockTest();
}

}  // end of namespace.