/*!
* \brief . 
*/

#include "pocket-ai/memory/allocator.hpp"

#include "gtest/gtest.h"

namespace {

using namespace pai::memory;

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
    // allocator.Show();
    
    std::vector<void *> ptrs2;
    for (uint32_t i=0; i<15; i++) {
        ptrs2.push_back(allocator.FastMalloc(i*40, true));
        // 200 => 40 / 80 / 120 / 160 / 200
        if (i > 0 && i <= 5)
            EXPECT_EQ(ptrs[1], ptrs2[i]);
        // 400 => 240 / 280 / 320 / 360 / 400
        if (i > 5 && i <= 10)
            EXPECT_EQ(ptrs[2], ptrs2[i]);
        // 600 => 440 / 480 / 520 / 560 / 600
        if (i>10)
            EXPECT_EQ(ptrs[3], ptrs2[i]);

        // allocator.Show();
        allocator.FastFree(ptrs2[i], true);
        // allocator.Show();

        // if (i==10)
        //     allocator.Shrink();
    }
    
    /////////
    // allocator.Show();
    std::vector<void *> ptrs3;
    for (uint32_t i=0; i<6; i++) {
        ptrs3.push_back(allocator.FastMalloc(i*40, true));
        // 200 => 40 / 80 bias 40 
        // 400 => 120 / 160 bias 120 / 200 / 240 / 280
        if (i == 1)
            EXPECT_EQ(ptrs[1], ptrs3[i]); // 40
        else if (i==2)
            EXPECT_EQ(ptrs[1] + 40, ptrs3[i]); // 80
        else if (i==3)
            EXPECT_EQ(ptrs[2], ptrs3[i]); // 120
        else if (i==4)
            EXPECT_EQ(ptrs[2] + 120, ptrs3[i]); // 160
        else if (i==5)
            EXPECT_EQ(ptrs[3], ptrs3[i]); // 200 
        else if (i==6)
            EXPECT_EQ(ptrs[3] + 200, ptrs3[i]); // 240 
        // allocator.Show();
    }
    for (uint32_t i=0; i<6; i++) {
        allocator.FastFree(ptrs3[i], true);
    }
    /////////

    allocator.Clear();
}

void AllocatorBindingPtrTest() {
    Allocator allocator;

    char *bptr = new char[1000];

    uint32_t id = allocator.InplacementCreate(1000, bptr); // 默认为free
    void *ptr0 = allocator.FastMalloc(900); // 整体复用 bptr
    void *ptr1 = allocator.FastMalloc(1100); // 新开一个1100的
    allocator.FastFree(ptr0);

    void *ptr2 = allocator.FastMallocSubBlock(200, id);
    void *ptr3 = allocator.FastMallocSubBlock(200, id);

    // void *ptr2 = allocator.FastMalloc(1000, true, id);
    allocator.Show();
    allocator.FastFree(ptr3, true);
    allocator.FastFree(ptr2, true);
    allocator.FastFree(ptr1);
    allocator.Show();

    // std::vector<void *> ptrs;

    allocator.Clear();    
    delete[] bptr;
}

TEST(MemoryTest, Allocator) {
    AllocatorScratchBufferTest();
    AllocatorScratchBufferTest2();
    AllocatorManuallySpecifyTest();
    AllocatorSubBlockTest();
    AllocatorBindingPtrTest();
}

}  // end of namespace.