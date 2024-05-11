#include "thread/thread_pool.hpp"

#include "gtest/gtest.h"

namespace {

using namespace pai::thread;

void CompTest(int32_t *a, int32_t *b, uint32_t len) {
    // printf("%d, ", a[0]);
    for (uint32_t i = 0; i < len; i++) {
        b[i] = a[i] + b[i];
    }
}

void Check(int32_t **b, uint32_t num, uint32_t len, uint32_t iter) {
    for (uint32_t i = 0; i < num; i++) {
        for (uint32_t j = 0; j < len; j++) {
            EXPECT_EQ(b[i][j], i*iter);
        }
    }
}

void ThreadPoolTest() {
    uint32_t len = 10;// 1000000;
    uint32_t num = 50;

    int32_t **a = (int32_t **)malloc(sizeof(int32_t *) * num);
    int32_t **b = (int32_t **)malloc(sizeof(int32_t *) * num);
    for (uint32_t i = 0; i < num; i++) {
        a[i] = (int32_t *)malloc(sizeof(int32_t) * len);
        b[i] = (int32_t *)malloc(sizeof(int32_t) * len);
    }

    for (uint32_t i = 0; i < num; i++) {
        for (uint32_t j = 0; j < len; j++) {
            a[i][j] = b[i][j] = i; // i*len + j;
        }
    }

    uint32_t thread_num = 15;
    ThreadPool thread_pool;
    thread_pool.CreateThreads(thread_num);

    /////////////    Process      /////////////
    auto func = [&](const uint32_t start, const uint32_t end) {
        // printf("start(%d),end(%d)\n", start, end);
        for (uint32_t idx = start; idx < end; idx++)
            CompTest(*(a + idx), *(b + idx), len);
    };

    // without thread pool.
    for (uint32_t idx = 0; idx < num; idx++)
        CompTest(*(a + idx), *(b + idx), len);
    Check(b, num, len, 2);

    // test TaskEnqueue
    std::future<void> res = thread_pool.TaskEnqueue(func, 0, num);
    res.wait();
    Check(b, num, len, 3);
    // printf("1.\n");

    // test ParallelFor
    thread_pool.ParallelFor(func, 0, num, 5);
    Check(b, num, len, 4);
    // printf("2.\n");

    // test ParallelFor
    thread_pool.ParallelFor(func, 0, num, 6);
    Check(b, num, len, 5);
    // printf("3.\n");

    // test ParallelFor
    thread_pool.ParallelFor(func, 0, num, 7);
    Check(b, num, len, 6);
    // printf("4.\n");
    
    // test ParallelFor
    thread_pool.ParallelFor(func, 0, num, 8);
    Check(b, num, len, 7);
    // printf("5.\n");

    /////////////    Clear      /////////////
    thread_pool.ClearPool();
    for (uint32_t i = 0; i < num; i++) {
        free(a[i]);
        free(b[i]);
    }
    free(a);
    free(b);
}

// TEST(ThreadTest, ThreadPool) {
//     ThreadPoolTest();
// }

}  // end of namespace.