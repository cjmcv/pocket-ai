/*!
* \brief . 
*/

#include "memory/blocking_queue.hpp"

#include "gtest/gtest.h"

namespace {

using namespace ptk::memory;

class MyClass {
public:
    MyClass(uint32_t value, std::string str) 
    : value_(value), str_(str) {
        // printf("Construct %d.\n", value_);
    }
    ~MyClass() {
        // printf("Deconstruct %d.\n", value_);
    }
    uint32_t value() { return value_; }
    std::string str() { return str_; }

private:
    uint32_t value_;
    std::string str_;
};

void BlockingQueueTest() {
    BlockingQueue<MyClass *> *bq = new BlockingQueue<MyClass *>();

    for (int i=0; i<500; i++) {
        std::string str = "abc" + std::to_string(i);
        bq->Push(new MyClass(i, str));
    }

    MyClass *c;
    bq->TryGetFront(&c);

    EXPECT_EQ(c->value(), 0);
    EXPECT_EQ(c->str(), "abc0");

    for (int i=0; i<500; i++) {
        MyClass *c;
        bq->BlockingPop(&c);
        EXPECT_EQ(c->value(), i);
        // printf("%d,", c->value());
        delete c;
    }

    delete bq;
}

TEST(MemoryTest, BlockingQueue) {
    BlockingQueueTest();
}

}  // end of namespace.