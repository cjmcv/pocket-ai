/*!
* \brief . 
*/

#include "thread/internal_thread.hpp"

#include <thread>
#include <queue>
#include "gtest/gtest.h"

namespace {

using namespace pai::thread;

// Producer and consumer
// You can use BlockingQueue to synchronize.
class MyClass : public InternalThread {
public:
    MyClass(uint32_t step): step_(step) {}
    // Consumer
    int32_t GetValue() {
        if (values_.size() <= 0) {
            return -1;
        }
        int32_t v = values_.front();
        values_.pop();
        return v;
    }

    // Override
    virtual void Entry() {
        // Producer
        while (!IsMustStop()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            static int32_t i=0;
            i += step_;
            values_.push(i);
        }
    }

private:
    uint32_t step_;
    std::queue<int32_t> values_;
};

void InternalThreadTest() {
    MyClass c1(1);
    c1.Start();

    int32_t i = 0;
    bool is_exit = false;
    while (!is_exit) {
        int32_t v = c1.GetValue();
        if (v != -1)
            EXPECT_EQ(v, i += 1);
        if (v > 20) {
            c1.Stop();
            is_exit = true;      
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

TEST(ThreadTest, InternalThread) {
    InternalThreadTest();
}

}  // end of namespace.