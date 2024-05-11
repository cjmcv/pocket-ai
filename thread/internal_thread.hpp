/*!
* \brief Internal Thread.
*        Mainly implemented by thread.
*/

#ifndef POCKET_AI_INTERNAL_THREAD_HPP_
#define POCKET_AI_INTERNAL_THREAD_HPP_

#include <stdio.h>
#include <thread>
#include <memory>

namespace pai {
namespace thread {

class InternalThread {
public:
    InternalThread() : thread_(), interrupt_flag_(false) {}
    virtual ~InternalThread() { Stop(); }

    // To chech wether the internal thread has been started. 
    inline bool is_started() const { return thread_ && thread_->joinable(); }

    bool Start() {
        if (is_started()) {
            printf("Threads should persist and not be restarted.");
            return false;
        }
        try {
            thread_.reset(new std::thread(&InternalThread::Entry, this));
        }
        catch (std::exception& e) {
            printf("Thread exception: %s", e.what());
        }

        return true;
    }

    void Stop() {        
        // This flag will work in must_stop.
        if (!is_started()) 
            return ;

        interrupt_flag_ = true;
        try {
            thread_->join();
        }
        catch (std::exception& e) {
            printf("Thread exception: %s", e.what());
        }
    }

protected:
    // Virtual function, should be override by the classes
    // which needs a internal thread to assist.
    virtual void Entry() {}
    bool IsMustStop()  {
        if (thread_ && interrupt_flag_) {
            interrupt_flag_ = false;
            return true;
        }
        return false;
    }

private:
    bool interrupt_flag_;
    std::shared_ptr<std::thread> thread_;
};

} // namespace thread
} // namespace pai

#endif // POCKET_AI_INTERNAL_THREAD_HPP_
