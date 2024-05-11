#ifndef POCKET_AI_MEMORY_BLOCKING_QUEUE_HPP_
#define POCKET_AI_MEMORY_BLOCKING_QUEUE_HPP_

#include <queue>
#include <mutex>
#include <condition_variable>

namespace pai {
namespace memory {

template <typename T>
class BlockingQueue {
public:
    BlockingQueue() : is_exit_(false) {};
    ~BlockingQueue() {};

    void Push(const T& t) {
        std::unique_lock <std::mutex> lock(mutex_);
        queue_.emplace(t); // push
        lock.unlock();
        cond_var_.notify_one();
    }

    bool TryGetFront(T* t) {
        std::unique_lock <std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;

        *t = queue_.front();
        return true;
    }
    
    bool TryPop(T* t) {
        std::unique_lock <std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;

        *t = queue_.front();
        queue_.pop();
        return true;
    }

    bool BlockingPop(T* t) {
        std::unique_lock <std::mutex> lock(mutex_);
        while (!is_exit_ && queue_.empty())
            cond_var_.wait(lock);

        if (is_exit_) return false;

        *t = queue_.front();
        queue_.pop();
        return true;
    }

    inline bool empty() const { return queue_.empty(); }
    inline int size() const { return queue_.size(); }
    inline void exit() { is_exit_ = true; cond_var_.notify_all(); }

private:
    bool is_exit_;
    mutable std::mutex mutex_;
    std::condition_variable cond_var_;
    std::queue<T> queue_;
};

} // namespace memory
} // namespace pai

#endif // POCKET_AI_MEMORY_BLOCKING_QUEUE_HPP_