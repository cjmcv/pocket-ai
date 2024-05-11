/*!
* \brief Thread Pool.
*        Mainly implemented by thread, queue, future and condition_variable.
*/


#ifndef POCKET_AI_THREAD_POOL_HPP_
#define POCKET_AI_THREAD_POOL_HPP_

#include <thread>
#include <queue>
#include <future>
#include <functional>
#include <condition_variable>

namespace pai {
namespace thread {

class ThreadPool {
public:
    ThreadPool() : is_created_(false) {}
    ~ThreadPool() {};
  
    void CreateThreads(uint32_t thread_num) {
        is_stop_ = false;
        if (is_created_ == true) {
            if (workers_.size() == thread_num)
                return;
            else {
                ClearPool();
                is_created_ = false;
            }
        }

        for (uint32_t i = 0; i < thread_num; ++i) {
            workers_.emplace_back([this] {
                // Threads live in this loop.
                while (1) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        // Waiting to be activated.
                        this->condition_.wait(lock,
                            [this] { return this->is_stop_ || !this->tasks_.empty(); });
                        // If the thread pool is closed and the task queue is empty.
                        if (this->is_stop_ && this->tasks_.empty())
                            return;
                        // Get a task from the tasks queue.
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    // Execute the task.
                    task();
                }
            });
        }

        is_created_ = true;
    }

    void ClearPool() {
        if (is_created_ == false)
            return;
        is_stop_ = true; // TODO: 是否有需要留着？

        // Activates all threads in the thread pool and
        // waits for all threads to complete their work.
        condition_.notify_all();
        for (std::thread &worker : workers_)
            worker.join();

        workers_.clear();
        tasks_ = decltype(tasks_)();

        is_created_ = false;
    }

    // Add a new task to the pool.
    // If there is an inactive thread, the task will be executed immediately.
    template<class F, class... Args>
    auto TaskEnqueue(F&& f, Args&&... args)
    ->std::future<typename std::result_of<F(Args...)>::type> {
        if (is_created_ == false)
            printf("Error: Please create a Thread Pool first.\n");

        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (is_stop_)
                throw std::runtime_error("TaskEnqueue on stopped ThreadPool");
            // Add a task to tasks queue.
            tasks_.emplace([task]() { (*task)(); });
        }
        // Activate/notify a thread.
        condition_.notify_one();
        return res;
    }

    // Use all threads in the thread pool to run the task.
    // The workload per thread is split evenly.
    // Note: The entire thread pool will serve this task 
    //       and no other tasks should be inserted.
    void ParallelFor(std::function<void(const uint32_t, const uint32_t)> func, 
                     const uint32_t start_idx, const uint32_t end_idx, 
                     const uint32_t threads_num) {
        if (start_idx > end_idx) {
            printf("[ ThreadPool::ParallelFor ]: start_idx > end_idx\n");
            return;
        }

        if (workers_.size() <= 1 || threads_num <= 1) {
            func(start_idx, end_idx);
        }
        else {
            uint32_t len = end_idx - start_idx;
            const uint32_t datum_per_thread = len / threads_num;

            uint32_t cur_start_idx = start_idx;
            futures_.clear();
            for (uint32_t i = 0; i < threads_num; i++) {
                int cur_end_idx = cur_start_idx + datum_per_thread;
  
                futures_.emplace_back(TaskEnqueue(func, cur_start_idx, cur_end_idx));

                cur_start_idx = cur_end_idx;
            }

            // Separate processing of remaining parts.
            uint32_t datum_remain = len - datum_per_thread * threads_num;
            if (datum_remain > 0)
                func(cur_start_idx, end_idx);

            // Waiting for the tasks to be completed.
            for (uint32_t i = 0; i < futures_.size(); i++)
                futures_[i].wait();
        }
    }

private:
    // The Threads generated in this thread pool.
    std::vector< std::thread > workers_;
    // Stores the functional tasks that need to be run.
    std::queue< std::function<void()> > tasks_;
    // Gets the result from the asynchronous task.
    std::vector<std::future<void>> futures_;
    // Synchronization.
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool is_stop_;
    bool is_created_;
};





} // thread
} // pai

#endif // POCKET_AI_THREAD_POOL_HPP_