/*!
* \brief timer.
*/

#ifndef PTK_PROF_TIMER_HPP_
#define PTK_PROF_TIMER_HPP_

#include <iostream>
#include <chrono>

#include <util/logger.hpp>

namespace ptk {
namespace prof {

// Timer for cpu.
class CpuTimer {
public:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::nanoseconds ns;

    inline void Start() { start_time_ = clock::now(); }
    inline void Stop() { stop_time_ = clock::now(); }
    inline float NanoSeconds() {
        return (float)std::chrono::duration_cast<ns>(stop_time_ - start_time_).count();
    }

    inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }
    inline float MicroSeconds() { return NanoSeconds() / 1000.f; }
    inline float Seconds() { return NanoSeconds() / 1000000000.f; }

protected:
    std::chrono::time_point<clock> start_time_;
    std::chrono::time_point<clock> stop_time_;
};

class Timer {
public:
    Timer(std::string name, uint32_t num)
        : name_(name), num_(num) {
        count_.resize(num);
        min_.resize(num);
        max_.resize(num);
        acc_.resize(num);
        msg_.resize(num);

        for (uint32_t i=0; i<num; i++) min_[i] = 999999;
    }

    ~Timer() {}

    void Start() {
        timer_.Start();
    }

    void Stop(uint32_t idx, std::string msg) {
        timer_.Stop();
        float time = timer_.MilliSeconds();

        if (idx >= num_) {
            PTK_LOGE("Timer::Stop -> idx(%d) >= num_(%d).\n", idx, num_);
        }

        if (time > max_[idx])
            max_[idx] = time;
        if (time < min_[idx])
            min_[idx] = time;

        acc_[idx] += time;
        count_[idx]++;
        msg_[idx] = msg;
    }

    void Print(uint32_t specify_id, uint32_t print_interval = 1) {
        if (print_interval != 0 && count_[specify_id] >= print_interval) {
            for (uint32_t i=0; i<num_; i++) {
                if (count_[i] <= 0)
                    continue;
                PTK_LOGS("Timer(%s-%s) idx(%d) cnt(%d): %.3f ms (min: %.3f, max: %.3f).\n", 
                    name_.c_str(), msg_[i].c_str(), i, count_[i], acc_[i] / count_[i], min_[i], max_[i]);
            }
            for (uint32_t i=0; i<num_; i++) {
                count_[i] = 0;
                max_[i] = 0;
                acc_[i] = 0;
                min_[i] = 999999;
            }
        }
    }

private:
    CpuTimer timer_;

    std::string name_;
    uint32_t num_;
    std::vector<uint32_t> count_;
    std::vector<std::string> msg_;

    std::vector<float> min_;
    std::vector<float> max_;
    std::vector<float> acc_;
};

/////////////////////////////////////////////////
//  auto func = [&]()
//  -> float {
//    timer.Start();
//    ptk::QueryDevices();
//    timer.Stop();
//    return timer.MilliSeconds();
//  };
//  ret = func();
// #define PTK_TIME_DIFF_RECORD(timer, ...)  \
//     [&]() -> void {                       \
//         timer.Start();                    \
//         {__VA_ARGS__}                     \
//         timer.Stop();                     \
//     }();

} // prof.
} // ptk.
#endif //PTK_PROF_TIMER_HPP_
