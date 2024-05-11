/*!
* \brief timer.
*/

#ifndef POCKET_AI_PROF_TIMER_HPP_
#define POCKET_AI_PROF_TIMER_HPP_

#include <iostream>
#include <vector>

#include <util/logger.hpp>

#define CXX_TIMER_CHRONO
// #define CXX_TIMER_SYS
// #define CXX_TIMER_FREERTOS

namespace pai {
namespace prof {

#ifdef CXX_TIMER_CHRONO
// Timer for cpu.
#include <chrono>
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
#endif // CXX_TIMER_CHRONO

#ifdef CXX_TIMER_SYS
#include <sys/time.h>
class CpuTimer {
public:
    inline void Start() { gettimeofday(&start_time_, NULL); }
    inline void Stop() { gettimeofday(&stop_time_, NULL); }
    inline float MilliSeconds() {
        uint64_t t = 1000000 * (stop_time_.tv_sec - start_time_.tv_sec) + stop_time_.tv_usec - start_time_.tv_usec;
        return (float)t / 1000.f;
    }

protected:
    struct timeval start_time_;
    struct timeval stop_time_;
};
#endif // CXX_TIMER_SYS

#ifdef CXX_TIMER_FREERTOS
#include <FreeRTOS.h>
class CpuTimer {
public:
    inline void Start() { start_time_ = xTaskGetTickCount(); }
    inline void Stop() { stop_time_ = xTaskGetTickCount(); }
    inline float MilliSeconds() {
        return (float)(stop_time_ - start_time_) * portTICK_PERIOD_MS;
    }

protected:
    uint32_t start_time_;
    uint32_t stop_time_;
};
#endif

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
        if (util::Logger::GetInstance()->min_log_level() > util::INFO_DEBUG)
            return;
        timer_.Start();
    }

    void Stop(uint32_t idx, std::string msg) {
        if (util::Logger::GetInstance()->min_log_level() > util::INFO_DEBUG)
            return;

        timer_.Stop();
        float time = timer_.MilliSeconds();

        if (idx >= num_) {
            PAI_LOGE("Timer::Stop -> idx(%d) >= num_(%d).\n", idx, num_);
        }

        if (time > max_[idx])
            max_[idx] = time;
        if (time < min_[idx])
            min_[idx] = time;

        acc_[idx] += time;
        count_[idx]++;
        msg_[idx] = msg;
    }

    void Print(uint32_t specify_id) {
        if (util::Logger::GetInstance()->min_log_level() > util::INFO_DEBUG)
            return;
        PAI_LOGS("Time %.3f ms.", acc_[specify_id] / count_[specify_id]);
    }

    void Print(uint32_t specify_id, uint32_t print_interval) {
        if (util::Logger::GetInstance()->min_log_level() > util::INFO_DEBUG)
            return;

        if (print_interval != 0 && count_[specify_id] >= print_interval) {
            for (uint32_t i=0; i<num_; i++) {
                if (count_[i] <= 0)
                    continue;
                PAI_LOGS("Timer(%s-%s) idx(%d) cnt(%d): %.3f ms (min: %.3f, max: %.3f).\n", 
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
//    pai::QueryDevices();
//    timer.Stop();
//    return timer.MilliSeconds();
//  };
//  ret = func();
// #define POCKET_AI_TIME_DIFF_RECORD(timer, ...)  \
//     [&]() -> void {                       \
//         timer.Start();                    \
//         {__VA_ARGS__}                     \
//         timer.Stop();                     \
//     }();

} // prof.
} // pai.
#endif //POCKET_AI_PROF_TIMER_HPP_
