/*!
* \brief Logger.
*/

#ifndef POCKET_AI_UTIL_LOGGER_HPP_
#define POCKET_AI_UTIL_LOGGER_HPP_

#include <iostream>
#include <sstream>
#include <mutex>

#ifdef POCKET_AI_PLATFORM_ANDRIOD
#include <android/log.h>
#endif // POCKET_AI_PLATFORM_ANDRIOD

namespace pai {
namespace util {

enum LogLevel {
    INFO_DEBUG = 0,
    INFO_SIMPLE,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static inline Logger *GetInstance() {
        static Logger instance;
        return &instance;
    }
    inline void SetMinLogLevel(LogLevel level) { min_log_level_ = level; }
    inline int min_log_level() { return min_log_level_; }
    inline char *buffer() { return buffer_; }
        
    void GenerateLogMessage(const char *fname, int line, int severity)  {
    #ifdef POCKET_AI_PLATFORM_ANDRIOD
        if (severity == LogLevel::ERROR)
            __android_log_print(ANDROID_LOG_ERROR, "com.pai", "[%s:%d] %s", fname, line, buffer_);
        else if (severity == LogLevel::WARNING)
            __android_log_print(ANDROID_LOG_INFO, "com.pai", "[%s:%d] %s", fname, line, buffer_);
        else
            __android_log_print(ANDROID_LOG_INFO, "com.pai", "[%s:%d] %s", fname, line, buffer_);
    #else
        if (severity != LogLevel::INFO_SIMPLE) {
            fprintf(stderr, "<%c>", "IIWE"[severity]);
            fprintf(stderr, " %s:%d] ", fname, line);        
        }
        fprintf(stderr, "%s", buffer_);
    #endif // POCKET_AI_PLATFORM_ANDRIOD

        if (severity == LogLevel::ERROR)
            std::abort();
    }

private:
    Logger() { buffer_ = new char[1024]; min_log_level_ = LogLevel::INFO_DEBUG; }
    ~Logger() { delete[] buffer_; }
    char *buffer_;
    
    int min_log_level_;
};

#define PAI_LOG_BODY(level, format, ...)                                      \
    do {                                                                      \
        util::Logger *log = util::Logger::GetInstance();                      \
        if (level < log->min_log_level())                                     \
            break;                                                            \
        static std::mutex m;                                                  \
        m.lock();                                                             \
        sprintf(log->buffer(), format, ##__VA_ARGS__);                        \
        log->GenerateLogMessage(__FILE__, __LINE__, level);                   \
        m.unlock();                                                           \
    } while (0)

#define PAI_LOGD(format, ...)      PAI_LOG_BODY(util::LogLevel::INFO_DEBUG, format, ##__VA_ARGS__);
#define PAI_LOGS(format, ...)      PAI_LOG_BODY(util::LogLevel::INFO_SIMPLE, format, ##__VA_ARGS__);
#define PAI_LOGI(format, ...)      PAI_LOG_BODY(util::LogLevel::INFO, format, ##__VA_ARGS__);
#define PAI_LOGW(format, ...)      PAI_LOG_BODY(util::LogLevel::WARNING, format, ##__VA_ARGS__);
#define PAI_LOGE(format, ...)      PAI_LOG_BODY(util::LogLevel::ERROR, format, ##__VA_ARGS__);

#define POCKET_AI_CNT_LOGI(cnt, format, ...)           \
    do {                                          \
        static int count = 0;                     \
        count++;                                  \
        if (count >= cnt) {                       \
            PAI_LOGI(format, ##__VA_ARGS__);     \
            count = 0;                            \
        } while (0)                               \
    }

} // util
} // pai

#endif // POCKET_AI_UTIL_LOGGER_HPP_