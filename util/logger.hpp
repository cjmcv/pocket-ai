/*!
* \brief Logger.
*/

#ifndef PTK_UTIL_LOGGER_HPP_
#define PTK_UTIL_LOGGER_HPP_

#include <iostream>
#include <sstream>
#include <mutex>

#ifdef PTK_PLATFORM_ANDRIOD
#include <android/log.h>
#endif // PTK_PLATFORM_ANDRIOD

namespace ptk {
namespace util {

enum LogLevel {
    INFO_SIMPLE = 0,
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
    #ifdef PTK_PLATFORM_ANDRIOD
        if (severity == LogLevel::ERROR)
            __android_log_print(ANDROID_LOG_ERROR, "com.ecas", "[%s:%d] %s", fname, line, buffer_);
        else if (severity == LogLevel::WARNING)
            __android_log_print(ANDROID_LOG_INFO, "com.ecas", "[%s:%d] %s", fname, line, buffer_);
        else
            __android_log_print(ANDROID_LOG_INFO, "com.ecas", "[%s:%d] %s", fname, line, buffer_);
    #else
        if (severity != LogLevel::INFO_SIMPLE) {
            fprintf(stderr, "<%c>", "IIWE"[severity]);
            fprintf(stderr, " %s:%d] ", fname, line);        
        }
        fprintf(stderr, "%s", buffer_);
    #endif // PTK_PLATFORM_ANDRIOD

        if (severity == LogLevel::ERROR)
            std::abort();
    }

private:
    Logger() { buffer_ = new char[1024]; min_log_level_ = LogLevel::INFO_SIMPLE; }
    ~Logger() { delete[] buffer_; }
    char *buffer_;
    
    int min_log_level_;
};

#define PTK_LOG_BODY(level, format, ...)                                      \
    do {                                                                      \
        util::Logger *log = util::Logger::GetInstance();                      \
        if (level < log->min_log_level())                            \
            break;                                                            \
        static std::mutex m;                                                  \
        m.lock();                                                             \
        sprintf(log->buffer(), format, ##__VA_ARGS__);                        \
        log->GenerateLogMessage(__FILE__, __LINE__, level);                   \
        m.unlock();                                                           \
    } while (0)

#define PTK_LOGS(format, ...)      PTK_LOG_BODY(util::LogLevel::INFO_SIMPLE, format, ##__VA_ARGS__);
#define PTK_LOGI(format, ...)      PTK_LOG_BODY(util::LogLevel::INFO, format, ##__VA_ARGS__);
#define PTK_LOGW(format, ...)      PTK_LOG_BODY(util::LogLevel::WARNING, format, ##__VA_ARGS__);
#define PTK_LOGE(format, ...)      PTK_LOG_BODY(util::LogLevel::ERROR, format, ##__VA_ARGS__);

#define PTK_CNT_LOGI(cnt, format, ...)           \
    do {                                          \
        static int count = 0;                     \
        count++;                                  \
        if (count >= cnt) {                       \
            PTK_LOGI(format, ##__VA_ARGS__);     \
            count = 0;                            \
        } while (0)                               \
    }

} // util
} // ptk

#endif // PTK_UTIL_LOGGER_HPP_