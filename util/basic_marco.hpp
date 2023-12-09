/*!
* \brief Util.
*/

#ifndef PTK_UTIL_BASIC_MARCO_HPP_
#define PTK_UTIL_BASIC_MARCO_HPP_

#include <iostream>
#include "logger.hpp"

namespace ptk {
namespace util {

#define PTK_DEBUG
#ifdef PTK_DEBUG
#define PTK_ASSERT(x)                                            \
    do {                                                         \
        int res = (x);                                           \
        if (!res) {                                              \
            PTK_LOGE("An error occurred.\n");                    \
        }                                                        \
    } while (0)
#else
#define PTK_ASSERT(x)
#endif

/////////////////////////////////////////////////////////////////////////////////////
// If this method is used, it will generate many warnings, such as unused-value.
// #define PRINT printf 
// #define PRINT 
//
// Recommended -> 
// #define PRINT PTK_PRINTF
// #define PRINT PTK_NO_PRINTF
#define PTK_PRINTF(format, ...)        \
    do {                               \
        printf(format, ##__VA_ARGS__); \
    } while(0); 
#define PTK_NO_PRINTF(format, ...)

// Usage -> 
// #define DEBUG_CALL PTK_DEBUG_CALL
// #define DEBUG_CALL PTK_DEBUG_NO_CALL
#define PTK_DEBUG_CALL(x)  \
    do {                   \
        {x;}               \
    } while(0); 
#define PTK_DEBUG_NO_CALL(x)

} // util
} // ptk.
#endif //PTK_UTIL_BASIC_MARCO_HPP_
