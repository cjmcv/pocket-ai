/*!
* \brief Util.
*/

#ifndef POCKET_AI_UTIL_BASIC_MARCO_HPP_
#define POCKET_AI_UTIL_BASIC_MARCO_HPP_

#include <iostream>
#include "logger.hpp"

namespace pai {
namespace util {

#define POCKET_AI_DEBUG
#ifdef POCKET_AI_DEBUG
#define POCKET_AI_ASSERT(x)                                            \
    do {                                                         \
        int res = (x);                                           \
        if (!res) {                                              \
            PAI_LOGE("An error occurred.\n");                    \
        }                                                        \
    } while (0)
#else
#define POCKET_AI_ASSERT(x)
#endif

/////////////////////////////////////////////////////////////////////////////////////
// If this method is used, it will generate many warnings, such as unused-value.
// #define PRINT printf 
// #define PRINT 
//
// Recommended -> 
// #define PRINT POCKET_AI_PRINTF
// #define PRINT POCKET_AI_NO_PRINTF
#define POCKET_AI_PRINTF(format, ...)        \
    do {                               \
        printf(format, ##__VA_ARGS__); \
    } while(0); 
#define POCKET_AI_NO_PRINTF(format, ...)

// Usage -> 
// #define DEBUG_CALL POCKET_AI_DEBUG_CALL
// #define DEBUG_CALL POCKET_AI_DEBUG_NO_CALL
#define POCKET_AI_DEBUG_CALL(x)  \
    do {                   \
        {x;}               \
    } while(0); 
#define POCKET_AI_DEBUG_NO_CALL(x)

} // util
} // pai.
#endif //POCKET_AI_UTIL_BASIC_MARCO_HPP_
