/*!
* \brief vulkan common. 
*/

#ifndef PTK_ENGINE_VULKAN_COMMON_HPP_
#define PTK_ENGINE_VULKAN_COMMON_HPP_

#include "../../util/logger.hpp"

// Used for validating return values of Vulkan API calls.
#define VK_CHECK(f) 	{																			\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)	{											        						\
        PTK_LOGE("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
    }																									\
}

namespace ptk {
namespace vk {

}  // end of namespace vk.
}  // end of namespace ptk.

#endif // PTK_ENGINE_VULKAN_COMMON_HPP_