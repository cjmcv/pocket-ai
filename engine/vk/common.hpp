/*!
* \brief vulkan common. 
*/

#ifndef POCKET_AI_ENGINE_VULKAN_COMMON_HPP_
#define POCKET_AI_ENGINE_VULKAN_COMMON_HPP_

#include "../../util/logger.hpp"

// Used for validating return values of Vulkan API calls.
#define VK_CHECK(f) 	{																			\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)	{											        						\
        PAI_LOGE("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
    }																									\
}

namespace pai {
namespace vk {

}  // end of namespace vk.
}  // end of namespace pai.

#endif // POCKET_AI_ENGINE_VULKAN_COMMON_HPP_