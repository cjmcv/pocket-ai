/*!
* \brief FrameShiftCache
*     用于处理帧长和帧移大小不一致时的数据缓存逻辑
*/

#ifndef PTK_MEMORY_FRAME_SHIFT_CACHE_HPP_
#define PTK_MEMORY_FRAME_SHIFT_CACHE_HPP_

#include <cstdlib>
#include <cstring>
#include "util/logger.hpp"

namespace ptk {
namespace memory {

class FrameShiftCache {
public:
    FrameShiftCache(int frame_size, int frame_shift_size) {
        frame_size_ = frame_size;
        frame_shift_size_ = frame_shift_size;
        capaticy_ = frame_size *2;// 两倍空间

        data_ = (char *)malloc(capaticy_);
        pushed_size_ = 0;
    }

    ~FrameShiftCache() {
        free(data_);
    }
    //在is_ready为true后，可以使用data。使用一次后需要Pop()
    inline char *data() { return data_; }
    inline bool is_ready() { return pushed_size_ >= frame_size_? true : false; }

    // push数据;如空间有余，则返回true，否则为false
    //在使用后需要pop数据，pop一次减少一次帧移数据
    bool Push(const char *data, const int size) {
        if (size > capaticy_ - pushed_size_)
            return false;
        memcpy(data_ + pushed_size_, data, size);
        pushed_size_ += size;
        return true;
    } 
    void Pop() {
        // printf("is_ready: %d.\n", pushed_size_ >= frame_size_? true : false);
        pushed_size_ -= frame_shift_size_;
        // printf("pop: %d, %d, %d.\n", data, data  + frame_shift size , pushed size_);
        memmove(data_, data_ + frame_shift_size_, pushed_size_);
    }

    void Reset() {
        memset(data_, 0, frame_size_);
        pushed_size_ = 0;
    }

public:
    int frame_size_;
    int frame_shift_size_;
    int capaticy_;
    char *data_;
    int pushed_size_;
};

} // memory.
} // ptk.
#endif //PTK_UTIL_FRAME_SHIFT_CACHE_HPP_
