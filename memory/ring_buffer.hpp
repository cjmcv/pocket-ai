/*!
* \brief RingBuffer with cond.
*/

#ifndef POCKET_AI_MEMORY_RINGBUFFER_HPP_
#define POCKET_AI_MEMORY_RINGBUFFER_HPP_

#include <thread>
#include <chrono>
#include <condition_variable>
#include <string.h>

#include "pocket-ai/util/logger.hpp"

namespace pai {
namespace memory {

// TODO: 可选禁用线程
class RingBuffer{
public:
    RingBuffer(uint32_t size) {
        data_ = (unsigned char *)malloc(size);
        if (data_ == nullptr) {
            PAI_LOGE("RingBuffer::RingBuffer:malloc for data fail!n");
        }

        start_ = data_;
        end_ = start_ + size;
        capacity_ = size;

        //    <head> empty [tail] data.data.data.data [head] empty <end>
        // or <head> data.data [head] empty.empty [tail] data.data <end>
        head_ = start_;
        tail_ = start_;
        payload_= 0;
    }

    ~RingBuffer() { 
        if(data_ != nullptr) {
            free(data_);
            data_ = nullptr;
        }
    }

    bool Write(char *data, const uint32_t size, bool is_blocking = true) {
        if ((nullptr == data) || (0 == size) || (size > capacity_))
            return false;

        std::unique_lock <std::mutex> lock(mutex_);
        // Wait if there is not enough space left
        while (size > (capacity_ - payload_)) {
            if (is_blocking)
                not_full_.wait(lock);
            else
                return false;
        }

        if(head_ >= tail_) {
            // tail d.d.d.d head e.e <end>
            if (size <= (end_ - head_)) {
                memcpy(head_, data, size);
                head_ += size;
            }
            else {
                unsigned int tmp_size = end_ - head_;
                memcpy(head_, data, tmp_size);
                head_ = start_;
                memcpy(head_, data + tmp_size, size - tmp_size);
                head_ += size - tmp_size;
            }
        }
        else if (head_ < tail_) {
            // from d.d head e.e.e tail d.d
            // to   d.d.d head e.e tail d.d
            memcpy(head_, data, size);
            head_ += size;
        }
        if(head_ >= end_) {
            head_ = start_;
        }
        payload_ += size;

        not_empty_.notify_one();
        return true;
    }

    bool Read(char *data, const uint32_t size, bool is_blocking = true) {
        if ((0 == size) || (data == nullptr) || (size > capacity_))
            return false;
            
        std::unique_lock <std::mutex> lock(mutex_);
        // Wait if there is not enough available data
        while(size > payload_) {
            if (is_blocking)
                not_empty_.wait(lock);
            else
                return false;
        }

        if(head_ > tail_) {
            memcpy(data, tail_, size);
            tail_ += size;
        }
        else if (head_ < tail_ || payload_ == capacity_) {
            // when payload_ == capacity_, head_ will be equal to tail_.
            if (size <= (end_ - tail_)) {
                memcpy(data, tail_, size);
                tail_ += size;
            }
            else {
                int tmp_size = end_ - tail_;
                memcpy(data, tail_, tmp_size);
                tail_ = start_; 
                memcpy(data + tmp_size, tail_, size - tmp_size);
                tail_ += size-tmp_size;
            }
        }
        if (tail_ >= end_) {
            tail_ = start_;
        }
        payload_ -= size;

        not_full_.notify_one();
        return true;
    }

    uint32_t GetPayloadSize() {
        std::unique_lock <std::mutex> lock(mutex_);
        uint32_t size = 0;
        size = payload_;
        return size;
    }

    uint32_t GetFreeSize() {         
        std::unique_lock <std::mutex> lock(mutex_);
        uint32_t size = 0;
        size = capacity_ - payload_;
        return size;
    }

    void Reset(){
        std::unique_lock <std::mutex> lock(mutex_);
        head_ = start_;
        tail_ = start_;
        payload_ = 0;
        not_full_.notify_all();
    }

private:
    unsigned char *data_;// 内存起始点
    // 内存起始点
    unsigned char *start_;
    unsigned char *end_;
    // 内存结束点
    uint32_t capacity_;// 总容量
    unsigned char *head_;
    unsigned char *tail_;
    uint32_t payload_;
    
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_; 
};

} // memory.
} // pai.
#endif //POCKET_AI_MEMORY_RINGBUFFER_HPP_
