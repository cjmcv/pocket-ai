/*!
* \brief Tensor. 
*        数据维度形态等，不涉及具体内存。具体内存Buffer中处理
*/

#ifndef PTK_ENGINE_GRAPH_TENSOR_HPP_
#define PTK_ENGINE_GRAPH_TENSOR_HPP_

#include <string>
#include <vector>
#include <string.h>

#include "../../util/type.hpp"
#include "../../util/logger.hpp"

namespace ptk {
namespace engine {

// Tensor 为Tensor的父类，对外数据交互, 隐藏细节，不允许外部构造
// Tensor 为Tensor子类，实际内存操作者，跨设备，异步拷贝等
// Buffer 实际内存管理提供者，多设备多类型内存
// Tensor绑定Buffer，而一般不持有Buffer, 除非使用外部内存，外部使用对Buffer无感。

// 基本数据计算与操作

class Tensor {

public:
    Tensor(std::vector<int> &&shape, util::Type type) { Create(shape, type); }
    Tensor(std::vector<int> &shape, util::Type type) { Create(shape, type); }
    ~Tensor() { Release(); }

    inline void SetId(int id) { id_ = id; }
    inline uint32_t id() { return id_; }
    inline util::MemoryLoc mem_loc() { return mem_loc_; }
    inline uint32_t size() { return size_; }
    inline std::vector<int> &shape() { return shape_; }

    void CopyFrom(Tensor *in) {
        // Check dimension.
        CheckDimension(in);
        // Check memory type,
        if (mem_loc_ != in->mem_loc()) {
            PTK_LOGE("Tensor::CloneFrom -> memory type mismatch.\n");
        }
        id_ = in->id();
        memcpy(GetData(), in->GetData(), size_);
    }

    void CopyTo(Tensor *out) {
        // Check dimension.
        CheckDimension(out);
        // Check memory type,
        if (mem_loc_ != out->mem_loc()) {
            PTK_LOGE("Tensor::CopyTo -> memory loc mismatch.\n");
        }
        out->SetId(id_);
        memcpy(out->GetData(), GetData(), size_);
    }
    
    // Use external memory.
    void BindHostDataPtr(void *data) {
        is_owned_data_ = false;
        data_ = data;
    }
    
    void *GetData(util::MemoryLoc mode = util::ON_HOST) {
        if (mode == util::ON_HOST) {
            if (mem_loc_ == util::ON_HOST)
                return data_;
            else {
                // TODO: push data from host to device.

            }
        }
        else {
            if (mem_loc_ == util::ON_HOST) {
                // TODO: push data from device to host
                return data_;
            }
            else {

            }
        }
    }

    void Print() {
        PTK_LOGS("\n====== Tensor %p ======\n", this);
        PTK_LOGS("\nShape: ");
        for (int i = 0; i < shape_.size(); i++) {
            PTK_LOGS("%d, ", shape_[i]);
        }

        PTK_LOGS("\nData: \n");
        int s[4] = {1, 1, 1, 1};
        memcpy(s + 4 - shape_.size(), &shape_[0], sizeof(uint32_t) * shape_.size());

        void *host_data = GetData(util::ON_HOST);
        TYPE_SWITCH(type_, T, {
            T *data = (T *)host_data;
            for (int n = 0; n < s[0]; n++) {
                int n_bias = n * s[1] * s[2] * s[3];
                for (int c = 0; c < s[1]; c++) {
                    int c_bias = c * s[2] * s[3];
                    PTK_LOGS("(n: %d, c: %d): ", n, c);
                    for (int h = 0; h < s[2]; h++) {
                        int h_bias = h * s[3];
                        for (int w = 0; w < s[3]; w++) {
                            std::cout << data[n_bias + c_bias + h_bias + w] << ", ";
                        }
                    }
                }
            }
        });
    }

private:
    void Create(std::vector<int> &shape, util::Type type) {
        id_ = -1;        
        type_ = type;
        shape_ = shape;

        TYPE_SWITCH(type, T, size_ = sizeof(T););
        for (int i=0; i < shape.size(); i++) {
            size_ *= shape[i];
        }
        if (size_ == 0)
            std::abort();
        
        is_owned_data_ = true;
        data_ = new int8_t[size_];
        mem_loc_ = util::ON_HOST;
    }

    void Release() {
        if (is_owned_data_ == true) {
            if (mem_loc_ == util::ON_HOST)
                delete[] data_;
        }
    }

    void CheckDimension(Tensor *target) {
        if (shape_.size() != target->shape().size()) {
            PTK_LOGE("Tensor::CloneFrom -> shape mismatch.\n");
        }
        for (int i=0; i<shape_.size(); i++) {
            if (shape_[i] != target->shape()[i]) {
                PTK_LOGE("Tensor::CloneFrom -> shape mismatch.\n");
            }
        }
    }

private:
    uint32_t id_;
    util::MemoryLoc mem_loc_;

    util::Type type_;
    uint32_t size_;
    std::vector<int> shape_; // n c h w

    bool is_owned_data_;
    void *data_;
};

}  // engine
}  // ptk

#endif // PTK_ENGINE_GRAPH_TENSOR_HPP_