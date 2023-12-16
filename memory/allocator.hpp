/*!
* \brief 
*/

#ifndef PTK_MEMORY_ALLOCATOR_HPP_
#define PTK_MEMORY_ALLOCATOR_HPP_

#include <stdlib.h>
#include <list>

#include <vector>
#include <mutex>

#include "align_alloc.hpp"
#include "util/logger.hpp"

namespace ptk {
namespace memory {

// 功能点：
// 1. 复用临时内存，每次使用调用 FastMalloc 和 FastFree 。内部做内存使用情况的标记。不会有频繁创建释放产生内存碎片的风险。
// 2. 手动关联复用：用于关联多处内存的使用，使其共用一块内存，以免除拷贝耗时。
//                 从第一处到最后一处之间，不允许其他地方申请。
//                 调用时指定 group_id，head 分配并给block设定id，body 中间，tail 释放并清空id，释放后可供其他地方使用
// 3. 自动关联复用：按生命周期，先plan(外部实现)，为每个调用点指定group_id和head/body/tail，使用时照常使用按2方式使用。

// 弊端：无法极致压缩内存，比如有一块大内存，其他都是小内存，这块大内存一般就只有申请的那个地方使用，其他地方申请不了。
//                           需要一块大内存可跟多块小内存复用，才能达到压缩目的。
// 多块小内存复用一块大内存的逻辑：每个block内用step复用。
// TODO
// 4. block内使用固定全局数组step代替malloc，用于使用如dtcm等高速内存。
// 5. shrink, 内存收缩，释放未使用的内存。
// 6. 常驻内存分配，inplacementMalloc

enum MemBindAttr {
    MEM_BIND_HEAD = 0,
    MEM_BIND_BODY = 1,
    MEM_BIND_TAIL = 2
};

enum MemTag {
    MEM_TAG_FREE = 0,
    MEM_TAG_FULL = 1,
    MEM_TAG_EMPTY = 2
};

struct Memblock {
    MemTag tag;
    void *ptr;
    uint32_t size;

    uint32_t group_id; // default 0 without binding
    std::vector<uint32_t> offset; // 块内偏移量，初始为空。即分配时，直接取首地址。如分段，则offset[0]为第一段长度
};

class Allocator {
public:
    Allocator() {
        compare_ratio_ = 0.75f; // 设为1时，强制申请大小与原有大小一致，可用于排查内存越界。
    }

    ~Allocator() {
        Clear();
    }

    void SetCompareRatio(float ratio) {
        compare_ratio_ = ratio;
    }

    void Show() {
        PTK_LOGS("\n####################################\n");
        uint32_t total_size = 0;
        for (uint32_t i=0; i<storage_.size(); i++) {
            Memblock *block = &storage_[i];
            if (block->tag != MEM_TAG_EMPTY) {
                total_size += block->size;
            }
        }
        PTK_LOGS("Allocator(%p), storage size: %lld, total malloc size: %u. ", this, storage_.size(), total_size);
        for (uint32_t i=0; i<storage_.size(); i++) {
            if (i % 5 == 0)
                printf("\n");
            Memblock *block = &storage_[i];
            printf("(%d: %d, %d, %d), ", i, block->size, block->group_id, block->tag);
        }
        PTK_LOGS("\n####################################\n");        
    }

    void Clear() {
        for (uint32_t i=0; i<storage_.size(); i++) {
            Memblock *block = &storage_[i];
            if (block->tag != MEM_TAG_EMPTY) {
                if (block->tag == MEM_TAG_FULL)
                   PTK_LOGW("%d,%p still in use.\n", i, block->ptr);

                AlignFree(block->ptr);
                block->tag = MEM_TAG_EMPTY;
            }
        }
        storage_.clear();
    }

    void* FastMalloc(uint32_t size, uint32_t group_id = 0, MemBindAttr attr = MEM_BIND_BODY) {

        if (group_id == 0) {
            // 先寻找合适大小能直接用的block
            void *ptr = GetFreeMemBlock(size, 0, 0);
            if (ptr != nullptr) {
                return ptr;
            }
            return CreateMemBlock(size, 0);
        }
        else {
            // 如果指定了id，则认为走绑定的模式。绑定模式下，不支持block内分段

            // head，则直接去查找合适大小且未被分组的内存块，没有就创建
            if (attr == MEM_BIND_HEAD) {
                void *ptr = GetFreeMemBlock(size, 0, group_id);
                if (ptr != nullptr) {
                    return ptr;
                }
                return CreateMemBlock(size, group_id);
            }
            // body，意味着前面肯定用过head，则直接寻找group_id一致的内存块，否则报错
            if (attr == MEM_BIND_BODY) {
                void *ptr = GetFreeMemBlock(size, group_id, group_id);
                if (ptr != nullptr) {
                    return ptr;
                }
                PTK_LOGE("Can not find group: %d with size: %d \n", group_id, size);
            }
            // tail, 结束点，同样意味着前面用过head，则直接寻找group_id一致的内存块，否则报错；
            // 在找到需要使用时，将其id号重新置为0，表示关联结束，free后可提供给其他地方使用
            if (attr == MEM_BIND_TAIL) {
                void *ptr = GetFreeMemBlock(size, group_id, 0);
                if (ptr != nullptr) {
                    return ptr;
                }
                PTK_LOGE("Can not find group: %d with size: %d \n", group_id, size);
            }        
        }
        return nullptr;
    }

    void FastFree(void* ptr) {
        for (uint32_t i=0; i<storage_.size(); i++) {

            Memblock *block = &storage_[i];
            if (block->tag == MEM_TAG_FULL && 
                block->ptr == ptr) {

                block->tag = MEM_TAG_FREE;
                return;
            }
        }

        PTK_LOGE("Allocator get wild %p", ptr);
        AlignFree(ptr);
    }

private:
    void *CreateMemBlock(uint32_t size, uint32_t group_id) {
        Memblock new_block;
        new_block.size = size;
        new_block.ptr = AlignMalloc(size);
        new_block.tag = MEM_TAG_FULL;
        new_block.group_id = group_id;
        new_block.offset.clear();
        storage_.push_back(new_block);

        return new_block.ptr;
    }

    void *GetFreeMemBlock(uint32_t size, uint32_t target_group_id, uint32_t set_group_id) {
        for (uint32_t i = 0; i < storage_.size(); i++) {
            Memblock *block = &storage_[i];
            if (block->tag == MEM_TAG_FREE && 
                block->group_id == target_group_id &&
                block->size >= size) {

                if ((block->size * compare_ratio_) <= size) {
                    block->group_id = set_group_id;
                    block->tag = MEM_TAG_FULL;
                    return block->ptr;                    
                }
            }
        }
        return nullptr;
    }

private:
    float compare_ratio_; // 0.0 ~ 1.0f

    // std::mutex mutex_;
    std::vector<Memblock> storage_;
};


} // memory.
} // ptk.
#endif // PTK_MEMORY_ALLOCATOR_HPP_
