/*!
* \brief 
*/
// 功能点：
// 1. 复用临时内存，每次使用调用 FastMalloc 和 FastFree 。内部做内存使用情况的标记。不会有频繁创建释放产生内存碎片的风险。
//                 void *FastMalloc(uint32_t size, bool is_use_subblock = false) 
// 2. 手动关联复用：用于关联多处内存的使用，使其共用一块内存，以免除拷贝耗时。
//                 从第一处的head到最后一处的tail之间，不允许其他地方申请。
//                 在设定head时，分配并给block设定id，body为中间，tail 释放并将id置0释放，释放后可供其他地方使用。
//                 void *FastMalloc(uint32_t size, uint32_t group_id, MemBindAttr attr)
// 4. block内复用，内存块MemBlock分两个层级，
//    第一级MemBlock直接从系统分配内存（或使用外部内存），块内内存连续，块间内存不连续；
//    第二级SubBlock从一级块内切分子块，按一级内存块内按偏移划分子块内存，子块间内存连续。
//    第二级的内存子块可随时按需刷新 RefreshSubBlock （如子块均为MEM_TAG_FREE时），可重新回收子块重新按需划分。
// 5. 针对高速内存，以dtcm为例，使用 void *FastMalloc(uint32_t size, void *binding_ptr) 
//    可直接将dtcm全局数组作为一个一级内存块，然后按需在该内存块内划分子块进行复用。
// 6. 可随时shrink, 内存收缩，释放未使用的内存。

// TODO: 一级二级搜索链路过长，影响效率，需要优化。可对vector进行划分或采用list

#ifndef POCKET_AI_MEMORY_ALLOCATOR_HPP_
#define POCKET_AI_MEMORY_ALLOCATOR_HPP_

#include <stdlib.h>
#include <list>

#include <vector>
#include <mutex>

#include "align_alloc.hpp"
#include "pocket-ai/util/logger.hpp"

namespace pai {
namespace memory {

// 用于关联多处调用点进行复用，使其共用一块内存，以免除拷贝耗时
enum MemBindAttr {
    MEM_BIND_HEAD = 0,
    MEM_BIND_BODY = 1,
    MEM_BIND_TAIL = 2
};

enum MemTag {
    MEM_TAG_FREE = 0, // 未被使用
    MEM_TAG_PART,     // 部分被使用，即部分子块被使用
    MEM_TAG_FULL,     // 整个内存块已被使用
    MEM_TAG_EMPTY     // 内存块未持有内存，无法使用
};

struct SubBlock {
    MemTag tag;
    void *ptr;
    uint32_t size;
};

struct Memblock {
    MemTag tag;
    void *ptr;
    uint32_t size;
    bool is_hold;
    // default 0 without binding
    uint32_t group_id; 

    // 块中块，最多去到两层
    std::vector<SubBlock> sub_storage;
    uint32_t rest_size;
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
        PAI_LOGS("\n####################################\n");
        uint32_t total_size = 0;
        for (uint32_t i=0; i<storage_.size(); i++) {
            Memblock *block = &storage_[i];
            if (block->tag != MEM_TAG_EMPTY) {
                total_size += block->size;
            }
        }
        PAI_LOGS("Allocator(%p), storage size: %lld, total malloc size: %u. ", this, storage_.size(), total_size);
        for (uint32_t i=0; i<storage_.size(); i++) {
            if (i % 5 == 0)
                PAI_LOGS("\n");
            Memblock *block = &storage_[i];
            PAI_LOGS("{%d: %d, %d, %d}", i, block->size, block->group_id, block->tag);
            if (block->sub_storage.size() != 0)
                PAI_LOGS("-");
            for (uint32_t si=0; si<block->sub_storage.size(); si++) {
                SubBlock *sub = &block->sub_storage[si];
                PAI_LOGS("(%p, %d,%d)", sub->ptr, sub->size, sub->tag);
            }
            PAI_LOGS(", ");
        }
        PAI_LOGS("\n####################################\n");        
    }

    void Clear() {
        for (uint32_t i=0; i<storage_.size(); i++) {
            Memblock *block = &storage_[i];
            if (block->tag != MEM_TAG_EMPTY) {
                if (block->tag == MEM_TAG_FULL)
                   PAI_LOGW("%d,%p still in use.\n", i, block->ptr);

                if (block->is_hold == true)
                    AlignFree(block->ptr);
                block->tag = MEM_TAG_EMPTY;
            }
        }
        storage_.clear();
    }

    // No binding, no sub block
    // Set group_id=0, attr=MEM_BIND_BODY, 
    void *FastMalloc(uint32_t size, bool is_use_subblock = false) {
        if (size == 0)
            return nullptr;

        // 先寻找合适大小能直接用的block
        void *ptr = GetFreeMemBlockL1(size, 0, 0);
        if (ptr != nullptr) {
            return ptr;
        }
        // 其次搜索第二层，找到合适的进行分配
        if (is_use_subblock) {
            void *ptr = GetFreeMemBlockL2(size);
            if (ptr != nullptr) {
                return ptr;
            }
        }
        // 最后都没有，则开辟L1的Memblock
        return CreateMemBlock(size, 0);
    }

    // malloc with binding ptr， 主要用于高速内存，如dtcm
    // 返回 id 号
    uint32_t InplacementCreate(uint32_t size, void *binding_ptr) {
        CreateMemBlock(size, 0, binding_ptr);
        return storage_.size() - 1;
    }
    
    // 指定Memblock的id号，分配子内存块, 返回 nullptr 时分配失败
    void *FastMallocSubBlock(uint32_t size, uint32_t specify_id) {
        if (size == 0)
            return nullptr;

        Memblock *block = &storage_[specify_id];
        RefreshSubBlock(&storage_[specify_id]);

        return GetFreeSubBlock(block, size);
    }

    // use binding, no sub block
    void *FastMalloc(uint32_t size, uint32_t group_id, MemBindAttr attr) {
        if (group_id == 0)
            PAI_LOGW("In the case of group_id == 0, please Call void *FastMalloc(uint32_t size) instead.");
        // 如果指定了id，则认为走绑定的模式。绑定模式下，不支持block内分段

        // head，则直接去查找合适大小且未被分组的内存块，没有就创建
        if (attr == MEM_BIND_HEAD) {
            void *ptr = GetFreeMemBlockL1(size, 0, group_id);
            if (ptr != nullptr) {
                return ptr;
            }
            return CreateMemBlock(size, group_id);
        }
        // body，意味着前面肯定用过head，则直接寻找group_id一致的内存块，否则报错
        else if (attr == MEM_BIND_BODY) {
            void *ptr = GetFreeMemBlockL1(size, group_id, group_id);
            if (ptr != nullptr) {
                return ptr;
            }
            PAI_LOGE("Can not find group: %d with size: %d \n", group_id, size);
        }
        // tail, 结束点，同样意味着前面用过head，则直接寻找group_id一致的内存块，否则报错；
        // 在找到需要使用时，将其id号重新置为0，表示关联结束，free后可提供给其他地方使用
        else if (attr == MEM_BIND_TAIL) {
            void *ptr = GetFreeMemBlockL1(size, group_id, 0);
            if (ptr != nullptr) {
                return ptr;
            }
            PAI_LOGE("Can not find group: %d with size: %d \n", group_id, size);
        }
    }

    void FastFree(void* ptr, bool is_use_subblock = false) {
        if (ptr == nullptr)
            return;
        // 首先按整块复用的方式去搜索对应块
        for (uint32_t i=0; i<storage_.size(); i++) {
            Memblock *block = &storage_[i];
            if (block->tag == MEM_TAG_FULL && 
                block->ptr == ptr) {

                block->tag = MEM_TAG_FREE;
                return;
            }
        }
        // 按子块复用的模式去搜索对应子块
        if (is_use_subblock) {
            for (uint32_t i=0; i<storage_.size(); i++) {
                Memblock *block = &storage_[i];
                if (block->tag == MEM_TAG_PART) {
                    for (uint32_t si=0; si<block->sub_storage.size(); si++) {
                        SubBlock *sub = &block->sub_storage[si];
                        if (sub->tag == MEM_TAG_FULL && sub->ptr == ptr) {
                            sub->tag = MEM_TAG_FREE;
                            return;
                        }
                    }
                }
            }
        }
        PAI_LOGE("Allocator get wild %p", ptr);
        AlignFree(ptr);
    }

    // Release the memory blocks marked as MEM_TAG_FREE
    void Shrink() {
        // std::vector<Memblock> temp_storage;
        for (uint32_t i = 0; i < storage_.size(); i++) {
            Memblock *block = &storage_[i];
            RefreshSubBlock(block);

            // 未持有的内存(即外部分配后指定的)，不能执行释放操作
            if (block->tag == MEM_TAG_FREE && block->is_hold == true) {
                block->tag = MEM_TAG_EMPTY;
                AlignFree(block->ptr);
            }
            // else {
            //     temp_storage.push_back(storage_[i]);
            // }
        }
        // storage_.swap(temp_storage);
    }

private:
    void *CreateMemBlock(uint32_t size, uint32_t group_id, void *ptr = nullptr) {
        Memblock new_block;
        new_block.size = size;
        if (ptr == nullptr) {
            new_block.ptr = AlignMalloc(size);
            new_block.is_hold = true;
            new_block.tag = MEM_TAG_FULL;     
        }
        else { // 外部持有内存时，首次分配默认为未识别
            new_block.ptr = ptr;
            new_block.is_hold = false;
            new_block.tag = MEM_TAG_FREE;      
        }
        new_block.group_id = group_id;
        new_block.sub_storage.clear();
        new_block.rest_size = size;
        storage_.push_back(new_block);
        // printf("Size: %d.\n", storage_.size());
        return new_block.ptr;
    }

    void *CreateSubBlock(Memblock *block, uint32_t size) {
        // new subblock
        SubBlock new_block;
        new_block.size = size;
        new_block.ptr = block->ptr + block->size - block->rest_size;
        new_block.tag = MEM_TAG_FULL;
        block->sub_storage.push_back(new_block);
        block->rest_size -= size;
        return new_block.ptr;
    }

    // 刷新子块，如某内存块中的所有子块都为free，则回收子块，恢复主块的正常使用
    void RefreshSubBlock(Memblock *block) {
        if (block->sub_storage.size() == 0)
            return;

        bool is_all_free = true;
        for (uint32_t si=0; si<block->sub_storage.size(); si++) {
            SubBlock *sub = &block->sub_storage[si];
            if (sub->tag != MEM_TAG_FREE) {
                is_all_free = false;
                break;
            }
        }
        // 所有子块都是free，则直接全部回收
        if (is_all_free) {
            block->sub_storage.clear();
            block->rest_size = block->size;
            block->tag = MEM_TAG_FREE;
        }
    }
    
    // 已有部分子模块的Block(MEM_TAG_PART), 不再参与整块使用
    void *GetFreeMemBlockL1(uint32_t size, uint32_t target_group_id, uint32_t set_group_id) {
        for (uint32_t i = 0; i < storage_.size(); i++) {
            Memblock *block = &storage_[i];
            RefreshSubBlock(block);

            if (block->tag == MEM_TAG_FREE && block->group_id == target_group_id && 
                block->size >= size && (block->size * compare_ratio_) <= size) {

                block->group_id = set_group_id;
                block->tag = MEM_TAG_FULL;
                return block->ptr;
            }
        }
        return nullptr;
    }
    
    // 寻找L2合适的内存子块，group_id必须为0时才使用子块
    void *GetFreeMemBlockL2(uint32_t size) {
        for (uint32_t i = 0; i < storage_.size(); i++) {
            Memblock *block = &storage_[i];
            if (block->group_id != 0)
                continue;

            void *ptr = GetFreeSubBlock(block, size);
            if (ptr != nullptr)
                return ptr;
        }
        return nullptr;
    }

    void *GetFreeSubBlock(Memblock *block, uint32_t size) {
        if (block->tag == MEM_TAG_PART || block->tag == MEM_TAG_FREE) {
            // 已有部分子模块
            for (uint32_t si = 0; si < block->sub_storage.size(); si++) {
                SubBlock *sub = &block->sub_storage[si];
                if (sub->tag == MEM_TAG_FREE &&
                    sub->size >= size &&
                    (sub->size * compare_ratio_) <= size) {
                    sub->tag = MEM_TAG_FULL;
                    return sub->ptr;
                }
            }

            // 找不到合适的块，如内存充足则新开一个
            block->tag = MEM_TAG_PART;
            if (block->rest_size >= size) {
                return CreateSubBlock(block, size);
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
} // pai.
#endif // POCKET_AI_MEMORY_ALLOCATOR_HPP_
