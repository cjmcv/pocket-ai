/*!
* \brief Compute node.
*/

#ifndef PTK_ENGINE_GRAPH_NODE_HPP_
#define PTK_ENGINE_GRAPH_NODE_HPP_

#include <string>
#include <vector>
#include <functional>

#include "tensor.hpp"
#include "../../memory/blocking_queue.hpp"

namespace ptk {
namespace engine {

// TODO: 内存管理，内存复用在其内部接allocator
// class Tensor;
// TODO: 入参，算法参数配置，按类型分组，
// struct Params;
// TODO: 可选择已注册的kernel函数，也可以外设自己的函数

struct BlockingQueuePair {
    std::string front_name;
    std::string rear_name;
    memory::BlockingQueue<Tensor *> free;
    memory::BlockingQueue<Tensor *> full;

    void Enqueue(Tensor *input) {
        Tensor *inside_free;
        free.wait_and_pop(&inside_free);
        inside_free->CopyFrom(input);
        full.push(inside_free);
    }

    void Dequeue(Tensor *output) {
        Tensor *inside_full;
        full.wait_and_pop(&inside_full);
        inside_full->CopyTo(output);
        free.push(inside_full);
    }

    void LoanOutFromFull(Tensor **out_full) {
        full.try_pop(out_full);
    }

    void RecycleToFree(Tensor *in_free) {
        free.push(in_free);
    }
};

using Task = std::function<void(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs)>;

class Node {
public:
    Node(const std::string &name, Task &&task,
         std::vector<std::vector<int>> &input_dims, 
         std::vector<std::vector<int>> &output_dims)
         : input_nodes_(nullptr), output_nodes_(nullptr) {
        name_ = name;
        task_ = task;

        input_dims_ = input_dims;
        output_dims_ = output_dims;
    }
    ~Node() {};
    inline void Run(void *usr, std::vector<Tensor *> &input, std::vector<Tensor *> &output) {
        task_(usr, input, output);
    }

    inline std::string &name() { return name_; }
    inline void SetInputNodes(std::vector<Node *> *input_nodes) { input_nodes_ = input_nodes; };
    inline void SetOutputNodes(std::vector<Node *> *output_nodes) { output_nodes_ = output_nodes; };

    inline std::vector<Node *> *input_nodes() { return input_nodes_; }
    inline std::vector<Node *> *output_nodes() { return output_nodes_; }

    inline std::vector<std::vector<int>> &input_dims() { return input_dims_; }
    inline std::vector<std::vector<int>> &output_dims() { return output_dims_; }

    inline void AppendInputs(BlockingQueuePair *bq) { input_queues_.push_back(bq); }
    inline void AppendOutputs(BlockingQueuePair *bq) { output_queues_.push_back(bq); }
    inline std::vector<BlockingQueuePair *> &input_queues() { return input_queues_; }
    inline std::vector<BlockingQueuePair *> &output_queues() { return output_queues_; }

    void ReorderInputQueues() {
        // Make the order of the input queues consistent with the order of the input nodes
        if (input_nodes_ != nullptr) {
            for (int ni = 0; ni < input_nodes_->size(); ni++) {
                std::string target_name = (*input_nodes_)[ni]->name();
                for (int qi = 0; qi < input_queues_.size(); qi++) {
                    if (target_name == input_queues_[qi]->front_name) {
                        if (ni == qi) 
                            continue;
                        else 
                            SwapQueueOrder(input_queues_, ni, qi);
                    }
                }
            }
        }
    }

    void ReorderOutputQueues() {
        // Make the order of the output queues consistent with the order of the output nodes
        if (output_nodes_ != nullptr) {
            for (int ni = 0; ni < output_nodes_->size(); ni++) {
                std::string target_name = (*output_nodes_)[ni]->name();
                for (int qi = 0; qi < output_queues_.size(); qi++) {
                    if (target_name == output_queues_[qi]->rear_name) {
                        if (ni == qi)
                            continue;
                        else
                            SwapQueueOrder(output_queues_, ni, qi);
                    }
                }
            }
        }
    }

    bool CheckIoIsReady() {
        bool is_ready = true;
        for (int i=0; i<input_queues_.size(); i++) {
            if (input_queues_[i]->full.empty())
                is_ready = false;
        }
        for (int i=0; i<output_queues_.size(); i++) {
            if (output_queues_[i]->free.empty())
                is_ready = false;
        }
        return is_ready;
    }

    bool BorrowIo(std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs){
        input_tensors_.clear();
        // printf("input_queues_.size: %d.\n", input_queues_.size());
        for (int i=0; i<input_queues_.size(); i++) {
            Tensor *inside_full;
            // printf("input_queues_[%d]->full.size : %d.\n", i, input_queues_[i]->full.size());
            bool is_ready = input_queues_[i]->full.wait_and_pop(&inside_full);
            if (!is_ready) return false;
            input_tensors_.push_back(inside_full);
        }
        output_tensors_.clear();
        // printf("output_queues_.size: %d.\n", output_queues_.size());
        for (int i=0; i<output_queues_.size(); i++) {
            Tensor *inside_free;
            // printf("output_queues_[%d]->free.size : %d.\n", i, output_queues_[i]->free.size());
            bool is_ready = output_queues_[i]->free.wait_and_pop(&inside_free);
            if (!is_ready) return false;
            output_tensors_.push_back(inside_free);
        }
        // Get ITensor, TODO 直接拷贝
        inputs.clear();
        for (int i=0; i<input_tensors_.size(); i++) {
            inputs.push_back(input_tensors_[i]);
        }
        outputs.clear();
        for (int i=0; i<output_tensors_.size(); i++) {
            outputs.push_back(output_tensors_[i]);
        }
        // Check id
        for (int i=1; i<inputs.size(); i++) {
            if (inputs[i]->id() != inputs[0]->id()) {
                PTK_LOGE("Node::BorrowIo -> The ID of Tensor in the same group is inconsistent.\n");
            }
        }
        // Pass id
        for (int i=0; i<outputs.size(); i++) {
            outputs[i]->SetId(inputs[0]->id());
        }
        return true;
    }

    void RecycleIo() {
        // TODO: 按需进行异步的跨设备内存拷贝。
        for (int i=0; i<input_queues_.size(); i++) {
            input_queues_[i]->free.push(input_tensors_[i]);
        }
        for (int i=0; i<output_queues_.size(); i++) {
            output_queues_[i]->full.push(output_tensors_[i]);
        }
    }

private:
    void SwapQueueOrder(std::vector<BlockingQueuePair *> &queues, int i, int j) {
        BlockingQueuePair *temp = queues[i];
        queues[i] = queues[j];
        queues[j] = temp;
    }

private:
    std::string name_;
    Task task_;
    
    std::vector<Node *> *input_nodes_;
    std::vector<Node *> *output_nodes_;

    // 0: data_type, 1, 2, 3...
    std::vector<std::vector<int>> input_dims_;
    std::vector<std::vector<int>> output_dims_;

    std::vector<BlockingQueuePair *> input_queues_; // It is also part of the output of the input node 
    std::vector<BlockingQueuePair *> output_queues_; // It is also part of the input of the output node

    std::vector<Tensor *> input_tensors_;
    std::vector<Tensor *> output_tensors_;
};

}  // engine
}  // ptk

#endif //PTK_ENGINE_GRAPH_NODE_HPP_