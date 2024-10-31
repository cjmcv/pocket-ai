#ifndef POCKET_AI_ENGINE_GRAPH_MAIN_HPP_
#define POCKET_AI_ENGINE_GRAPH_MAIN_HPP_

#include <string>
#include <vector>
#include <map>
#include <thread>

#include "scheduler.hpp"
#include "pocket-ai/util/logger.hpp"

namespace pai {
namespace engine {

class Graph {
public:
    Graph(const std::string &name, int num_thread) {
        name_ = name;
        num_thread_ = num_thread;

        nodes_.clear();

        input_node_ = nullptr;
        output_node_ = nullptr;
    }

    ~Graph() {
        for (std::map<std::string, Node *>::iterator iter = nodes_.begin();
            iter != nodes_.end(); iter++) {
            delete iter->second;
        }
        nodes_.clear();

        for (uint32_t i=0; i<bq_pairs_.size(); i++) {
            BlockingQueuePair *bqp = bq_pairs_[i];
            bqp->full.exit();
            bqp->free.exit();

            while (!bqp->full.empty()) {
                Tensor *t;
                bqp->full.TryPop(&t);
                delete t;
            }
            while (!bqp->free.empty()) {
                Tensor *t;
                bqp->free.TryPop(&t);
                delete t;
            }
            delete bqp;
        }
    }

    void CreateNode(const std::string &name, Task &&task, 
                    std::vector<std::vector<int>> &&input_dims, 
                    std::vector<std::vector<int>> &&output_dims,
                    int group_id = 0) {
        Node *n = new Node(name, std::forward<Task>(task), input_dims, output_dims);

        scheduler_.MarkGroupId(n, group_id);
        nodes_.insert(std::make_pair(name, n));
        // printf("CreateNode: %s.\n", n->name().c_str());
    }

    void BuildTopology(std::vector<std::vector<std::string>> &&relation) {
        // for (uint32_t i=0; i<relation.size(); i++) {
        //     for (uint32_t j=1; j<relation[i].size(); j++) {
        //         printf("%s->%s.\n", relation[i][j-1].c_str(), relation[i][j].c_str());    
        //     }
        // }

        for (uint32_t i=0; i<relation.size(); i++) {
            for (uint32_t j=1; j<relation[i].size(); j++) {
                std::map<std::string, Node*>::iterator nodes_iter;
                std::map<Node*, std::vector<Node*>>::iterator io_iter;

                // Find the target node.
                Node *target = nullptr;
                nodes_iter = nodes_.find(relation[i][j-1].c_str());
                if(nodes_iter != nodes_.end())
                    target = nodes_iter->second;
                else
                    PAI_LOGE("Can not find node named %s .\n", relation[i][j-1].c_str());
                
                // Find the output node of the target.
                Node *n_out = nullptr;
                nodes_iter = nodes_.find(relation[i][j].c_str());
                if(nodes_iter != nodes_.end())
                    n_out = nodes_iter->second;
                else
                    PAI_LOGE("Can not find node named %s .\n", relation[i][j].c_str());

                // Set output.
                io_iter = output_map_.find(target);
                if(io_iter != output_map_.end()) {
                    io_iter->second.push_back(n_out);
                }
                else {
                    std::vector<Node*> vec = {n_out};
                    output_map_.insert(std::make_pair(target, vec));
                }
                
                // Set Input.
                io_iter = input_map_.find(n_out);
                if(io_iter != input_map_.end()) {
                    io_iter->second.push_back(target);
                }
                else {
                    std::vector<Node*> vec = {target};
                    input_map_.insert(std::make_pair(n_out, vec));
                }
            }
        }

        // Specify inputs and outputs for each node according to the constructed topology.
        std::map<std::string, Node*>::iterator iter;
        for(iter = nodes_.begin(); iter != nodes_.end(); iter++) {
            Node *node = iter->second;
            std::map<Node *, std::vector<Node *>>::iterator io_iter;
            // printf("Node: %s.\n", node->name().c_str());

            io_iter = input_map_.find(node);
            if (io_iter != input_map_.end())
                node->SetInputNodes(&(io_iter->second));
            else
                node->SetInputNodes(nullptr);

            io_iter = output_map_.find(node);
            if (io_iter != output_map_.end())
                node->SetOutputNodes(&(io_iter->second));
            else
                node->SetOutputNodes(nullptr);
        }
    }

    void BuildGraph(std::vector<std::vector<std::string>> &&relation) {
        // Build topology
        BuildTopology(std::forward<std::vector<std::vector<std::string>>>(relation));

        // Find the graph IO nodes according to the number of inputs and outputs.
        // Input node of the graph: no input.
        // Output node of the graph: no output.
        // Only one input node and one output node are allowed
        // Nodes with neither input nor output are not included in the graph.
        std::map<std::string, Node*>::iterator iter;
        for(iter = nodes_.begin(); iter != nodes_.end(); iter++) {
            if (iter->second->input_nodes() == nullptr && iter->second->output_nodes() == nullptr) {
                // independent, not included in the graph
            }
            else if (iter->second->input_nodes() == nullptr) {
                if (input_node_ != nullptr) 
                    PAI_LOGE("BuildGraph -> Only one input node is allowed.\n");
                input_node_ = iter->second;
            }
            else if (iter->second->output_nodes() == nullptr) {
                if (output_node_ != nullptr) 
                    PAI_LOGE("BuildGraph -> Only one output node is allowed.\n");
                output_node_ = iter->second;
            }
        }

        // printf("input_node_: %s.\n", input_node_->name().c_str());
        // printf("output_node_: %s.\n", output_node_->name().c_str());

        // Group nodes.
        scheduler_.UpdateGroups();
        scheduler_.GetGraphNodes(graph_nodes_);

        // Check shape && allocate memory && reorder
        SetupInteractTensors();
        SetupIoTensors();
        ReorderTensors();
        PAI_LOGI("Finish Graph::BuildGraph.\n");
    }
    void ShowInfo() {
        PAI_LOGS("\n>>>>>>>>> Graph ShowInfo >>>>>>>>>\n");
        PAI_LOGS("Graph: %s.\n", name_.c_str());
        PAI_LOGS("Input node: %s.\n", input_node_->name().c_str());
        PAI_LOGS("Output node: %s.\n", output_node_->name().c_str());

        std::map<std::string, Node*>::iterator iter;
        for(iter = nodes_.begin(); iter != nodes_.end(); iter++) {
            Node *n = iter->second;
            // std::vector<Node *> *ins = n->input_nodes();
            // std::vector<Node *> *outs = n->output_nodes();
            PAI_LOGS("node: %s (%p) -> in: [", n->name().c_str(), n);
            for (uint32_t i = 0; i<n->input_dims().size(); i++) {
                PAI_LOGS("%d(", n->input_dims()[i][0]);
                for (uint32_t j=1; j<n->input_dims()[i].size(); j++) {
                    PAI_LOGS("%d", n->input_dims()[i][j]);
                    if (j != n->input_dims()[i].size() - 1) PAI_LOGS(",");
                }
                PAI_LOGS(")");
                if (i != n->input_dims().size() - 1) PAI_LOGS(",");
            }
            PAI_LOGS("], out: [");
            for (uint32_t i = 0; i<n->output_dims().size(); i++) {
                PAI_LOGS("%d(", n->output_dims()[i][0]);
                for (uint32_t j=1; j<n->output_dims()[i].size(); j++) {
                    PAI_LOGS("%d", n->output_dims()[i][j]);
                    if (j != n->output_dims()[i].size() - 1) PAI_LOGS(",");
                }
                PAI_LOGS(")");
                if (i != n->output_dims().size() - 1) PAI_LOGS(",");
            }
            PAI_LOGS("]\n");
        }
        PAI_LOGS("\n");
        //
        PAI_LOGS("Node Relationship: \n");
        for(iter = nodes_.begin(); iter != nodes_.end(); iter++) {
            Node *n = iter->second;
            std::vector<Node *> *ins = n->input_nodes();
            std::vector<Node *> *outs = n->output_nodes();
            PAI_LOGS("%s -> in: [", n->name().c_str());
            if (ins != nullptr) {
                for (uint32_t i=0; i<ins->size(); i++) {
                    PAI_LOGS("%s", (*ins)[i]->name().c_str());
                    if (i != ins->size() - 1) PAI_LOGS(", ");
                }
            }
            PAI_LOGS("], out: [");
            if (outs != nullptr) {
                for (uint32_t i=0; i<outs->size(); i++) {
                    PAI_LOGS("%s", (*outs)[i]->name().c_str());
                    if (i != outs->size() - 1) PAI_LOGS(", ");
                }
            }
            PAI_LOGS("].\n");
        }
        PAI_LOGS("\n");
        //
        PAI_LOGS("Tensors: \n");
        for(iter = nodes_.begin(); iter != nodes_.end(); iter++) {
            Node *n = iter->second;
            std::vector<BlockingQueuePair *> ins = n->input_queues();
            std::vector<BlockingQueuePair *> outs = n->output_queues();
            if (ins.size() == 0 && outs.size() == 0)
                continue;
            PAI_LOGS("%s -> in: [", n->name().c_str());
            for (uint32_t i = 0; i < ins.size(); i++) {
                PAI_LOGS("%p(%s)", ins[i], ins[i]->front_name.c_str());
                if (i != ins.size() - 1)
                    PAI_LOGS(", ");
            }
            PAI_LOGS("], out: [");
            for (uint32_t i = 0; i < outs.size(); i++) {
                PAI_LOGS("%p(%s)", outs[i], outs[i]->rear_name.c_str());
                if (i != outs.size() - 1)
                    PAI_LOGS(", ");
            }
            PAI_LOGS("].\n");
        }

        PAI_LOGS("\n");
        scheduler_.ShowGroups();
        PAI_LOGS(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");
    }

    void Start(void *usr) {
        // Start all task threads.
        scheduler_.TasksSpawn(usr);
        // StartProfile();      
    }
    void Stop() {        
        // Stop all task threads.
        scheduler_.TasksStop();

        for (uint32_t i=0; i<bq_pairs_.size(); i++) {
            BlockingQueuePair *bqp = bq_pairs_[i];
            bqp->full.exit();
            bqp->free.exit();
        }
        scheduler_.TasksJoin();
        // EndProfile();
        PAI_LOGI("Graph::Stop().\n");
    }

    // Asynchronous function.
    void Feed(Tensor *in) {
        // PAI_LOGI("Graph Running: %s, %d, %d.\n", name_.c_str(), p->mode, p->num_thread);
        input_node_->input_queues()[0]->Enqueue(in);
        // scheduler_.BfsExecute(input_node_, &in);
    }
    
    // Get the result after calling the Feed.
    void GetResult(Tensor *out) {
        output_node_->output_queues()[0]->Dequeue(out);
    }

private:
    BlockingQueuePair *CreateBlockingQueue(std::vector<int> &shape, util::Type type) {
        uint32_t queue_size = 10;
        BlockingQueuePair *bqp = new BlockingQueuePair;
        for (uint32_t i = 0; i < queue_size; i++) {
            Tensor *t = new Tensor(shape, type);
            bqp->free.Push(t);
        }
        bq_pairs_.push_back(bqp);
        return bqp;
    }

    // Check whether the shapes match and create tensors for node interaction.
    void SetupInteractTensors() {
        for (uint32_t i = 0; i < graph_nodes_.size(); i++) {
            Node *n = graph_nodes_[i];
            std::vector<std::vector<int>> input_dims = n->input_dims();
            std::vector<Node *> *input_nodes = n->input_nodes();

            // if input_nodes == nullptr, it indicates that it is an input node of the graph
            // and does not need to check the output shape.
            if (input_nodes == nullptr) {
                continue;
            }

            // The number of input shapes and input nodes should be the same.
            if (input_nodes->size() != input_dims.size())
                PAI_LOGE("SetupInteractTensors -> output_nodes->size() != output_dims.size(): %lld vs %lld.\n",
                        input_nodes->size(), input_dims.size());

            // Check each of input nodes.
            for (uint32_t si = 0; si < input_nodes->size(); si++) {
                Node *in_node = (*input_nodes)[si];
                std::vector<std::vector<int>> need_match_dims = in_node->output_dims();

                if (need_match_dims.size() == 1) {
                    if (need_match_dims[0] != input_dims[si]) {
                        PAI_LOGE("SetupInteractTensors -> Shape check failed (node %s to %s).\n",
                                n->name().c_str(), in_node->name().c_str());
                    }
                }
                else if (need_match_dims.size() > 1) {
                    // TODO: 目前其中一个能匹配就算匹配上了，但实际上看可能会在a+b=>c时，a和b的输出维度一致和c的其中一个输入吻合，c的另一个不吻合，则应该是不匹配的。但目前的策略是会判断为匹配的。
                    bool is_pass = false;
                    for (uint32_t ni = 0; ni < need_match_dims.size(); ni++) {
                        if (need_match_dims[ni] == input_dims[si])
                            is_pass = true;
                    }
                    if (is_pass == false) {
                        PAI_LOGE("SetupInteractTensors -> Shape check failed (node %s to %s).\n",
                                n->name().c_str(), in_node->name().c_str());
                    }
                }
                else {
                    PAI_LOGE("SetupInteractTensors -> Shape check failed: need_match_dims.size() <= 0 \
                            (node %s to %s).\n", n->name().c_str(), in_node->name().c_str());
                }
                // Check passed and allocate BlockingQueuePair.
                std::vector<int> tensor_shapes;
                tensor_shapes.assign(input_dims[si].begin() + 1, input_dims[si].end());
                BlockingQueuePair *bqp = CreateBlockingQueue(tensor_shapes, (util::Type)input_dims[si][0]);
                bqp->front_name = in_node->name();
                bqp->rear_name = n->name();
                in_node->AppendOutputs(bqp);
                n->AppendInputs(bqp);
            }
        }
    }

    void SetupIoTensors() {
        if (input_node_ == nullptr || output_node_ == nullptr)
            PAI_LOGE("SetupIoTensors -> Both input and output nodes must exist.\n");
        if (input_node_->input_dims().size() != 1 || output_node_->output_dims().size() != 1)
            PAI_LOGE("SetupIoTensors -> Input node has one input, output node has one output.\n");

        std::vector<int> tensor_shapes;
        BlockingQueuePair *bqp;

        // Skip data type saved in shape[0].
        tensor_shapes.assign(input_node_->input_dims()[0].begin() + 1, input_node_->input_dims()[0].end());
        bqp = CreateBlockingQueue(tensor_shapes, (util::Type)input_node_->input_dims()[0][0]);
        bqp->front_name = "input";
        bqp->rear_name = input_node_->name();
        input_node_->AppendInputs(bqp);

        tensor_shapes.assign(output_node_->output_dims()[0].begin() + 1, output_node_->output_dims()[0].end());
        bqp = CreateBlockingQueue(tensor_shapes, (util::Type)output_node_->output_dims()[0][0]);
        bqp->front_name = output_node_->name();
        bqp->rear_name = "output";
        output_node_->AppendOutputs(bqp);
    }

    void ReorderTensors() {
        for (uint32_t i = 0; i < graph_nodes_.size(); i++) {
            graph_nodes_[i]->ReorderInputQueues();
            graph_nodes_[i]->ReorderOutputQueues();
        }
    }
    //
    // void StartProfile() {
    //     if (is_profiler_start_ == true)
    //         return;

    //     is_profiler_start_ = true;
    //     std::thread *t = new std::thread([this]() -> void {
    //         while (is_profiler_start_ == true) {
    //             allocator_->PrintInfo();
    //             std::this_thread::sleep_for(std::chrono::seconds(3));
    //         }
    //     });
    // }

    void EndProfile() {
        is_profiler_start_ = false;
    }

private:
    std::string name_;
    int num_thread_;

    std::map<std::string, Node*> nodes_; // 包含普通节点和组合节点
    Node *input_node_;
    Node *output_node_;

    // topology: <target, the outputs/inputs of the target>
    std::map<Node*, std::vector<Node*>> output_map_;
    std::map<Node*, std::vector<Node*>> input_map_;

    std::vector<Node *> graph_nodes_; // 参与组建图的节点

    void *usr_;
    Scheduler scheduler_;

    //
    std::vector<BlockingQueuePair *> bq_pairs_; // 用于节点间数据交互

    bool is_profiler_start_;
};


} // namespace engine
} // namespace pai

#endif // POCKET_AI_ENGINE_GRAPH_MAIN_HPP_