/*!
* \brief Scheduler. 
*        提供节点的调度方案, 多线程管理，包含多线程间数据内存交互
*/

#ifndef PTK_ENGINE_GRAPH_SCHEDULER_HPP_
#define PTK_ENGINE_GRAPH_SCHEDULER_HPP_

#include <string>
#include <vector>
#include <thread>
#include <map>

#include "node.hpp"
#include "../../util/logger.hpp"

namespace ptk {
namespace engine {

class Scheduler {

public:
    Scheduler() {
        is_stop_ = false;
        groups_.clear();
    }
    ~Scheduler() {}

    void MarkGroupId(Node *node, uint32_t group_id) {
        static uint32_t init_size = 10;
        if (groups_temp_.size() != init_size) {
            groups_temp_.resize(init_size);
        }
        if (group_id > groups_temp_.size()) {
            PTK_LOGE("group_id should be smaller than %lld.\n", groups_temp_.size());
        }
        groups_temp_[group_id].push_back(node);
    }

    void UpdateGroups() {
        groups_.clear();
        for (uint32_t i=0; i<groups_temp_.size(); i++) {
            if (groups_temp_[i].size() == 0)
                continue;
            groups_.push_back(groups_temp_[i]);
        }
    }

    void GetGraphNodes(std::vector<Node *> &graph_nodes) {
        graph_nodes.clear();
        for (uint32_t i = 0; i < groups_.size(); i++) {
            for (uint32_t j = 0; j < groups_[i].size(); j++) {
                graph_nodes.push_back((Node *)groups_[i][j]);
            }
        }
    }

    ////////////////////////
    /// Serial Execution
    // Breadth First.
    void BuildBfsPass(Node *input_node) {}
    void BfsExecute(Node *input_node, Tensor *input_data) {
        std::queue<Node *> tasks;
        tasks.push(input_node);
        while (!tasks.empty()) {
            Node *t = tasks.front();
            // Tensor *input;
            // Tensor *output;
            // t->Run(input, output);
            std::vector<Node *> *outs = t->output_nodes();
            if (outs != nullptr) {
                for (unsigned int i=0; i<outs->size(); i++) {
                    tasks.push((*outs)[i]);
                }            
            }
            tasks.pop();
        }
    }

    ////////////////////////
    /// Parallel execution
    // Group nodes, and each group uses one thread.
    void BuildGroup(std::map<std::string, Node*> &nodes, 
                    std::vector<std::vector<std::string>> &&groups) {
        // groups_[group_id][node_ptr]
        groups_.resize(groups.size());
        for (uint32_t i = 0; i < groups.size(); i++) {
            groups_[i].resize(groups[i].size());
            for (uint32_t j = 0; j < groups[i].size(); j++) {
                std::map<std::string, Node *>::iterator iter = nodes.find(groups[i][j]);
                if (iter != nodes.end()) {
                    groups_[i][j] = iter->second;
                }
                else {
                    PTK_LOGI("BuildGroup -> Can not find node named %s .\n", groups[i][j].c_str());
                }
            }
        }
    }
    void ShowGroups() {
        PTK_LOGS("Groups: \n");
        for (uint32_t i = 0; i < groups_.size(); i++) {
            if (groups_[i].size() == 0)
                PTK_LOGE("ShowGroups -> groups_[%d].size() == 0.\n", i)

            PTK_LOGS("%d -> ", i);
            for (uint32_t j = 0; j < groups_[i].size(); j++) {
                PTK_LOGS("%s", ((Node *)groups_[i][j])->name().c_str()); // groups_[i][j]
                if (j != groups_[i].size() - 1) PTK_LOGS(", ");
            }
            PTK_LOGS("\n");
        }
    }
    inline int group_size() { return groups_.size(); }
    // inline std::vector<std::vector<Node *>> &group_nodes() { return groups_; };
    void TasksSpawn(void *usr) {
        if (groups_.size() == 0) {
            PTK_LOGE("TasksSpawn -> groups_.size() == 0, please call function BuildGraph first.\n");
        }

        std::vector<std::vector<Node *>> &groups = groups_;
        PTK_LOGI("group size: %lld.\n", groups_.size());
        for (unsigned i = 0; i < groups.size(); ++i) {
            threads_.emplace_back([this, i, groups, usr]() -> void {
                std::vector<Tensor *> inputs;
                std::vector<Tensor *> outputs;
                while (!is_stop_) {
                    // 同一组的按顺序堵塞执行，不会有帧差
                    for (uint32_t ni=0; ni<groups[i].size(); ni++) {
                        Node *n = groups[i][ni];
                        // printf("Start BorrowIo: %s.\n", n->name().c_str());
                        bool ret = n->BorrowIo(inputs, outputs);
                        if (ret == false) break;
                        // printf("%s -> (%d, %d).\n", n->name().c_str(), inputs.size(), outputs.size());
                        n->Run(usr, inputs, outputs);
                        n->RecycleIo();
                    }
                }
                PTK_LOGI("groups %d exit.\n", i);
            });
        }
        // PTK_LOGI("Scheduler::TasksSpawn End.\n");
    }

    void TasksStop()  {
        is_stop_ = true;
    }

    void TasksJoin() {
        for (auto& t : threads_) {
            t.join();
        }
    }


private:
    /// Serial Execution
    std::vector<Node *> bfs_nodes_;

    /// Parallel execution
    // groups_[group_id][node_ptr]
    std::vector<std::vector<Node *>> groups_;
    std::vector<std::vector<Node *>> groups_temp_;

    std::vector<std::thread> threads_;
    bool is_stop_;
};

}  // engine
}  // ptk

#endif // PTK_ENGINE_GRAPH_SCHEDULER_HPP_