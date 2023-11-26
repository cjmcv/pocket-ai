/*!
* \brief . 
*/

#include "engine/graph/graph.hpp"
#include "prof/timer.hpp"

#include "gtest/gtest.h"

namespace {

using namespace ptk;
using namespace ptk::util;
using namespace ptk::engine;

class AlgoTasks {
public:
    AlgoTasks() {
        // for task B and C
        gemm_b_ = new Tensor({600, 300}, Type::FP32);
        float *data = (float *)gemm_b_->GetData();
        for (int j=0; j<600*300; j++) {
            data[j] = 1;
        }
        // for task D
        dot_a_ = new Tensor({300}, Type::FP32);
        dot_b_ = new Tensor({300}, Type::FP32);
    }

public:
    Tensor *gemm_b_;
    void *gemm_ptr_;

    Tensor *dot_a_;
    Tensor *dot_b_;
    void *dot_ptr_;
};

// 转置 分割 -> 乘法  ->  累加 转置？
//          -> 乘法 
// 600 * 600 -> 转置 -> 分割 200 * 600 ， 400 * 600
void TaskA(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;
    static int count = 0;

    float *data = (float *)inputs[0]->GetData(); // 600, 600
    int rows = inputs[0]->shape()[0];
    int cols = inputs[0]->shape()[1];

    for (int i=0; i<rows; i++) {
        for (int j=i+1; j<cols; j++) {
            float temp = data[i*cols + j];
            data[i*cols + j] = data[j*cols + i];
            data[j*cols + i] = temp;
        }
    }

    float *out_data0 = (float *)outputs[0]->GetData(); // 200, 600
    float *out_data1 = (float *)outputs[1]->GetData(); // 400, 600
    for (int i=0; i<rows*1/3; i++) {
        for (int j=0; j<cols; j++) {
            out_data0[i*cols + j] = data[i*cols + j];
        }
    }
    for (int i=rows*1/3; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            out_data1[(i-rows*1/3)*cols + j] = data[i*cols + j] + 1;
        }
    }
    printf("TaskA: %d (%d).\n", count++, std::this_thread::get_id());
}

// 200 * 600 -> gemm（600 * 300）-> 200 * 300 
void TaskB(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;

    static int count = 0;

    // float *A = (float *)inputs[0]->GetData();
    // float *B = (float *)ins->gemm_b_;
    // float *C = (float *)outputs[0];
    // uint32_t M = 200;
    // uint32_t N = 300;
    // uint32_t K = 600;
    // for (uint32_t i=0; i<M; i++) {
    //     for (uint32_t j=0; j<N; j++) {
    //         for (uint32_t k=0; k<K; k++) {
    //             C[i*N + j] += A[i*K + k] * B[k*N + j];
    //         }
    //     }
    // }
    // outputs[0]->Print();
    printf("TaskB: %d (%d).\n", count++, std::this_thread::get_id());
}

// 400 * 600 -> gemm（600 * 300）-> 400 * 300
void TaskC(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;

    static int count = 0;

    // float *A = (float *)inputs[0]->GetData();
    // float *B = (float *)ins->gemm_b_;
    // float *C = (float *)outputs[0];
    // uint32_t M = 400;
    // uint32_t N = 300;
    // uint32_t K = 600;
    // for (uint32_t i=0; i<M; i++) {
    //     for (uint32_t j=0; j<N; j++) {
    //         for (uint32_t k=0; k<K; k++) {
    //             C[i*N + j] += A[i*K + k] * B[k*N + j];
    //         }
    //     }
    // }

    // outputs[0]->Print();
    printf("TaskC: %d (%d).\n", count++, std::this_thread::get_id());
}

// 200 * 300， 400 * 300 合并 点积分
void TaskD(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;
    static int count = 0;

    float *data0 = (float *)inputs[0]->GetData();
    float *data1 = (float *)inputs[1]->GetData();
    ins->dot_a_->BindHostDataPtr(data0);
    ins->dot_b_->BindHostDataPtr(data1);

    // std::vector<Tensor *> new_inputs;
    // new_inputs.push_back(ins->dot_a_);
    // new_inputs.push_back(ins->dot_b_);
    // std::vector<ecas::Param> params;
    // ins->session()->OpRun(ins->dot_ptr_, params, new_inputs, outputs);

    // printf("inner id: %d.\n", );
    outputs[0]->Print();
    printf("TaskD: %d (%d).\n", count++, std::this_thread::get_id());
}

void GraphBaseTest() {

    Graph *graph = new Graph("s1", 1);

    graph->CreateNode("n1", TaskA, {{Type::FP32, 600, 600}}, {{Type::FP32, 200, 600}, {Type::FP32, 400, 600}}, 0);
    graph->CreateNode("n2", TaskB, {{Type::FP32, 200, 600}}, {{Type::FP32, 200, 300}}, 1);
    graph->CreateNode("n3", TaskC, {{Type::FP32, 400, 600}}, {{Type::FP32, 400, 300}}, 0);
    graph->CreateNode("n4", TaskD, {{Type::FP32, 200, 300}, {Type::FP32, 400, 300}}, {{Type::FP32, 1}}, 0);
    
    graph->BuildGraph({{"n1", "n2"}, {"n1", "n3"}, {"n2", "n4"}, {"n3", "n4"}});
    graph->ShowInfo();

    Tensor *in = new Tensor({600, 600}, Type::FP32);
    Tensor *out = new Tensor({1}, Type::FP32);    
    float *in_data = (float *)in->GetData();        
    for (int j=0; j<600*600; j++) {
        in_data[j] = 1;
    }


    AlgoTasks algo;
    graph->Start((void *)&algo);

    prof::Timer timer("GraphBaseTest", 2);
    timer.Start();
    for (int i=0; i<5; i++) {

        in->SetId(i);
        graph->Feed(in);
    }
    printf("hello.\n");
    for (int i=0; i<5; i++) {
        printf("hello: %d.\n", i);
        graph->GetResult(out);
        printf("out id: %d, %f.\n", out->id(), ((float *)out->GetData())[0]);
    }
    timer.Stop(0, 1);

    printf("Call stop.\n");
    graph->Stop();

    delete in;
    delete out;
}

TEST(EngineTest, Graph) {
    GraphBaseTest();
}

}  // end of namespace.