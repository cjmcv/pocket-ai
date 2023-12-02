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

void Gemm(const int M, const int N, const int K,
            const float *A, const int lda,
            const float *B, const int ldb,
            float *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
            }
        }
    }
}

class AlgoTasks {
public:
    AlgoTasks() {
        // for task B and C
        gemm_b_ = new Tensor({600, 300}, Type::FP32);
        float *data = (float *)gemm_b_->GetData();
        for (int j=0; j<600*300; j++) {
            data[j] = 1;
        }
    }

public:
    Tensor *gemm_b_;
};

// 转置 分割 -> 乘法  ->  累加 转置？
//          -> 乘法 
// 600 * 600 -> 转置 -> 分割 200 * 600 ， 400 * 600
void TaskA(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;
    static int count = 0;

    float *in_data = (float *)inputs[0]->GetData(); // 600, 600
    int rows = inputs[0]->shape()[Dim::HEIGHT];
    int cols = inputs[0]->shape()[Dim::WIDTH];

    for (int i=0; i<rows; i++) {
        for (int j=i+1; j<cols; j++) {
            float temp = in_data[i*cols + j];
            in_data[i*cols + j] = in_data[j*cols + i];
            in_data[j*cols + i] = temp;
        }
    }

    float *out_data0 = (float *)outputs[0]->GetData(); // 200, 600
    float *out_data1 = (float *)outputs[1]->GetData(); // 400, 600
    for (int i=0; i<rows*1/3; i++) {
        for (int j=0; j<cols; j++) {
            out_data0[i*cols + j] = in_data[i*cols + j];
        }
    }
    for (int i=rows*1/3; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            out_data1[(i-rows*1/3)*cols + j] = in_data[i*cols + j] + 1;
        }
    }

    // inputs[0]->Print();
    // outputs[0]->Print();
    // outputs[1]->Print();

    std::ostringstream id;
    id << std::this_thread::get_id();
    printf("TaskA: %d (%s).\n", count++, id.str().c_str());
}

// 200 * 600 -> gemm（600 * 300）-> 200 * 300 
void TaskB(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;

    // // Check inputs
    // // inputs[0]->Print();
    // if (inputs[0]->shape()[Dim::HEIGHT] != 200 || inputs[0]->shape()[Dim::WIDTH] != 600 || 
    //     inputs[0]->size() != 200 * 600 * sizeof(float))
    //     printf("Error: input shape mismatch.\n");
    // // outputs[0]->Print();
    // if (outputs[0]->shape()[Dim::HEIGHT] != 200 || outputs[0]->shape()[Dim::WIDTH] != 300 || 
    //     outputs[0]->size() != 200 * 300 * sizeof(float))
    //     printf("Error: output shape mismatch.\n");

    static int count = 0;
    float *A = (float *)inputs[0]->GetData();
    float *B = (float *)ins->gemm_b_->GetData();
    float *C = (float *)outputs[0]->GetData();
    uint32_t M = outputs[0]->shape()[Dim::HEIGHT]; // 200
    uint32_t N = outputs[0]->shape()[Dim::WIDTH];  // 300
    uint32_t K = inputs[0]->shape()[Dim::WIDTH];   // 600
    Gemm(M, N, K, A, K, B, N, C, N);
    
    // outputs[0]->Print();

    std::ostringstream id;
    id << std::this_thread::get_id();
    printf("TaskB: %d (%s).\n", count++, id.str().c_str());
}

// 400 * 600 -> gemm（600 * 300）-> 400 * 300
void TaskC(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;

    static int count = 0;

    float *A = (float *)inputs[0]->GetData();
    float *B = (float *)ins->gemm_b_->GetData();
    float *C = (float *)outputs[0]->GetData();
    uint32_t M = outputs[0]->shape()[Dim::HEIGHT];
    uint32_t N = outputs[0]->shape()[Dim::WIDTH];
    uint32_t K = inputs[0]->shape()[Dim::WIDTH];
    Gemm(M, N, K, A, K, B, N, C, N);

    // outputs[0]->Print();

    std::ostringstream id;
    id << std::this_thread::get_id();
    printf("TaskC: %d (%s).\n", count++, id.str().c_str());
}

// 200 * 300， 400 * 300 合并 点积分
void TaskD(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;
    static int count = 0;

    uint32_t len = inputs[0]->len();
    float *data0 = (float *)inputs[0]->GetData();
    float *data1 = (float *)inputs[1]->GetData();
    float *out = (float *)outputs[0]->GetData();

    *out = 0;
    for (uint32_t i=0; i<len; i++) {
        *out += data0[i] * data1[i];
    }

    // outputs[0]->Print();

    std::ostringstream id;
    id << std::this_thread::get_id();
    printf("TaskD: %d (%s).\n", count++, id.str().c_str());
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
    // in->Print();

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