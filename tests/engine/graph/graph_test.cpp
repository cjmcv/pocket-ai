/*!
* \brief . 
*/

#include "engine/graph/graph.hpp"
#include "prof/timer.hpp"

#include "util/basic_marco.hpp"
#include "gtest/gtest.h"

namespace {

using namespace ptk;
using namespace ptk::util;
using namespace ptk::engine;

// #define PRINTF PTK_PRINTF
#define PRINTF PTK_NO_PRINTF
// #define DEBUG_CALL PTK_DEBUG_CALL
#define DEBUG_CALL PTK_DEBUG_NO_CALL

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
    count++;

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
    PRINTF("TaskA: %d (%s).\n", count, id.str().c_str());
}

// 200 * 600 -> gemm（600 * 300）-> 200 * 300 
void TaskB(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;

    // // Check inputs
    // // inputs[0]->Print();
    // if (inputs[0]->shape()[Dim::HEIGHT] != 200 || inputs[0]->shape()[Dim::WIDTH] != 600 || 
    //     inputs[0]->size() != 200 * 600 * sizeof(float))
    //     PRINTF("Error: input shape mismatch.\n");
    // // outputs[0]->Print();
    // if (outputs[0]->shape()[Dim::HEIGHT] != 200 || outputs[0]->shape()[Dim::WIDTH] != 300 || 
    //     outputs[0]->size() != 200 * 300 * sizeof(float))
    //     PRINTF("Error: output shape mismatch.\n");

    static int count = 0;
    count++;

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
    PRINTF("TaskB: %d (%s).\n", count, id.str().c_str());
}

// 400 * 600 -> gemm（600 * 300）-> 400 * 300
void TaskC(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;

    static int count = 0;
    count++;

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
    PRINTF("TaskC: %d (%s).\n", count, id.str().c_str());
}

// 200 * 300， 400 * 300 合并 点积分
void TaskD(void *usr, std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
    AlgoTasks *ins = (AlgoTasks *)usr;
    static int count = 0;
    count++;

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
    PRINTF("TaskD: %d (%s).\n", count, id.str().c_str());
}

class SerialPass {
public:
    void Initialize(AlgoTasks *ins) {
        ins_ = ins;
        a_in_ = new Tensor({600, 600}, Type::FP32);
        a_out_b_in_ = new Tensor({200, 600}, Type::FP32);
        a_out_c_in_ = new Tensor({400, 600}, Type::FP32);

        b_out_d_in_ = new Tensor({200, 300}, Type::FP32);
        c_out_d_in_ = new Tensor({400, 300}, Type::FP32);

        d_out_ = new Tensor({1}, Type::FP32);
        //
        a_vout_.push_back(a_out_b_in_);
        a_vout_.push_back(a_out_c_in_);

        b_vin_.push_back(a_out_b_in_);
        b_vout_.push_back(b_out_d_in_);

        c_vin_.push_back(a_out_c_in_);
        c_vout_.push_back(c_out_d_in_);

        d_vin_.push_back(b_out_d_in_);
        d_vin_.push_back(c_out_d_in_);
        // d_vout_.push_back(d_out_);
    }

    void Cleanup() {
        delete a_in_;
        delete a_out_b_in_;
        delete a_out_c_in_;

        delete b_out_d_in_;
        delete c_out_d_in_;
        delete d_out_;
    }

    void Run(std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs) {
        TaskA(ins_, inputs, a_vout_);
        TaskB(ins_, b_vin_, b_vout_);
        TaskC(ins_, c_vin_, c_vout_);
        TaskD(ins_, d_vin_, outputs);
    }
    
private:
    AlgoTasks *ins_;

    Tensor *a_in_;
    Tensor *a_out_b_in_;
    Tensor *a_out_c_in_;

    Tensor *b_out_d_in_;
    Tensor *c_out_d_in_;

    Tensor *d_out_;

    std::vector<Tensor *> a_vin_;
    std::vector<Tensor *> a_vout_;

    std::vector<Tensor *> b_vin_;
    std::vector<Tensor *> b_vout_;

    std::vector<Tensor *> c_vin_;
    std::vector<Tensor *> c_vout_;

    std::vector<Tensor *> d_vin_;
    std::vector<Tensor *> d_vout_;
};

void GraphBaseTest() {

    uint32_t loop_num = 10;
    Tensor *in = new Tensor({600, 600}, Type::FP32);
    Tensor *out = new Tensor({1}, Type::FP32);    
    float *in_data = (float *)in->GetData();        
    float first_out = 0;
    prof::Timer timer("GraphBaseTest", 2);

    // Graph demo
    Graph *graph = new Graph("s1", 1);

    graph->CreateNode("n1", TaskA, {{Type::FP32, 600, 600}}, {{Type::FP32, 200, 600}, {Type::FP32, 400, 600}}, 0);
    graph->CreateNode("n2", TaskB, {{Type::FP32, 200, 600}}, {{Type::FP32, 200, 300}}, 1);
    graph->CreateNode("n3", TaskC, {{Type::FP32, 400, 600}}, {{Type::FP32, 400, 300}}, 2);
    graph->CreateNode("n4", TaskD, {{Type::FP32, 200, 300}, {Type::FP32, 400, 300}}, {{Type::FP32, 1}}, 3);
    
    graph->BuildGraph({{"n1", "n2"}, {"n1", "n3"}, {"n2", "n4"}, {"n3", "n4"}});
    DEBUG_CALL(graph->ShowInfo());

    AlgoTasks algo;
    graph->Start((void *)&algo);

    timer.Start();
    for (uint32_t i=0; i<loop_num; i++) {
        for (int j=0; j<600*600; j++) {
            in_data[j] = 1;
        }
        in->SetId(i);
        graph->Feed(in);
    }

    for (uint32_t i=0; i<loop_num; i++) {
        graph->GetResult(out);

        EXPECT_EQ(out->id(), i);
        if (i == 0)
            first_out = ((float *)out->GetData())[0];
        EXPECT_EQ(first_out, ((float *)out->GetData())[0]);
        PRINTF("out id: %d, %f.\n", out->id(), ((float *)out->GetData())[0]);
    }
    timer.Stop(0, "graph");

    PRINTF("Call stop.\n");
    graph->Stop();
    delete graph;

    // SerialPass demo.
    SerialPass sp;
    sp.Initialize(&algo);
    std::vector<Tensor *> a_vin;
    a_vin.push_back(in);
    std::vector<Tensor *> d_vout;
    d_vout.push_back(out);

    timer.Start();
    for (uint32_t i=0; i<loop_num; i++) {
        for (uint32_t j=0; j<600*600; j++) {
            in_data[j] = 1;
        }
        in->SetId(i+loop_num);
        sp.Run(a_vin, d_vout);

        EXPECT_EQ(first_out, ((float *)out->GetData())[0]);
        PRINTF("out id: %d, %f.\n", out->id(), ((float *)out->GetData())[0]);
    }
    timer.Stop(1, "serial");
    timer.Print(1, 1);
    sp.Cleanup();

    delete in;
    delete out;
}

TEST(EngineTest, Graph) {
    GraphBaseTest();
}

}  // end of namespace.