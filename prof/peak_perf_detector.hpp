
/*!
* \brief Peak performance evaluation.
*/

#ifndef POCKET_AI_PROFILER_PEAK_PERF_DETECTOR_HPP_
#define POCKET_AI_PROFILER_PEAK_PERF_DETECTOR_HPP_

#include <time.h>
#include <stdio.h>

namespace pai {
namespace prof {
    
// neon:      
// C内嵌汇编: https://blog.alex.balgavy.eu/a-practical-guide-to-gcc-inline-assembly/
// 浮点峰值:  https://zhuanlan.zhihu.com/p/28226956

/////////////////////////////////////////////////////////////////////////////////////
// 实例1: Cortex-A77中fmla的指令延迟是4个时钟周期,可双发射,则意味着4*2条fmla指令可打满峰值, 指令继续往上增加, 得出的GLOPS不会有变化. (注意需要warmup)
//       其中主频为2.6G
//       理论峰值: 2(port, 双发射) * 4 (fmla处理4个元素) * 2(fmla = mul+add) * 频率 ( * 核心数 )
//              = 2 * 4 * 2 * 2.6G = 41.6G
//       测试输出 [Warnup] perf: 38.507451 GFLOPS, 1.662016 s
//                        perf: 41.408368 GFLOPS, 1.545581 s
//
// 实例2: Cortex-A55, 主频为2G, 在Q格式下, 仅能单发射fmla, 则理论峰值应为 4*2*2=16G. 只需要4条fmla即可打满峰值.
//       Q格式下单发射fmla, 参考官方文档34页 https://developer.arm.com/documentation/EPM128372/0300/?lang=en
//       测试输出8条fmla时 [Warnup] perf: 13.981993 GFLOPS, 4.577316 s
//                                 perf: 14.114482 GFLOPS, 4.534350 s
//              4条fmla时 [Warnup] perf: 12.400137 GFLOPS, 2.580617 s
//                                 perf: 12.720032 GFLOPS, 2.515717 s
//
// 注意: 测试手机的小核时, 数值容易波动, 可能是因为在低功耗状态下, 有不少后台应用在小核上运行, 影响了峰值性能的评估. 可能也是小核最多只能跑出14+G, 与峰值16G有偏差有原因.
// (挖坑: 为什么4条fmla一般只能达到12.7G, 8条却能达到14G ?)

class PeakPerfDetector {
public:
    void RunFmla(int instr_cnt) {
        const int loop = static_cast<int>(1e9); // 1e9 即1G, 乘以op_float，即为 G FLOP
        for (int wi=0; wi<2; wi++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);

            int op_float;
            if (instr_cnt == 4)
                op_float = Fmla4(loop);
            else if (instr_cnt == 8)
                op_float = Fmla8(loop);
            else if (instr_cnt == 12)
                op_float = Fmla12(loop);

            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            double time_used = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9; // 1e-9 将 ns 转为 s
            // 1e-9 将 time_used 的单位转化为秒，1e-9 / time_used即为每秒。
            if (wi == 0)
                printf("[Warnup] ");
            else
                printf("         ");
            // GFLOPS, G floating-point operations per second
            // 乘以1e-9, 是op_float个G的FLOP,单位转为GFLOP,除以秒得到GFLOPS
            printf("perf: %.6lf GFLOPS, %lf s\r\n", static_cast<float>(loop) * op_float * 1e-9 / time_used, time_used);
        }
    }

private:
    int Fmla4(int loop) {
        for (int i=0; i<loop; i++) {
            asm volatile(
                "fmla v0.4s, v0.4s, v0.4s \n"
                "fmla v1.4s, v1.4s, v1.4s \n"
                "fmla v2.4s, v2.4s, v2.4s \n"
                "fmla v3.4s, v3.4s, v3.4s \n"
                :
                :
                : "memory", "v0", "v1", "v2", "v3"  // 使用到的寄存器都需要标记,否则可能会被外面标量用到
            );
        }
        return 32; // fmla含4个元素的一次乘和一次加, 共4*2=8次浮点操作; 4条fmla, 则有32次
    }

    int Fmla8(int loop) {
        for (int i=0; i<loop; i++) {
            asm volatile(
                "fmla v0.4s, v0.4s, v0.4s \n"
                "fmla v1.4s, v1.4s, v1.4s \n"
                "fmla v2.4s, v2.4s, v2.4s \n"
                "fmla v3.4s, v3.4s, v3.4s \n"

                "fmla v4.4s, v4.4s, v4.4s \n"
                "fmla v5.4s, v5.4s, v5.4s \n"
                "fmla v6.4s, v6.4s, v6.4s \n"
                "fmla v7.4s, v7.4s, v7.4s \n"
                :
                :
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            );
        }
        return 64;
    }

    int Fmla12(int loop) {
        for (int i=0; i<loop; i++) {
            asm volatile(
                "fmla v0.4s, v0.4s, v0.4s \n"
                "fmla v1.4s, v1.4s, v1.4s \n"
                "fmla v2.4s, v2.4s, v2.4s \n"
                "fmla v3.4s, v3.4s, v3.4s \n"

                "fmla v4.4s, v4.4s, v4.4s \n"
                "fmla v5.4s, v5.4s, v5.4s \n"
                "fmla v6.4s, v6.4s, v6.4s \n"
                "fmla v7.4s, v7.4s, v7.4s \n"

                "fmla v8.4s, v8.4s, v8.4s \n"
                "fmla v9.4s, v9.4s, v9.4s \n"
                "fmla v10.4s, v10.4s, v10.4s \n"
                "fmla v11.4s, v11.4s, v11.4s \n"
                :
                :
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
            );
        }
        return 96;
    }
};

} // prof.
} // pai.
#endif // POCKET_AI_PROFILER_PEAK_PERF_DETECTOR_HPP_