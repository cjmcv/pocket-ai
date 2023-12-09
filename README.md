# PTK - Portable Toolkit

极简可拆分无耦合工具箱

test -> 单元测试
ptk::engine -> graph / vk / cl / cu
ptk::type -> Vector (相同使用方式，添加约束进行加速) / Mat
ptk::thread -> pthread（gemmlowp-pthread_everywhere） / InternalThread / ThreadPool
ptk::memory -> Align / RingBuffer / FrameShift / BlockingQueue / MemReuser / MemPart / MemPool / Compressor(数组/权重压缩)
ptk::prof -> AsmPeakPerf / Timer（gflops） / MemRecoder (直接替换malloc/calloc) 
ptk::util -> basic_marco / Logger / Type / PcmReader / BmpReader
scrip -> obfuscator (.a / .hpp)

---

TODO
1. graph test 补充
2. 排查thread_pool_test偶发卡住的问题

## 使用

直接链接头文件，即可使用。

```bash
#include "hcs/executor.hpp"
```