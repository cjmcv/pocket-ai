# PTK - Portable ToolKit for AI / edge computing / hpc

极简可拆分无耦合工具箱

test -> 单元测试

ptk::engine -> graph / vk / cl / cu

ptk::type -> Vector (相同使用方式，添加约束进行加速) / Mat

ptk::thread -> pthread（gemmlowp-pthread_everywhere） / InternalThread / ThreadPool

ptk::memory -> Align / RingBuffer / FrameShift / BlockingQueue / MemReuser(内存复用 / 内存分区，用于管理高速缓存的使用) / MemPool(内存池) / Compressor(数组/权重压缩)

ptk::prof -> AsmPeakPerf(宏定义代码块) / Timer（gflops） / MemRecorder (直接替换malloc/calloc) 

ptk::util -> basic_marco / Logger / Type / PcmReader / BmpReader

scrip -> obfuscator (.a / .hpp)

---

TODO: 排查thread_pool_test偶发卡住的问题

## 使用

直接链接头文件，即可使用。

如:

```bash
#include "ptk/engine/graph/graph.hpp"
#include "ptk/engine/cl/engine.hpp"
#include "ptk/engine/vk/engine.hpp"
```