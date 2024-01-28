# PTK - Portable ToolKit

A Portable Toolkit for deploying Edge AI and HPC 

<details>
<summary>ptk::engine</summary>

* [cl](https://github.com/cjmcv/ptk/tree/master/engine/cl): A small computing framework based on opencl. 
This framework is designed to help you quickly call Opencl API to do the calculations you need.

* [vk](https://github.com/cjmcv/ptk/tree/master/engine/vk): A small computing framework based on vulkan. 
This framework is designed to help you quickly call vulkan's computing API to do the calculations you need.

* [graph](https://github.com/cjmcv/ptk/tree/master/engine/graph): A small multitasking scheduler that can quickly build efficient pipelines for your multiple tasks.

</details>

<details>
<summary>ptk::memory</summary>

* align_alloc
* allocator
* blocking_queue
* frame_shift_cache
* huffman_encoder
* ring_buffer

</details>

<details>
<summary>ptk::thread</summary>

* internal_thread
* thread_pool

</details>

<details>
<summary>tests</summary>

Unit tests
```bash
cd tests && ./build_win_x86.bat # windows
cd tests && ./build.sh          # linux
# run
.\bin\unit_tests
```
</details>


# NOTE
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

## Usage

直接链接头文件，即可使用。

如:

```bash
#include "ptk/engine/graph/graph.hpp"
#include "ptk/engine/cl/engine.hpp"
#include "ptk/engine/vk/engine.hpp"
```