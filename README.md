# Pocket AI

A Portable Toolkit for deploying AI and HPC 

<details>
<summary>engine</summary>

* [cl](https://github.com/cjmcv/pai/tree/master/engine/cl): A small computing framework based on opencl. 
This framework is designed to help you quickly call Opencl API to do the calculations you need.

* [vk](https://github.com/cjmcv/pai/tree/master/engine/vk): A small computing framework based on vulkan. 
This framework is designed to help you quickly call vulkan's computing API to do the calculations you need.

* [graph](https://github.com/cjmcv/pai/tree/master/engine/graph): A small multitasking scheduler that can quickly build efficient pipelines for your multiple tasks.

* [infer](https://github.com/cjmcv/pai/tree/master/engine/infer): A tiny inference engine for microprocessors, with a library size of only 10K+.

</details>

<details>
<summary>eval</summary>

* [llm](https://github.com/cjmcv/pai/tree/master/eval/llm): A small tool is used to quickly verify whether the end-to-end calculation results are correct when accelerating and optimizing the large language model (LLM) inference engine.

</details>

<details>
<summary>memory</summary>

* align_alloc
* allocator
* blocking_queue
* frame_shift_cache
* huffman_encoder
* ring_buffer

</details>

<details>
<summary>prof</summary>

* timer

</details>

<details>
<summary>signal</summary>

* rfft / irfft

</details>

<details>
<summary>thread</summary>

* internal_thread
* thread_pool

</details>

<details>
<summary>util</summary>

* bmp_reader
* logger

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

pai::engine -> graph / vk / cl / infer

pai::type -> Vector (相同使用方式，添加约束进行加速) / Mat

pai::thread -> pthread（gemmlowp-pthread_everywhere） / InternalThread / ThreadPool

pai::memory -> Align / RingBuffer / FrameShift / BlockingQueue / MemReuser(内存复用 / 内存分区，用于管理高速缓存的使用) / MemPool(内存池) / Compressor(数组/权重压缩)

pai::prof -> AsmPeakPerf(宏定义代码块) / Timer（gflops） / MemRecorder (直接替换malloc/calloc) 

pai::util -> basic_marco / Logger / Type / PcmReader / BmpReader

scrip -> obfuscator (.a / .hpp)

---

## Usage

直接链接头文件，即可使用。

如:

```bash
#include "pocket-ai/engine/graph/graph.hpp"
#include "pocket-ai/engine/cl/engine.hpp"
#include "pocket-ai/engine/vk/engine.hpp"
```