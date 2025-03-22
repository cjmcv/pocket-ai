# Pocket AI

A Portable Toolkit for AI Infra

<details>
<summary>engine</summary>

* [cl](https://github.com/cjmcv/pocket-ai/tree/master/engine/cl): A small computing framework based on opencl. 
This framework is designed to help you quickly call Opencl API to do the calculations you need.

```cpp
// Usage: After ensuring that your OpenCL is ready, you only need to include this header file. 
//        For more details, please refer to example/cl.
#include "pocket-ai/engine/cl/engine.hpp"
```

* [vk](https://github.com/cjmcv/pocket-ai/tree/master/engine/vk): A small computing framework based on vulkan. 
This framework is designed to help you quickly call vulkan's computing API to do the calculations you need.

```cpp
// Usage: After ensuring that your Vulkan is ready, you only need to include this header file. 
//        For more details, please refer to example/vk.
#include "pocket-ai/engine/vk/engine.hpp"
```

* [graph](https://github.com/cjmcv/pocket-ai/tree/master/engine/graph): A small multitasking scheduler that can quickly build efficient pipelines for your multiple tasks.

```cpp
// Usage: Only need to include this header file.
//        For more details, please refer to tests/engine/graph/graph_test.cpp.
#include "pocket-ai/engine/graph/graph.hpp"
```

* [infer](https://github.com/cjmcv/pocket-ai/tree/master/engine/infer): A tiny inference engine for microprocessors, with a library size of only 10K+.

```cpp
// Usage:1. Export your tflite model to a header file, 
//          as shown in file example/infer/export_model.sh. 
//          You will get a model structure file named x_model.h 
//          and a model weight file named x_model_params.h. 
//       2. Directly #include "x_model.h" and call Init, Run, and Deinit for utilization.
//
// More: 1. Use generate_test_models_tinynn.py to generate 
//          a custom model based on TinyNN for experiments. 
//          As shown in file example/infer/generate_model.sh.
//       2. Generate test data based on TFLite to verify whether 
//          the inference is correct. As shown in file 
//          example/infer/export_model.sh and example/infer/infer_main.cpp
```

</details>

<details>
<summary>eval</summary>

* [llm](https://github.com/cjmcv/pocket-ai/tree/master/eval/llm): A small tool is used to quickly verify whether the end-to-end calculation results are correct when accelerating and optimizing the large language model (LLM) inference engine.

```bash
# Usage: Refer to eval/llm/run.sh.
```
</details>

<details>
<summary>memory</summary>

* align_alloc: Byte-aligned memory allocation.
* allocator: Memory management for memory reuse.
* blocking_queue: A blocking producer-consumer queue using condition_variable and queue.
* frame_shift_cache: Manage the frame shift in audio processing.
* huffman_encoder: Compress the large arrays such as DL models.
* ring_buffer: A blocking ringbuffer.

</details>

<details>
<summary>prof</summary>

* cpu_selector: Choose which CPU core you want to run your code on.
* peak_perf_detector: Evaluate the peak CPU performance using FMLA instructions.
* timer

</details>

<details>
<summary>script</summary>

* obfuscator_a: An immature solution for obfuscating the .a static library. It's necessary to verify the usability of the generated library.

</details>

<details>
<summary>signal</summary>

* rfft / irfft: kiss_fft warpper.

</details>

<details>
<summary>thread</summary>

* internal_thread: It is commonly used for GPU data prefetching.
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