# engine/cl

A small computing framework based on opencl. 

This framework is designed to help you quickly call Opencl API to do the calculations you need.

## How to use

* Refer to example/cl/cl_engine_example.cpp && example/cl/CMakeLists.txt

# OpenCL - Open Computing Language
OpenCL™ is the first open, royalty-free standard for cross-platform, parallel programming of modern processors found in personal computers, servers and handheld/embedded devices.
* Main Page - [link](https://www.khronos.org/opencl/)
* The OpenCL 1.1 Quick Reference card - [link](http://www.khronos.org/files/opencl-1-1-quick-reference-card.pdf)
* Intel SDK for OpenCL applications - [link](https://software.intel.com/en-us/opencl-sdk/choose-download)
* Intel(R) Graphics Compute Runtime for OpenCL(TM) - [link](https://github.com/intel/compute-runtime)
---

# Depens

## Windows

For Intel graphics cards on the Windows platform, OpenCl can also be used through the OneAPI. The installation is as follows:

[oneapi](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

After installation, you can set:

set OPENCL_SDK_INCLUDE=D:/software/oneAPI/compiler/2024.0/include/sycl

set OPENCL_SDK_LIB=D:/software/oneAPI/compiler/2024.0/lib

## Android

1. Install android-ndk.

2. Lib: Using script example/cl/opencl/pull_cl_lib32_win.bat to copy libs (32bits) from device to host.

3. Header: Download from [OpenCL](https://registry.khronos.org/OpenCL/) / [OpenCL-Header](https://github.com/KhronosGroup/OpenCL-Headers)

4. Set the path of OPENCL_HEADER and OPENCL_LIB for example/cl/build.sh

5. Run ./build.sh