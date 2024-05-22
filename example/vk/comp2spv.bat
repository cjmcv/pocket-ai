pushd shaders
md spv
popd

set GLSLANG_VALIDATIOR_PATH=D:\software\VulkanSDK\1.3.231.1\Bin

%GLSLANG_VALIDATIOR_PATH%\glslangValidator.exe -V shaders\engine_test.comp -o shaders/spv/engine_test.spv
%GLSLANG_VALIDATIOR_PATH%\glslangValidator.exe -V shaders\gemm_v1.comp -o shaders/spv/gemm_v1.spv
%GLSLANG_VALIDATIOR_PATH%\glslangValidator.exe -V shaders\gemm_v2.comp -o shaders/spv/gemm_v2.spv
%GLSLANG_VALIDATIOR_PATH%\glslangValidator.exe -V shaders\gemm_v3.comp -o shaders/spv/gemm_v3.spv
pause