
rm a.out
nvcc -arch=sm_89 -O3 -I../../../ main.cu -o a.out && ./a.out