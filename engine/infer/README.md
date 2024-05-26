
# Setup

pip install tensorflow
pip install tflite  # https://pypi.org/project/tflite/  https://zhenhuaw.me/tflite/docs/ https://github.com/zhenhuaw-me/tflite/ 

pip list:
numpy                        1.24.3
tensorflow                   2.13.1
tflite                       2.10.0

# depend

MiniMalloc

输入:
lower和upper是占用时间，左闭右开，size是占用内存
id,lower,upper,size
b1,0,3,4
b2,3,9,4
b3,0,9,4
b4,9,21,4
b5,0,21,4
分析：b5占用4字节，因其占用时间从0到21全过程，则该4字节只能独占，无法复用。
     b1/b2/b4，b3/b4 在时间上无冲突，可以复用。
即：b5独占4字节，b1/b2复用4字节，b3/b4复用4字节。共12字节
    b5独占4字节，b1/b4复用4字节，但b2和b3有冲突，不可复用，各占4字节，则为16字节。


输出:
id,lower,upper,size,offset
b1,0,3,4,8
b2,3,9,4,8
b3,0,9,4,4
b4,9,21,4,4
b5,0,21,4,0

   0    3        9             21
4  b1---|b2------|             |
4  b3------------|b4-----------|
4  b5--------------------------|

纵轴为内存占用，横轴为时间占用。b5独占4字节，b1/b2复用4字节，b3/b4复用4字节。共12字节

