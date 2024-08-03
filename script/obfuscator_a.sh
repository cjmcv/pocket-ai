#!/usr/bin/env bash

# 不成熟方案，生成的新库需要自行确认是否可用！

if [ -n "$1" ]; then
    echo Input static lib: $1
else
    echo "Input argument is missing."
fi

target_name=infer

ar -x $1                                                          #  解开得到.o
i=0;
for f in *.o;                                                     # 遍历所有.o，并重命名
do
    mv "$f" ${target_name}_${i}.o; ((i++));
done
ar r lib${target_name}_temp.a *.o                                 # 重新合成.a，完成.o的混淆

# readelf -s libabc.a | grep "FUNC" > a.txt                       # 列出所有函数符号的整行（符号太长了会显示不全！）
# awk '/_Z/ {print $8}' a.txt  > b.txt                            # 只取出第8列，即函数符号本身，并筛选出_Z字符的内容
# sort -u b.txt > c.txt                                           # 排序去重

objdump -t lib${target_name}_temp.a | grep ".text" > a.txt        # 找出text段内容
awk '{print $6}' a.txt  > b.txt                                   # 取第6列，即为函数符号
awk '/_Z/' b.txt  > c.txt                                         # 筛选出带_Z的名字，认为这些名字都是非接口的函数？后面针对这些名字进行修改

cp lib${target_name}_temp.a lib${target_name}.a

i=0
for line in $(cat c.txt)                                          # 遍历所有函数符号
do
    let i++
    msg="$i : $line -> obufunc_$i"
    echo $msg
    objcopy --redefine-sym $line=obufunc_$i lib${target_name}.a   # 重命名函数符号，起到函数名混淆的作用
done

rm *.o
rm a.txt b.txt c.txt

echo 按任意键继续
read -n 1
