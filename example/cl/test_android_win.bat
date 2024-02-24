adb shell "rm -r /data/local/tmp/ptk/"

adb push ./bin/ptk_cl /data/local/tmp/ptk/bin/ptk_cl
adb push ./kernels /data/local/tmp/ptk/kernels

adb shell "chmod 777 -R /data/local/tmp/ptk/ && cd /data/local/tmp/ptk/ && export LD_LIBRARY_PATH=/system/lib/:/system/vender/lib/:$LD_LIBRARY_PATH && ./bin/ptk_cl"