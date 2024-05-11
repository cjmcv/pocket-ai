adb shell "rm -r /data/local/tmp/pai/"

adb push ./bin/ptk_vk /data/local/tmp/pai/bin/ptk_vk
adb push ./shaders /data/local/tmp/pai/shaders

adb shell "chmod 777 -R /data/local/tmp/pai/ && cd /data/local/tmp/pai/ && ./bin/ptk_vk"
