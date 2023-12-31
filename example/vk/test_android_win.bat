adb push ./bin/ptk_vk /data/local/tmp/ptk/bin/ptk_vk
adb push ./shaders /data/local/tmp/ptk/shaders
adb shell "chmod 777 -R /data/local/tmp/ptk/ && cd /data/local/tmp/ptk/ && ./bin/ptk_vk"
