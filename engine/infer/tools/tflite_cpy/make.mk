CPY_INCLUDE := -I/usr/include/python3.8/ \
              -I/usr/local/lib/python3.8/dist-packages/numpy/core/include/
CPY_LDFLAGS := -L/usr/lib/x86_64-linux-gnu/ -lpython3.8
CPY_MARCOFLAGS := -DNPY_NO_DEPRECATED_API

## Example
# include tflite_cpy.mk
# INCLUDE += $(CPY_INCLUDE)
# LDFLAGS += $(CPY_LDFLAGS)
# MARCOFLAGS += $(CPY_MARCOFLAGS)