POCKET_AI_DIR := ../../

CXX_TOOL := g++
CC_TOOL  := gcc
AR_TOOL  := ar
CXXFLAGS := -Werror -fno-exceptions -fPIC -std=c++11
LDFLAGS  := -lstdc++ -lm
MARCOFLAGS := 
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := infer_main

CXXFLAGS += -g -O0
# CXXFLAGS += -O3

CXX := $(CXX_TOOL)

INCLUDE  := -I. \
			-I$(POCKET_AI_DIR)

SRC      :=                      \
	$(wildcard $(CURDIR)/infer_main.cpp)

# cpy
include $(POCKET_AI_DIR)engine/infer/tools/tflite_cpy/make.mk
INCLUDE += $(CPY_INCLUDE)
LDFLAGS += $(CPY_LDFLAGS)
MARCOFLAGS += $(CPY_MARCOFLAGS)
#

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES \
         := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MARCOFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MARCOFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"