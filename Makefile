# Makefile for gpu_image_filter (GPU Specialization Capstone Project)
#
# Usage:
#   make            # build ./gpu_image_filter
#   make clean      # remove build artifacts and generated outputs
#
# Override the target GPU architecture if needed, e.g.:
#   make ARCH=sm_75

NVCC      := nvcc
ARCH      := sm_50
CXXSTD    := c++14
NVCC_FLAGS := -O3 -std=$(CXXSTD) -arch=$(ARCH) -Wno-deprecated-gpu-targets

TARGET  := gpu_image_filter
SRC_DIR := src
SRCS    := $(SRC_DIR)/main.cu $(SRC_DIR)/ppm_io.cu $(SRC_DIR)/kernels.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
	rm -f output/*.ppm output/log.csv
