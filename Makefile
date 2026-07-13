NVCC := nvcc
NVCC_FLAGS := -O3 -g -G -std=c++14 -arch=sm_70
# -arch=sm_70 is a conservative default (Volta). Change to match your
# GPU: sm_80 (Ampere/A100), sm_86 (RTX 30xx), sm_90 (Hopper), etc.
LIBS := -lcublas

SRCS := lbfgs.cu line_search.cu example_driver.cu
TARGET := fwi_lbfgs

$(TARGET): $(SRCS) lbfgs.h line_search.h
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
