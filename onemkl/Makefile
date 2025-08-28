# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra

# MKL paths (allow override via environment variable)
MKL_DIR ?= /home/jin/intel/oneapi/mkl/latest
#MKL_DIR ?= /media/jin/nvme1n1p1/intel/oneapi/mkl/latest

INCLUDES := -I$(MKL_DIR)/include
LIB_DIRS := -L$(MKL_DIR)/lib/intel64
LIBS := -Wl,--start-group \
        $(MKL_DIR)/lib/intel64/libmkl_intel_ilp64.a \
        $(MKL_DIR)/lib/intel64/libmkl_gnu_thread.a \
        $(MKL_DIR)/lib/intel64/libmkl_core.a \
        -Wl,--end-group -lgomp -lpthread -lm -ldl

# Sources and target
SRC := onemkl_sgemm.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := onemkl_sgemm

.PHONY: all clean

all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LIBS)

# Compile source to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean object files and executable
clean:
	@echo "[make] cleaning..."
	rm -f $(OBJ) $(TARGET)

