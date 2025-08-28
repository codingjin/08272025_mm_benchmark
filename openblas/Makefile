#---------------------------------------------------------------------
# Makefile for building the OpenBLAS SGEMM
#---------------------------------------------------------------------

# Compiler and flags
CXX          := g++
CXXFLAGS     := -std=c++20 -Wall -Wextra

# OpenBLAS installation prefix (override if needed)
#OPENBLAS_PREFIX ?= /media/jin/nvme1n1p1/OpenBLAS/install
OPENBLAS_PREFIX ?= /home/jin/OpenBLAS/install

# Include and library paths
INCLUDES     := -I$(OPENBLAS_PREFIX)/include/openblas
LDFLAGS      := -L$(OPENBLAS_PREFIX)/lib
LDLIBS       := -lopenblas

# Source and target
SRCS         := openblas_sgemm.cpp
TARGET       := openblas_sgemm

# Phony targets
.PHONY: all clean

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Clean up
clean:
	rm -f $(TARGET) *.o