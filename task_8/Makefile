EXECUTABLE_NAME = main

NVCC = nvcc
CXXFLAGS = -std=c++20 -arch=native

BOOST_INCLUDE_DIR = /usr/include
BOOST_LIB_DIR = /usr/lib
CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64

LIBS = -lboost_program_options -lcudart

SRCS = lab.cu

OBJS = $(SRCS:.cu=.o)

all: $(EXECUTABLE_NAME)

$(EXECUTABLE_NAME): $(OBJS)
	$(NVCC) $(CXXFLAGS) -o $@ $^ -L$(BOOST_LIB_DIR) -L$(CUDA_LIB_DIR) $(LIBS)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -I$(BOOST_INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXECUTABLE_NAME)
