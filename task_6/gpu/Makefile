CXX=pgc++
CXXFLAGS=-fast -acc -ta=tesla  -Minfo=accel -lboost_program_options

NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/

all: exe

exe: laplace2d.o jacobi.o
	$(CXX) $(CXXFLAGS) -o $@ $^ ${NVTXLIB}

run: exe
	CUDA_VISIBLE_DEVICES=3 ./exe --n 128 --iter 1000000 --err 1e-6

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .cpp .o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${NVTXLIB} 
