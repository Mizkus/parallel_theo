FLAGS = -fopenmp

all: 
	@echo "dgemv.exe, integration.exe, sle.exe, sle2.exe, sle3.exe"

dgemv.exe:
	g++ $(FLAGS) dgemv.cpp -o $@

integration.exe:
	g++ $(FLAGS) integration.cpp -o $@

sle.exe:
	g++ $(FLAGS) SLE.cpp -o $@

sle2.exe:
	g++ $(FLAGS) SLE2.cpp -o $@

sle3.exe:
	g++ $(FLAGS) SLE3.cpp -o $@

clean:
	rm *.exe
