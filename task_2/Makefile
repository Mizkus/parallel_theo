FLAGS =  -o
EXE = out.exe


all: 
	@echo("build_dgemv or build_integration")

build_dgemv:
	g++ $(FLAGS) $(EXE) dgemv.cpp -fopenmp

build_integration:
	g++ $(FLAGS) $(EXE) integration.cpp -fopenmp

build_sle:
	g++ $(FLAGS) $(EXE) SLE.cpp -fopenmp

build_sle2:
	g++ $(FLAGS) $(EXE) SLE2.cpp -fopenmp

run:
	./$(EXE)

clean:
	rm $(EXE)a
