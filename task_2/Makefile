FLAGS =  -o
EXE = out.exe


all: 
	@echo("build_dgemv or build_integration")

build_dgemv:
	g++ $(FLAGS) $(EXE) $(DEF) dgemv.cpp -fopenmp

build_integration:
	g++ $(FLAGS) $(EXE) $(DEF) integration.cpp -fopenmp

run:
	./$(EXE)

clean:
	rm $(EXE)a