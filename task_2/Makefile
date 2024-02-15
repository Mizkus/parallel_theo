FLAGS =  -o
EXE = out.exe


all: build

build:
	g++ $(FLAGS) $(EXE) $(DEF) dgemv.cpp -fopenmp

run:
	./$(EXE)

clean:
	rm $(EXE)a