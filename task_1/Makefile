FLAGS = -Wall -Wextra -Werror -o
EXE = out.exe
DEF = -D FLOAT_TYPE


all:
	@echo "Build: make build_double/float then make run"
 
build_double:
	g++ $(FLAGS) $(EXE) task1.cpp

build_float:
	g++ $(FLAGS) $(EXE) $(DEF) task1.cpp

run:
	./$(EXE)

clean:
	rm $(EXE)
