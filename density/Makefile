OPT = -g
ContDens : main.o data.o typeUnit.o algo.o
	gcc $(OPT) -Wall -fopenmp main.o data.o algo.o typeUnit.o -o ContDens -lm

main.o: main.c data.h
	gcc $(OPT) -c main.c

data.o : data.c data.h
	gcc $(OPT) -c data.c

algo.o : algo.c algo.h
	gcc $(OPT) -Wall -fopenmp -c algo.c

typeUnit.o : typeUnit.c typeUnit.h
	gcc $(OPT) -c typeUnit.c

cleanProg:
	rm -rf *.o
