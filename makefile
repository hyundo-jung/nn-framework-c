build:
	rm -f ./a.out
	gcc -std=c11 -g -Wall -pedantic -Werror -Wno-newline-eof nn.c -lm -Wno-unused-variable -Wno-unused-function


run:
	./a.out