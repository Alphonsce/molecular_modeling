final.out: my_vector.o cpp_main.o
	g++ my_vector.o cpp_main.o -o final.out

cpp_main.o: cpp_main.cpp
	g++ -c cpp_main.cpp

my_vector.o: ./includes/my_vector.cpp
	g++ -c ./includes/my_vector.cpp

clean:
	rm *.o final.out