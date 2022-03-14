final.out: my_vector.o cpp_main.o particle.o
	g++ -std=c++11 my_vector.o cpp_main.o particle.o -o final.out

cpp_main.o: cpp_main.cpp
	g++ -std=c++11 -c cpp_main.cpp

particle.o: ./includes/particle.cpp
	g++ -std=c++11 -c ./includes/particle.cpp

my_vector.o: ./includes/my_vector.cpp
	g++ -std=c++11 -c ./includes/my_vector.cpp

clean:
	rm *.o final.out