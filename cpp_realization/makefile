final.out: my_vector.o cpp_main.o particle.o
	g++ my_vector.o cpp_main.o particle.o -o final.out

cpp_main.o: cpp_main.cpp
	g++ -c cpp_main.cpp

particle.o: ./includes/particle.cpp
	g++ -c ./includes/particle.cpp

my_vector.o: ./includes/my_vector.cpp
	g++ -c ./includes/my_vector.cpp

clean:
	rm *.o final.out