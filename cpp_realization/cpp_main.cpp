#include <iostream>
#include <vector>
#include <string>

#include "./includes/my_constants.hpp"      // You don't need to compile header file with constants, when you build everything
#include "./includes/my_vector.hpp"
#include "./includes/particle.hpp"

int main()
{
    std::cout << my_constants::L << std::endl;

    My_vector vec1({-1.1, 2, 3});
    My_vector vec2({1, 2.98, 3});
    vec1 += vec1;
    vec1.printVector();
    
    My_vector pos({1, 2, 3});
    My_vector vel({1, 2, 3});
    My_vector acc({1, 2, 3});

    Particle p(pos, vel, acc);

    p.getPos().printVector();
    return 0;
}