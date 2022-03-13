#include <iostream>
#include <vector>
#include <string>

#include "./includes/my_constants.hpp"      // You don't need to compile header file with constants, when you build everything
#include "./includes/my_vector.hpp"

int main()
{
    My_vector vec1({-1.1, 2, 3});
    My_vector vec2({1, 2.98, 3});
    vec1 += vec1;
    vec1.printVector();

    std::cout << my_constants::L << std::endl;
    return 0;
}