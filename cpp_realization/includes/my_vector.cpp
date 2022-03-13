#include <vector>
#include <iostream>

#include "my_vector.hpp"

My_vector::My_vector(std::vector <long double> v_input) {      // it is constructor (it is like __init__)
    _vect = v_input;
    _size = _vect.size();
}

My_vector::My_vector() {                // default constructor
    std::vector<long double> v = {0, 0, 0};
    _vect = v;
}

void My_vector::setVect(std::vector <long double> v_set) {
    _vect = v_set;
}

std::vector <long double> My_vector::getVect() {
    return _vect;
}

int My_vector::getSize() {
    return _size;
}

void My_vector::printVector(){
    for (int i = 0; i < _size; i++)
        std::cout << _vect[i] << ' ';
    std::cout << std::endl;
}

My_vector My_vector::operator + (My_vector& other_vector) {
    std::vector <long double> sum_of_vec(_size, 0);
    if (_size == other_vector.getSize())
        for (int i = 0; i < _size; i ++)
            sum_of_vec[i] = _vect[i] + other_vector.getVect()[i];
    else
        std::cout << "DIFFERENT SIZES OF VECTORS" << std::endl;

    My_vector result_vec(sum_of_vec);
    return result_vec;        
}

void My_vector::operator += (My_vector& other_vector) {
    std::vector <long double> sum_of_vec(_size, 0);
    if (_size == other_vector.getSize())
        for (int i = 0; i < _size; i ++)
            sum_of_vec[i] = _vect[i] + other_vector.getVect()[i];
    else
        std::cout << "DIFFERENT SIZES OF VECTORS" << std::endl;
    
    _vect = sum_of_vec;
}

My_vector My_vector::operator * (const double& scalar_value) {
    std::vector <long double> mult_vec(_size, 1);
    for (int i = 0; i < _size; i ++)
        mult_vec[i] = _vect[i] * scalar_value;
    
    My_vector result_vec(mult_vec);
    return result_vec;
}