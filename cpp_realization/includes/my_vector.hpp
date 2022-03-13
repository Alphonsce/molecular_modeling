#ifndef MY_VECTOR_HPP
#define MY_VECTOR_HPP

#include <vector>

class My_vector {

    private:
        std::vector <long double> _vect;
        int _size;
    
    public:
        My_vector(std::vector <long double> v_input);

        My_vector();
    
        void setVect(std::vector <long double> v_set);

        std::vector <long double> getVect();

        int getSize();

        void printVector();

        My_vector operator + (My_vector& other_vector);

        void operator += (My_vector& other_vector);

        My_vector operator * (const double& scalar_value);

};

#endif