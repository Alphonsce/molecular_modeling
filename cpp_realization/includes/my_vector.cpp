#include <vector>
#include <iostream>

class My_vector {

    private:
        std::vector <long double> vect;
        int size;
    
    public:
        My_vector(std::vector <long double> v_input) {
            vect = v_input;
            size = vect.size();
        }
    
        void setVect(std::vector <long double> v_set) {
            vect = v_set;
        }

        std::vector <long double> getVect() {
            return vect;
        }

        int getSize() {
            return size;
        }

        void printVector(){
            for (int i = 0; i < size; i++)
                std::cout << vect[i] << ' ';
            std::cout << std::endl;
        }

        My_vector operator + (My_vector& other_vector) {
            std::vector <long double> sum_of_vec(size, 0);
            if (size == other_vector.getSize())
                for (int i = 0; i < size; i ++)
                    sum_of_vec[i] = vect[i] + other_vector.getVect()[i];
            else
                std::cout << "DIFFERENT SIZES OF VECTORS" << std::endl;

            return sum_of_vec;        
        }

};

int main()
{
    My_vector vec1({1, 2, 3});
    My_vector vec2({1, 2, 3});
    My_vector vec3 = vec1 + vec2;
    vec3.printVector();
    // std::cout << vec1.getSize() << std::endl;
    return 0;
}