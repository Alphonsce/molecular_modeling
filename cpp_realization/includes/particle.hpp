#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <vector>

#include "my_vector.hpp"

class Particle {
    private:
        My_vector _pos;
        My_vector _vel;
        My_vector _acc;

        long double _kin_energy;
        long double _pot_energy;

    public:
        Particle(My_vector pos, My_vector vel, My_vector acc);

        My_vector getPos();

        My_vector getVel();

        My_vector getAcc();

        void setPos(My_vector& vector);

        void setVel(My_vector& vector);

        void setAcc(My_vector& vector);
};

#endif