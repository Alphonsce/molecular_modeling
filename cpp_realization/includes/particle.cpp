#include <vector>

#include "my_vector.hpp"
#include "particle.hpp"


Particle::Particle(My_vector pos, My_vector vel, My_vector acc) {
    My_vector _pos = pos;
    My_vector _vel = vel;
    My_vector _acc = acc;
}

My_vector Particle::getPos() {
    return _pos;
}

My_vector Particle::getVel() {
    return _vel;
}

My_vector Particle::getAcc() {
    return _acc;
}

void Particle::setPos(My_vector& vector) {
    _pos = vector;
}

void Particle::setVel(My_vector& vector) {
    _vel = vector;
}

void Particle::setAcc(My_vector& vector) {
    _acc = vector;
}
