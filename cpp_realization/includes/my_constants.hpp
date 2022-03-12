#ifndef MY_CONSTANTS_H      // it is an preprocessor directive, so we don't accidentally include it twice
#define MY_CONSTANTS_H      // #pragma_once is basically the same, but in one line

#include <math.h>

namespace my_constants
{
    constexpr float SIGMA = 1.;
    constexpr float EPSILON = 1.;
    constexpr float M = 1.;

    constexpr unsigned N = 20;
    constexpr long int TIME_STEPS = 1000;

    const double L = 3 * pow(N, 0.333333333333);
    constexpr float r_cut = 10.;

    constexpr double dt = 0.0005;   // 0.001 is neutral
}

#endif