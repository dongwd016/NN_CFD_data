
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include "keras_model.h" // change file name to desired header file

using Scalar = double;

int main() {
    std::array<Scalar, 12> input = {
        2.672759398903868714e+03,
        6.094697246420008874e+00,
        -4.844101359138212715e+00,
        -2.567899733682797780e+00,
        -4.017229331889003774e+00,
        -1.036033058638364590e+00,
        -3.623349802636398387e+00,
        -2.724017374557429871e+00,
        -5.866017617623208835e+00,
        -6.939431887127089027e+00,
        -8.122801971325522175e+00,
        -9.000983929310528708e+00
    }; // change input to desired features
    
    auto output = keras_model<Scalar>(input); // change input to desired features
    std::cout << "Output: ";
    for(const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}

/*
clang++ -std=c++2b -o test test.cpp
./test
*/
