
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include "keras_model.h" // change file name to desired header file

using Scalar = double;

int main() {
    std::array<Scalar, 12> mean_input = {
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
    }; // average of input vector

    std::array<Scalar, 12> std_input = {
        8.696648380114651218e+02,
        2.523623360913707980e+00,
        1.851689784311611575e+00,
        4.895923842287467909e-01,
        2.415243873212422621e+00,
        7.885952128472254463e-01,
        2.703946780628826208e+00,
        3.131510758084573354e+00,
        9.805477382785572349e-01,
        9.513022326237198234e-01,
        6.318167637816615168e-01,
        5.775138840740737800e-01
    }; // standard deviation of input vector

    std::array<Scalar, 12> mean_output = {
        2.226339711777933772e+00,
        4.472529415919860446e-03,
        6.154352149586567872e-02,
        -3.551763249790321122e-03,
        8.286383247320015799e-02,
        -5.343197527196321371e-03,
        9.009658262265089756e-02,
        1.005506124718058697e-01,
        7.352331975353487703e-02,
        6.375771774403653225e-02,
        2.392409226376049011e-02
    }; // average of output vector

    std::array<Scalar, 12> std_output = {
        6.618284126548039659e+00,
        1.484254814800395166e-02,
        2.105046270556587273e-01,
        1.550626078964396914e-02,
        2.814502995476770297e-01,
        2.244889510215105147e-02,
        3.153211662278485039e-01,
        3.662108155453713820e-01,
        2.871888183443607190e-01,
        2.467127348453634073e-01,
        9.224268515595179796e-02
    }; // standard deviation of output vector

    std::array<Scalar, 12> input_original = {
        1800.0, // Temperature, K
        5.0, // Pressure, atm
        0.0, // Mass fraction of 9 species, starts, H
        0.11190674, // H2
        0.0,
        0.88809326, // O2
        0.0,
        0.0,
        0.0,
        0.0,
        0.0, // Mass fraction, ends, O3
        -9.0 // log10(time step, s)
    }; // state properties at current time

    std::cout << "Input:  ";
    for(const auto& val : input_original) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::array<Scalar, 12> input_real;
    for (int i = 0; i < 12; i++) {
        if (i>=2 && i<11) {
            input_real[i] = (pow(input_original[i], 0.1) - 1) / 0.1; // Boxcox transform of mass fractions
        } else {
            input_real[i] = input_original[i];
        }
    }
    std::array<Scalar, 12> input_nn;
    for (int i = 0; i < 12; i++) {
        input_nn[i] = (input_real[i] - mean_input[i]) / std_input[i]; // normalize input
    }
    
    auto output_nn = keras_model<Scalar>(input_nn); // change input to desired features

    std::array<Scalar, 11> output_real;
    for (int i = 0; i < 11; i++) {
        output_real[i] = output_nn[i] * std_output[i] + mean_output[i]; // denormalize output
    }
    for (int i = 0; i < 11; i++) {
        output_real[i] = output_real[i] + input_real[i]; // NN outputs change of state properties, transferred it to real values
    }
    std::array<Scalar, 11> output;
    for (int i = 0; i < 11; i++) {
        if (i>=2 && i<11) {
            output[i] = pow(output_real[i] * 0.1 + 1, 10.0); // Inverse Boxcox transform of mass fractions
        } else {
            output[i] = output_real[i];
        }
    }

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
