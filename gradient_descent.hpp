#pragma once

#include <vector>
#include <array>
#include <functional>
#include <iostream>
#include <math.h>
#include <random>

class GradientDescent
{
public:
    static std::vector<double> run(const std::vector<double> &dataX, const std::vector<double> &dataY, int degree = 1, const int &batchSize = 1, const int &epochs = 1, const double &learningRate = 1e-4, const double &costThreshold = 1e-5);

private:

    static void lossFunction(const std::vector<double> &dataX, const std::vector<double> &dataY, const int &lowerLimitInclusive, const int &upperLimitExclusive, const std::vector<double> &currentCoefficients, std::vector<double> *lossResult);
};