#pragma once

#include <vector>
#include <array>
#include <functional>
#include <math.h>

class GradientDescent
{
public:
    static std::vector<double> run(const std::vector<double> &dataX, const std::vector<double> &dataY, const int &degree = 2, const int &batchSize = 1, const int &epochs = 1, const double &learningRate = 0.0003);

private:

    static void lossFunction(const std::vector<double> &dataX, const std::vector<double> &dataY ,const std::vector<double> &currentCoefficients, std::vector<double> *lossResult, const int &degree, const int &batchSize);
};