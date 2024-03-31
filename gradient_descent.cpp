#include "gradient_descent.hpp"

std::vector<double> GradientDescent::run(const std::vector<double> &dataX, const std::vector<double> &dataY, const int &degree, const int &batchSize, const int &epochs, const double &learningRate)
{
    std::vector<double> currentCoefficients(degree, 0.0); 
    std::vector<double> loss(degree, 0.0);
    double stepSize{};

    for (int i = 0; i < epochs; i++)
    {
        GradientDescent::lossFunction(dataX, dataY, &loss, degree, batchSize);
        for (int j = 0; j < degree; j++)
        {
            stepSize = loss[j] * learningRate;
            currentCoefficients[j] = currentCoefficients[j] - stepSize;
        }
    }
    
    return currentCoefficients;
}

void GradientDescent::lossFunction(const std::vector<double> &dataX, const std::vector<double> &dataY, std::vector<double> *lossResult, const int &degree, const int &batchSize)
{
    double sumOfSquaredResiduals{};

    int dataSize = dataX.size();

    for (int i = 0; i < batchSize; i++)
    {
        //take random samples.
    }
    
    return 0.0;
}
