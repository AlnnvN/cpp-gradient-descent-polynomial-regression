#include "gradient_descent.hpp"

std::vector<double> GradientDescent::run(const std::vector<double> &dataX, const std::vector<double> &dataY, const int &degree, const int &batchSize, const int &epochs, const double &learningRate)
{
    std::vector<double> currentCoefficients(degree, 0.0); 
    std::vector<double> loss(degree, 0.0);
    double stepSize{};

    for (int i = 0; i < epochs; i++)
    {
        GradientDescent::lossFunction(dataX, dataY, currentCoefficients, &loss, degree, batchSize);
        for (int j = 0; j < degree; j++)
        {
            stepSize = loss[j] * learningRate;
            loss[j] = 0.0;
            currentCoefficients[j] = currentCoefficients[j] - stepSize;
        }
    }
    
    return currentCoefficients;
}

void GradientDescent::lossFunction(const std::vector<double> &dataX, const std::vector<double> &dataY, const std::vector<double> &currentCoefficients, std::vector<double> *lossResult, const int &degree, const int &batchSize)
{
    /*
        Sum of squared residuals = sum(y_expected - y_prediction)²
                                 = sum(y_expected - (a_i * x^n))²

        d(SSR)/d(a_i) = sum -2x^n(a_i * x^n) -> partial derivative for every coefficient
    */

    double squaredResidual{};

    int dataSize = dataX.size();

    double chainRule{};

    double x{};

    double y{};

    // batch descent
    if(batchSize == dataSize)
    {
        for (int i = 0; i < dataSize; i++)
        {
            x = dataX[i];
            y = dataY[i];

            chainRule = 0.0;
            for (int j = 0; j < degree; j++)
            {
                chainRule += std::pow(x, j) * currentCoefficients[j];
            }
            chainRule = y - chainRule;

            for (int j = 0; j < degree; j++)
            {
                squaredResidual = -2 * std::pow(x, j) * chainRule;
                (*lossResult)[j] += squaredResidual;
            }
        }
    }
}
