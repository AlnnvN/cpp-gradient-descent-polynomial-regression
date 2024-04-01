#include "gradient_descent.hpp"

std::vector<double> GradientDescent::run(const std::vector<double> &dataX, const std::vector<double> &dataY, int degree, const int &batchSize, const int &epochs, const double &learningRate, const double &costThreshold)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    degree += 1; //adds independent term

    std::vector<double> currentCoefficients(degree, 0); 

    std::vector<double> loss(degree, 0.0);

    int dataSize = dataX.size();

    double stepSize{};

    bool isEpochDone{};
    bool isOptimizedEnough{};

    int lowerLimitInclusive{};
    int upperLimitExclusive{};

    int currentIndex{};
    int maxIndex = std::ceil(dataSize / (float)batchSize) - 1;

    double meanCost{};

    /* indexes list that will be shuffled */
    std::vector<int> indexes(maxIndex + 1, 0);
    for (int i = 0; i < maxIndex + 1; i++)
    {
        indexes[i] = i;
    }

    bool isBatch = batchSize >= dataSize;

    /* runs epochs */
    for (int i = 0; i < epochs; i++)
    {
        isEpochDone = false;
        currentIndex = 0;
        std::shuffle(indexes.begin(), indexes.end(), rng);

        while(!isEpochDone)
        {
            meanCost = 0.0;

            /* creates the batch dataset */
            if(isBatch) // batch descent
            {
                lowerLimitInclusive = 0;
                upperLimitExclusive = dataSize;
            }
            else // stochastic or minibatch
            {
                lowerLimitInclusive = indexes[currentIndex] * batchSize;
                upperLimitExclusive = std::min(lowerLimitInclusive + batchSize, dataSize);
            }

            /* runs loss function */
            GradientDescent::lossFunction(dataX, dataY, lowerLimitInclusive, upperLimitExclusive, currentCoefficients, &loss);

            /* checks for loss threshold */
            for (int j = 0; j < degree; j++)
            {
                meanCost += std::abs(loss[j]);   
            }
            meanCost /= degree;

            if(isBatch && meanCost <= costThreshold)
            {
                i = epochs; //triggers outer loop finishing condition
                break;
            }
            
            /* generates new step */
            for (int j = 0; j < degree; j++)
            {
                stepSize = loss[j] * learningRate;
                loss[j] = 0.0;
                currentCoefficients[j] = currentCoefficients[j] - stepSize;
            }

            currentIndex++;
            /* checks if epoch is done */
            if(isBatch || currentIndex >= maxIndex)
            {
                isEpochDone = true;
            }
        }
    }
    
    return currentCoefficients;
}

void GradientDescent::lossFunction(const std::vector<double> &dataX, const std::vector<double> &dataY, const int &lowerLimitInclusive, const int &upperLimitExclusive, const std::vector<double> &currentCoefficients, std::vector<double> *lossResult)
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

    int degree = currentCoefficients.size();


    for (int i = lowerLimitInclusive; i < upperLimitExclusive; i++)
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