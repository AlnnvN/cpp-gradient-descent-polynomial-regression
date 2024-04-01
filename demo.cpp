#include <iostream>
#include "gradient_descent.hpp"
#include <random>
#include <chrono>

int main()
{
    int totalX = 15;
    int maxX = totalX * 0.2;
    double step = 0.01;

    std::vector<double> iterations;
    iterations.reserve(totalX / step);
    std::vector<double> yData;
    yData.reserve(totalX / step);

    double a = -1;
    double b = 15;
    double c = 0;
    double noise = 0.3;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution distribution{0.0, noise};
    

    std::cout << "Creating dataset" << std::endl;

    for (double x = 0; x < maxX; x+=step)
    {
        iterations.push_back(x);
        double rawY = (a * std::pow(x,2) + b * x + c);
        double noisyY = rawY + distribution(gen);
        yData.push_back(noisyY);
    }

    std::vector<double> coefficients(2);
    
    int dataSize = iterations.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    coefficients = GradientDescent::run(iterations, yData, 2, 32, 2000, 0.0001, 0.01);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Total elapsed gradient descent time -> " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n\n\n";
    std::cout << "Coefficients -> ";
    for (int i = 0; i < coefficients.size(); i++)
    {
        std::cout << ", " << coefficients[i]; 
    }
    std::cout << std::endl;
    

    return 0;
}