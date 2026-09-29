#include "structures.hpp"
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

//MIG Function
double mig(const double& width, const double& height, const std::vector<Function>& grad_mag){

    /*
    
    Function: MIG
    Used to compute the mean intensity gradient as a diagnostic for assessing speckle pattern quality.

    Takes in: 
    double width                --> Width of the ROI. (x)
    double height               --> Height of the ROI. (y)
    Function vector grad_mag    --> Gradient magnitude function. 

    Outputs:
    double mig                  --> Mean intensity gradient. (Single value)
    
    */

    //Defining helper variables. 
    int nx = grad_mag[0].nx;
    int ny = grad_mag[0].ny;

    //Create empty variable for MIG.
    double MIG{0.0};
    
    //Loop to get un-nromalized MIG
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){

            int index = i * ny + j;
            MIG += grad_mag[index].f;

        }
    }

    MIG /= (nx * ny);

    return MIG;

} 

void histogram_generator(const int& nbins, const std::vector<Function>& function, std::vector<Histogram>& histogram, std::vector<Histogram>& probability_density, double& mean, double& variance, double& sentropy){

    /*
    
    Function: Histogram Generator
    Takes the speckle pattern function, spits it into bins to make a histogram. 
    Computes mean, variance, shannon entropy and outputs that.

    Takes in: 
    int nbins                   --> Width of the ROI. (x)
    Function vector function    --> Function for the speckle pattern.

    Overwrites:
    Histogram vector histogram  --> Pre-initialized histogram vector.
    Histogram vector p_density  --> Pre-initialized vector to store the probability density, (Variable name is shortened here)
    double mean                 --> Mean speckle intensity value.
    double variance             --> Variance in speckle intensity,
    sentropy                    --> Shannon entropy.
    
    */

    //Create bin width. 
    double min_f {0.0};
    double max_f {1.0};
    double df {(max_f - min_f) / nbins};

    //Create actual grid.
    for (int k = 0; k < nbins; k++){
        
        histogram[k].centre = min_f + (k + 0.5) * df;

    }

    //Loop over image point to sort into appropriate bin. 
    for (const auto& point: function){

        int bin_index = static_cast<int>(std::floor((point.f - min_f) / df));
        bin_index = std::clamp(bin_index, 0, nbins - 1);

        histogram[bin_index].f += 1.0;

    }

    std::cout << "Finished generating histogram with " << nbins << " bins." << std::endl;

    //Get element counts.
    double N = function.size();

    probability_density = histogram;

    //Normalize to get the probability density.
    for (auto& bin: probability_density){
        bin.f /= N;
    }

    //Generating mean intensity, variance.
    for (int k = 0; k < nbins; k++){

        double a_k {min_f + (k + 0.5) * df};
        mean += a_k * probability_density[k].f;

    }

    for (int k = 0; k < nbins; k++){

        double a_k {min_f + (k + 0.5) * df};
        variance += std::pow((a_k - mean), 2) * probability_density[k].f;
        
        if (probability_density[k].f > 0.0){
            sentropy -= probability_density[k].f * std::log2(probability_density[k].f);
        }

    }

}

void autocorrelator_ssd(std::vector<Points>& displacements, const std::vector<Function> function, const double subpixel){

    /*
    
    Function: SSD Autocorrelator
    Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the SSD correlation function.

    Takes in: 
    double subpixel             --> Subpixel accuracy we want to go to.
    Function vector function    --> Speckle pattern function.

    Overwrites:
    Points vector displacement  --> Pre-initialized displacements vector.
    
    */

    //Console outputs
    std::cout << "SSD (Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                displacement.r += std::pow((function[index].f - shifted_function[index].f),2);

            }
        }

        //Normalize the correlated value.
        displacement.r /= function.size();
        displacement.r = (1 - displacement.r);

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_scc(std::vector<Points>& displacements, const std::vector<Function> function, const double subpixel){

    /*
    
    Function: SCC Autocorrelator
    Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the SCC correlation function.

    Takes in: 
    double subpixel             --> Subpixel accuracy we want to go to.
    Function vector function    --> Speckle pattern function.

    Overwrites:
    Points vector displacement  --> Pre-initialized displacements vector.
    
    */

    //Console outputs
    std::cout << "SCC (Standard Cross-Correlation) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Compute square sum for normalization. 
    double f_squared_sum = 0.0;
    for (const auto& pt : function) {
        f_squared_sum += pt.f * pt.f;
    }

    //Loop over displacements. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                displacement.r += function[index].f * shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        displacement.r /= f_squared_sum;

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}