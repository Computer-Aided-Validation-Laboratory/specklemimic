#include "structures.hpp"
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

double mig(const std::vector<Function>& grad_mag){

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

void histogram_generator(const int nbins, const std::vector<Function>& function, std::vector<Histogram>& histogram, std::vector<Histogram>& probability_density, double mean, double variance, double sentropy){

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
        variance += ((a_k - mean) * (a_k - mean)) * probability_density[k].f;
        
        if (probability_density[k].f > 0.0){
            sentropy -= probability_density[k].f * std::log2(probability_density[k].f);
        }

    }

}

void autocorrelator_ssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "SSD (Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
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
                displacement.r += ((function[index].f - shifted_function[index].f) * (function[index].f - shifted_function[index].f));

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

void autocorrelator_scc(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

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

void autocorrelator_nssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "NSSD (Normalized Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the norms.
        double f_m {0};
        double g_m {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
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
                f_m += function[index].f * function[index].f;
                g_m += shifted_function[index].f * shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m = std::sqrt(f_m);
        g_m = std::sqrt(g_m);

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the NSSD value.
                displacement.r += (((function[index].f / f_m) - (shifted_function[index].f / g_m)) * ((function[index].f / f_m) - (shifted_function[index].f / g_m)));

            }
        }

        //Invert the correlation function.
        displacement.r = (1 - (displacement.r / 2));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_zssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "ZSSD (Zero Mean Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the mean.
        double f_m {0};
        double g_m {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
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
                f_m += function[index].f;
                g_m += shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m /= function.size();
        g_m /= function.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the ZSSD value.
                displacement.r += ((function[index].f - f_m) - (shifted_function[index].f - g_m)) * ((function[index].f - f_m) - (shifted_function[index].f - g_m));

            }
        }

        //Invert and normalize the correlation function. 
        displacement.r /= function.size();
        displacement.r = (1 - (displacement.r));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_znssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "ZNSSD (Zero Mean Normalized Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the mean.
        double f_m {0};
        double g_m {0};

        //To compute the zero mean normalization. 
        double delta_f {0};
        double delta_g {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
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
                f_m += function[index].f;
                g_m += shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m /= function.size();
        g_m /= function.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the deltas
                delta_f += (function[index].f - f_m) * (function[index].f - f_m);
                delta_g += (shifted_function[index].f - g_m) * (shifted_function[index].f - g_m);

            }
        }

        //Normalize the deltas. 
        delta_f = std::sqrt(delta_f);
        delta_g = std::sqrt(delta_g);


        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the ZNSSD value.
                displacement.r += (((function[index].f - f_m) / delta_f) - ((shifted_function[index].f - g_m) / delta_g)) * 
                (((function[index].f - f_m) / delta_f) - ((shifted_function[index].f - g_m) / delta_g));

            }
        }

        //Invert the correlation function.
        displacement.r = (1 - (displacement.r / 2));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_zncc(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "ZNCC (Zero Normalized Cross-Correlation) Autocorrelator" << std::endl;
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

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the mean.
        double f_m {0};
        double g_m {0};

        //To compute the zero mean normalization. 
        double delta_f {0};
        double delta_g {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
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
                f_m += function[index].f;
                g_m += shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m /= function.size();
        g_m /= function.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the deltas
                delta_f += (function[index].f - f_m) * (function[index].f - f_m);
                delta_g += (shifted_function[index].f - g_m) * (shifted_function[index].f - g_m);

            }
        }

        //Normalize the deltas. 
        delta_f = std::sqrt(delta_f);
        delta_g = std::sqrt(delta_g);


        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the ZNCC value.
                displacement.r += ((function[index].f - f_m) * (shifted_function[index].f - g_m)) / (delta_f * delta_g);

            }
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}