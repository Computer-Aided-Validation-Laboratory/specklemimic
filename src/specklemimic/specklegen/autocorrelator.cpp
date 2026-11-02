#include "structures.hpp"
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numbers>

//Value of the displaced function at point (i, j). The displaced grid is never built: the value is read straight from the original field.
//Where the displaced point falls outside the grid the value is 0.
static inline double shifted_value(
    const Field& function,
    const int i,
    const int j,
    const double u,
    const double v)
{

    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
        //Mappin from the old function to the new function.
        int shifted_index = (i - u) * ny + (j - v);
        return function.f[shifted_index];
    }

    //Adding a default 0 value.
    return 0.0;

}

void autocorrelator_ssd(
    std::vector<Points>& displacements,
    const Field& function,
    const double subpixel)
{

    //Console outputs
    std::cout << "(SSD) SSD (Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "(SSD) Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables.

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE.
    int nx = function.grid.nx;
    int ny = function.grid.ny;

    //Loop over displacements.
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the SSD value.
                displacement.r += ((function.f[index] - shifted) * (function.f[index] - shifted));

            }
        }

        //Normalize the correlated value.
        displacement.r /= function.f.size();
        displacement.r = (1 - displacement.r);

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "(SSD) Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_scc(
    std::vector<Points>& displacements,
    const Field& function,
    const double subpixel)
{

    //Console outputs
    std::cout << "(SCC) SCC (Standard Cross-Correlation) Autocorrelator" << std::endl;
    std::cout << "(SCC) Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables.
    int nx = function.grid.nx;
    int ny = function.grid.ny;

    //Compute square sum for normalization.
    double f_squared_sum = 0.0;
    for (const double value : function.f) {
        f_squared_sum += value * value;
    }

    //Loop over displacements.
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the SSD value.
                displacement.r += function.f[index] * shifted;

            }
        }

        //Normalize the correlated value.
        displacement.r /= f_squared_sum;

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "(SCC) Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_nssd(
    std::vector<Points>& displacements,
    const Field& function,
    const double subpixel)
{

    //Console outputs
    std::cout << "(NSSD) NSSD (Normalized Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "(NSSD) Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables.

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE.
    int nx = function.grid.nx;
    int ny = function.grid.ny;

    //Loop over displacements again to get the actual correlation.
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the norms.
        double f_m {0};
        double g_m {0};

        //Compute some norms for NSSD.
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the SSD value.
                f_m += function.f[index] * function.f[index];
                g_m += shifted * shifted;

            }
        }

        //Normalize the correlated value.
        f_m = std::sqrt(f_m);
        g_m = std::sqrt(g_m);

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the NSSD value.
                displacement.r += (((function.f[index] / f_m) - (shifted / g_m))
                * ((function.f[index] / f_m) - (shifted / g_m)));

            }
        }

        //Invert the correlation function.
        displacement.r = (1 - (displacement.r / 2));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "(NSSD) Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_zssd(
    std::vector<Points>& displacements,
    const Field& function,
    const double subpixel)
{

    //Console outputs
    std::cout << "(ZSSD) ZSSD (Zero Mean Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "(ZSSD) Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables.

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE.
    int nx = function.grid.nx;
    int ny = function.grid.ny;

    //Loop over displacements again to get the actual correlation.
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the mean.
        double f_m {0};
        double g_m {0};

        //Compute the means.
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the SSD value.
                f_m += function.f[index];
                g_m += shifted;

            }
        }

        //Normalize the correlated value.
        f_m /= function.f.size();
        g_m /= function.f.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the ZSSD value.
                displacement.r += ((function.f[index] - f_m) - (shifted - g_m))
                 * ((function.f[index] - f_m) - (shifted - g_m));

            }
        }

        //Invert and normalize the correlation function.
        displacement.r /= function.f.size();
        displacement.r = (1 - (displacement.r));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "(ZSSD) Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_znssd(
    std::vector<Points>& displacements,
    const Field& function,
    const double subpixel)
{

    //Console outputs
    std::cout << "(ZNSSD) ZNSSD (Zero Mean Normalized Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "(ZNSSD) Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables.

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE.
    int nx = function.grid.nx;
    int ny = function.grid.ny;

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

        //Compute the means.
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the SSD value.
                f_m += function.f[index];
                g_m += shifted;

            }
        }

        //Normalize the correlated value.
        f_m /= function.f.size();
        g_m /= function.f.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the deltas
                delta_f += (function.f[index] - f_m) * (function.f[index] - f_m);
                delta_g += (shifted - g_m) * (shifted - g_m);

            }
        }

        //Normalize the deltas.
        delta_f = std::sqrt(delta_f);
        delta_g = std::sqrt(delta_g);


        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the ZNSSD value.
                displacement.r += (((function.f[index] - f_m) / delta_f) - ((shifted - g_m) / delta_g)) *
                (((function.f[index] - f_m) / delta_f) - ((shifted - g_m) / delta_g));

            }
        }

        //Invert the correlation function.
        displacement.r = (1 - (displacement.r / 2));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "(ZNSSD) Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_zncc(
    std::vector<Points>& displacements,
    const Field& function,
    const double subpixel)
{

    //Console outputs
    std::cout << "(ZNCC) ZNCC (Zero Normalized Cross-Correlation) Autocorrelator" << std::endl;
    std::cout << "(ZNCC) Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables.
    int nx = function.grid.nx;
    int ny = function.grid.ny;

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

        //Compute the means.
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the SSD value.
                f_m += function.f[index];
                g_m += shifted;

            }
        }

        //Normalize the correlated value.
        f_m /= function.f.size();
        g_m /= function.f.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the deltas
                delta_f += (function.f[index] - f_m) * (function.f[index] - f_m);
                delta_g += (shifted - g_m) * (shifted - g_m);

            }
        }

        //Normalize the deltas.
        delta_f = std::sqrt(delta_f);
        delta_g = std::sqrt(delta_g);


        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                double shifted = shifted_value(function, i, j, u, v);

                //Computing the ZNCC value.
                displacement.r += ((function.f[index] - f_m) * (shifted - g_m)) / (delta_f * delta_g);

            }
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "(ZNCC) Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}
