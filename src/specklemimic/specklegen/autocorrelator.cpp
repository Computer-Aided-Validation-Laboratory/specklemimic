/**
 * @file autocorrelator.cpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Autocorrelation functions (implementation).
 *
 * @version 0.2
 * @date 2026-11-05
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "structures.hpp"
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numbers>

//The displaced function is never built. A displacement (u, v), in grid points, moves point (i, j) to (i - u, j - v).
//Whether that point is inside the grid is decided with index ranges: a row i is inside if 0 <= i - u < nx, and in such a row
//the columns j with 0 <= j - v < ny are inside. Outside the grid the displaced function is 0.
struct ShiftWindow {
    int u {0};          //Displacement along i, in grid points.
    int v {0};          //Displacement along j, in grid points.
    int jlo {0};        //First column of an inside row that is inside the grid.
    int jhi {0};        //One past the last column of an inside row that is inside the grid.
};

static inline ShiftWindow make_window(
    const double u,
    const double v,
    const int nx,
    const int ny)
{

    //Displacements are whole numbers of grid points (an integer pixel displacement divided by a subpixel of 1/k).
    //Anything else is rounded (see warn_if_rounded).
    ShiftWindow window;
    window.u = static_cast<int>(std::llround(u));
    window.v = static_cast<int>(std::llround(v));
    window.jlo = std::clamp(std::max(0, window.v), 0, ny);
    window.jhi = std::clamp(std::min(ny, window.v + ny), window.jlo, ny);

    return window;

}

//Says once, before the displacements are processed, if a displacement has to be rounded to a whole number of grid points.
static void warn_if_rounded(const std::vector<Points>& displacements, const double subpixel)
{

    for (const auto& displacement : displacements) {
        const double u = displacement.x / subpixel;
        const double v = displacement.y / subpixel;
        if (u != std::round(u) || v != std::round(v)) {
            std::cout << "WARNING: a displacement is not a whole number of grid points, it is rounded to the nearest grid point." << std::endl;
            return;
        }
    }

}

static inline bool row_inside(const ShiftWindow& window, const int i, const int nx)
{
    return (i - window.u >= 0) && (i - window.u < nx);
}

//Mean of the function itself, the same for every displacement.
static double function_mean(const Field& function)
{

    double mean {0};
    for (const double value : function.f) {
        mean += value;
    }
    mean /= function.f.size();

    return mean;

}

//Norm of the function around its mean, the same for every displacement.
static double function_delta(const Field& function, const double mean)
{

    double delta {0};
    for (const double value : function.f) {
        delta += (value - mean) * (value - mean);
    }

    return std::sqrt(delta);

}

//Mean of the displaced function. Outside the grid it is 0 and adds nothing, so only the inside is summed (in the same order).
static double shifted_mean(const Field& function, const ShiftWindow& window)
{

    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    double mean {0};

    for (int i = 0; i < nx; i++){

        if (!row_inside(window, i, nx)) {
            continue;
        }

        const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

        for (int j = window.jlo; j < window.jhi; j++){
            mean += function.f[shifted_row + static_cast<size_t>(j - window.v)];
        }
    }

    mean /= function.f.size();

    return mean;

}

//Norm of the displaced function around its mean. Every point counts, in grid order, including the ones outside the grid (value 0).
static double shifted_delta(const Field& function, const ShiftWindow& window, const double mean)
{

    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    const double outside_deviation = (0.0 - mean);
    double delta {0};

    for (int i = 0; i < nx; i++){

        if (!row_inside(window, i, nx)) {
            for (int j = 0; j < ny; j++){
                delta += outside_deviation * outside_deviation;
            }
            continue;
        }

        const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

        for (int j = 0; j < window.jlo; j++){
            delta += outside_deviation * outside_deviation;
        }
        for (int j = window.jlo; j < window.jhi; j++){
            const double shifted = function.f[shifted_row + static_cast<size_t>(j - window.v)];
            delta += (shifted - mean) * (shifted - mean);
        }
        for (int j = window.jhi; j < ny; j++){
            delta += outside_deviation * outside_deviation;
        }
    }

    return std::sqrt(delta);

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
    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    //Loop over displacements.
    warn_if_rounded(displacements, subpixel);

    //Displacements are independent, each one only reads the function and writes its own value, so they are shared out between threads.
    #pragma omp parallel for
    for (long long displacement_index = 0; displacement_index < static_cast<long long>(displacements.size()); displacement_index++){

        auto& displacement = displacements[displacement_index];

        const ShiftWindow window = make_window(displacement.x/subpixel, displacement.y/subpixel, nx, ny);
        displacement.r = 0.0;

        for (int i = 0; i < nx; i++){

            const size_t row = static_cast<size_t>(i) * ny;

            //Outside the grid the displaced function is 0, so the difference is the function itself.
            if (!row_inside(window, i, nx)) {
                for (int j = 0; j < ny; j++){
                    displacement.r += function.f[row + j] * function.f[row + j];
                }
                continue;
            }

            const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

            for (int j = 0; j < window.jlo; j++){
                displacement.r += function.f[row + j] * function.f[row + j];
            }
            for (int j = window.jlo; j < window.jhi; j++){
                const double difference = function.f[row + j] - function.f[shifted_row + static_cast<size_t>(j - window.v)];
                displacement.r += difference * difference;
            }
            for (int j = window.jhi; j < ny; j++){
                displacement.r += function.f[row + j] * function.f[row + j];
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
    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    //Compute square sum for normalization.
    double f_squared_sum = 0.0;
    for (const double value : function.f) {
        f_squared_sum += value * value;
    }

    //Loop over displacements.
    warn_if_rounded(displacements, subpixel);

    //Displacements are independent, each one only reads the function and writes its own value, so they are shared out between threads.
    #pragma omp parallel for
    for (long long displacement_index = 0; displacement_index < static_cast<long long>(displacements.size()); displacement_index++){

        auto& displacement = displacements[displacement_index];

        const ShiftWindow window = make_window(displacement.x/subpixel, displacement.y/subpixel, nx, ny);
        displacement.r = 0.0;

        //Outside the grid the displaced function is 0 and the product adds nothing, so only the inside is summed (in the same order).
        for (int i = 0; i < nx; i++){

            if (!row_inside(window, i, nx)) {
                continue;
            }

            const size_t row = static_cast<size_t>(i) * ny;
            const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

            for (int j = window.jlo; j < window.jhi; j++){
                displacement.r += function.f[row + j] * function.f[shifted_row + static_cast<size_t>(j - window.v)];
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
    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    //Norm of the function itself, the same for every displacement.
    double f_norm {0};
    for (const double value : function.f) {
        f_norm += value * value;
    }
    f_norm = std::sqrt(f_norm);

    //Loop over displacements again to get the actual correlation.
    warn_if_rounded(displacements, subpixel);

    //Displacements are independent, each one only reads the function and writes its own value, so they are shared out between threads.
    #pragma omp parallel for
    for (long long displacement_index = 0; displacement_index < static_cast<long long>(displacements.size()); displacement_index++){

        auto& displacement = displacements[displacement_index];

        const ShiftWindow window = make_window(displacement.x/subpixel, displacement.y/subpixel, nx, ny);
        displacement.r = 0.0;

        //Norm of the displaced function. Outside the grid it is 0 and adds nothing.
        double g_norm {0};
        for (int i = 0; i < nx; i++){

            if (!row_inside(window, i, nx)) {
                continue;
            }

            const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

            for (int j = window.jlo; j < window.jhi; j++){
                const double shifted = function.f[shifted_row + static_cast<size_t>(j - window.v)];
                g_norm += shifted * shifted;
            }
        }
        g_norm = std::sqrt(g_norm);

        //Normalized value of the displaced function outside the grid.
        const double outside_term = 0.0 / g_norm;

        for (int i = 0; i < nx; i++){

            const size_t row = static_cast<size_t>(i) * ny;

            if (!row_inside(window, i, nx)) {
                for (int j = 0; j < ny; j++){
                    const double difference = (function.f[row + j] / f_norm) - outside_term;
                    displacement.r += difference * difference;
                }
                continue;
            }

            const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

            for (int j = 0; j < window.jlo; j++){
                const double difference = (function.f[row + j] / f_norm) - outside_term;
                displacement.r += difference * difference;
            }
            for (int j = window.jlo; j < window.jhi; j++){
                const double shifted = function.f[shifted_row + static_cast<size_t>(j - window.v)];
                const double difference = (function.f[row + j] / f_norm) - (shifted / g_norm);
                displacement.r += difference * difference;
            }
            for (int j = window.jhi; j < ny; j++){
                const double difference = (function.f[row + j] / f_norm) - outside_term;
                displacement.r += difference * difference;
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
    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    //Mean of the function itself, the same for every displacement.
    const double f_m = function_mean(function);

    //Loop over displacements again to get the actual correlation.
    warn_if_rounded(displacements, subpixel);

    //Displacements are independent, each one only reads the function and writes its own value, so they are shared out between threads.
    #pragma omp parallel for
    for (long long displacement_index = 0; displacement_index < static_cast<long long>(displacements.size()); displacement_index++){

        auto& displacement = displacements[displacement_index];

        const ShiftWindow window = make_window(displacement.x/subpixel, displacement.y/subpixel, nx, ny);
        displacement.r = 0.0;

        const double g_m = shifted_mean(function, window);

        //Deviation of the displaced function from its mean outside the grid.
        const double outside_deviation = (0.0 - g_m);

        for (int i = 0; i < nx; i++){

            const size_t row = static_cast<size_t>(i) * ny;

            if (!row_inside(window, i, nx)) {
                for (int j = 0; j < ny; j++){
                    const double difference = (function.f[row + j] - f_m) - outside_deviation;
                    displacement.r += difference * difference;
                }
                continue;
            }

            const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

            for (int j = 0; j < window.jlo; j++){
                const double difference = (function.f[row + j] - f_m) - outside_deviation;
                displacement.r += difference * difference;
            }
            for (int j = window.jlo; j < window.jhi; j++){
                const double shifted = function.f[shifted_row + static_cast<size_t>(j - window.v)];
                const double difference = (function.f[row + j] - f_m) - (shifted - g_m);
                displacement.r += difference * difference;
            }
            for (int j = window.jhi; j < ny; j++){
                const double difference = (function.f[row + j] - f_m) - outside_deviation;
                displacement.r += difference * difference;
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
    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    //Mean and norm of the function itself, the same for every displacement.
    const double f_m = function_mean(function);
    const double delta_f = function_delta(function, f_m);

    //Loop over displacements again to get the actual correlation.
    warn_if_rounded(displacements, subpixel);

    //Displacements are independent, each one only reads the function and writes its own value, so they are shared out between threads.
    #pragma omp parallel for
    for (long long displacement_index = 0; displacement_index < static_cast<long long>(displacements.size()); displacement_index++){

        auto& displacement = displacements[displacement_index];

        const ShiftWindow window = make_window(displacement.x/subpixel, displacement.y/subpixel, nx, ny);
        displacement.r = 0.0;

        //Mean and norm of the displaced function.
        const double g_m = shifted_mean(function, window);
        const double delta_g = shifted_delta(function, window, g_m);

        //Normalized deviation of the displaced function outside the grid.
        const double outside_term = (0.0 - g_m) / delta_g;

        for (int i = 0; i < nx; i++){

            const size_t row = static_cast<size_t>(i) * ny;

            if (!row_inside(window, i, nx)) {
                for (int j = 0; j < ny; j++){
                    const double difference = ((function.f[row + j] - f_m) / delta_f) - outside_term;
                    displacement.r += difference * difference;
                }
                continue;
            }

            const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

            for (int j = 0; j < window.jlo; j++){
                const double difference = ((function.f[row + j] - f_m) / delta_f) - outside_term;
                displacement.r += difference * difference;
            }
            for (int j = window.jlo; j < window.jhi; j++){
                const double shifted = function.f[shifted_row + static_cast<size_t>(j - window.v)];
                const double difference = ((function.f[row + j] - f_m) / delta_f) - ((shifted - g_m) / delta_g);
                displacement.r += difference * difference;
            }
            for (int j = window.jhi; j < ny; j++){
                const double difference = ((function.f[row + j] - f_m) / delta_f) - outside_term;
                displacement.r += difference * difference;
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
    const int nx = function.grid.nx;
    const int ny = function.grid.ny;

    //Mean and norm of the function itself, the same for every displacement.
    const double f_m = function_mean(function);
    const double delta_f = function_delta(function, f_m);

    //Loop over displacements again to get the actual correlation.
    warn_if_rounded(displacements, subpixel);

    //Displacements are independent, each one only reads the function and writes its own value, so they are shared out between threads.
    #pragma omp parallel for
    for (long long displacement_index = 0; displacement_index < static_cast<long long>(displacements.size()); displacement_index++){

        auto& displacement = displacements[displacement_index];

        const ShiftWindow window = make_window(displacement.x/subpixel, displacement.y/subpixel, nx, ny);
        displacement.r = 0.0;

        //Mean and norm of the displaced function.
        const double g_m = shifted_mean(function, window);
        const double delta_g = shifted_delta(function, window, g_m);

        //Deviation of the displaced function from its mean outside the grid, and the normalization of the product.
        const double outside_deviation = (0.0 - g_m);
        const double normalization = delta_f * delta_g;

        for (int i = 0; i < nx; i++){

            const size_t row = static_cast<size_t>(i) * ny;

            if (!row_inside(window, i, nx)) {
                for (int j = 0; j < ny; j++){
                    displacement.r += ((function.f[row + j] - f_m) * outside_deviation) / normalization;
                }
                continue;
            }

            const size_t shifted_row = static_cast<size_t>(i - window.u) * ny;

            for (int j = 0; j < window.jlo; j++){
                displacement.r += ((function.f[row + j] - f_m) * outside_deviation) / normalization;
            }
            for (int j = window.jlo; j < window.jhi; j++){
                const double shifted = function.f[shifted_row + static_cast<size_t>(j - window.v)];
                displacement.r += ((function.f[row + j] - f_m) * (shifted - g_m)) / normalization;
            }
            for (int j = window.jhi; j < ny; j++){
                displacement.r += ((function.f[row + j] - f_m) * outside_deviation) / normalization;
            }
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "(ZNCC) Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}
