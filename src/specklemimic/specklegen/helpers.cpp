#include <algorithm>
#include <cmath>
#include <numbers>
#include <fstream>
#include <iostream>
#include <vector>
#include "structures.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

int speckle_numbers(
    const double width, 
    const double height, 
    const double size, 
    double ratio)
{

    //Calculate areas and then divide them to get output. 
    double area{width * height};
    double speckleArea{std::numbers::pi * ((size/2) * (size/2))};

    int output = std::round((ratio * area)/speckleArea);

    return output;
    
}

void file_loader(
    const Field& points, 
    const std::string filename)
{

    std::ofstream file(filename);

    file << "x,y,f \n";

    const GridInfo& grid {points.grid};

    //Same order as the storage: index = i * ny + j. The positions are computed from the indices.
    for(int i = 0; i < grid.nx; i++){
        for(int j = 0; j < grid.ny; j++){

            size_t index = static_cast<size_t>(i) * grid.ny + j;
            file << grid.x(i) << "," << grid.y(j) << "," << points.f[index] << "\n";

        }
    }

    std::cout << "Saved file to: " << filename << std::endl;
}

void gradient_x(
    const Field& function, 
    Field& grad_x)
{

    //Define some helper variables.
    int nx = function.grid.nx;
    int ny = function.grid.ny;
    double dx = function.grid.dx;

    //The output lives on the same grid. Every value is written below, so it does not need to be cleared.
    grad_x.grid = function.grid;
    grad_x.f.resize(function.f.size());

    //Loop over all non-boundary points. 
    //Rows are independent, they only read the input and write their own output values.
    #pragma omp parallel for
    for(int i = 1; i < (nx - 1); i++){
        for(int j = 0; j < ny; j++){

            int index = i * ny + j;

            //Compute Central Difference Approximation.
            grad_x.f[index] = (function.f[index + ny] - function.f[index - ny]) / (2.0 * dx);

        }
    }

    //Consider the boundary points. 
    for(int j = 0; j < ny; j++){

        int indexleft = j;
        int indexright = ny * (nx - 1) + j;

        grad_x.f[indexleft] = (function.f[indexleft + ny] - function.f[indexleft]) / (dx);
        grad_x.f[indexright] = (function.f[indexright] - function.f[indexright - ny]) / (dx);

    }

}

void gradient_y(
    const Field& function, 
    Field& grad_y)
{

    //Define some helper variables. 
    int nx = function.grid.nx;
    int ny = function.grid.ny;
    double dy = function.grid.dy;

    //The output lives on the same grid. Every value is written below, so it does not need to be cleared.
    grad_y.grid = function.grid;
    grad_y.f.resize(function.f.size());

    //Loop over all non-buondary points. 
    //Columns of a row are independent, rows are independent as well.
    #pragma omp parallel for
    for(int i = 0; i < nx; i++){
        for(int j = 1; j < (ny - 1); j++){

            int index = i * ny + j;

            //Compute the central difference.
            grad_y.f[index] = (function.f[index + 1] - function.f[index - 1]) / (2.0 * dy);

        }
    }

    //Consider the boundary points. 
    for(int i = 0; i < nx; i++){

        int indexbot = i * ny;
        int indextop = i * ny + (ny - 1);

        grad_y.f[indexbot] = (function.f[indexbot + 1] - function.f[indexbot]) / (dy);
        grad_y.f[indextop] = (function.f[indextop] - function.f[indextop - 1]) / (dy);

    }

}

void gradient_mag(
    const Field& grad_x, 
    const Field& grad_y, 
    Field& grad_mag)
{

    //Define some helper variables.
    int nx = grad_x.grid.nx;
    int ny = grad_y.grid.ny;

    //The output lives on the same grid. Every value is written below, so it does not need to be cleared.
    grad_mag.grid = grad_x.grid;
    grad_mag.f.resize(grad_x.f.size());

    #pragma omp parallel for
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){

            int index = i * ny + j;

            //Populate the gradient magnitude.
            grad_mag.f[index] = std::sqrt((grad_x.f[index] * grad_x.f[index]) + (grad_y.f[index] * grad_y.f[index]));

        }
    }

}

void gradient_magnitude_inplace(Field& function)
{

    const int nx = function.grid.nx;
    const int ny = function.grid.ny;
    const double dx = function.grid.dx;
    const double dy = function.grid.dy;

    //A gradient needs two neighbours in every direction, so there is nothing to compute on a grid thinner than 2 points.
    if (nx < 2 || ny < 2) {
        std::fill(function.f.begin(), function.f.end(), 0.0);
        return;
    }

    //The rows are cut into one block per thread. A block needs the ORIGINAL rows just outside it, and the neighbouring block overwrites
    //them with its results, so those two rows are copied before anything is overwritten.
    int nblocks = 1;
    #ifdef _OPENMP
    nblocks = std::min(omp_get_max_threads(), nx);
    #endif

    std::vector<int> block_start(nblocks + 1);
    for (int k = 0; k <= nblocks; k++) {
        block_start[k] = static_cast<int>(static_cast<long long>(nx) * k / nblocks);
    }

    std::vector<double> halo_above(static_cast<size_t>(nblocks) * ny);   //Original row start - 1 of every block.
    std::vector<double> halo_below(static_cast<size_t>(nblocks) * ny);   //Original row end of every block (the first row of the next block).
    for (int k = 0; k < nblocks; k++) {
        if (block_start[k] > 0) {
            std::copy_n(&function.f[static_cast<size_t>(block_start[k] - 1) * ny], ny, &halo_above[static_cast<size_t>(k) * ny]);
        }
        if (block_start[k + 1] < nx) {
            std::copy_n(&function.f[static_cast<size_t>(block_start[k + 1]) * ny], ny, &halo_below[static_cast<size_t>(k) * ny]);
        }
    }

    //Blocks do not depend on each other, so they run in parallel. The result does not depend on the number of blocks.
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < nblocks; k++) {

        //Original values of the previous row and of the current row, copied before the current row is overwritten.
        std::vector<double> previous(ny);
        std::vector<double> current(ny);

        if (block_start[k] > 0) {
            std::copy_n(&halo_above[static_cast<size_t>(k) * ny], ny, previous.begin());
        }

        for (int i = block_start[k]; i < block_start[k + 1]; i++) {

            std::copy_n(&function.f[static_cast<size_t>(i) * ny], ny, current.begin());

            //Original next row: still untouched inside the block, a saved copy at the end of the block.
            const double* next = nullptr;
            if (i + 1 < block_start[k + 1]) {
                next = &function.f[static_cast<size_t>(i + 1) * ny];
            }
            else if (i + 1 < nx) {
                next = &halo_below[static_cast<size_t>(k) * ny];
            }

            for (int j = 0; j < ny; j++) {

                //Same differences as gradient_x and gradient_y: central inside, one sided on the boundary.
                double grad_x {0.0};
                if (i == 0) {
                    grad_x = (next[j] - current[j]) / (dx);
                }
                else if (i == nx - 1) {
                    grad_x = (current[j] - previous[j]) / (dx);
                }
                else {
                    grad_x = (next[j] - previous[j]) / (2.0 * dx);
                }

                double grad_y {0.0};
                if (j == 0) {
                    grad_y = (current[j + 1] - current[j]) / (dy);
                }
                else if (j == ny - 1) {
                    grad_y = (current[j] - current[j - 1]) / (dy);
                }
                else {
                    grad_y = (current[j + 1] - current[j - 1]) / (2.0 * dy);
                }

                function.f[static_cast<size_t>(i) * ny + j] = std::sqrt((grad_x * grad_x) + (grad_y * grad_y));

            }

            previous.swap(current);

        }
    }

}


std::vector<Points> displacement_generator(const int pixelshift){

    int side = 2 * pixelshift + 1;
    
    std::vector<Points> displacements;
    displacements.resize(side * side);
    
    int index = 0;
    for (int i = -pixelshift; i <= pixelshift; i++) {
        for (int j = -pixelshift; j <= pixelshift; j++) {
            displacements[index].x = i;
            displacements[index].y = j;
            index++;
        }
    }

    /*
    This loop will return displacements that follow this indexing logic:
    index = (i * ny) + j;
    Provided of course that i,j here run from zero. 

    In this case nx = ny = 2 * pixelshift + 1;
    So the index formula here is: 
    index = (i * (2 * pixelshift - 1)) + j;
    */
    
    return displacements;

}

void file_loader_points(
    const std::vector<Points>& points, 
    const std::string filename)
{
   
    std::ofstream file(filename);

    file << "x,y,f \n";

    for(const auto &p : points){
        file << p.x << "," << p.y << "," << p.r << "\n";
    }

    std::cout << "Saved file to: " << filename << std::endl;

}

int optimal_sector_count(
    const int pixelshift, 
    const double safety, 
    const int fallback)
{

    int sectors {static_cast<int>(std::ceil(2.0 * std::numbers::pi * pixelshift * safety))};

    return std::max(sectors, fallback);

}

void file_loader_binned_points(
    const std::vector<BinnedPoint>& points, 
    const std::string filename)
{
   
    std::ofstream file(filename);

    file << "x,y,f \n";

    for(const auto &p : points){
        file << p.x << "," << p.y << "," << p.r << "\n";
    }

    std::cout << "Saved file to: " << filename << std::endl;

}