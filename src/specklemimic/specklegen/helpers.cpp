#include <cmath>
#include <numbers>
#include <fstream>
#include <iostream>
#include <vector>
#include "structures.hpp"

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
    const std::vector<Function>& points, 
    const std::string filename)
{

    std::ofstream file(filename);

    file << "x,y,f \n";

    for(const auto &p : points){
        file << p.x << "," << p.y << "," << p.f << "\n";
    }

    std::cout << "Saved file to: " << filename << std::endl;
}

void gradient_x(
    const std::vector<Function>& function, 
    std::vector<Function>& grad_x)
{

    //Define some helper variables.
    int nx = function[0].nx;
    int ny = function[0].ny;
    double dx = function[0 + ny].x - function[0].x;

    //Loop over all non-boundary points. 
    for(int i = 1; i < (nx - 1); i++){
        for(int j = 0; j < ny; j++){

            int index = i * ny + j;

            //Compute Central Difference Approximation.
            grad_x[index].f = (function[index + ny].f - function[index - ny].f) / (2.0 * dx);
            grad_x[index].nx = nx;
            grad_x[index].ny = ny; 

        }
    }

    //Consider the boundary points. 
    for(int j = 0; j < ny; j++){

        int indexleft = j;
        int indexright = ny * (nx - 1) + j;

        grad_x[indexleft].f = (function[indexleft + ny].f - function[indexleft].f) / (dx);
        grad_x[indexleft].nx = nx;
        grad_x[indexleft].ny = ny;
        grad_x[indexright].f = (function[indexright].f - function[indexright - ny].f) / (dx);
        grad_x[indexright].nx = nx;
        grad_x[indexright].ny = ny;

    }

}

void gradient_y(
    const std::vector<Function>& function, 
    std::vector<Function>& grad_y)
{

    //Define some helper variables. 
    int nx = function[0].nx;
    int ny = function[0].ny;
    double dy = function[1].y - function[0].y;

    //Loop over all non-buondary points. 
    for(int i = 0; i < nx; i++){
        for(int j = 1; j < (ny - 1); j++){

            int index = i * ny + j;

            //Compute the central difference.
            grad_y[index].f = (function[index + 1].f - function[index - 1].f) / (2.0 * dy);
            grad_y[index].ny = ny;
            grad_y[index].nx = nx;

        }
    }

    //Consider the boundary points. 
    for(int i = 0; i < nx; i++){

        int indexbot = i * ny;
        int indextop = i * ny + (ny - 1);

        grad_y[indexbot].f = (function[indexbot + 1].f - function[indexbot].f) / (dy);
        grad_y[indexbot].ny = ny;
        grad_y[indexbot].nx = nx;
        grad_y[indextop].f = (function[indextop].f - function[indextop - 1].f) / (dy);
        grad_y[indextop].ny = ny;
        grad_y[indextop].nx = nx;

    }

}

void gradient_mag(
    const std::vector<Function>& grad_x, 
    const std::vector<Function>& grad_y, 
    std::vector<Function>& grad_mag)
{

    //Define some helper variables.
    int nx = grad_x[0].nx;
    int ny = grad_y[0].ny;

    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){

            int index = i * ny + j;

            //Populate the gradient magnitude.
            grad_mag[index].f = std::sqrt((grad_x[index].f * grad_x[index].f) + (grad_y[index].f * grad_y[index].f));
            grad_mag[index].nx = nx;
            grad_mag[index].ny = ny;

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