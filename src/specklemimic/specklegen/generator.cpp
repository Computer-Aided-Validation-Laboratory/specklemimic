#include "structures.hpp"
#include <vector>
#include <cmath>
#include "halton/halton.hpp"
#include <iostream>

std::vector<Points> generate_speckles_random(
    const double width, 
    const double height, 
    const int count, 
    const double size)
{
    
    //Create points vector
    std::vector<Points> points;
    points.resize(count);

    //Use halton library to generate a sequence.
    for (size_t i = 0; i < count; i++){
        double *r {halton(i + 1, 3)};

        //Rescale
        points[i].x = r[0] * width;
        points[i].y = r[1] * height;
        points[i].r = (r[2]/r[2]) * size/2;
        
        delete[] r; //Delete to clear memory.
    }

    std::cout << "Seeded " << count << " speckles." << std::endl;

    return points;

}

std::vector<Function> background_generator(
    const double width, 
    const double height, 
    const double subpixel)
{

    //Generate 1D arrays
    int nx = std::round((width / subpixel));
    int ny = std::round((height / subpixel));
    
    double dx {width / nx};
    double dy {height / ny};

    std::vector<Function> background;
    background.resize(nx * ny);

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){

            int index = i * ny + j;

            background[index].x = i * dx;
            background[index].y = j * dy;
            background[index].f = 0.0; 
            background[index].nx = nx;
            background[index].ny = ny;

        }
    }
    
    return background;

}

void speckle_filler(
    std::vector<Function>& background, 
    const std::vector<Points>& seeds) 
{

    for (auto& point : background) { 
        for (const auto& seed : seeds) {

            double dx = point.x - seed.x;
            double dy = point.y - seed.y;
            
            if (dx * dx + dy * dy <= seed.r * seed.r) {
                point.f = 1.0;
                break;
            }

        }
    }

    std::cout << "Finished filling all seeds." << std::endl;

}