#include "structures.hpp"
#include <vector>
#include <cmath>
#include "halton/halton.hpp"
#include <iostream>

std::vector<Points> generate_speckles_random(const double& width, const double&height, const int& count, const double& size){

    /*
    
    Function: Speckle Seed Generator
    Creates the seed points for a speckle pattern. 

    Takes in: 
    double width                --> Width of the ROI. (x)
    double height               --> Height of the ROI. (y)
    int count                   --> Number of seeds required.
    double size                 --> Radius of the speckle. 

    Outputs:
    Points vector               --> Generated seed points in the points structure.
    
    */
    
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

std::vector<Function> background_generator(const double& width, const double& height, const double& subpixel){

    /*
    
    Function: Background Generator
    Fills out the x,y grid for a vector with the function structure. This is the initializer for a bunch of functions. 

    Takes in: 
    double width                --> Width of the ROI. (x)
    double height               --> Height of the ROI. (y)
    double subpixel             --> Subpixel resolution we are going till.

    Outputs:
    Function vector             --> Background generated function vector.
    
    */

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

void speckle_filler(std::vector<Function>& background, const std::vector<Points>& seeds) {

    /*
    
    Function: Speckle Filler
    Fills out the speckle pattern based on given speckle seeds. 

    Takes in: 
    Points vector seeds         --> Seed points for the speckle pattern.

    Overwrites:
    Function vector background  --> Background generated function vector.
    
    */

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