#include "helpers.hpp"
#include <fstream>
#include <iostream>

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