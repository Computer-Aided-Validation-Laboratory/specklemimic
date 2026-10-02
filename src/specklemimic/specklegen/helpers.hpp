#include <vector>
#include <iostream>
#include "structures.hpp"

int speckle_numbers(const double width, const double height, const double size, double ratio);
void file_loader(const std::vector<Function>& points, const std::string filename);
void gradient_x(const std::vector<Function>& function, std::vector<Function>& grad_x);
void gradient_y(const std::vector<Function>& function, std::vector<Function>& grad_y);
void gradient_mag(const std::vector<Function>& grad_x, const std::vector<Function>& grad_y, std::vector<Function>& grad_mag);
std::vector<Points> displacement_generator(const int pixelshift);
void file_loader_points(const std::vector<Points>& points, const std::string filename);