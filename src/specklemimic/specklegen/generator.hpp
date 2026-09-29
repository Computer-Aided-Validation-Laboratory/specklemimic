#include "structures.hpp"
#include <vector>

std::vector<Points> generate_speckles_random(const double& width, const double& height, const int& count, const double& size);
std::vector<Function> background_generator(const double& width, const double& height, const double& subpixel);
void speckle_filler(std::vector<Function>& background, const std::vector<Points>& seeds);