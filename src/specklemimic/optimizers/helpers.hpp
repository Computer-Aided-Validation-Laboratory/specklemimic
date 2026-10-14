#pragma once

#include <vector>
#include <string>

/**
 * @brief Structure for point data.
 * 
 * Takes x coordinate, y coordinate and f value.
 * Used for particle history tracking.
 */
struct Function {
    double x;
    double y;
    double f;
};

void file_loader(const std::vector<Function>& points, const std::string filename);