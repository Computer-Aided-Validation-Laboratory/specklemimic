/**
 * @file structures.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief File to store structures.
 * 
 * Currently stores the Points, GridInfo, Field and Histogram structures.
 * These structures are extensively used throughout the speckle generator code.
 * 
 * @version 0.2
 * @date 2026-11-05
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <vector>

/**
 * @brief Structure called Points
 * 
 * Used for anything that requires a x,y value and a related function r.
 * In the speckle generator, it is used to initialize seed points with r being a radius. 
 * It is also used to generate displacements for the autocorrelation.
 * 
 */
struct Points{
    double x;
    double y;
    double r;
};

/**
 * @brief Structure called GridInfo
 *
 * Describes a uniform grid, once, instead of storing the grid in every point.
 * Points are stored in the order index = i * ny + j, and the position of a point is computed from its indices:
 * x = x0 + i * dx and y = y0 + j * dy.
 *
 */
struct GridInfo{
    int nx {0};
    int ny {0};
    double x0 {0.0};
    double y0 {0.0};
    double dx {1.0};
    double dy {1.0};

    double x(const int i) const {return x0 + i * dx;}
    double y(const int j) const {return y0 + j * dy;}
};

/**
 * @brief Structure called Field
 *
 * Houses a discrete representation of a function on a uniform grid: the grid description and the function values f only.
 * It is used extensively through the speckle generator.
 * Needs 8 bytes per point, the grid itself is never stored point by point.
 *
 */
struct Field{
    GridInfo grid;
    std::vector<double> f;
};

/**
 * @brief Structure called Histogram
 * 
 * This structure is primarily used for the histogram generation as a speckle diagnostic.
 * Contains places to put in the bin centre locations, the functional value, the number of bins (n) and the difference in function values between bin centres (df).
 * 
 */
struct Histogram{
    double centre;
    double f;
    int n;
    int df;
};

struct BinnedPoint{
    double x;
    double y;
    double r;
    double theta;
};