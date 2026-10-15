/**
 * @file structures.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief File to store structures.
 * 
 * Currently stores the Points, Function and Histogram structures. 
 * These structures are extensively used throughout the speckle generator code.
 * 
 * @version 0.1
 * @date 2026-10-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

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
 * @brief Structure called Function
 * 
 * Houses a discrete representation of a function. Contains details about its grid values and the grid.
 * It is used extensively through the speckle generator.
 * 
 */
struct Function{
    double x;
    double y;
    double f;
    int nx;
    int ny;
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