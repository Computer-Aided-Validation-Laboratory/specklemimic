/**
 * @file helpers.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Helper functions for Speckle Generator
 * 
 * Currently contains functions to save files, generate speckle seed counts, initialize autocorrelation displacements and compute gradients.
 * 
 * @version 0.1
 * @date 2026-10-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <vector>
#include <iostream>
#include "structures.hpp"


/**
 * @brief Speckle Number Calculator
 * 
 * Calculates how many speckles are needed to cover a given ratio of the ROI area. 
 * 
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param size Diameter of the speckle. 
 * @param ratio Target coverage ratio of the ROI.
 * @return int Number of speckles required.
 */
int speckle_numbers(
    const double width, 
    const double height, 
    const double size, 
    double ratio);


/**
 * @brief Function File Loader
 * 
 * Writes the x, y and f values of a function vector to a csv file. 
 * 
 * @param points Function vector to be saved.
 * @param filename Name of the output file.
 */
void file_loader(
    const std::vector<Function>& points, 
    const std::string filename);


/**
 * @brief X Gradient Calculator
 * 
 * Computes the x gradient of a function using central differences, with one sided differences at the boundaries. 
 * 
 * @param function Function vector to differentiate.
 * @param grad_x Gradient of the function in x.
 */
void gradient_x(
    const std::vector<Function>& function, 
    std::vector<Function>& grad_x);


/**
 * @brief Y Gradient Calculator
 * 
 * Computes the y gradient of a function using central differences, with one sided differences at the boundaries. 
 * 
 * @param function Function vector to differentiate.
 * @param grad_y Gradient of the function in y.
 */
void gradient_y(
    const std::vector<Function>& function, 
    std::vector<Function>& grad_y);


/**
 * @brief Gradient Magnitude Calculator
 * 
 * Computes the magnitude of the gradient from its x and y components. 
 * 
 * @param grad_x Gradient of the function in x.
 * @param grad_y Gradient of the function in y.
 * @param grad_mag Magnitude of the gradient.
 */
void gradient_mag(
    const std::vector<Function>& grad_x, 
    const std::vector<Function>& grad_y, 
    std::vector<Function>& grad_mag);


/**
 * @brief Displacement Generator
 * 
 * Creates a square grid of integer pixel displacements from -pixelshift to +pixelshift in x and y. 
 * 
 * @param pixelshift Maximum displacement in pixels.
 * @return std::vector<Points> Generated displacements in the points structure.
 */
std::vector<Points> displacement_generator(const int pixelshift);


/**
 * @brief Points File Loader
 * 
 * Writes the x, y and r values of a points vector to a csv file. 
 * 
 * @param points Points vector to be saved.
 * @param filename Name of the output file.
 */
void file_loader_points(
    const std::vector<Points>& points, 
    const std::string filename);


/**
 * @brief Function to count optimal number of radial sectors. 
 * 
 * This is used when trying to bin stuff radially while doing autocorrelation calculations. Especially in the watershed radius.
 * 
 * @param pixelshift Maximum displacement in pixels.
 * @param safety Safety factor to overpredict the bins by.
 * @param fallback The number of bins should not go below this fallback to keep things reasonable.
 * @return int The number of sectors needed by the autocorrelation diagnostic.
 */
int optimal_sector_count(
    const int pixelshift, 
    const double safety, 
    const int fallback);


/**
 * @brief Binned Point File Loader
 * 
 * Writes the x, y and f (r) values of a binned point vector to a csv file.
 * 
 * @param points Binned point vector to be saved.
 * @param filename Name of the output file.
 */
void file_loader_binned_points(
    const std::vector<BinnedPoint>& points, 
    const std::string filename);