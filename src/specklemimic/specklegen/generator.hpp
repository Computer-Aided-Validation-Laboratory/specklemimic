/**
 * @file generator.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Contains functions needed to generate speckle patterns.
 * 
 * Currently allows for the generation of speckle seeds using halton sampling, initialization of structures and filling out a speckle pattern.
 * Speckle pattern filling is done very naively and needs improvements using bounding boxes.
 * 
 * @version 0.1
 * @date 2026-10-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "structures.hpp"
#include <vector>

/**
 * @brief Speckle Seed Generator
 * 
 * Creates the seed points for a speckle pattern. 
 * 
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param count Number of seeds required.
 * @param size Radius of the speckle. 
 * @return std::vector<Points> Generated seed points in the points structure.
 */
std::vector<Points> generate_speckles_random(const double width, const double height, const int count, const double size);


/**
 * @brief Background Generator
 * 
 * Fills out the x,y grid for a vector with the function structure. This is the initializer for a bunch of functions. 
 * 
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param subpixel Subpixel resolution we are going till.
 * @return std::vector<Function> Background generated function vector.
 */
std::vector<Function> background_generator(const double width, const double height, const double subpixel);


/**
 * @brief Speckle Filler
 * 
 * Fills out the speckle pattern based on given speckle seeds. 
 * 
 * @param background Background generated function vector.
 * @param seeds Seed points for the speckle pattern.
 */
void speckle_filler(std::vector<Function>& background, const std::vector<Points>& seeds);