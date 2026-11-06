/**
 * @file generator.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Contains functions needed to generate speckle patterns.
 * 
 * Currently allows for the generation of speckle seeds using halton sampling, initialization of structures and filling out a speckle pattern.
 * Speckle pattern filling is done very naively and needs improvements using bounding boxes.
 * 
 * @version 0.2
 * @date 2026-11-05
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

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
 * @param verbose Print the number of seeded speckles. Turn off when called inside an optimizer loop.
 * @return std::vector<Points> Generated seed points in the points structure.
 */
std::vector<Points> generate_speckles_random(
    const double width,
    const double height,
    const int count,
    const double size,
    const bool verbose = true);


/**
 * @brief Halton points of the unit square.
 *
 * Halton point i depends only on i, never on the speckle size, the ratio or the ROI.
 * So the first N points can be computed once and reused for every pattern that needs at most N seeds.
 * The table is read only, so it can be shared by many threads.
 */
struct HaltonTable {
    std::vector<double> x;  //x of point i, in [0, 1).
    std::vector<double> y;  //y of point i, in [0, 1).
};


/**
 * @brief Halton Table Generator
 *
 * Computes the first count points of the Halton sequence, with the same function generate_speckles_random uses.
 *
 * @param count Number of points.
 * @return HaltonTable Points in the unit square.
 */
HaltonTable generate_halton_table(const int count);


/**
 * @brief Speckle Seeds from a Halton Table
 *
 * Scales the stored points to the ROI. This is exactly what generate_speckles_random does, without recomputing the points.
 * If more seeds are needed than the table holds, the missing points are computed (nothing shared is written).
 *
 * @param table Stored Halton points.
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param count Number of seeds required.
 * @param size Diameter of the speckle.
 * @return std::vector<Points> Generated seed points in the points structure.
 */
std::vector<Points> seeds_from_table(
    const HaltonTable& table,
    const double width,
    const double height,
    const int count,
    const double size);


/**
 * @brief Grid Generator
 *
 * Describes the uniform grid for a region and subpixel resolution. The grid is only described (GridInfo), never stored point by point.
 *
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param subpixel Subpixel resolution we are going till.
 * @return GridInfo Grid of the ROI. The origin is 0, 0.
 */
GridInfo grid_generator(
    const double width,
    const double height,
    const double subpixel);


/**
 * @brief Background Generator
 *
 * Creates an all zero field on the grid of the ROI. This is the initializer for a bunch of functions.
 *
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param subpixel Subpixel resolution we are going till.
 * @return Field Background field, all values 0.
 */
Field background_generator(
    const double width,
    const double height,
    const double subpixel);


/**
 * @brief Speckle Filler
 *
 * Fills out the speckle pattern based on given speckle seeds.
 * The speckles are written on top of what is already in the field, so the field can be reused after clearing it.
 *
 * @param background Field to fill, grid set and f of size nx * ny.
 * @param seeds Seed points for the speckle pattern.
 */
void speckle_filler(
    Field& background,
    const std::vector<Points>& seeds);