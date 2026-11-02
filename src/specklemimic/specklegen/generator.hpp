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
 * @brief Background Generator
 * 
 * Fills out the x,y grid for a vector with the function structure. This is the initializer for a bunch of functions. 
 * 
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param subpixel Subpixel resolution we are going till.
 * @return std::vector<Function> Background generated function vector.
 */
std::vector<Function> background_generator(
    const double width, 
    const double height, 
    const double subpixel);


/**
 * @brief Speckle Filler
 * 
 * Fills out the speckle pattern based on given speckle seeds. 
 * 
 * @param background Background generated function vector.
 * @param seeds Seed points for the speckle pattern.
 */
void speckle_filler(
    std::vector<Function>& background,
    const std::vector<Points>& seeds);


/**
 * @brief Uniform grid description.
 *
 * Everything the grid layout needs, stored once instead of in every point: index = i * ny + j, x = x0 + i * dx, y = y0 + j * dy.
 */
struct GridInfo {
    int nx {0};
    int ny {0};
    double x0 {0.0};
    double y0 {0.0};
    double dx {1.0};
    double dy {1.0};
};


/**
 * @brief Grid Description from a Background
 *
 * Recovers the layout with the same arithmetic speckle_filler uses.
 *
 * @param background Background generated function vector.
 * @return GridInfo Layout of that background.
 */
GridInfo grid_info(const std::vector<Function>& background);


/**
 * @brief Speckle Filler on gray values only
 *
 * Same as speckle_filler (same bounding boxes, same coordinates, same distance test), but writes into a plain array of gray values.
 * Needs 8 bytes per point instead of 32, and the array can be reused between calls.
 *
 * @param field Gray values, nx * ny entries, index = i * ny + j. Speckles are written on top of what is already there.
 * @param grid Layout of the grid.
 * @param seeds Seed points for the speckle pattern.
 */
void speckle_filler_field(
    std::vector<double>& field,
    const GridInfo& grid,
    const std::vector<Points>& seeds);