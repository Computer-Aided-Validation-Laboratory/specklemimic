/**
 * @file generator.cpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Speckle pattern generation (implementation).
 *
 * @version 0.2
 * @date 2026-11-05
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "generator.hpp"
#include "structures.hpp"
#include <vector>
#include <cmath>
#include "halton/halton.hpp"
#include <iostream>
#include <algorithm>

std::vector<Points> generate_speckles_random(
    const double width, 
    const double height, 
    const int count,
    const double size,
    const bool verbose)
{
    
    //Generate the sequence once and scale it to the ROI.
    HaltonTable table {generate_halton_table(count)};
    std::vector<Points> points {seeds_from_table(table, width, height, count, size)};

    if (verbose == true){
        std::cout << "Seeded " << count << " speckles." << std::endl;
    }

    return points;

}

HaltonTable generate_halton_table(const int count)
{

    HaltonTable table;
    if (count <= 0){
        return table;
    }

    table.x.resize(count);
    table.y.resize(count);

    //Use halton library to generate a sequence.
    for (int i = 0; i < count; i++){
        double *r {halton(i + 1, 3)};

        table.x[i] = r[0];
        table.y[i] = r[1];

        delete[] r; //Delete to clear memory.
    }

    return table;

}

std::vector<Points> seeds_from_table(
    const HaltonTable& table,
    const double width,
    const double height,
    const int count,
    const double size)
{

    //Create points vector
    std::vector<Points> points;
    if (count <= 0){
        return points;
    }
    points.resize(count);

    const size_t stored {table.x.size()};

    for (size_t i = 0; i < points.size(); i++){

        double unit_x {0.0};
        double unit_y {0.0};

        if (i < stored){
            unit_x = table.x[i];
            unit_y = table.y[i];
        }
        else{
            //Beyond the table: compute the point, nothing shared is written.
            double *r {halton(i + 1, 3)};
            unit_x = r[0];
            unit_y = r[1];
            delete[] r;
        }

        //Rescale
        points[i].x = unit_x * width;
        points[i].y = unit_y * height;
        points[i].r = size / 2;

    }

    return points;

}


GridInfo grid_generator(
    const double width, 
    const double height, 
    const double subpixel)
{

    //Generate 1D arrays
    GridInfo grid;
    grid.nx = std::round((width / subpixel));
    grid.ny = std::round((height / subpixel));

    grid.x0 = 0.0;
    grid.y0 = 0.0;
    grid.dx = width / grid.nx;
    grid.dy = height / grid.ny;

    return grid;

}

Field background_generator(
    const double width, 
    const double height, 
    const double subpixel)
{

    Field background;
    background.grid = grid_generator(width, height, subpixel);
    background.f.assign(static_cast<size_t>(background.grid.nx) * background.grid.ny, 0.0);

    return background;

}

void speckle_filler(
    Field& background,
    const std::vector<Points>& seeds)
{
 
    //Nothing to fill.
    if (background.f.empty()) {
        return;
    }
 
    //Grid layout: index = i * ny + j, with x = x0 + i * dx and y = y0 + j * dy. Nothing is stored per point.
    const GridInfo& grid {background.grid};
    const int nx {grid.nx};
    const int ny {grid.ny};
    const double x0 {grid.x0};
    const double y0 {grid.y0};
    const double dx {grid.dx};
    const double dy {grid.dy};
 
    //Loop over seeds, and only visit the background points near each one.
    //Seeds are independent, so they are shared out between threads (one fork for the whole fill, not one per speckle).
    #pragma omp parallel for
    for (long long s = 0; s < static_cast<long long>(seeds.size()); s++) {

        const Points& seed = seeds[s];
 
        //Bounding box of the speckle in grid indices.
        //Padded by one cell so rounding can never cut off an edge point. The exact distance test below still decides.
        double i_lo {std::floor((seed.x - seed.r - x0) / dx) - 1.0};
        double i_hi {std::ceil((seed.x + seed.r - x0) / dx) + 1.0};
        double j_lo {std::floor((seed.y - seed.r - y0) / dy) - 1.0};
        double j_hi {std::ceil((seed.y + seed.r - y0) / dy) + 1.0};
 
        //Clamp to the grid (in double first so the int cast is always safe).
        int imin {static_cast<int>(std::clamp(i_lo, 0.0, nx - 1.0))};
        int imax {static_cast<int>(std::clamp(i_hi, 0.0, nx - 1.0))};
        int jmin {static_cast<int>(std::clamp(j_lo, 0.0, ny - 1.0))};
        int jmax {static_cast<int>(std::clamp(j_hi, 0.0, ny - 1.0))};
 
        double r2 {seed.r * seed.r};
 
        //i is the outer loop and j the inner one, so memory is read in order (index = i * ny + j).
        for (int i = imin; i <= imax; i++) {

            double ddx = grid.x(i) - seed.x;

            for (int j = jmin; j <= jmax; j++) {
 
                double ddy = grid.y(j) - seed.y;
 
                //Inside the speckle, set it. Overlapping speckles just set the same value again, so no break is needed.
                //Every thread writes the same 1.0, the atomic write only makes that overlap formally race free.
                if (ddx * ddx + ddy * ddy <= r2) {
                    #pragma omp atomic write
                    background.f[static_cast<size_t>(i) * ny + j] = 1.0;
                }
            }
        }
    }
}
