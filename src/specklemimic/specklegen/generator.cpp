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

std::vector<Function> background_generator(
    const double width, 
    const double height, 
    const double subpixel)
{

    //Generate 1D arrays
    int nx = std::round((width / subpixel));
    int ny = std::round((height / subpixel));
    
    double dx {width / nx};
    double dy {height / ny};

    std::vector<Function> background;
    background.resize(nx * ny);

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){

            int index = i * ny + j;

            background[index].x = i * dx;
            background[index].y = j * dy;
            background[index].f = 0.0; 
            background[index].nx = nx;
            background[index].ny = ny;

        }
    }
    
    return background;

}

void speckle_filler(
    std::vector<Function>& background,
    const std::vector<Points>& seeds)
{
 
    //Nothing to fill.
    if (background.empty()) {
        return;
    }
 
    //Grid layout is set by background_generator: index = i * ny + j, with x = i * dx and y = j * dy.
    int nx = background[0].nx;
    int ny = background[0].ny;
 
    //Recover the grid origin and spacing from the background itself, so the inputs stay the same.
    double x0 {background[0].x};
    double y0 {background[0].y};
    double dx {1.0};
    double dy {1.0};
 
    if (nx > 1) {
        dx = background[ny].x - x0;     //Point (i = 1, j = 0).
    }
    if (ny > 1) {
        dy = background[1].y - y0;      //Point (i = 0, j = 1).
    }
 
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
            for (int j = jmin; j <= jmax; j++) {
 
                Function& point = background[static_cast<size_t>(i) * ny + j];
 
                double ddx = point.x - seed.x;
                double ddy = point.y - seed.y;
 
                //Inside the speckle, set it. Overlapping speckles just set the same value again, so no break is needed.
                //Every thread writes the same 1.0, the atomic write only makes that overlap formally race free.
                if (ddx * ddx + ddy * ddy <= r2) {
                    #pragma omp atomic write
                    point.f = 1.0;
                }
            }
        }
    }
}
GridInfo grid_info(const std::vector<Function>& background)
{

    GridInfo grid;

    //Nothing to describe.
    if (background.empty()) {
        return grid;
    }

    //Same recovery as in speckle_filler.
    grid.nx = background[0].nx;
    grid.ny = background[0].ny;
    grid.x0 = background[0].x;
    grid.y0 = background[0].y;

    if (grid.nx > 1) {
        grid.dx = background[grid.ny].x - grid.x0;     //Point (i = 1, j = 0).
    }
    if (grid.ny > 1) {
        grid.dy = background[1].y - grid.y0;           //Point (i = 0, j = 1).
    }

    return grid;

}

void speckle_filler_field(
    std::vector<double>& field,
    const GridInfo& grid,
    const std::vector<Points>& seeds)
{

    //Nothing to fill.
    if (field.empty()) {
        return;
    }

    const int nx {grid.nx};
    const int ny {grid.ny};
    const double x0 {grid.x0};
    const double y0 {grid.y0};
    const double dx {grid.dx};
    const double dy {grid.dy};

    //Same loop as speckle_filler. The coordinates are computed (x0 + i * dx), which is the value background_generator stores.
    #pragma omp parallel for
    for (long long s = 0; s < static_cast<long long>(seeds.size()); s++) {

        const Points& seed = seeds[s];

        double i_lo {std::floor((seed.x - seed.r - x0) / dx) - 1.0};
        double i_hi {std::ceil((seed.x + seed.r - x0) / dx) + 1.0};
        double j_lo {std::floor((seed.y - seed.r - y0) / dy) - 1.0};
        double j_hi {std::ceil((seed.y + seed.r - y0) / dy) + 1.0};

        int imin {static_cast<int>(std::clamp(i_lo, 0.0, nx - 1.0))};
        int imax {static_cast<int>(std::clamp(i_hi, 0.0, nx - 1.0))};
        int jmin {static_cast<int>(std::clamp(j_lo, 0.0, ny - 1.0))};
        int jmax {static_cast<int>(std::clamp(j_hi, 0.0, ny - 1.0))};

        double r2 {seed.r * seed.r};

        for (int i = imin; i <= imax; i++) {

            double point_x {x0 + i * dx};
            double ddx = point_x - seed.x;

            for (int j = jmin; j <= jmax; j++) {

                double point_y {y0 + j * dy};
                double ddy = point_y - seed.y;

                //Every thread writes the same 1.0, the atomic write only makes overlapping speckles formally race free.
                if (ddx * ddx + ddy * ddy <= r2) {
                    #pragma omp atomic write
                    field[static_cast<size_t>(i) * ny + j] = 1.0;
                }
            }
        }
    }
}
