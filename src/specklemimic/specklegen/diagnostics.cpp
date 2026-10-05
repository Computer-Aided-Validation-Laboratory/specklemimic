#include "structures.hpp"
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numbers>

double mig(const std::vector<Function>& grad_mag){

    //Defining helper variables. 
    int nx = grad_mag[0].nx;
    int ny = grad_mag[0].ny;

    //Create empty variable for MIG.
    double MIG{0.0};
    
    //Loop to get un-nromalized MIG
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){

            int index = i * ny + j;
            MIG += grad_mag[index].f;

        }
    }

    MIG /= (nx * ny);

    return MIG;

} 

void histogram_generator(const int nbins, const std::vector<Function>& function, std::vector<Histogram>& histogram, std::vector<Histogram>& probability_density, double mean, double variance, double sentropy){

    //Create bin width. 
    double min_f {0.0};
    double max_f {1.0};
    double df {(max_f - min_f) / nbins};

    //Create actual grid.
    for (int k = 0; k < nbins; k++){
        
        histogram[k].centre = min_f + (k + 0.5) * df;

    }

    //Loop over image point to sort into appropriate bin. 
    for (const auto& point: function){

        int bin_index = static_cast<int>(std::floor((point.f - min_f) / df));
        bin_index = std::clamp(bin_index, 0, nbins - 1);

        histogram[bin_index].f += 1.0;

    }

    std::cout << "Finished generating histogram with " << nbins << " bins." << std::endl;

    //Get element counts.
    double N = function.size();

    probability_density = histogram;

    //Normalize to get the probability density.
    for (auto& bin: probability_density){
        bin.f /= N;
    }

    //Generating mean intensity, variance.
    for (int k = 0; k < nbins; k++){

        double a_k {min_f + (k + 0.5) * df};
        mean += a_k * probability_density[k].f;

    }

    for (int k = 0; k < nbins; k++){

        double a_k {min_f + (k + 0.5) * df};
        variance += ((a_k - mean) * (a_k - mean)) * probability_density[k].f;
        
        if (probability_density[k].f > 0.0){
            sentropy -= probability_density[k].f * std::log2(probability_density[k].f);
        }

    }

}

void autocorrelator_ssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "SSD (Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                displacement.r += ((function[index].f - shifted_function[index].f) * (function[index].f - shifted_function[index].f));

            }
        }

        //Normalize the correlated value.
        displacement.r /= function.size();
        displacement.r = (1 - displacement.r);

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_scc(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "SCC (Standard Cross-Correlation) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Compute square sum for normalization. 
    double f_squared_sum = 0.0;
    for (const auto& pt : function) {
        f_squared_sum += pt.f * pt.f;
    }

    //Loop over displacements. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                displacement.r += function[index].f * shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        displacement.r /= f_squared_sum;

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_nssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "NSSD (Normalized Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the norms.
        double f_m {0};
        double g_m {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                f_m += function[index].f * function[index].f;
                g_m += shifted_function[index].f * shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m = std::sqrt(f_m);
        g_m = std::sqrt(g_m);

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the NSSD value.
                displacement.r += (((function[index].f / f_m) - (shifted_function[index].f / g_m)) 
                * ((function[index].f / f_m) - (shifted_function[index].f / g_m)));

            }
        }

        //Invert the correlation function.
        displacement.r = (1 - (displacement.r / 2));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_zssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "ZSSD (Zero Mean Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the mean.
        double f_m {0};
        double g_m {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                f_m += function[index].f;
                g_m += shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m /= function.size();
        g_m /= function.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the ZSSD value.
                displacement.r += ((function[index].f - f_m) - (shifted_function[index].f - g_m))
                 * ((function[index].f - f_m) - (shifted_function[index].f - g_m));

            }
        }

        //Invert and normalize the correlation function. 
        displacement.r /= function.size();
        displacement.r = (1 - (displacement.r));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_znssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "ZNSSD (Zero Mean Normalized Sum of Squared Differences) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 

    //NOTE TO SELF: COULD ADD SOME FALLBACKS HERE. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the mean.
        double f_m {0};
        double g_m {0};

        //To compute the zero mean normalization. 
        double delta_f {0};
        double delta_g {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                f_m += function[index].f;
                g_m += shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m /= function.size();
        g_m /= function.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the deltas
                delta_f += (function[index].f - f_m) * (function[index].f - f_m);
                delta_g += (shifted_function[index].f - g_m) * (shifted_function[index].f - g_m);

            }
        }

        //Normalize the deltas. 
        delta_f = std::sqrt(delta_f);
        delta_g = std::sqrt(delta_g);


        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the ZNSSD value.
                displacement.r += (((function[index].f - f_m) / delta_f) - ((shifted_function[index].f - g_m) / delta_g)) * 
                (((function[index].f - f_m) / delta_f) - ((shifted_function[index].f - g_m) / delta_g));

            }
        }

        //Invert the correlation function.
        displacement.r = (1 - (displacement.r / 2));

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

void autocorrelator_zncc(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel){

    //Console outputs
    std::cout << "ZNCC (Zero Normalized Cross-Correlation) Autocorrelator" << std::endl;
    std::cout << "Starting autocorrelation landscape generation for " << displacements.size() << " displacements." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //Define some helper variables. 
    int nx = function[0].nx;
    int ny = function[0].ny;

    //Compute square sum for normalization. 
    double f_squared_sum = 0.0;
    for (const auto& pt : function) {
        f_squared_sum += pt.f * pt.f;
    }

    //Loop over displacements again to get the actual correlation. 
    for (auto& displacement: displacements){

        //More helper variables for convenience.
        double u = displacement.x/subpixel;
        double v = displacement.y/subpixel;
        displacement.r = 0.0;

        //To compute the mean.
        double f_m {0};
        double g_m {0};

        //To compute the zero mean normalization. 
        double delta_f {0};
        double delta_g {0};

        //Initialize shifted function. 
        std::vector<Function> shifted_function = function;

        //Compute some norms for NSSD and construct the shifted function. 
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Mappin from the old function to the new function. 
                int shifted_index = (i - u) * ny + (j - v);

                if (i - u >= 0 && i - u < nx && j - v >= 0 && j - v < ny){
                    //Shifting function properly
                    shifted_function[index].f = function[shifted_index].f;
                }
                else{
                    //Adding a default 0 value.
                    shifted_function[index].f = 0.0;
                }

                //Computing the SSD value.
                f_m += function[index].f;
                g_m += shifted_function[index].f;

            }
        }

        //Normalize the correlated value.
        f_m /= function.size();
        g_m /= function.size();

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the deltas
                delta_f += (function[index].f - f_m) * (function[index].f - f_m);
                delta_g += (shifted_function[index].f - g_m) * (shifted_function[index].f - g_m);

            }
        }

        //Normalize the deltas. 
        delta_f = std::sqrt(delta_f);
        delta_g = std::sqrt(delta_g);


        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){

                int index = i * ny + j;

                //Computing the ZNCC value.
                displacement.r += ((function[index].f - f_m) * (shifted_function[index].f - g_m)) / (delta_f * delta_g);

            }
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "Finished generating the autocorrelation landscape in " << duration_ms.count()/1000 << "s" << std::endl;

}

std::vector<Points> autocorrelation_grad_r(
    const std::vector<Points>& displacements, 
    const int pixelshift)
{ 

    //Get nx = ny for index logic. 
    int ny {2 * pixelshift + 1}; 
    
    //Initialize array. 
    std::vector<Points> grad_r; 
    grad_r.resize(ny * ny); 

    //Loop over points in autocorrelation landscape to compute gradient. 
    for(int i = 0; i < ny; i++){ 
        for(int j = 0; j < ny; j++){ 
            int index = (ny * i) + j; 
            
            //Initialize gradients in x,y. 
            double grad_x {0.0}; 
            double grad_y {0.0}; 
            
            //Get the current radius. 
            double rad = std::sqrt((displacements[index].x * displacements[index].x) + (displacements[index].y * displacements[index].y)); 

            //1. Gradient in X (outer loop i, stride = ny)
            if(i == 0){ 
                //Forward difference in x. 
                grad_x = (displacements[index + ny].r - displacements[index].r) / (displacements[index + ny].x - displacements[index].x); 
            }

            else if(i == ny - 1){ 
                //Backward difference in x. 
                grad_x = (displacements[index].r - displacements[index - ny].r) / (displacements[index].x - displacements[index - ny].x); 
            } 

            else{ 
                //Central difference in x. 
                grad_x = (displacements[index + ny].r - displacements[index - ny].r) / (displacements[index + ny].x - displacements[index - ny].x); 
            }

            //2. Gradient in Y (inner loop j, stride = 1)
            if(j == 0){ 
                //Forward difference in y. 
                grad_y = (displacements[index + 1].r - displacements[index].r) / (displacements[index + 1].y - displacements[index].y); 
            } 

            else if(j == ny - 1){ 
                //Backward difference in y. 
                grad_y = (displacements[index].r - displacements[index - 1].r) / (displacements[index].y - displacements[index - 1].y); 
            } 

            else{ 
                //Central difference in y. 
                grad_y = (displacements[index + 1].r - displacements[index - 1].r) / (displacements[index + 1].y - displacements[index - 1].y); 
            } 

            //Compute the radial gradient. 
            if(rad < 1e-9){
                grad_r[index].r = 0.0;
            } 

            else{
                grad_r[index].r = ((displacements[index].x / rad) * grad_x) + ((displacements[index].y / rad) * grad_y); 
            }
            
            grad_r[index].x = displacements[index].x; 
            grad_r[index].y = displacements[index].y; 
            
        } 
    } 

    //Return computed gradient. 
    return grad_r; 

}

std::vector<BinnedPoint> watershed_gradient(
    const std::vector<Points>& grad_r, 
    const std::vector<Points>& grad_2r, 
    const std::vector<Points>& displacements, 
    const int pixelshift, 
    const int optimalsectors,
    double& watershed_radius
)
{

    //Initialize empty vector.
    std::vector<BinnedPoint> minima;
    
    //Initialize grid information. 
    int ny = (2 * pixelshift) + 1;

    //Bin Minima.
    double dtheta {(2 * std::numbers::pi) / optimalsectors};

    //Construct output vector for angular bins.
    //Fallback to boundadry if nothing correct is found.
    minima.resize(optimalsectors);

    //Track minimum distance found per sector
    const double max_domain_r = std::sqrt(2.0) * pixelshift;
    std::vector<double> min_found_radius(optimalsectors, max_domain_r);

    //Loop over function to isolate boundary.
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            
            //Check boundaries only.
            if (i == 0 || i == ny - 1 || j == 0 || j == ny - 1) {
                int index = (ny * i) + j;

                double dist = std::sqrt((grad_r[index].x * grad_r[index].x) + (grad_r[index].y * grad_r[index].y));
                double theta = std::atan2(grad_r[index].y, grad_r[index].x);
                if (theta < 0.0) theta += 2.0 * std::numbers::pi;

                int sector_idx = static_cast<int>(std::floor(theta / dtheta)) % optimalsectors;

                if (dist <= min_found_radius[sector_idx]) {
                    min_found_radius[sector_idx] = dist;
                    minima[sector_idx].x = grad_r[index].x;
                    minima[sector_idx].y = grad_r[index].y;
                    minima[sector_idx].r = displacements[index].r;
                    minima[sector_idx].theta = theta;
                }
            }

        }
    }

    //Construct Minima Function
    for(int i = 0; i < ny; i++){ 
        for(int j = 0; j < ny; j++){ 

            int index = (ny * i) + j; 

            double dist = std::sqrt((grad_r[index].x * grad_r[index].x) + (grad_r[index].y * grad_r[index].y));
            
            //Avoid the centre. That would be bad.
            if (dist < 1e-3) continue;

            //Sort for minima
            bool minima_condition {(std::abs(grad_r[index].r) < 1e-3) && (grad_2r[index].r > 0.0)};
            if (minima_condition == true){

                //Sort minima into bins.
                double theta = std::atan2(grad_r[index].y, grad_r[index].x);
                if (theta < 0.0) theta += 2.0 * std::numbers::pi; //So that all angles are in the domain.

                double angular_width = std::atan2(1.0, dist);

                //Here basically, we are finding the overlap of a pixel with multiple bins. 
                //If there is an overlap, we assign the same pixel to multiple bins to avoid any gaps as we move along the grid.
                int start_sector = static_cast<int>(std::floor((theta - angular_width) / dtheta));
                int end_sector   = static_cast<int>(std::ceil((theta + angular_width) / dtheta));

                for (int s = start_sector; s <= end_sector; ++s) {
                    int sector_idx = (s % optimalsectors + optimalsectors) % optimalsectors;

                    if (dist < min_found_radius[sector_idx]) {
                        min_found_radius[sector_idx] = dist;
                        minima[sector_idx].x = grad_r[index].x;
                        minima[sector_idx].y = grad_r[index].y;
                        minima[sector_idx].r = displacements[index].r;
                        minima[sector_idx].theta = theta;
                    }
                }
            }

        }
    }

    // GENAI : Post-processing: Remove unassigned (0, 0) origin points
    std::erase_if(minima, [](const BinnedPoint& p) {
        return (std::abs(p.x) < 1e-6 && std::abs(p.y) < 1e-6);
    });

    //Get watershed area.
    double area {};

    for(auto point: minima){

        double dist = std::sqrt((point.x * point.x) + (point.y * point.y));
        area += (dist * dist * (dtheta)) / (2.0);

    }

    watershed_radius = std::sqrt(area / std::numbers::pi);

    return minima;
}

double autocorrelation_peak(const std::vector<Points>& grad_r, const int pixelshift){

    //Initialize required variables. 
    int ny = (2 * pixelshift) + 1;
    double auto_peak {0.0};
    int perimeter_count {0};

    //This metric doesn't work if the pixel shift is too small.
    if (pixelshift < 4){
        return auto_peak;
    }

    //Loop Logic
    //Need to go 4 pixels up and down. 
    //index = (i * ny) + j;
    //0,0 at: (ny/2). (j/2)
    //step 4 on either side. 
    for(int i = (ny/2 - 4); i <= (ny / 2) + 4; i++){
        for(int j = (ny/2 - 4); j <= (ny / 2) + 4; j++){

            int index = (i * ny) + j;

            if(i == ((ny/2) - 4) || i == ((ny/2) + 4) || j == ((ny/2) - 4) || j == ((ny/2) + 4)){
                auto_peak += std::abs(grad_r[index].r);
                perimeter_count++;
            }

        }
    }

    auto_peak /= perimeter_count;

    return auto_peak;

}