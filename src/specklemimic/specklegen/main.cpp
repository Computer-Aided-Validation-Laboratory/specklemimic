#include <iostream>
#include <vector>
#include "structures.hpp"
#include "helpers.hpp"
#include "generator.hpp"    
#include "diagnostics.hpp"
#include <chrono>

//MAIN FUNCTION - Used to run the code. 
int main(){

    //Get the user defined parameters. 
    double width {500}; //Specify width in pixels. 
    double height {500}; //Specify height in pixels. 
    double speckleSize {20}; //Specify desired speckle size in pixels.
    std::string speckle_filename {"output/speckles.csv"}; //Specify speckle filename output.
    double subpixel {0.5}; //Specify what fraction of pixel you want to resolve.
    double ratio {0.7}; //Black to white ratio
    double nbins {20}; //How many bins are needed for speckle intensity statistics.
    int pixelshift {20}; //How many autocorrelation shifts.

    //CONSOLE OUTPUTS TO START CODE
    std::cout << "-------------" << std::endl;
    std::cout << "SPECKLE GENERATOR" << std::endl;
    std::cout << "-------------" << std::endl;

    std::cout << "Generating speckles to: " << speckle_filename << std::endl;
    std::cout << "Pattern Dimensions: " << "Width = " << width << "px | " << "Height = " << height << "px " << std::endl;
    std::cout << "Target speckle size: " << speckleSize << "px" << std::endl;

    //FILENAMES FOR SAVING
    std::string gradx_filename {"output/grad_x.csv"};
    std::string grady_filename {"output/grad_y.csv"};
    std::string gradmag_filename {"output/grad_mag.csv"};
    std::string ssdautocorrelator_filename {"output/ssd_autocorrelator.csv"};
    std::string sccautocorrelator_filename {"output/scc_autocorrelator.csv"};

    std::cout << "-------------" << std::endl;
    std::cout << "Starting generator" << std::endl;
    std::cout << "-------------" << std::endl;

    //Start validation timer. 
    auto start = std::chrono::high_resolution_clock::now();


    //GENERATE BASE SPECKLE PATTERN
    //Get the required number of speckles. 
    int speckleCount {0};
    speckle_numbers(width, height, speckleSize, speckleCount, ratio);

    //Generate the speckles.
    std::vector<Points> points {generate_speckles_random(width, height, speckleCount, speckleSize)}; //Create vector with the structure. 

    //Create a background. 
    std::vector<Function> function {background_generator(width, height, subpixel)};

    //Fill background. 
    speckle_filler(function, points);

    //Export raw pattern to a .csv file.
    file_loader(function, speckle_filename);

    std::cout << "-------------" << std::endl;
    std::cout << "Generation finished, moving to gradient generation" << std::endl;
    std::cout << "-------------" << std::endl;

    //GENERATE GRADIENTS FOR DIAGNOSTICS
    //Create a gradient vectors.
    std::vector<Function> grad_X {background_generator(width, height, subpixel)};
    std::vector<Function> grad_Y {background_generator(width, height, subpixel)};

    //Populate gradient vectors.
    gradient_x(function, grad_X);
    gradient_y(function, grad_Y);

    //Export gradients to a .csv file.
    file_loader(grad_X, gradx_filename);
    file_loader(grad_Y, grady_filename);

    //Compute the gradient magnitude. 
    std::vector<Function> grad_Mag {background_generator(width, height, subpixel)};
    gradient_mag(grad_X, grad_Y, grad_Mag);

    //Export gradient magnitudes to a .csv file.
    file_loader(grad_Mag, gradmag_filename);

    std::cout << "-------------" << std::endl;
    std::cout << "Gradient generation finished, moving to histogram generation and basic statistics" << std::endl;
    std::cout << "-------------" << std::endl;

    //GENERATE HISTOGRAM AND RELATED DIAGNOSTICS
    std::vector<Histogram> intensity_distribution {};
    intensity_distribution.resize(nbins);
    std::vector<Histogram> probability_distribution {};
    probability_distribution.resize(nbins);
    double mean{0.0};
    double variance{0.0};
    double sentropy {0.0};

    histogram_generator(nbins, function, intensity_distribution, probability_distribution, mean, variance, sentropy);

    std::cout << "Basic Speckle Statistics: " << "Mean = " << mean << " | Variance = " << variance << std::endl;

    std::cout << "-------------" << std::endl;
    std::cout << "Histogram generation finished, moving to autocorrelation." << std::endl;
    std::cout << "-------------" << std::endl;

    //AUTOCORRELATION!!
    std::vector<Points> displacements {displacement_generator(pixelshift)};

    //Autocorrelation studies using SSD as a correlation function.
    autocorrelator_ssd(displacements, function, subpixel);
    file_loader_points(displacements, ssdautocorrelator_filename);

    //Autocorrelation studies using SCC as a correlation function. 
    autocorrelator_scc(displacements, function, subpixel);
    file_loader_points(displacements, sccautocorrelator_filename);

    //NOTE: I am still planning on adding more autocorrelations, these will be cool studies to conduct :)

    std::cout << "-------------" << std::endl;
    std::cout << "Autocorrelation finished, moving to diagnostics." << std::endl;
    std::cout << "-------------" << std::endl;

    //Generate the MIG (mean intensity gradient).
    double MIG {mig(width, height, grad_Mag)};    

    std::cout << "Single Valued Diagnostics: " << "MIG = " << MIG << " | Shannon Entropy = " << sentropy << std::endl;

    //End benchmarking timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "-------------" << std::endl;
    std::cout << "PROGRAM FINISHED." << std::endl;
    std::cout << "-------------" << std::endl;

    std::cout << "Program finished analyzing and generating a "<< ((width * height) / 1000000) << "MPx image in: " << duration_ms.count()/1000 << "s" << std::endl;
    
}