/**
 * @file main.cpp
 * @author Lavya Ghanwat (Pizzaman2629)
 * @brief Main Speckle Generator File
 * 
 * The user can specify a bunch of important speckle related parameters here and run the speckle generator.
 * It is possible to direct where the speckle information is saved to a custom folder for the run number.
 * 
 * @version 0.2
 * @date 2026-10-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <iostream>
#include <vector>
#include "structures.hpp"
#include "helpers.hpp"
#include "generator.hpp"    
#include "diagnostics.hpp"
#include <chrono>
#include "autocorrelator.hpp"
#include <filesystem>

//MAIN FUNCTION.
int main(){

    //---------------------------------------------------------------------------

    //USER DEFINED PARAMETERS. 
    //THIS IS THE AREA OF INTEREST

    double width {500}; //Specify width in pixels. 
    double height {500}; //Specify height in pixels. 
    double speckleSize {25}; //Specify desired speckle size in pixels.
    std::string runfolder {"pattern1"}; //Which folder should all the results be saved in.
    double subpixel {0.5}; //Specify what fraction of pixel you want to resolve.
    double ratio {0.7}; //Black to white ratio
    int nbins {20}; //How many bins are needed for speckle intensity statistics.
    int pixelshift {20}; //How many autocorrelation shifts.
    bool save_speckle {true};
    bool save_grads {true};
    bool save_autocorrelators {true};
    int ss_size {20}; //Subset Size for SSSIG Heatmap Calculations.
    int st_size {20}; //Step Size for SSIG Heatmap Calculations.

    double safety {1.25}; //Safety factor for binning in watershed radius calculations. 
    int fallback {40}; //Number of bins to fall back to for watershed radius calculation.

    //AUTOCORRELATOR FLAGS
    //Multiple autocorrelators can be turned on or off for diagnostics purposes.
    bool ssd {false};
    bool scc {false};
    bool nssd {false};
    bool zssd {false};
    bool zncc {false};
    bool znssd {true};

    //---------------------------------------------------------------------------

    //FILESAVING
    std::string grad_directory {runfolder + "/" + "gradients"};
    std::string autoc_directory {runfolder + "/" + "autocorrelation"};
    std::string sssig_directory {runfolder + "/" + "sssig"};

    std::filesystem::create_directories(runfolder); //Main Directory
    std::filesystem::create_directories(grad_directory); //Gradients
    std::filesystem::create_directories(autoc_directory); //Autocorrelation
    std::filesystem::create_directories(sssig_directory); //SSSIG

    std::string speckle_filename {runfolder + "/" "speckles.csv"};
    std::string gradx_filename {grad_directory + "/" + "grad_x.csv"};
    std::string grady_filename {grad_directory + "/" + "grad_y.csv"};
    std::string gradmag_filename {grad_directory + "/" + "grad_mag.csv"};
    std::string ssdautocorrelator_filename {autoc_directory + "/" + "ssd_autocorrelator.csv"};
    std::string sccautocorrelator_filename {autoc_directory + "/" + "scc_autocorrelator.csv"};
    std::string nssdautocorrelator_filename {autoc_directory + "/" + "nssd_autocorrelator.csv"};
    std::string zssdautocorrelator_filename {autoc_directory + "/" + "zssd_autocorrelator.csv"};
    std::string znssdautocorrelator_filename {autoc_directory + "/" + "znssd_autocorrelator.csv"};
    std::string znccautocorrelator_filename {autoc_directory + "/" + "zncc_autocorrelator.csv"};
    //Radial gradient and watershed files are saved per autocorrelator so they do not overwrite each other.
    std::string ssdautogradr_filename {autoc_directory + "/" + "ssd_autogradr.csv"};
    std::string ssdwatershedgrad_filename {autoc_directory + "/" + "ssd_wshedgrad.csv"};
    std::string sccautogradr_filename {autoc_directory + "/" + "scc_autogradr.csv"};
    std::string sccwatershedgrad_filename {autoc_directory + "/" + "scc_wshedgrad.csv"};
    std::string nssdautogradr_filename {autoc_directory + "/" + "nssd_autogradr.csv"};
    std::string nssdwatershedgrad_filename {autoc_directory + "/" + "nssd_wshedgrad.csv"};
    std::string zssdautogradr_filename {autoc_directory + "/" + "zssd_autogradr.csv"};
    std::string zssdwatershedgrad_filename {autoc_directory + "/" + "zssd_wshedgrad.csv"};
    std::string znccautogradr_filename {autoc_directory + "/" + "zncc_autogradr.csv"};
    std::string znccwatershedgrad_filename {autoc_directory + "/" + "zncc_wshedgrad.csv"};
    std::string znssdautogradr_filename {autoc_directory + "/" + "znssd_autogradr.csv"};
    std::string znssdwatershedgrad_filename {autoc_directory + "/" + "znssd_wshedgrad.csv"};
    std::string sssig_filename {sssig_directory + "/" + "sssig_heatmap.csv"};
    std::string sssigdelta_filename {sssig_directory + "/" + "sssig_deltamap.csv"};

    //---------------------------------------------------------------------------

    //START GENERATOR AND BENCHMARKING
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "SPECKLE GENERATOR" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    std::cout << "Generating speckles to: " << speckle_filename << std::endl;
    std::cout << "Pattern Dimensions: " << "Width = " << width << "px | " << "Height = " << height << "px " << std::endl;
    std::cout << "Target speckle size: " << speckleSize << "px" << std::endl;

    std::cout << "---------------------------------------------------------------------------------------------------"<< std::endl;
    std::cout << "Starting generator" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    //Start validation timer. 
    auto start = std::chrono::high_resolution_clock::now();

    //---------------------------------------------------------------------------

    //GENERATE BASE SPECKLE PATTERN
    //Get the required number of speckles. 
    int speckleCount {speckle_numbers(width, height, speckleSize, ratio)};

    //Generate the speckles.
    std::vector<Points> points {generate_speckles_random(width, height, speckleCount, speckleSize)}; //Create vector with the structure. 

    //Create a background. 
    std::vector<Function> function {background_generator(width, height, subpixel)};

    //Fill background. 
    speckle_filler(function, points);

    //Export raw pattern to a .csv file.
    if (save_speckle == true){
        file_loader(function, speckle_filename);
    }

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Basic pattern generation finished, moving to gradient generation" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    //---------------------------------------------------------------------------

    //GENERATE GRADIENTS FOR DIAGNOSTICS
    //Create a gradient vectors.
    std::vector<Function> grad_X {background_generator(width, height, subpixel)};
    std::vector<Function> grad_Y {background_generator(width, height, subpixel)};

    //Populate gradient vectors.
    gradient_x(function, grad_X);
    gradient_y(function, grad_Y);

    //Export gradients to a .csv file.
    if (save_grads == true){
        file_loader(grad_X, gradx_filename);
        file_loader(grad_Y, grady_filename);
    }
    
    //Compute the gradient magnitude. 
    std::vector<Function> grad_Mag {background_generator(width, height, subpixel)};
    gradient_mag(grad_X, grad_Y, grad_Mag);

    //Export gradient magnitudes to a .csv file.
    if (save_grads == true){
        file_loader(grad_Mag, gradmag_filename);
    }

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Gradient generation finished, moving to histogram generation and basic statistics" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    //---------------------------------------------------------------------------

    //GENERATE HISTOGRAM AND RELATED DIAGNOSTICS
    std::vector<Histogram> intensity_distribution {};
    intensity_distribution.resize(nbins);
    std::vector<Histogram> probability_distribution {};
    probability_distribution.resize(nbins);
    double mean{0.0};
    double variance{0.0};
    double sentropy {0.0};

    histogram_generator(nbins, function, intensity_distribution, probability_distribution, mean, variance, sentropy);

    std::cout << "Finished computing some basic speckle statistics: " << std::endl;
    std::cout <<  " | Mean = " << mean << " | Variance = " << variance << std::endl;

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Histogram and basic statistic generation finished, moving to autocorrelation." << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    //---------------------------------------------------------------------------

    //AUTOCORRELATION
    std::vector<Points> displacements {displacement_generator(pixelshift)};

    //---------------------------------------------------------------------------

    //Autocorrelation using SSD.
    if (ssd == true){

        std::cout << "(SSD) Starting autocorrelation workflow using SSD..." << std::endl;

        autocorrelator_ssd(displacements, function, subpixel);

        if(save_autocorrelators == true){
            file_loader_points(displacements, ssdautocorrelator_filename);
        }

        std::cout << "(SSD) Generating Radial Gradients..." << std::endl;

        int optimalsectors {optimal_sector_count(pixelshift, safety, fallback)};
        double watershed_radius {0.0};

        std::vector<Points> autograd_r {autocorrelation_grad_r(displacements, pixelshift)};

        std::vector<Points> autograd_2r {autocorrelation_grad_r(autograd_r, pixelshift)};

        if(save_autocorrelators == true){
            file_loader_points(autograd_r, ssdautogradr_filename);
        }

        std::cout << "(SSD) Computing Autocorrelation Diagnostics..." << std::endl;

        std::vector<BinnedPoint> watershed_surface_grad {
            watershed_gradient(autograd_r, autograd_2r, displacements, pixelshift, optimalsectors, watershed_radius)
        };

        if(save_autocorrelators == true){
            file_loader_binned_points(watershed_surface_grad, ssdwatershedgrad_filename);
        }

        double auto_peaksharp {autocorrelation_peak(autograd_r, pixelshift)};

        double peak_max {0.0};
        double peak_var {0.0};

        autocorrelation_peak_comp(autograd_r, autograd_2r, optimalsectors, pixelshift, displacements, peak_max, peak_var);

        std::cout << "(SSD) Diagnostics: " << std::endl;
        std::cout << " | Watershed Radius = " << watershed_radius
                  << " | Peak Sharpness = " << auto_peaksharp
                  << " | Autocorrelation Peak Comparison = " << peak_max
                  << " | Autocorrelation Standard Deviation Comparison = " << peak_var << " |" << std::endl;

        std::cout << "(SSD) Autocorrelation Workflow Finished." << std::endl;

    }

    //---------------------------------------------------------------------------

    //Autocorrelation using SCC.
    if (scc == true){

        std::cout << "(SCC) Starting autocorrelation workflow using SCC..." << std::endl;

        autocorrelator_scc(displacements, function, subpixel);

        if(save_autocorrelators == true){
            file_loader_points(displacements, sccautocorrelator_filename);
        }

        std::cout << "(SCC) Generating Radial Gradients..." << std::endl;

        int optimalsectors {optimal_sector_count(pixelshift, safety, fallback)};
        double watershed_radius {0.0};

        std::vector<Points> autograd_r {autocorrelation_grad_r(displacements, pixelshift)};

        std::vector<Points> autograd_2r {autocorrelation_grad_r(autograd_r, pixelshift)};

        if(save_autocorrelators == true){
            file_loader_points(autograd_r, sccautogradr_filename);
        }

        std::cout << "(SCC) Computing Autocorrelation Diagnostics..." << std::endl;

        std::vector<BinnedPoint> watershed_surface_grad {
            watershed_gradient(autograd_r, autograd_2r, displacements, pixelshift, optimalsectors, watershed_radius)
        };

        if(save_autocorrelators == true){
            file_loader_binned_points(watershed_surface_grad, sccwatershedgrad_filename);
        }

        double auto_peaksharp {autocorrelation_peak(autograd_r, pixelshift)};

        double peak_max {0.0};
        double peak_var {0.0};

        autocorrelation_peak_comp(autograd_r, autograd_2r, optimalsectors, pixelshift, displacements, peak_max, peak_var);

        std::cout << "(SCC) Diagnostics: " << std::endl;
        std::cout << " | Watershed Radius = " << watershed_radius
                  << " | Peak Sharpness = " << auto_peaksharp
                  << " | Autocorrelation Peak Comparison = " << peak_max
                  << " | Autocorrelation Standard Deviation Comparison = " << peak_var << " |" << std::endl;

        std::cout << "(SCC) Autocorrelation Workflow Finished." << std::endl;

    }

    //---------------------------------------------------------------------------

    //Autocorrelation using NSSD.
    if (nssd == true){

        std::cout << "(NSSD) Starting autocorrelation workflow using NSSD..." << std::endl;

        autocorrelator_nssd(displacements, function, subpixel);

        if(save_autocorrelators == true){
            file_loader_points(displacements, nssdautocorrelator_filename);
        }

        std::cout << "(NSSD) Generating Radial Gradients..." << std::endl;

        int optimalsectors {optimal_sector_count(pixelshift, safety, fallback)};
        double watershed_radius {0.0};

        std::vector<Points> autograd_r {autocorrelation_grad_r(displacements, pixelshift)};

        std::vector<Points> autograd_2r {autocorrelation_grad_r(autograd_r, pixelshift)};

        if(save_autocorrelators == true){
            file_loader_points(autograd_r, nssdautogradr_filename);
        }

        std::cout << "(NSSD) Computing Autocorrelation Diagnostics..." << std::endl;

        std::vector<BinnedPoint> watershed_surface_grad {
            watershed_gradient(autograd_r, autograd_2r, displacements, pixelshift, optimalsectors, watershed_radius)
        };

        if(save_autocorrelators == true){
            file_loader_binned_points(watershed_surface_grad, nssdwatershedgrad_filename);
        }

        double auto_peaksharp {autocorrelation_peak(autograd_r, pixelshift)};

        double peak_max {0.0};
        double peak_var {0.0};

        autocorrelation_peak_comp(autograd_r, autograd_2r, optimalsectors, pixelshift, displacements, peak_max, peak_var);

        std::cout << "(NSSD) Diagnostics: " << std::endl;
        std::cout << " | Watershed Radius = " << watershed_radius
                  << " | Peak Sharpness = " << auto_peaksharp
                  << " | Autocorrelation Peak Comparison = " << peak_max
                  << " | Autocorrelation Standard Deviation Comparison = " << peak_var << " |" << std::endl;

        std::cout << "(NSSD) Autocorrelation Workflow Finished." << std::endl;

    }

    //---------------------------------------------------------------------------

    //Autocorrelation using ZSSD.
    if (zssd == true){

        std::cout << "(ZSSD) Starting autocorrelation workflow using ZSSD..." << std::endl;

        autocorrelator_zssd(displacements, function, subpixel);

        if(save_autocorrelators == true){
            file_loader_points(displacements, zssdautocorrelator_filename);
        }

        std::cout << "(ZSSD) Generating Radial Gradients..." << std::endl;

        int optimalsectors {optimal_sector_count(pixelshift, safety, fallback)};
        double watershed_radius {0.0};

        std::vector<Points> autograd_r {autocorrelation_grad_r(displacements, pixelshift)};

        std::vector<Points> autograd_2r {autocorrelation_grad_r(autograd_r, pixelshift)};

        if(save_autocorrelators == true){
            file_loader_points(autograd_r, zssdautogradr_filename);
        }

        std::cout << "(ZSSD) Computing Autocorrelation Diagnostics..." << std::endl;

        std::vector<BinnedPoint> watershed_surface_grad {
            watershed_gradient(autograd_r, autograd_2r, displacements, pixelshift, optimalsectors, watershed_radius)
        };

        if(save_autocorrelators == true){
            file_loader_binned_points(watershed_surface_grad, zssdwatershedgrad_filename);
        }

        double auto_peaksharp {autocorrelation_peak(autograd_r, pixelshift)};

        double peak_max {0.0};
        double peak_var {0.0};

        autocorrelation_peak_comp(autograd_r, autograd_2r, optimalsectors, pixelshift, displacements, peak_max, peak_var);

        std::cout << "(ZSSD) Diagnostics: " << std::endl;
        std::cout << " | Watershed Radius = " << watershed_radius
                  << " | Peak Sharpness = " << auto_peaksharp
                  << " | Autocorrelation Peak Comparison = " << peak_max
                  << " | Autocorrelation Standard Deviation Comparison = " << peak_var << " |" << std::endl;

        std::cout << "(ZSSD) Autocorrelation Workflow Finished." << std::endl;

    }

    //---------------------------------------------------------------------------

    //Autocorrelation using ZNCC.
    if (zncc == true){

        std::cout << "(ZNCC) Starting autocorrelation workflow using ZNCC..." << std::endl;

        autocorrelator_zncc(displacements, function, subpixel);

        if(save_autocorrelators == true){
            file_loader_points(displacements, znccautocorrelator_filename);
        }

        std::cout << "(ZNCC) Generating Radial Gradients..." << std::endl;

        int optimalsectors {optimal_sector_count(pixelshift, safety, fallback)};
        double watershed_radius {0.0};

        std::vector<Points> autograd_r {autocorrelation_grad_r(displacements, pixelshift)};

        std::vector<Points> autograd_2r {autocorrelation_grad_r(autograd_r, pixelshift)};

        if(save_autocorrelators == true){
            file_loader_points(autograd_r, znccautogradr_filename);
        }

        std::cout << "(ZNCC) Computing Autocorrelation Diagnostics..." << std::endl;

        std::vector<BinnedPoint> watershed_surface_grad {
            watershed_gradient(autograd_r, autograd_2r, displacements, pixelshift, optimalsectors, watershed_radius)
        };

        if(save_autocorrelators == true){
            file_loader_binned_points(watershed_surface_grad, znccwatershedgrad_filename);
        }

        double auto_peaksharp {autocorrelation_peak(autograd_r, pixelshift)};

        double peak_max {0.0};
        double peak_var {0.0};

        autocorrelation_peak_comp(autograd_r, autograd_2r, optimalsectors, pixelshift, displacements, peak_max, peak_var);

        std::cout << "(ZNCC) Diagnostics: " << std::endl;
        std::cout << " | Watershed Radius = " << watershed_radius
                  << " | Peak Sharpness = " << auto_peaksharp
                  << " | Autocorrelation Peak Comparison = " << peak_max
                  << " | Autocorrelation Standard Deviation Comparison = " << peak_var << " |" << std::endl;

        std::cout << "(ZNCC) Autocorrelation Workflow Finished." << std::endl;

    }

    //---------------------------------------------------------------------------

    //Autocorrelation using ZNSSD.
    if (znssd == true){

        std::cout << "(ZNSSD) Starting autocorrelation workflow using ZNSSD..." << std::endl;

        autocorrelator_znssd(displacements, function, subpixel);

        if(save_autocorrelators == true){
            file_loader_points(displacements, znssdautocorrelator_filename);
        }

        std::cout << "(ZNSSD) Generating Radial Gradients..." << std::endl;

        int optimalsectors {optimal_sector_count(pixelshift, safety, fallback)};
        double watershed_radius {0.0};

        std::vector<Points> autograd_r {autocorrelation_grad_r(displacements, pixelshift)};

        std::vector<Points> autograd_2r {autocorrelation_grad_r(autograd_r, pixelshift)};

        if(save_autocorrelators == true){
            file_loader_points(autograd_r, znssdautogradr_filename);
        }

        std::cout << "(ZNSSD) Computing Autocorrelation Diagnostics..." << std::endl;

        std::vector<BinnedPoint> watershed_surface_grad {
            watershed_gradient(autograd_r, autograd_2r, displacements, pixelshift, optimalsectors, watershed_radius)
        };

        if(save_autocorrelators == true){
            file_loader_binned_points(watershed_surface_grad, znssdwatershedgrad_filename);
        }

        double auto_peaksharp {autocorrelation_peak(autograd_r, pixelshift)};

        double peak_max {0.0};
        double peak_var {0.0};

        autocorrelation_peak_comp(autograd_r, autograd_2r, optimalsectors, pixelshift, displacements, peak_max, peak_var);

        std::cout << "(ZNSSD) Diagnostics: " << std::endl;
        std::cout << " | Watershed Radius = " << watershed_radius
                  << " | Peak Sharpness = " << auto_peaksharp
                  << " | Autocorrelation Peak Comparison = " << peak_max
                  << " | Autocorrelation Standard Deviation Comparison = " << peak_var << " |" << std::endl;

        std::cout << "(ZNSSD) Autocorrelation Workflow Finished." << std::endl;

    }

    std::cout << "Finished all autocorrelation workflows. " << std::endl;

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Autocorrelation finished, moving to diagnostics." << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    //---------------------------------------------------------------------------

    //Generate the MIG (mean intensity gradient).
    double MIG {mig(grad_Mag)};

    std::cout << "Single Valued Diagnostics: " << std::endl;
    std::cout << " | MIG = " << MIG << " | Shannon Entropy = " << sentropy << " |" << std::endl;

    std::vector<Points> sssig_heatmap {
        sssig_heatmap_generator(ss_size, st_size, grad_Mag, subpixel, width, height)
    };

    file_loader_points(sssig_heatmap, sssig_filename);

    double deltamap_mean {0.0};
    double deltamap_std {0.0};
    std::vector<Points> sssig_deltamap = sssig_heatmap;

    sssig_deltamap_generator(sssig_deltamap, deltamap_mean, deltamap_std, MIG);

    file_loader_points(sssig_deltamap, sssigdelta_filename);

    std::cout << "Finished SSSIG Related Diagnostics: " << std::endl;
    std::cout << " | Delta Map Mean = " << deltamap_mean << " | Delta Map Standard Deviation = " << deltamap_std << std::endl;

    //---------------------------------------------------------------------------

    //End benchmarking timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "PROGRAM FINISHED." << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    std::cout << "Program finished analyzing and generating a "<< ((width * height) / 1000000) << "MPx image in: " << duration_ms.count()/1000 << "s" << std::endl;
    
}