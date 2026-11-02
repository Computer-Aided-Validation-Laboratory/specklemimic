#include <iostream>
#include <chrono>
#include <vector>
#include "loss.hpp"
#include "study.hpp"


int main(){

    //Start the total runtime timer.
    auto start = std::chrono::high_resolution_clock::now();

    //---------------------------------------------------------------------------

    //GROUND TRUTH CONFIGURATIONS.
    //Each one is reconstructed with every loss switched on below. {folder name, speckle size (px), ratio}
    std::vector<PatternConfig> configs {
        {"baseline_20px",     20.0, 0.7},   //Small, dense speckles.
        {"large_sparse_60px", 60.0, 0.3},   //Polar opposite: large, sparse speckles.
    };

    //LOSS FUNCTION LEVERS.
    bool run_graylevel        {true};   //Relative gray level MSE.
    bool run_sssig_mse        {true};   //Relative SSSIG MSE.
    bool run_sssig_var        {true};   //Relative SSSIG MSE + residual variance.
    bool run_autocorr_mse     {true};   //Relative ZNSSD autocorrelation MSE. (Expensive)
    bool run_autocorr_var     {true};   //Relative ZNSSD autocorrelation MSE + residual variance. (Expensive)
    bool run_autocorr_scalars {true};   //Relative autocorrelation scalars. (Expensive)

    //STUDY SETTINGS.
    StudySettings settings;
    settings.study_folder = "loss_study";   //Runs go to <study_folder>/<configuration>/<loss>/

    //Grid. Fixed for every run so all patterns live on the same array.
    settings.loss.width = 500;
    settings.loss.height = 500;
    settings.loss.subpixel = 0.5;

    //SSSIG.
    settings.loss.ss_size = 20;     //Subset size (px).
    settings.loss.ss_step = 20;     //Step size (px).

    //Autocorrelation (ZNSSD).
    settings.loss.pixelshift = 20;  //Window (px). Must be larger than the speckles to see the whole peak.
    settings.loss.safety = 1.25;    //Safety factor for the watershed sector count.
    settings.loss.fallback = 40;    //Fallback watershed sector count.

    //Particle swarm.
    settings.optimizer.iterations = 100;
    settings.optimizer.particles = 20;
    settings.optimizer.inertia = 0.729;
    settings.optimizer.acceleration1 = 1.49445;
    settings.optimizer.acceleration2 = 1.49445;
    settings.optimizer.seed = 42;

    //Search bounds {speckle size (px), ratio}.
    settings.optimizer.size_lo = 3.0;
    settings.optimizer.size_hi = 100.0;
    settings.optimizer.ratio_lo = 0.001;
    settings.optimizer.ratio_hi = 1.0;

    //Budget for the autocorrelation losses (about 43s per evaluation serially at pixelshift 20).
    settings.autocorr_iterations = 100;
    settings.autocorr_particles = 20;

    //---------------------------------------------------------------------------

    std::vector<LossType> losses;
    if(run_graylevel)        losses.push_back(LossType::GrayLevel);
    if(run_sssig_mse)        losses.push_back(LossType::SSSIGMSE);
    if(run_sssig_var)        losses.push_back(LossType::SSSIGVar);
    if(run_autocorr_mse)     losses.push_back(LossType::AutocorrMSE);
    if(run_autocorr_var)     losses.push_back(LossType::AutocorrVar);
    if(run_autocorr_scalars) losses.push_back(LossType::AutocorrScalars);

    std::vector<RunRecord> records {run_loss_study(configs, losses, settings)};

    //End the total runtime timer.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = end - start;

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "STUDY FINISHED: " << records.size() << " run(s)" << std::endl;
    for(const RunRecord& record : records){
        std::cout << " | " << record.config << " | " << record.loss << " | " << record.runtime_s << "s |" << std::endl;
    }
    std::cout << "Total runtime: " << total.count() << "s" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    return 0;

}
