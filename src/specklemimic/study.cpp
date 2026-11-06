/**
 * @file study.cpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief TEMPORARY: runner for the loss function study (implementation).
 *
 * @version 0.2
 * @date 2026-11-05
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "study.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "optimizers/pso.hpp"
#include "specklegen/generator.hpp"
#include "specklegen/helpers.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

static const char* SUMMARY_HEADER {
    "config,loss,true_size,true_ratio,best_size,best_ratio,best_loss,graylevel_relmse,runtime_s,evaluations,iterations,particles\n"
};

static void write_record(std::ofstream& file, const RunRecord& record)
{
    file << std::setprecision(10)
         << record.config << "," << record.loss << ","
         << record.true_size << "," << record.true_ratio << ","
         << record.best_size << "," << record.best_ratio << ","
         << record.best_loss << "," << record.graylevel_relmse << ","
         << record.runtime_s << "," << record.evaluations << ","
         << record.iterations << "," << record.particles << "\n";
}

//Reads back a finished run (for resuming). False if the run is missing or incomplete.
static bool load_record(const std::string& runfolder, RunRecord& record)
{

    std::ifstream file(runfolder + "/run_summary.csv");
    if(!file || !std::filesystem::exists(runfolder + "/finalresult.csv")){
        return false;
    }

    std::string header;
    std::string row;
    std::getline(file, header);
    if(!std::getline(file, row)){
        return false;
    }

    std::vector<std::string> cells;
    std::stringstream stream(row);
    std::string cell;
    while(std::getline(stream, cell, ',')){
        cells.push_back(cell);
    }
    if(cells.size() != 12){
        return false;
    }

    try{
        record.config = cells[0];
        record.loss = cells[1];
        record.true_size = std::stod(cells[2]);
        record.true_ratio = std::stod(cells[3]);
        record.best_size = std::stod(cells[4]);
        record.best_ratio = std::stod(cells[5]);
        record.best_loss = std::stod(cells[6]);
        record.graylevel_relmse = std::stod(cells[7]);
        record.runtime_s = std::stod(cells[8]);
        record.evaluations = static_cast<size_t>(std::stoull(cells[9]));
        record.iterations = std::stoi(cells[10]);
        record.particles = static_cast<size_t>(std::stoull(cells[11]));
    }
    catch(const std::exception&){
        return false;
    }

    return true;

}

//Rewrites the table of every run finished so far, so a long study always has usable results on disk.
static void save_summary(const std::vector<RunRecord>& records, const StudySettings& settings)
{

    std::ofstream summary(settings.study_folder + "/study_summary.csv");
    summary << SUMMARY_HEADER;
    for(const RunRecord& record : records){
        write_record(summary, record);
    }

}

static RunRecord run_single(
    const PatternConfig& config,
    const LossType loss,
    const LossContext& context,
    const StudySettings& settings,
    const std::string& runfolder)
{

    const OptimizerSettings& opt {settings.optimizer};
    bool autocorr {needs_autocorrelation(loss)};
    int iterations {autocorr ? settings.autocorr_iterations : opt.iterations};
    size_t particles {autocorr ? settings.autocorr_particles : opt.particles};

    std::filesystem::create_directories(runfolder);

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "RUN: " << config.name << " | LOSS: " << loss_name(loss)
              << " | " << iterations << " iterations x " << particles << " particles" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    //Threads. Autocorrelation: one particle at a time and every thread inside the map (a map has 1681 independent displacements,
    //a swarm only 20 particles). Other losses: a few threads on the particle loop, nothing inside the evaluation.
    int total_threads {1};
#ifdef _OPENMP
    total_threads = settings.threads > 0 ? settings.threads : omp_get_num_procs();
#endif
    const int outer_threads {autocorr ? 1 : std::min(settings.cheap_threads, total_threads)};
    const int inner_threads {autocorr ? total_threads : 0};
    std::cout << "Threads: " << outer_threads << " for the particle loop";
    if(inner_threads > 0){
        std::cout << ", " << inner_threads << " inside every evaluation";
    }
    std::cout << std::endl;

    //The optimizer may call this from several threads, so it must not write to shared state.
    auto objective = [&](const double* x, size_t d) -> double{
#ifdef _OPENMP
        //Team size of the parallel loops inside this evaluation. They only open a team when the particle loop has one thread.
        if(inner_threads > 0){
            omp_set_num_threads(inner_threads);
        }
#endif
        return params_loss(loss, x, d, context);
    };

    std::vector<double> lo {opt.size_lo, opt.ratio_lo};
    std::vector<double> hi {opt.size_hi, opt.ratio_hi};

#ifdef _OPENMP
    omp_set_num_threads(outer_threads);
#endif

    auto start = std::chrono::high_resolution_clock::now();

    PSOResult result{spsoptimzer(
        objective,
        iterations,
        particles,
        2,
        lo,
        hi,
        false,
        opt.inertia,
        opt.acceleration1,
        opt.acceleration2,
        opt.seed,
        true,           //Verbose
        runfolder,      //Optimizer files go to <runfolder>/optimizer/
        true            //Save loss history
    )};

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;

#ifdef _OPENMP
    omp_set_num_threads(total_threads);
#endif

    //Rebuild and save the best pattern, next to a copy of the ground truth so the run folder stands alone.
    Field reconstructed {patternreconstructor_params(
        result,
        context.settings.width,
        context.settings.height,
        context.settings.subpixel
    )};
    file_loader(context.ground_truth, runfolder + "/groundtruth.csv");
    file_loader(reconstructed, runfolder + "/finalresult.csv");

    RunRecord record;
    record.config = config.name;
    record.loss = loss_name(loss);
    record.true_size = config.speckleSize;
    record.true_ratio = config.ratio;
    record.best_size = result.best_pos[0];
    record.best_ratio = result.best_pos[1];
    record.best_loss = result.best_val;
    record.graylevel_relmse = graylevel_mse(context.ground_truth, reconstructed);
    record.runtime_s = runtime.count();
    record.evaluations = static_cast<size_t>(iterations) * particles;   //Exact: every particle is evaluated once per iteration.
    record.iterations = iterations;
    record.particles = particles;

    std::ofstream summary(runfolder + "/run_summary.csv");
    summary << SUMMARY_HEADER;
    write_record(summary, record);

    std::cout << " | Best Speckle Size = " << record.best_size << "px (True = " << config.speckleSize << "px)"
              << " | Best Ratio = " << record.best_ratio << " (True = " << config.ratio << ")"
              << " | Runtime = " << record.runtime_s << "s |" << std::endl;

    return record;

}

std::vector<RunRecord> run_loss_study(
    const std::vector<PatternConfig>& configs,
    const std::vector<LossType>& losses,
    const StudySettings& settings)
{

    std::vector<RunRecord> records;

    //Only pay for the ground truth diagnostics that some loss actually uses.
    bool need_sssig {false};
    bool need_autocorr {false};
    for(LossType loss : losses){
        need_sssig = need_sssig || loss == LossType::SSSIGMSE || loss == LossType::SSSIGVar;
        need_autocorr = need_autocorr || needs_autocorrelation(loss);
    }

    std::filesystem::create_directories(settings.study_folder);

    for(const PatternConfig& config : configs){

        std::string config_folder {settings.study_folder + "/" + config.name};
        std::filesystem::create_directories(config_folder);

        std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "CONFIGURATION: " << config.name << " | Speckle Size = " << config.speckleSize
                  << "px | Ratio = " << config.ratio << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

        //Ground truth and its diagnostics are built once per configuration.
        //The Halton table has to cover the most speckles the search bounds allow.
        LossSettings ls {settings.loss};
        ls.min_speckle_size = settings.optimizer.size_lo;
        ls.max_ratio = settings.optimizer.ratio_hi;
        Field background {background_generator(ls.width, ls.height, ls.subpixel)};
        Field ground_truth {generate_pattern(config.speckleSize, config.ratio, background.grid, ls.width, ls.height)};
        file_loader(ground_truth, config_folder + "/groundtruth.csv");

        std::cout << "Ground truth has " << speckle_numbers(ls.width, ls.height, config.speckleSize, config.ratio)
                  << " speckles. Computing its diagnostics..." << std::endl;
        LossContext context {build_loss_context(ground_truth, ls, need_sssig, need_autocorr)};

        for(LossType loss : losses){
            std::string runfolder {config_folder + "/" + loss_name(loss)};

            //A run that is already finished is read back, not repeated.
            RunRecord finished;
            if(settings.resume && load_record(runfolder, finished)){
                std::cout << "RUN: " << config.name << " | LOSS: " << loss_name(loss) << " | already finished, skipped." << std::endl;
                records.push_back(finished);
            }
            else{
                records.push_back(run_single(config, loss, context, settings, runfolder));
            }

            save_summary(records, settings);
        }

    }

    //Every run in one table (also saved after every single run above).
    save_summary(records, settings);
    std::cout << "Saved study summary to: " << settings.study_folder << "/study_summary.csv" << std::endl;

    return records;

}
