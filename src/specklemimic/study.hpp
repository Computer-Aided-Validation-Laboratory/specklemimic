/**
 * @file study.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief TEMPORARY: runner for the loss function study.
 *
 * Runs every chosen parameter loss on every chosen ground truth configuration, one after another,
 * and saves each run to its own folder: <study_folder>/<configuration>/<loss>/.
 * This file (and study.cpp) only exists for the loss benchmarking study and will be removed afterwards.
 *
 * @version 0.2
 * @date 2026-11-05
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include "loss.hpp"


/**
 * @brief One ground truth speckle pattern configuration.
 */
struct PatternConfig {
    std::string name;       //Folder name for this configuration.
    double speckleSize;     //Ground truth speckle size (px).
    double ratio;           //Ground truth black to white ratio.
};


/**
 * @brief Particle swarm settings and search bounds.
 */
struct OptimizerSettings {
    int iterations {100};
    size_t particles {20};
    double inertia {0.729};
    double acceleration1 {1.49445};
    double acceleration2 {1.49445};
    unsigned long long seed {42};   //Same seed for every run, so every loss starts from the same swarm.
    double size_lo {3.0};
    double size_hi {100.0};
    double ratio_lo {0.001};
    double ratio_hi {1.0};
};


/**
 * @brief Everything the study needs.
 *
 * The autocorrelation losses get their own iteration and particle budget because they are far more expensive.
 */
struct StudySettings {
    std::string study_folder {"loss_study"};
    LossSettings loss;
    OptimizerSettings optimizer;
    int autocorr_iterations {100};
    size_t autocorr_particles {20};

    //Threads. The autocorrelation losses run one particle at a time with all threads inside the map (the best split measured),
    //the cheap losses use a few threads for the particle loop (more does not help, they are limited by memory).
    int threads {0};            //Total threads for the autocorrelation maps. 0 = all the machine has.
    int cheap_threads {4};      //Particle threads for the gray level and SSSIG losses.
    bool resume {true};         //Skip runs that already have a run_summary.csv, so a stopped study can be started again.
};


/**
 * @brief Summary of one optimization run.
 */
struct RunRecord {
    std::string config;
    std::string loss;
    double true_size {0.0};
    double true_ratio {0.0};
    double best_size {0.0};
    double best_ratio {0.0};
    double best_loss {0.0};
    double graylevel_relmse {0.0};  //Relative gray level MSE of the reconstruction, comparable across losses.
    double runtime_s {0.0};         //Optimizer wall time.
    size_t evaluations {0};
    int iterations {0};
    size_t particles {0};
};


/**
 * @brief Loss Study Runner
 *
 * For every configuration: builds the ground truth and its diagnostics once, then runs the PSO with every loss.
 * Saves groundtruth.csv, finalresult.csv, optimizer/ (history, domain, loss) and run_summary.csv per run,
 * and study_summary.csv with every run in the study folder.
 *
 * @param configs Ground truth configurations.
 * @param losses Losses to run.
 * @param settings Study settings.
 * @return std::vector<RunRecord> One record per run.
 */
std::vector<RunRecord> run_loss_study(
    const std::vector<PatternConfig>& configs,
    const std::vector<LossType>& losses,
    const StudySettings& settings);
