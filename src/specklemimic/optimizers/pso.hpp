#pragma once
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdint>
#include "specklegen/helpers.hpp"

/**
 * @brief Structure for a Particle Swarm with Global Swarm Topology
 * 
 * Contains two size_t variables: n and d which contain the number of particles and problem dimensionality respectively.
 * 
 * Contains double vectors:
 * pos --> Position vectors for all particles. Flattened index = n_ * d + d_
 * vel --> Velocity vectors for all particles. Flattened index = n_ * d + d_
 * pbest_pos --> Best position vectors for all particles. Flattened index = n_ * d + d_
 * pbest_val --> Best function value for each particle. Indexed as n_
 * gbest_pos --> Global best position vector. Indexed as d_
 * gbest_val --> Global best value. 
 * Here, n_ and d_ are meant to be indices you can loop over to extract particle number and coordinate dimension respectively.
 * 
 * Contains pointers:
 * x --> Acts as an easy way to go to the first coordinate of a particle without much indexing.
 * y --> Similar to x but for velocity.
 * 
 * Used extensively in the PSO optimizer to describe the particle swarm state.
 */
struct GBSwarmTopology {
    size_t n;
    size_t d;

    std::vector<double> pos;
    std::vector<double> vel;
    std::vector<double> pbest_pos;
    std::vector<double> pbest_val;
    std::vector<double> gbest_pos;
    double gbest_val = std::numeric_limits<double>::infinity();

    GBSwarmTopology(size_t n_, size_t d_)
        : n(n_), d(d_),
          pos(n_*d_), vel(n_*d_),
          pbest_pos(n_*d_),
          pbest_val(n_, std::numeric_limits<double>::infinity()),
          gbest_pos(d_) {}

    double* x(size_t i) { return &pos[i*d]; }
    double* v(size_t i) { return &vel[i*d]; }
};


/**
 * @brief Result from PSO
 * 
 * Contains an array to store the minima locations and the smallest obtained functional value.
 */
struct PSOResult {
    std::vector<double> best_pos;   //Best position in PHYSICAL coordinates.
    double best_val;                //Objective value at best_pos.
};


/**
 * @brief Swarm Initializer
 * 
 * Takes the swarm state and initializes it.
 * Uses random numbers to initialize a swarm that follows the global best topology.
 * 
 * @param swarmState Input swarm state, overwritten to be initialized.
 * @param gen Random number generator for initialization (seeded ideally)
 */
void gbinitializer(GBSwarmTopology& swarmState, std::mt19937_64& gen);


/**
 * @brief Boundary Function
 * 
 * Deals with the boundary in the PSO iteration loop.
 * Reflects the particle back so it always bounces around in the unit hypercube.
 * 
 * @param x Particle position.
 * @param v Particle velocity.
 */
void reflect_unit(double& x, double& v);


/**
 * @brief Standard Particle Swarm Optimizer
 * 
 * This function runs a standard particle swarm optimization on an input loss function. 
 * It needs external functions and structures to work properly.
 * The external functions are needed to initialize the swarm and deal with clamping to a domain.
 * 
 * @tparam ObjectiveFunction 
 * @param f Input objective function. Needs to be shaped like (double array, int dimension)
 * @param it Iterations to run the loop for.
 * @param n Number of particles.
 * @param d Number of dimensions.
 * @param lo Array for lower bounds. (These should be physical for proper resizing)
 * @param hi Array for upper bounds. (Should be physical for proper resizing)
 * @param default_bounds Flag to turn on if the problem is purely in the unit hypercube.
 * @param inertia Inertia term of the PSO. Default is 0.729 to ensure stability.
 * @param acceleration1 Acceleration (Cognitive Influence) term of PSO. Defaults to 1.49445 (stability).
 * @param acceleration2 Acceleration (Social Influence) term of PSO. Defaults to 1.49445 (stability).
 * @param seed Seed for reproducibility.
 * @param verbose More print statements!
 * @param optimizer_name Run folder the optimizer files are saved in (inside its optimizer/ subfolder).
 *                       2D runs also save history.csv (iteration, x, y, f for every particle) and domain.csv (the bounds) when this is set.
 *                       If empty, only the loss history is saved, to ./optimizer/.
 * @param loss_history Save the global best value of every iteration to <optimizer_name>/optimizer/loss.csv (columns: iteration, f).
 * @return PSOResult 
 */
template <typename ObjectiveFunction>
PSOResult spsoptimzer(
    ObjectiveFunction f,
    int it,
    size_t n,
    size_t d,
    const std::vector<double>& lo,
    const std::vector<double>& hi,
    bool default_bounds,
    double inertia = 0.729,
    double acceleration1 = 1.49445,
    double acceleration2 = 1.49445,
    unsigned long long seed = 42,
    bool verbose = true,
    const std::string& optimizer_name = "",
    bool loss_history = false)
{

    std::cout << " | Starting Particle Swarm Optimization" << std::endl;

    //Some control flows for the input.
    if(n == 0 || d == 0) throw std::invalid_argument("Number of particles and dimensions must not be 0!");
    if(it < 0) throw std::invalid_argument("Iterations must be greater than 0");

    //Create a lo_, hi_ array that is used in this optimizer.
    //It is overwritten with user passed defaults.
    std::vector<double> lo_(d, 0.0), hi_(d, 1.0);
    if(!default_bounds){
        //Size mismatch arguments.
        if(lo.size() != d || hi.size() != d)
            throw std::invalid_argument("Lower bound and upper bound arrays must have d entries");
        
        //Ensure that lower bounds are actually lower than higher bounds.
        for(size_t k = 0; k < d; k++){
            if(!(lo[k] < hi[k])) throw std::invalid_argument("Need the lower bound to be smaller than the higher bound!");
        }
        lo_ = lo;
        hi_ = hi;
    }

    //Create the particle swarm, seed and random number generator.
    GBSwarmTopology swarmState(n,d);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    //Initialize the swarm.
    gbinitializer(swarmState, gen);

    //All optimizer files go in <optimizer_name>/optimizer/, so they sit next to the rest of the run's data.
    //With no name they go in ./optimizer/.
    std::filesystem::path optimizerDir = std::filesystem::path(optimizer_name) / "optimizer";

    //Setup history and domain tracking for 2D if an optimizer name is provided
    bool trackHistory = (d == 2 && !optimizer_name.empty());
    std::string historyFilename = "";
    if (trackHistory) {
        std::filesystem::create_directories(optimizerDir);
        historyFilename = (optimizerDir / "history.csv").string();
        std::ofstream histFile(historyFilename);
        histFile << "iteration,x,y,f\n";

        std::ofstream domFile(optimizerDir / "domain.csv");
        domFile << "lo_x,lo_y,hi_x,hi_y\n";
        domFile << lo_[0] << "," << lo_[1] << "," << hi_[0] << "," << hi_[1] << "\n";
    }

    //Setup loss history tracking. Stores the global best value after every iteration.
    std::vector<double> lossHistory;
    if (loss_history) {
        lossHistory.reserve(static_cast<size_t>(it));
    }

    //Iteration Loop.
    for(int i = 0; i < it; i++){
        auto iter_start = std::chrono::high_resolution_clock::now();

        //One slot per particle for iteration tracking to keep things thread safe.
        std::vector<Function> iterationPoints;
        if (trackHistory) {
            iterationPoints.resize(n);
        }

        //Particles are independent within an iteration, so they are evaluated in parallel.
        //Static scheduling is being used here for some more robustness.
        #pragma omp parallel for schedule(static, 1)
        for(size_t n_ = 0; n_ < n; n_++){

            //Every (iteration, particle) pair gets its own generator, seeded from the run seed.
            //This is thread-safe, and the numbers drawn do not depend on how many threads run or which one gets the particle.
            std::seed_seq seed_sequence{
                static_cast<std::uint32_t>(seed),
                static_cast<std::uint32_t>(seed >> 32),
                static_cast<std::uint32_t>(i),
                static_cast<std::uint32_t>(n_)
            };
            std::mt19937_64 gen_p(seed_sequence);
            std::uniform_real_distribution<double> dist_p(0.0, 1.0);

            //Physical coordinates of this particle, private to this thread.
            std::vector<double> xphys(d);

            //Loop over dimensions.
            for(size_t d_ = 0; d_ < d; d_++){

                size_t index {n_ * d + d_};

                double cognitive_influence = swarmState.pbest_pos[index] - swarmState.pos[index];
                double social_influence = swarmState.gbest_pos[d_] - swarmState.pos[index];

                double r1 = dist_p(gen_p);
                double r2 = dist_p(gen_p);

                //Velocity Update (skipped on i == 0 as a slight burn in)
                if(i != 0){
                swarmState.vel[index] = (inertia * swarmState.vel[index])   //Inertia Term.
                + (acceleration1 * r1 * cognitive_influence)                //Cognitive Influence Term.
                + (acceleration2 * r2 * social_influence);                  //Social Influence Term.
                }

                swarmState.vel[index] = std::clamp(swarmState.vel[index], -0.15, 0.15);

                //Particle Position Update
                swarmState.pos[index] = swarmState.pos[index] + swarmState.vel[index];

                //Boundary pass. Basically, if we hit any boundaries of the unit hypercube, reflect things back.
                reflect_unit(swarmState.pos[index], swarmState.vel[index]);
            }

            //Pointers to this particle's d contiguous coordinates.
            double* pos_ptr {swarmState.x(n_)};
            double* bpos_ptr {&swarmState.pbest_pos[n_*d]};

            //Map to physical coordinates for the objective.
            for(size_t d_ = 0; d_ < d; d_++){
                xphys[d_] = lo_[d_] + pos_ptr[d_] * (hi_[d_] - lo_[d_]);
            }

            //Compute Local Best
            double f_val = f(xphys.data(), d);
            if(f_val < swarmState.pbest_val[n_]){
                swarmState.pbest_val[n_] = f_val;
                std::copy(pos_ptr, pos_ptr + d, bpos_ptr);
            }

            // Record physical coordinates and function value for history tracking
            if (trackHistory) {
                iterationPoints[n_] = {xphys[0], xphys[1], f_val};
            }
        }

        //Check for global best.
        for(size_t n_ = 0; n_ < n; n_++){
            if(swarmState.pbest_val[n_] < swarmState.gbest_val){
                swarmState.gbest_val = swarmState.pbest_val[n_];
                std::copy(&swarmState.pbest_pos[n_*d], &swarmState.pbest_pos[n_*d] + d, swarmState.gbest_pos.begin());
            }
        }

        //Record the global best value of this iteration for the loss history.
        if (loss_history) {
            lossHistory.push_back(swarmState.gbest_val);
        }

        // Append iteration history to CSV file
        if (trackHistory) {
            std::ofstream histFile(historyFilename, std::ios::app);
            for (const auto& p : iterationPoints) {
                histFile << i << "," << p.x << "," << p.y << "," << p.f << "\n";
            }
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = iter_end - iter_start;

        //Print values out each iteration.
        if(verbose && i > 0){
            std::cout << " | Iteration = " << i << " | Global Best Value = " << swarmState.gbest_val << " | Time = " << elapsed.count() << " ms |" << std::endl;
        }

    }

    //Write the loss history to a csv file: iteration in the first column, global best value in the second.
    if (loss_history) {
        std::filesystem::create_directories(optimizerDir);
        std::string lossFilename = (optimizerDir / "loss.csv").string();
        std::ofstream lossFile(lossFilename);
        lossFile << "iteration,f\n";
        for (size_t k = 0; k < lossHistory.size(); k++) {
            lossFile << k << "," << lossHistory[k] << "\n";
        }
        std::cout << " | Saved loss history to: " << lossFilename << std::endl;
    }

    //Return the best point in physical coordinates.
    PSOResult result;
    result.best_val = swarmState.gbest_val;
    result.best_pos.resize(d);
    for(size_t k = 0; k < d; k++){
        result.best_pos[k] = lo_[k] + swarmState.gbest_pos[k] * (hi_[k] - lo_[k]);
    }
    return result;
}