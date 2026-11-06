/**
 * @file loss.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Loss functions for speckle pattern reconstruction.
 *
 * Contains the objective functions passed to the optimizers and the matching pattern reconstructors.
 * Seed loss: optimizes the x, y coordinates of every speckle seed (d = 2 * speckle count).
 * Parameter losses: optimize the speckle generator parameters (speckle size, ratio) so d = 2.
 * Every loss is a relative error (normalized by the ground truth), so losses with different units can be compared.
 *
 * @version 0.2
 * @date 2026-11-05
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include "optimizers/pso.hpp"
#include "specklegen/structures.hpp"
#include "specklegen/generator.hpp"


/**
 * @brief Parameter Loss Types
 *
 * GrayLevel        --> Relative gray level MSE.
 * SSSIGMSE         --> Relative MSE of the SSSIG heatmap.
 * SSSIGVar         --> SSSIGMSE + relative variance of the SSSIG residual.
 * AutocorrMSE      --> Relative MSE of the ZNSSD autocorrelation map.
 * AutocorrVar      --> AutocorrMSE + relative variance of the autocorrelation residual.
 * AutocorrScalars  --> Sum of squared relative errors of the autocorrelation scalars.
 */
enum class LossType {
    GrayLevel,
    SSSIGMSE,
    SSSIGVar,
    AutocorrMSE,
    AutocorrVar,
    AutocorrScalars
};


/**
 * @brief Scalars describing an autocorrelation landscape.
 */
struct AutocorrScalars {
    double watershed_radius {0.0};  //W
    double peak_sharpness {0.0};    //Lambda
    double peak_comparison {0.0};   //delta Lambda
    double peak_stddev {0.0};       //sigma
};


/**
 * @brief Fixed settings shared by every parameter loss.
 *
 * Width, height and subpixel fix the grid. SSSIG and autocorrelation settings fix the diagnostics grids.
 */
struct LossSettings {
    double width {500};
    double height {500};
    double subpixel {0.5};
    int ss_size {20};       //SSSIG subset size (px).
    int ss_step {20};       //SSSIG step size (px).
    int pixelshift {20};    //Autocorrelation window (px).
    double safety {1.25};   //Safety factor for the watershed sector count.
    int fallback {40};      //Fallback watershed sector count.
    double min_speckle_size {3.0};  //Smallest speckle size (px) the optimizer can ask for.
    double max_ratio {1.0};         //Largest ratio the optimizer can ask for. Together they set how many Halton points are stored.
};


/**
 * @brief Everything a parameter loss needs, with the ground truth diagnostics computed once.
 */
struct LossContext {
    LossSettings settings;
    HaltonTable halton;                     //Seed points, computed once (see generator.hpp), read only and shared by all threads.
    Field ground_truth;                     //Ground truth pattern: the grid (stored once) and the gray values. Read only and shared by all threads.
    double truth_norm {0.0};                //Sum of the squared ground truth gray values, summed in grid order.
    std::vector<Points> truth_sssig;        //Ground truth SSSIG heatmap (empty if not needed).
    std::vector<Points> truth_autocorr;     //Ground truth ZNSSD map (empty if not needed).
    AutocorrScalars truth_scalars;          //Ground truth autocorrelation scalars.(empty if not needed)
};


/**
 * @brief Loss Name
 *
 * @param type Loss type.
 * @return std::string Short name, also used as the output folder name.
 */
std::string loss_name(LossType type);


/**
 * @brief Autocorrelation Check
 *
 * @param type Loss type.
 * @return true If the loss needs the (expensive) autocorrelation map.
 */
bool needs_autocorrelation(LossType type);


/**
 * @brief Loss Context Builder
 *
 * Stores the ground truth and computes its diagnostics so they are not recomputed every evaluation.
 *
 * @param ground_truth Ground truth pattern.
 * @param settings Fixed loss settings.
 * @param need_sssig Compute the ground truth SSSIG heatmap.
 * @param need_autocorr Compute the ground truth ZNSSD map and scalars (expensive).
 * @return LossContext Context passed to every parameter loss.
 */
LossContext build_loss_context(
    const Field& ground_truth,
    const LossSettings& settings,
    const bool need_sssig,
    const bool need_autocorr);


/**
 * @brief Pattern Generator
 *
 * Runs the speckle generator for a speckle size and ratio: count, seed, fill.
 *
 * @param speckleSize Speckle size (diameter in px).
 * @param ratio Black to white ratio.
 * @param grid Grid of the ROI.
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param table Stored Halton points to reuse. If null, or for any point beyond the table, the point is computed. The pattern is identical either way.
 * @return Field Filled speckle pattern.
 */
Field generate_pattern(
    const double speckleSize,
    const double ratio,
    const GridInfo& grid,
    const double width,
    const double height,
    const HaltonTable* table = nullptr);


/**
 * @brief Pattern Generator into an existing field
 *
 * Same as generate_pattern, but clears and fills a field that already exists, so the memory can be reused for the next pattern.
 *
 * @param pattern Field to fill. Its grid has to be set, its values are cleared first.
 * @param speckleSize Speckle size (diameter in px).
 * @param ratio Black to white ratio.
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param table Stored Halton points to reuse, see generate_pattern.
 */
void generate_pattern_into(
    Field& pattern,
    const double speckleSize,
    const double ratio,
    const double width,
    const double height,
    const HaltonTable* table = nullptr);


/**
 * @brief SSSIG Heatmap of a Pattern
 *
 * Computes the gradients of the pattern and its SSSIG heatmap (mean gradient magnitude per subset).
 *
 * @param pattern Speckle pattern.
 * @param settings Fixed loss settings.
 * @return std::vector<Points> SSSIG heatmap.
 */
std::vector<Points> compute_sssig(
    const Field& pattern,
    const LossSettings& settings);


/**
 * @brief SSSIG Heatmap of a Pattern, consuming the pattern
 *
 * Same result as compute_sssig, but the gradient magnitude is computed in place over the pattern (no copy, no gradient fields).
 * The pattern is destroyed: afterwards it holds the gradient magnitude.
 *
 * @param pattern Speckle pattern. Overwritten.
 * @param settings Fixed loss settings.
 * @return std::vector<Points> SSSIG heatmap.
 */
std::vector<Points> compute_sssig_consume(
    Field& pattern,
    const LossSettings& settings);


/**
 * @brief ZNSSD Autocorrelation Map of a Pattern
 *
 * @param pattern Speckle pattern.
 * @param settings Fixed loss settings.
 * @return std::vector<Points> Autocorrelation map over (2 * pixelshift + 1)^2 displacements.
 */
std::vector<Points> compute_autocorrelation(
    const Field& pattern,
    const LossSettings& settings);


/**
 * @brief Autocorrelation Scalars of a Map
 *
 * Watershed radius, peak sharpness, peak comparison and peak standard deviation.
 *
 * @param autocorr Autocorrelation map.
 * @param settings Fixed loss settings.
 * @return AutocorrScalars Scalars of the landscape.
 */
AutocorrScalars compute_autocorr_scalars(
    const std::vector<Points>& autocorr,
    const LossSettings& settings);


/**
 * @brief Relative Gray Level MSE
 *
 * sum((truth - candidate)^2) / sum(truth^2).
 *
 * @param ground_truth Ground truth pattern.
 * @param candidate Candidate pattern. Needs the same nx, ny as the ground truth.
 * @return double Relative mean squared error.
 */
double graylevel_mse(
    const Field& ground_truth,
    const Field& candidate);


/**
 * @brief Relative MSE of two maps
 *
 * sum((truth - candidate)^2) / sum(truth^2) over the r values of two maps on the same grid.
 *
 * @param truth Ground truth map.
 * @param candidate Candidate map.
 * @return double Relative mean squared error.
 */
double relative_mse(
    const std::vector<Points>& truth,
    const std::vector<Points>& candidate);


/**
 * @brief Relative Residual Variance of two maps
 *
 * var(truth - candidate) / var(truth) over the r values of two maps on the same grid.
 *
 * @param truth Ground truth map.
 * @param candidate Candidate map.
 * @return double Relative variance of the residual.
 */
double relative_residual_variance(
    const std::vector<Points>& truth,
    const std::vector<Points>& candidate);


/**
 * @brief Relative Scalar Error
 *
 * Sum of squared relative errors of the autocorrelation scalars, all weights set to 1.
 *
 * @param truth Ground truth scalars.
 * @param candidate Candidate scalars.
 * @return double Scalar loss.
 */
double relative_scalar_error(
    const AutocorrScalars& truth,
    const AutocorrScalars& candidate);


/**
 * @brief Seed Loss (Gray Level MSE)
 *
 * Places speckles at the seed coordinates in x and compares the filled pattern to the ground truth.
 *
 * @param x Seed coordinates laid out as x0, y0, x1, y1, ...
 * @param d Number of dimensions (2 * n).
 * @param r Radius of every speckle.
 * @param ground_truth Ground truth pattern.
 * @param background Empty background. Passed by value so every call starts from a clean grid.
 * @param subpixel Subpixel resolution.
 * @param n Number of speckles.
 * @return double Relative gray level MSE.
 */
double graylevelmse_seeds(
    const double* x,
    size_t d,
    const double r,
    const Field& ground_truth,
    Field background,
    const double subpixel,
    const size_t n);


/**
 * @brief Seed Pattern Reconstructor
 *
 * Rebuilds the pattern from the best seed coordinates found by the optimizer.
 *
 * @param result Optimizer result.
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param subpixel Subpixel resolution.
 * @param n Number of speckles.
 * @param d Number of dimensions (2 * n).
 * @param r Radius of every speckle.
 * @return Field Reconstructed pattern.
 */
Field patternreconstructor_seeds(
    const PSOResult& result,
    const int width,
    const int height,
    const double subpixel,
    const int n,
    const int d,
    const double r);


/**
 * @brief Parameter Loss
 *
 * Generates a speckle pattern from the generator parameters in x and compares it to the ground truth with the chosen loss.
 * x[0] = speckle size (diameter in px), x[1] = black to white ratio.
 * Width, height and subpixel are fixed so the grid always matches the ground truth.
 *
 * @param type Which loss to use.
 * @param x Generator parameters {speckle size, ratio}.
 * @param d Number of dimensions (2).
 * @param context Ground truth and fixed settings (see build_loss_context).
 * @return double Loss value.
 */
double params_loss(
    const LossType type,
    const double* x,
    size_t d,
    const LossContext& context);


/**
 * @brief Parameter Pattern Reconstructor
 *
 * Rebuilds the pattern from the best generator parameters found by the optimizer.
 *
 * @param result Optimizer result. best_pos = {speckle size, ratio}.
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param subpixel Subpixel resolution.
 * @return Field Reconstructed pattern.
 */
Field patternreconstructor_params(
    const PSOResult& result,
    const double width,
    const double height,
    const double subpixel);