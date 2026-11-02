#include "loss.hpp"
#include <cmath>
#include "specklegen/generator.hpp"
#include "specklegen/helpers.hpp"
#include "specklegen/diagnostics.hpp"
#include "specklegen/autocorrelator.hpp"

//Below this a ground truth value counts as zero, and relative errors fall back to absolute ones.
const double RELATIVE_FLOOR {1e-12};

std::string loss_name(LossType type)
{
    switch(type){
        case LossType::GrayLevel:       return "graylevel";
        case LossType::SSSIGMSE:        return "sssig_mse";
        case LossType::SSSIGVar:        return "sssig_var";
        case LossType::AutocorrMSE:     return "autocorr_mse";
        case LossType::AutocorrVar:     return "autocorr_var";
        case LossType::AutocorrScalars: return "autocorr_scalars";
    }
    return "unknown";
}

bool needs_autocorrelation(LossType type)
{
    return type == LossType::AutocorrMSE || type == LossType::AutocorrVar || type == LossType::AutocorrScalars;
}

LossContext build_loss_context(
    const std::vector<Function>& ground_truth,
    const LossSettings& settings,
    const bool need_sssig,
    const bool need_autocorr)
{

    LossContext context;
    context.settings = settings;
    context.ground_truth = ground_truth;
    context.background = background_generator(settings.width, settings.height, settings.subpixel);

    //Enough Halton points for the most speckles the optimizer bounds can ask for.
    //Gray values and grid layout for the loss that only needs gray values.
    //The sum of squares is accumulated in the same order graylevel_mse uses, so the normalisation is the same number.
    context.grid = grid_info(context.background);
    context.truth_f.resize(ground_truth.size());
    for(size_t k = 0; k < ground_truth.size(); k++){
        context.truth_f[k] = ground_truth[k].f;
        context.truth_norm += ground_truth[k].f * ground_truth[k].f;
    }

    context.halton = generate_halton_table(speckle_numbers(settings.width, settings.height, settings.min_speckle_size, settings.max_ratio));

    if(need_sssig == true){
        context.truth_sssig = compute_sssig(ground_truth, settings);
    }

    if(need_autocorr == true){
        context.truth_autocorr = compute_autocorrelation(ground_truth, settings);
        context.truth_scalars = compute_autocorr_scalars(context.truth_autocorr, settings);
    }

    return context;

}

//---------------------------------------------------------------------------
//DIAGNOSTICS. Same machinery as the speckle generator (gen.cpp).

std::vector<Function> generate_pattern(
    const double speckleSize,
    const double ratio,
    std::vector<Function> background,
    const double width,
    const double height,
    const HaltonTable* table)
{

    //Same pipeline as the generator: count, seed, fill. A count of 0 just leaves an empty pattern.
    int speckleCount {speckle_numbers(width, height, speckleSize, ratio)};

    //With a table the stored points are reused, otherwise they are computed. The seeds are the same either way.
    std::vector<Points> seeds {table == nullptr
        ? generate_speckles_random(width, height, speckleCount, speckleSize, false)
        : seeds_from_table(*table, width, height, speckleCount, speckleSize)};

    speckle_filler(background, seeds);

    return background;

}

std::vector<Points> compute_sssig(
    const std::vector<Function>& pattern,
    const LossSettings& settings)
{

    //Gradient vectors take their grid from a background.
    std::vector<Function> grad_X {background_generator(settings.width, settings.height, settings.subpixel)};
    std::vector<Function> grad_Y {grad_X};
    std::vector<Function> grad_Mag {grad_X};

    gradient_x(pattern, grad_X);
    gradient_y(pattern, grad_Y);
    gradient_mag(grad_X, grad_Y, grad_Mag);

    return sssig_heatmap_generator(
        settings.ss_size,
        settings.ss_step,
        grad_Mag,
        settings.subpixel,
        static_cast<int>(settings.width),
        static_cast<int>(settings.height)
    );

}

std::vector<Points> compute_autocorrelation(
    const std::vector<Function>& pattern,
    const LossSettings& settings)
{

    std::vector<Points> displacements {displacement_generator(settings.pixelshift)};
    autocorrelator_znssd(displacements, pattern, settings.subpixel);

    return displacements;

}

AutocorrScalars compute_autocorr_scalars(
    const std::vector<Points>& autocorr,
    const LossSettings& settings)
{

    AutocorrScalars scalars;

    int optimalsectors {optimal_sector_count(settings.pixelshift, settings.safety, settings.fallback)};

    std::vector<Points> autograd_r {autocorrelation_grad_r(autocorr, settings.pixelshift)};
    std::vector<Points> autograd_2r {autocorrelation_grad_r(autograd_r, settings.pixelshift)};

    watershed_gradient(autograd_r, autograd_2r, autocorr, settings.pixelshift, optimalsectors, scalars.watershed_radius);

    scalars.peak_sharpness = autocorrelation_peak(autograd_r, settings.pixelshift);

    autocorrelation_peak_comp(autograd_r, autograd_2r, optimalsectors, settings.pixelshift, autocorr, scalars.peak_comparison, scalars.peak_stddev);

    return scalars;

}

//---------------------------------------------------------------------------
//RELATIVE ERRORS.

double graylevel_mse(
    const std::vector<Function>& ground_truth,
    const std::vector<Function>& candidate)
{

    //Loop and compute the relative gray level MSE.
    int nx = candidate[0].nx;
    int ny = candidate[0].ny;

    double loss {0.0};
    double norm {0.0};

    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){

            int index = i * ny + j;
            loss += (ground_truth[index].f - candidate[index].f) * (ground_truth[index].f - candidate[index].f);
            norm += ground_truth[index].f * ground_truth[index].f;

        }
    }

    if(norm < RELATIVE_FLOOR){
        return loss / (nx * ny);
    }

    return loss / norm;

}

double relative_mse(
    const std::vector<Points>& truth,
    const std::vector<Points>& candidate)
{

    double loss {0.0};
    double norm {0.0};

    for(size_t k = 0; k < truth.size(); k++){
        loss += (truth[k].r - candidate[k].r) * (truth[k].r - candidate[k].r);
        norm += truth[k].r * truth[k].r;
    }

    if(norm < RELATIVE_FLOOR){
        return loss / truth.size();
    }

    return loss / norm;

}

double relative_residual_variance(
    const std::vector<Points>& truth,
    const std::vector<Points>& candidate)
{

    double n {static_cast<double>(truth.size())};

    //Means of the residual and of the ground truth.
    double residual_mean {0.0};
    double truth_mean {0.0};
    for(size_t k = 0; k < truth.size(); k++){
        residual_mean += truth[k].r - candidate[k].r;
        truth_mean += truth[k].r;
    }
    residual_mean /= n;
    truth_mean /= n;

    //Variances of the residual and of the ground truth.
    double residual_var {0.0};
    double truth_var {0.0};
    for(size_t k = 0; k < truth.size(); k++){
        double residual {truth[k].r - candidate[k].r - residual_mean};
        double deviation {truth[k].r - truth_mean};
        residual_var += residual * residual;
        truth_var += deviation * deviation;
    }
    residual_var /= n;
    truth_var /= n;

    if(truth_var < RELATIVE_FLOOR){
        return residual_var;
    }

    return residual_var / truth_var;

}

//Squared relative error of one scalar, absolute if the ground truth is zero.
static double squared_relative_error(const double truth, const double candidate)
{
    double scale {std::abs(truth) < RELATIVE_FLOOR ? 1.0 : truth};
    double error {(truth - candidate) / scale};
    return error * error;
}

double relative_scalar_error(
    const AutocorrScalars& truth,
    const AutocorrScalars& candidate)
{

    //All weights (alpha_i) are 1.
    return squared_relative_error(truth.watershed_radius, candidate.watershed_radius)
         + squared_relative_error(truth.peak_sharpness, candidate.peak_sharpness)
         + squared_relative_error(truth.peak_comparison, candidate.peak_comparison)
         + squared_relative_error(truth.peak_stddev, candidate.peak_stddev);

}

//---------------------------------------------------------------------------
//SEED LOSS. Optimizes the x, y coordinates of every seed.

double graylevelmse_seeds(
    const double* x,
    size_t d,
    const double r,
    const std::vector<Function>& ground_truth,
    std::vector<Function> background,
    const double subpixel,
    const size_t n)
{

    std::vector<Points> seeds;
    seeds.resize(n);
    //Assign seed points.
    for(size_t i = 0; i < d; i += 2){
        seeds[i / 2].x = x[i];
        seeds[i / 2].y = x[i + 1];
        seeds[i / 2].r = r;
    }

    speckle_filler(background, seeds);

    return graylevel_mse(ground_truth, background);

}

std::vector<Function> patternreconstructor_seeds(
    const PSOResult& result,
    const int width,
    const int height,
    const double subpixel,
    const int n,
    const int d,
    const double r)
{

    //Create a background
    std::vector<Function> function {background_generator(width, height, subpixel)};

    //Get the seeds.
    std::vector<Points> seeds;
    seeds.resize(n);

    for(int i = 0; i < d; i+= 2){
        seeds[i / 2].x = result.best_pos[i];
        seeds[i / 2].y = result.best_pos[i + 1];
        seeds[i / 2].r = r;
    }

    speckle_filler(function, seeds);

    return function;

}

//---------------------------------------------------------------------------
//PARAMETER LOSSES. Optimize the speckle generator parameters {speckle size, ratio}.

//Relative gray level MSE on plain arrays of gray values. Same arithmetic and same order as graylevel_mse, so the same number.
static double graylevel_mse_field(const LossContext& context, const std::vector<double>& candidate)
{

    double loss {0.0};

    for(size_t index = 0; index < candidate.size(); index++){
        loss += (context.truth_f[index] - candidate[index]) * (context.truth_f[index] - candidate[index]);
    }

    if(context.truth_norm < RELATIVE_FLOOR){
        return loss / (context.grid.nx * context.grid.ny);
    }

    return loss / context.truth_norm;

}

//Gray level loss without building a full Function grid. Only the gray values are generated, in a buffer each thread keeps and reuses.
static double graylevel_params_loss(const double* x, const LossContext& context)
{

    const LossSettings& settings {context.settings};

    int speckleCount {speckle_numbers(settings.width, settings.height, x[0], x[1])};
    std::vector<Points> seeds {seeds_from_table(context.halton, settings.width, settings.height, speckleCount, x[0])};

    //One buffer per thread, allocated the first time and then only cleared. The gray values are 8 bytes per point (the Function grid was 32).
    static thread_local std::vector<double> field;
    field.assign(context.truth_f.size(), 0.0);

    speckle_filler_field(field, context.grid, seeds);

    return graylevel_mse_field(context, field);

}

double params_loss(
    const LossType type,
    const double* x,
    size_t d,
    const LossContext& context)
{

    //The gray level loss only needs gray values, so it skips the full grid.
    if(type == LossType::GrayLevel){
        return graylevel_params_loss(x, context);
    }

    const LossSettings& settings {context.settings};

    std::vector<Function> pattern {generate_pattern(x[0], x[1], context.background, settings.width, settings.height, &context.halton)};

    switch(type){

        case LossType::GrayLevel:
            return graylevel_mse(context.ground_truth, pattern);

        case LossType::SSSIGMSE:
            return relative_mse(context.truth_sssig, compute_sssig(pattern, settings));

        case LossType::SSSIGVar: {
            std::vector<Points> sssig {compute_sssig(pattern, settings)};
            return relative_mse(context.truth_sssig, sssig) + relative_residual_variance(context.truth_sssig, sssig);
        }

        case LossType::AutocorrMSE:
            return relative_mse(context.truth_autocorr, compute_autocorrelation(pattern, settings));

        case LossType::AutocorrVar: {
            std::vector<Points> autocorr {compute_autocorrelation(pattern, settings)};
            return relative_mse(context.truth_autocorr, autocorr) + relative_residual_variance(context.truth_autocorr, autocorr);
        }

        case LossType::AutocorrScalars: {
            std::vector<Points> autocorr {compute_autocorrelation(pattern, settings)};
            return relative_scalar_error(context.truth_scalars, compute_autocorr_scalars(autocorr, settings));
        }
    }

    return 0.0;

}

std::vector<Function> patternreconstructor_params(
    const PSOResult& result,
    const double width,
    const double height,
    const double subpixel)
{

    std::vector<Function> background {background_generator(width, height, subpixel)};

    return generate_pattern(result.best_pos[0], result.best_pos[1], background, width, height);

}
