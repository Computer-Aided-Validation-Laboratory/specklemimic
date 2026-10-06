/**
 * @file diagnostics.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Contains functions for a bunch of diagnostics. 
 * 
 * Currently allows the user to calculate the mean intensity gradient (MIG), speckle intensity histogram and autocorrelation landscapes.
 * The autocorrelation functions allowed so far are: SSD, SCC
 * 
 * @version 0.1
 * @date 2026-10-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "structures.hpp"
#include <vector>

/**
 * @brief Used to compute the mean intensity gradient as a diagnostic for assessing speckle pattern quality.
 * 
 * @param width Width of the ROI. (x)
 * @param height Height of the ROI. (y)
 * @param grad_mag Gradient magnitude function. 
 * @return double Mean intensity gradient. (Single value)
 */
double mig(const std::vector<Function>& grad_mag);


/**
 * @brief Takes the speckle pattern function, spits it into bins to make a histogram.
 * 
 * Computes mean, variance, shannon entropy and outputs that.
 * 
 * @param nbins Number of bins needed for the speckle histogram.
 * @param function Function for the speckle pattern.
 * @param histogram Pre-initialized histogram vector.
 * @param probability_density Pre-initialized vector to store the probability density, (Variable name is shortened here)
 * @param mean Mean speckle intensity value.
 * @param variance Variance in speckle intensity.
 * @param sentropy Shannon entropy.
 */
void histogram_generator(
    const int nbins, 
    const std::vector<Function>& function, 
    std::vector<Histogram>& histogram, 
    std::vector<Histogram>& probability_density, 
    double mean, 
    double variance, 
    double sentropy);


/**
 * @brief SSD Autocorrelator
 * 
 * Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the SSD correlation function.
 * 
 * @param displacements Pre-initialized displacements vector.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_ssd(
    std::vector<Points>& displacements, 
    const std::vector<Function> function, 
    const double subpixel);


/**
 * @brief SCC Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the SCC correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_scc(
    std::vector<Points>& displacements, 
    const std::vector<Function>& function, 
    const double subpixel);


/**
 * @brief NSSD Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the NSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_nssd(
    std::vector<Points>& displacements, 
    const std::vector<Function>& function, 
    const double subpixel);


/**
 * @brief ZSSD Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the ZSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_zssd(
    std::vector<Points>& displacements, 
    const std::vector<Function>& function, 
    const double subpixel);


/**
 * @brief ZNSSD Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the ZNSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_znssd(
    std::vector<Points>& displacements, 
    const std::vector<Function>& function, 
    const double subpixel);


/**
 * @brief ZNCC Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the ZNSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_zncc(
    std::vector<Points>& displacements, 
    const std::vector<Function>& function, 
    const double subpixel);


/**
 * @brief Radial Gradient for Autocorrelation Landscapes.
 * 
 * This function computes the radial derivative of any given autocorrelation landscape. This is used in further diagnostics. 
 * It employs central differencing for interior points while resorting to backward/forward differencing for boundaries.
 * 
 * @param displacements Input autocorrelation landscape.
 * @param pixelshift Used to calculate displacement domain.
 * @return std::vector<Points> Output gradient.
 */
std::vector<Points> autocorrelation_grad_r(
    const std::vector<Points>& displacements, 
    const int pixelshift);


/**
 * @brief Watershed Contour using Gradients.
 * 
 * Uses the closest minima as a way to compute the watershed surface of an autocorrelation landscape. 
 * This currently has the drawback that it is very unstable to noisy landscapes, this can be improved via smoothing, etc.
 * 
 * @param grad_r First order gradient of autocorrelation landscape.
 * @param grad_2r Second order gradient of autocorrelation landscape.
 * @param displacements Input autocorrelation landscape.
 * @param pixelshift Used to calculate displacement domain.
 * @param optimalsectors Number of sectors needed for binning to resolve grid properly with a 1px delta.
 * @return std::vector<BinnedPoint> Output watershed surface/line.
 */
std::vector<BinnedPoint> watershed_gradient(
    const std::vector<Points>& grad_r, 
    const std::vector<Points>& grad_2r, 
    const std::vector<Points>& displacements, 
    const int pixelshift, 
    const int optimalsectors,
    double& watershed_radius
);


/**
 * @brief Autocorrelation Peak Sharpness
 * 
 * Spans a 4 pixel wide rectangular ring around the peak and finds the average gradient there. 
 * This acts as an indicator of peak sharpness. If it is higher, then we have a sharper peak.
 * 
 * @param grad_r Input gradients for autocorrelation landscape.
 * @param pixelshift Used to calculate displacement domain.
 * @return double Autocorrelation Peak Sharpness.
 */
double autocorrelation_peak(
    const std::vector<Points>& grad_r, 
    const int pixelshift);


/**
 * @brief Autocorrelation Primary and Secondary Peak Comparison
 * 
 * Calculates the distance between primary peak and biggest secondary peak.
 * Also calculates the variance between all maxima in the landscape to give some good indicators.
 * Boundaries default as maxima.
 * 
 * @param grad_r First order radial gradient of autocrrelation landscape.
 * @param grad_2r Second order radial gradient of autocorrelation landscape.
 * @param optimalsectors Number of sectors needed for binning to resolve grid properly with a low delta.
 * @param pixelshift Used to calculate displacement domain.
 * @param displacements Input autocorrelation landscape
 * @param peak_max Output peak comparison between biggest secondary peak.
 * @param peak_comp Output standard deviation of all secondary peaks.
 */
void autocorrelation_peak_comp(
    const std::vector<Points>& grad_r, 
    const std::vector<Points> & grad_2r, 
    const int optimalsectors, 
    const int pixelshift, 
    const std::vector<Points>& displacements,
    double& peak_max, 
    double& peak_comp);