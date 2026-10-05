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
void histogram_generator(const int nbins, const std::vector<Function>& function, std::vector<Histogram>& histogram, std::vector<Histogram>& probability_density, double mean, double variance, double sentropy);


/**
 * @brief SSD Autocorrelator
 * 
 * Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the SSD correlation function.
 * 
 * @param displacements Pre-initialized displacements vector.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_ssd(std::vector<Points>& displacements, const std::vector<Function> function, const double subpixel);


/**
 * @brief SCC Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the SCC correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_scc(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel);


/**
 * @brief NSSD Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the NSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_nssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel);


/**
 * @brief ZSSD Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the ZSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_zssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel);


/**
 * @brief ZNSSD Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the ZNSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_znssd(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel);


/**
 * @brief ZNCC Autocorrelator
 * 
 * @param displacements Shifts speckle patterns with artificial displacements to create an autocorrelation landscape using the ZNSSD correlation function.
 * @param function Speckle pattern function.
 * @param subpixel Subpixel accuracy we want to go to.
 */
void autocorrelator_zncc(std::vector<Points>& displacements, const std::vector<Function>& function, const double subpixel);