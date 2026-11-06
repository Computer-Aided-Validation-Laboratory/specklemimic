/**
 * @file autocorrelator.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Speckle pattern autocorrelation functions.
 * 
 * Houses the functions used to generate autocorrelation landscapes. 
 * Compatible with SSD, SCC, ZSSD, NSSD, ZNSSD, ZNCC correlation functions.
 * 
 * @version 0.2
 * @date 2026-11-05
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "structures.hpp"
#include <vector>

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
    const Field& function, 
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
    const Field& function, 
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
    const Field& function, 
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
    const Field& function, 
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
    const Field& function, 
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
    const Field& function, 
    const double subpixel);