#include "structures.hpp"
#include <vector>

double mig(const double width, const double height, const std::vector<Function>& grad_mag);
std::vector<Histogram> histogram_generator(const int nbins, const std::vector<Function>& function);
void histogram_generator(const int nbins, const std::vector<Function>& function, std::vector<Histogram>& histogram, std::vector<Histogram>& probability_density, double mean, double variance, double sentropy);
void autocorrelator_ssd(std::vector<Points>& displacements, const std::vector<Function> function, const double subpixel);
void autocorrelator_scc(std::vector<Points>& displacements, const std::vector<Function> function, const double subpixel);