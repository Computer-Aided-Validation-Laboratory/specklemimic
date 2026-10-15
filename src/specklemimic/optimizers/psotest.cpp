#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <chrono>
#include <numbers>
#include "optimizers/pso.hpp"

//---------------------------------------------------------------------------
//USER DEFINED PARAMETERS.
const int ITERATIONS {2000};       //Iterations per optimizer run.
const int NRUNS {20};              //Number of repeated runs (different seeds) per test.
const double TOL {1e-4};           //A run counts as a success if (best value - known minimum) < TOL.
const bool DEFAULTS {false};       //true = use the library default coefficients.
const double INERTIA {0.729};      //Clerc-Kennedy constriction values.
const double ACCELERATION1 {1.49445};
const double ACCELERATION2 {1.49445};
//---------------------------------------------------------------------------
//OBJECTIVE FUNCTIONS.
//Every objective function MUST have exactly this signature:
//
//      double f(const double* x, size_t d)
//
//  x : pointer to d consecutive doubles holding the position in PHYSICAL coordinates
//      (the optimizer has already undone the unit hypercube scaling).
//  d : the dimension.
//
//It returns ONE number and the optimizer minimizes it. Anything the objective needs
//(shifts, constants) lives inside the function or is a global constant.
//
//Several functions have a shift so the minimum is not at the centre of the domain.
//Swarms are known to be biased towards the centre of the search space, so an unshifted
//test at the origin can look better than it should.
//---------------------------------------------------------------------------
//CONVEX.
//Sphere. Convex, unimodal, separable.
//Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
double sphere(const double* x, size_t d){
    double shift {2.0};
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += z * z;
    }
    return sum;
}

//Axis parallel hyper-ellipsoid. Convex, unimodal, but ill conditioned (weights 1 to d).
//Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
double ellipsoid(const double* x, size_t d){
    double shift {2.0};
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += static_cast<double>(k + 1) * z * z;
    }
    return sum;
}

//---------------------------------------------------------------------------
//NON-CONVEX.
//Rosenbrock. Non-convex, narrow curved valley, hard to converge along the valley floor.
//Domain: [-5, 10]^d. Minimum f = 0 at x_k = shift + 1.
double rosenbrock(const double* x, size_t d){
    double shift {2.0};
    double sum {0.0};
    for(size_t k = 0; k + 1 < d; k++){
        double z1 {x[k] - shift};
        double z2 {x[k + 1] - shift};
        double a {z2 - z1 * z1};
        double b {1.0 - z1};
        sum += 100.0 * a * a + b * b;
    }
    return sum;
}

//Rastrigin. Non-convex, highly multimodal with a regular grid of local minima.
//Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
double rastrigin(const double* x, size_t d){
    double shift {2.0};
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += z * z - 10.0 * std::cos(2.0 * std::numbers::pi * z);
    }
    return 10.0 * static_cast<double>(d) + sum;
}

//Ackley. Non-convex, many shallow local minima around one narrow global basin.
//Domain: [-32.768, 32.768]^d. Minimum f = 0 at x_k = shift. (a = 20, b = 0.2, c = 2 pi)
double ackley(const double* x, size_t d){
    double shift {2.0};
    double a {20.0};
    double b {0.2};
    double c {2.0 * std::numbers::pi};
    double sum1 {0.0};
    double sum2 {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum1 += z * z;
        sum2 += std::cos(c * z);
    }
    double dd {static_cast<double>(d)};
    return -a * std::exp(-b * std::sqrt(sum1 / dd)) - std::exp(sum2 / dd) + a + std::exp(1.0);
}

//Griewank. Non-convex, many regularly spaced local minima, couples dimensions through a product.
//Domain: [-600, 600]^d. Minimum f = 0 at x_k = shift.
double griewank(const double* x, size_t d){
    double shift {100.0};
    double sum {0.0};
    double prod {1.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += z * z / 4000.0;
        prod *= std::cos(z / std::sqrt(static_cast<double>(k + 1)));
    }
    return sum - prod + 1.0;
}

//Schwefel. Non-convex and deceptive: the global minimum is far from the next best local minima.
//Not shifted, its minimum is already far from the centre.
//Domain: [-500, 500]^d. Minimum f = 0 at x_k = 420.9687462275036.
double schwefel(const double* x, size_t d){
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        sum += x[k] * std::sin(std::sqrt(std::fabs(x[k])));
    }
    return 418.982887272433799807913601398 * static_cast<double>(d) - sum;
}

//---------------------------------------------------------------------------
//TEST RUNNER.
//Runs the optimizer NRUNS times on one function and prints a summary line.
//lo, hi : domain bounds applied to every dimension.
//xopt   : known location of the minimum in every dimension.
//fopt   : known minimum value.
void run_test(const std::string& name, double (*f)(const double*, size_t), size_t d, double lo, double hi, double xopt, double fopt){
    size_t n {20 + 2 * d};   //Particle count grows with dimension.
    std::vector<double> lo_vec(d, lo);
    std::vector<double> hi_vec(d, hi);
    std::vector<double> x_opt(d, xopt);

    //Sanity check on the objective itself: f at the claimed optimum must equal the known minimum.
    double check {f(x_opt.data(), d) - fopt};
    int successes {0};
    double sum_err {0.0};
    double best_err {std::numeric_limits<double>::infinity()};
    double worst_err {0.0};

    for(int run = 0; run < NRUNS; run++){
        // Enable history tracking for the first run when d = 2, using a distinct filename tag per landscape
        std::string tag = "";
        if (d == 2 && run == 0) {
            tag = name + "_run"; // e.g. "Sphere_run", "Rosenbrock_run", etc.
        }

        PSOResult result {spsoptimzer(f, ITERATIONS, n, d, lo_vec, hi_vec, DEFAULTS, INERTIA, ACCELERATION1, ACCELERATION2, 1000 + run, false, tag)};
        double err {result.best_val - fopt};
        if(err < TOL){
            successes++;
        }
        sum_err += err;
        if(err < best_err){
            best_err = err;
        }
        if(err > worst_err){
            worst_err = err;
        }
    }

    std::cout << std::left << std::setw(22) << name << std::right << " | d = " << std::setw(2) << d << " | n = " << std::setw(3) << n << " | f(x*) check = " << std::scientific << std::setprecision(1) << check << " | success = " << std::setw(2) << successes << "/" << NRUNS << " | mean err = " << std::setprecision(2) << sum_err / NRUNS << " | best err = " << best_err << " | worst err = " << worst_err << " |" << std::endl;
}

//---------------------------------------------------------------------------
//MAIN FUNCTION.
int main(){
    int dims[3] {2, 10, 30};

    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "PSO BENCHMARK TESTS" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Iterations = " << ITERATIONS << " | Runs per test = " << NRUNS << " | Success tolerance = " << TOL << std::endl;
    std::cout << "Inertia = " << INERTIA << " | Acceleration 1 = " << ACCELERATION1 << " | Acceleration 2 = " << ACCELERATION2 << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;

    //Start timer.
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 3; i++){
        size_t d {static_cast<size_t>(dims[i])};

        //CONVEX.
        run_test("Sphere",      sphere,     d,   -5.12,    5.12,   2.0,               0.0);
        run_test("Ellipsoid",   ellipsoid,  d,   -5.12,    5.12,   2.0,               0.0);

        //NON-CONVEX.
        run_test("Rosenbrock",  rosenbrock, d,   -5.0,     10.0,   3.0,               0.0);
        run_test("Rastrigin",   rastrigin,  d,   -5.12,    5.12,   2.0,               0.0);
        run_test("Ackley",      ackley,     d,   -32.768,  32.768, 2.0,               0.0);
        run_test("Griewank",    griewank,   d,   -600.0,   600.0,  100.0,             0.0);
        run_test("Schwefel",    schwefel,   d,   -500.0,   500.0,  420.9687462275036, 0.0);

        std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    }

    //End timer.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    std::cout << "PROGRAM FINISHED." << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << duration_ms.count() / 1000 << "s" << std::endl;
}