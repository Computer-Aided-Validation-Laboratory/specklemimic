#include "pso.hpp"

void gbinitializer(GBSwarmTopology& swarmState, std::mt19937_64& gen){
    
    //Clamp v max to be 0.02, we don't want the particle to whizz around like crazy.
    const double vmax = 0.02;

    //Create distributions to sample from.
    std::uniform_real_distribution<double> pos_dist(0.0, 1.0);
    std::uniform_real_distribution<double> vel_dist(-vmax, vmax);

    //Loop over all particles and dimensions and initialize them with some random number.
    for(size_t idx = 0; idx < swarmState.n * swarmState.d; idx++){

        swarmState.pos[idx] = pos_dist(gen);
        swarmState.vel[idx] = vel_dist(gen);

    }
}


void reflect_unit(double& x, double& v){

    //Check x < 0.0 bound
    if(x < 0.0){
        x = 0.0;    //Component gets set to 0.
        v = -v;     //Velocity gets flipped to act like it bounced back.
    }

    //Similar for other condition.
    else if(x > 1.0){
        x = 1.0;    //Component gets set to 1.0
        v = -v;     //Velocity gets flipped.
    }

}


//---------------------------------------------------------------------------
//PYTHON BINDING (pybind11)
//
//Only compiled when PSO_PYTHON_BINDINGS is defined, so the library still builds without pybind11.
//The optimizer itself (pso.hpp) is untouched, this just wraps a Python callable so it looks like a C++ objective.
//
//Build (from the folder that contains both specklegen/ and optimizers/):
//  g++ -O3 -std=c++23 -shared -fPIC -fvisibility=hidden -DPSO_PYTHON_BINDINGS -I. $(python -m pybind11 --includes) optimizers/pso.cpp -o pso_py$(python3-config --extension-suffix)
//On Windows the output is pso_py<suffix>.pyd (suffix from sysconfig EXT_SUFFIX) and you also need to link your Python: -L<python dir>/libs -lpython3XX
//
//Use from Python:
//  import pso_py
//  res = pso_py.optimize(objective, iterations, n_particles, d, lo, hi)
//  objective(x) gets a 1D numpy array of length d in PHYSICAL coordinates and returns a float.
//  res.best_pos is a numpy array (physical coordinates), res.best_val is a float.

#ifdef PSO_PYTHON_BINDINGS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <sstream>

namespace py = pybind11;

namespace {

//While alive, everything written to std::cout is swallowed (pso.hpp prints a start banner on every call).
struct CoutSilencer {
    std::ostringstream sink;
    std::streambuf* old;

    CoutSilencer() : old(std::cout.rdbuf(sink.rdbuf())) {}
    ~CoutSilencer() { std::cout.rdbuf(old); }
};

PSOResult optimize_python(
    py::function objective,
    int iterations,
    size_t n_particles,
    size_t d,
    const std::vector<double>& lo,
    const std::vector<double>& hi,
    bool default_bounds,
    double inertia,
    double acceleration1,
    double acceleration2,
    unsigned long long seed,
    bool verbose,
    const std::string& optimizer_name,
    bool loss_history)
{

    //Wrap the Python callable so it has the C++ objective shape: double f(const double* x, size_t d).
    //The d values are copied into a new numpy array, so Python can keep or modify x without touching the optimizer.
    auto f = [&objective](const double* x, size_t dim) -> double {
        py::array_t<double> arr(static_cast<py::ssize_t>(dim), x);
        return objective(arr).cast<double>();
    };

    auto run = [&]() -> PSOResult {
        return spsoptimzer(f, iterations, n_particles, d, lo, hi, default_bounds,
                           inertia, acceleration1, acceleration2, seed, verbose, optimizer_name, loss_history);
    };

    //The GIL stays held for the whole run, because the objective needs it on every call.
    if(verbose){
        py::scoped_ostream_redirect redirect;   //Send std::cout to Python's sys.stdout.
        return run();
    }

    CoutSilencer silence;
    return run();
}

} //namespace


PYBIND11_MODULE(pso_py, m){

    m.doc() = "Particle swarm optimizer (C++) that accepts a Python objective function.";

    py::class_<PSOResult>(m, "PSOResult")
        .def_property_readonly("best_pos", [](const PSOResult& r){
            return py::array_t<double>(static_cast<py::ssize_t>(r.best_pos.size()), r.best_pos.data());
        })
        .def_readonly("best_val", &PSOResult::best_val)
        .def("__repr__", [](const PSOResult& r){
            return "<PSOResult best_val=" + std::to_string(r.best_val) + ">";
        });

    m.def("optimize", &optimize_python,
        py::arg("objective"),
        py::arg("iterations"),
        py::arg("n_particles"),
        py::arg("d"),
        py::arg("lo"),
        py::arg("hi"),
        py::arg("default_bounds") = false,
        py::arg("inertia") = 0.729,
        py::arg("acceleration1") = 1.49445,
        py::arg("acceleration2") = 1.49445,
        py::arg("seed") = 42,
        py::arg("verbose") = false,
        py::arg("optimizer_name") = "",
        py::arg("loss_history") = false,
        "Run the particle swarm optimizer on a Python objective f(x) -> float, x is a 1D numpy array in physical coordinates.");

}

#endif