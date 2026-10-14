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