#pragma once

//Structure for the generated seed points.
struct Points{
    double x;
    double y;
    double r;
};

//Structure for any 2D function.
struct Function{
    double x;
    double y;
    double f;
    int nx;
    int ny;
};

//Structure for a histogram (1D function).
struct Histogram{
    double centre;
    double f;
    int n;
    int df;
};