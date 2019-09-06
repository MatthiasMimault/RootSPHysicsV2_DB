//
// Created by augustin on 31/07/19.
//

#ifndef CONTINUEDDENSITY_ALGO_H
#define CONTINUEDDENSITY_ALGO_H

#include "typeUnit.h"
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define PI23 0.17958712212
#define PI_1 0.31830988618

double indToCoord(int ind, double min, double dx);
//void computeDensity2D(double** dens, LPart* l, double h, double xMin, int nbX, double zMin, int nbZ);
void computeDensity3D(double*** dens, LPart* l, double h, double xMin, int nbX, double yMin, int nbY, double zMin, int nbZ, double dx);
LPoint* computeRidge(double*** dens, double xMin, int nbX, double yMin, int nbY, double zMin, int nbZ, double dx);

#endif //CONTINUEDDENSITY_ALGO_H
