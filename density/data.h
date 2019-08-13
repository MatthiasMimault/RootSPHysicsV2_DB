//
// Created by augustin on 31/07/19.
//

#ifndef CONTINUEDDENSITY_DATA_H
#define CONTINUEDDENSITY_DATA_H

#include <stdio.h>
#include "typeUnit.h"

LPart* getData(char *fileName, double *xMin, double *xMax, double *yMin, double *yMax, double *zMin, double *zMax);
void saveVTK(char* name, LPoint* l);

#endif //CONTINUEDDENSITY_DATA_H
