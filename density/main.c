
#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "algo.h"
#include "typeUnit.h"

int main() {
    double xMin, xMax, yMin, yMax, zMin, zMax;
    LPart* l = getData("4-P200_0100.csv", &xMin, &xMax, &yMin, &yMax, &zMin, &zMax);
    printf("%f %f %f %f %f %f %f\n", xMin, xMax, yMin, yMax, zMin, zMax, DX);

    // creation of the final data


    int nbX = (int) ((xMax - xMin) / DX) + 1;
    int nbY = (int) ((yMax - yMin) / DX) + 1;
    int nbZ = (int) ((zMax - zMin) / DX) + 1;

    printf("4-P200_0100.csv : ");
    printf("%d points de calcul\n", (nbX * nbY * nbZ));

    // float*** posValues = malloc(nbX * sizeof(double**));
    // for (int i = 0 ; i < nbX ; i++) {
    //     posValues[i] = malloc(nbY * sizeof(double*));
    //     for (int j = 0 ; j < nbY ; j++) {
    //         posValues[i][j] = malloc(nbZ  * sizeof(double));
    //     }
    // }

    // faire une fonction de conversion indices <-> position

    double*** dens = malloc(nbX * sizeof(double**));
    for (int i = 0 ; i < nbX ; i++) {
        dens[i] = malloc(nbY * sizeof(double*));
        for (int j = 0 ; j < nbY ; j++) {
            dens[i][j] = malloc(nbZ  * sizeof(double));
        }
    }
    /*double** dens = malloc(nbX * sizeof(double*));
    for (int i = 0 ; i < nbX ; i++) {
        dens[i] = malloc(nbZ * sizeof(double));
    }*/
    double h = 0.025;
    Position3D pos;
    computeDensity3D(dens, l, h, xMin, nbX, yMin, nbY, zMin, nbZ);
    for (int i = 0 ; i < nbX ; i++) {
        pos[0] = indToCoord(i, xMin);
        for (int j = 0 ; j < nbY ; j++) {
            pos[1] = indToCoord(j, yMin);
            for (int k = 0 ; k < nbZ ; k++) {
                pos[2] = indToCoord(k, zMin);
                printf("(%lf, %lf, %lf) -> %lf\n", pos[0], pos[1], pos[2], dens[i][j][k]);
            }
        }
    }
    LPoint* lP;
    lP = computeRidge(dens, xMin, nbX, yMin, nbY, zMin, nbZ);
    saveVTK("4-P200_0100vis.vtk", lP);

    return 0;
}
