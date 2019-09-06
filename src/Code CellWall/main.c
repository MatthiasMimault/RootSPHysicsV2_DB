
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "data.h"
#include "algo.h"
#include "typeUnit.h"

int main(int argc, char *argv[]) {
    // arguments : 1) names (without .csv), 2) distance between particules, 3) number of the last time step data
    double xMin, xMax, yMin, yMax, zMin, zMax, h, dx;
    char nameCsv[100], nameVtk[100];
    int lastSim = 0;
    if (argc == 4) lastSim = atoi(argv[3]);
    h = atof(argv[2]);
    dx = 0.001;
    for (int i = 0 ; i <= lastSim ; i++) {
        if (argc == 4)
            sprintf(nameCsv, "%s_%04d.csv", argv[1], i);
        else
            sprintf(nameCsv, "%s.csv", argv[1]);
        LPart* l = getData(nameCsv, &xMin, &xMax, &yMin, &yMax, &zMin, &zMax);
        printf("Recup faite\n");
        printf("%f %f %f %f %f %f %f\n", xMin, xMax, yMin, yMax, zMin, zMax, dx);

        // creation of the final data


        int nbX = (int) ((xMax - xMin) / dx) + 1;
        int nbY = (int) ((yMax - yMin) / dx) + 1;
        int nbZ = (int) ((zMax - zMin) / dx) + 1;

        printf("%s : ", nameCsv);
        printf("%d points de calcul\n", (nbX * nbY * nbZ));

        // float*** posValues = malloc(nbX * sizeof(double**));
        // for (int i = 0 ; i < nbX ; i++) {
        //     posValues[i] = malloc(nbY * sizeof(double*));
        //     for (int j = 0 ; j < nbY ; j++) {
        //         posValues[i][j] = malloc(nbZ  * sizeof(double));
        //     }
        // }

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
        Position3D pos;
        computeDensity3D(dens, l, h, xMin, nbX, yMin, nbY, zMin, nbZ, dx);
        /*for (int i = 0 ; i < nbX ; i++) {
            pos[0] = indToCoord(i, xMin);
            for (int j = 0 ; j < nbY ; j++) {
                pos[1] = indToCoord(j, yMin);
                for (int k = 0 ; k < nbZ ; k++) {
                    pos[2] = indToCoord(k, zMin);
                    printf("(%lf, %lf, %lf) -> %lf\n", pos[0], pos[1], pos[2], dens[i][j][k]);
                }
            }
        }*/
        LPoint* lP;
        lP = computeRidge(dens, xMin, nbX, yMin, nbY, zMin, nbZ, dx);

        if (argc == 4)
            sprintf(nameVtk, "%sWAll_%04d.vtk", argv[1], i);
        else
            sprintf(nameVtk, "%sW.vtk", argv[1]);
        saveVTK(nameVtk, lP);

        supprimer(&l);
        supprimerP(&lP);
        for (int i = 0 ; i < nbX ; i++) {
            for (int j = 0 ; j < nbY ; j++) {
                free(dens[i][j]);
            }
            free(dens[i]);
        }
        free(dens);
    }

    return 0;
}
