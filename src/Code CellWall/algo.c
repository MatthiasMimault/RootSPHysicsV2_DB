//
// Created by augustin on 31/07/19.
//

#include "algo.h"

double W2D(Position2D x, BilForm bil, double h) {
    //return PI23 * exp(-bilFormSym(x, bil)/(h*h))/(h*h*h);
    //printf("%lf", h);
    return PI_1/(h*h) * exp(-(x[0]*x[0] + x[1]*x[1])/(h*h));
}

double W3D(Position3D x, BilForm bil, double h) {
    //return PI23 * exp(-bilFormSym(x, bil)/(h*h))/(h*h*h);
    //printf("%lf", h);
    return bil.sqrtDet * PI23/(h*h*h) * exp(-bilFormSym3D(x, bil)/(h*h));
}

double density2D(Position2D r, LPart* l, double h) {
    double sum = 0.;
    Position2D pos;
    Position3D pos3d;
    while (!(estVide(l))) {
        pos3d[0] = r[0];
        pos3d[1] = 0;
        pos3d[2] = r[1];
        difference3D(pos3d, l->part.pos, pos3d);
        pos[0] = pos3d[0];
        pos[1] = pos3d[2];
        //printf("\t%lf\n", (pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]));
        //printf("\t%lf\t%lf\t%lf\t%lf\t%.30lf\t%lf\t%lf\n", pos[0], pos[1], pos[2], l->part.mass, W(pos, l->part.shape, h), (l->part.mass * W(pos, l->part.shape, h)), sum);
        sum += l->part.mass * W2D(pos, l->part.shape, h);
        //printf("%lf\n", sum);
        l = obtenirListeSuivante(l);
    }
    return sum;
}

double density3D(Position3D r, LPart* l, double h) {
    double sum = 0.;
    Position3D pos;
    while (!(estVide(l))) {
        difference3D(r, l->part.pos, pos);
        //printf("\t%lf\n", (pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]));
        //printf("\t%lf\t%lf\t%lf\t%lf\t%.30lf\t%lf\t%lf\n", pos[0], pos[1], pos[2], l->part.mass, W(pos, l->part.shape, h), (l->part.mass * W(pos, l->part.shape, h)), sum);
        sum += l->part.mass * W3D(pos, l->part.shape, h * l->part.h);
        //printf("%lf\n", sum);
        l = obtenirListeSuivante(l);
    }
    return sum;
}

double indToCoord(int ind, double min, double dx) {
    return min + (double) ind * dx;
}

void computeDensity3D(double*** dens, LPart* l, double h, double xMin, int nbX, double yMin, int nbY, double zMin, int nbZ, double dx) {
    Position3D pos;
    int i, j, k;
    #pragma omp parallel for private(pos, i, j, k)
    for (i = 0 ; i < nbX ; i++) {
        pos[0] = indToCoord(i, xMin, dx);
        printf("Beginning of step %d / %d, thread %d / %d\tPosX : %lf\n", (i+1), nbX, omp_get_thread_num(), omp_get_num_threads(), pos[0]);
        for (j = 0 ; j < nbY ; j++) {
            pos[1] = indToCoord(j, yMin, dx);
            for (k = 0 ; k < nbZ ; k++) {
                pos[2] = indToCoord(k, zMin, dx);
                //printf("%lf %lf %lf\n", pos[0], pos[1], pos[2]);
                dens[i][j][k] = density3D(pos, l, h);
                //printf("%lf\n", dens[i][j][k]);
                //printf("(%lf, %lf, %lf) -> %.15lf\n", pos[0], pos[1], pos[2], dens[i][j][k]);
            }
        }
        printf("\t%d / %d finished\n", (i+1), nbX);
    }
    printf("Densities are computed\n");
}

/*void computeDensity2D(double** dens, LPart* l, double h, double xMin, int nbX, double zMin, int nbZ) {
    Position2D pos;
    for (int i = 0 ; i < nbX ; i++) {
        for (int k = 0 ; k < nbZ ; k++) {
            pos[0] = indToCoord(i, xMin);
            pos[1] = indToCoord(k, zMin);
            //printf("%lf %lf %lf\n", pos[0], pos[1], pos[2]);
            dens[i][k] = density2D(pos, l, h);
            printf("(%lf, %lf) -> %lf\n", pos[0], pos[1], dens[i][k]);
            //printf("%lf\n", dens[i][k]);
        }
        printf("%d / %d\n", (i+1), nbX);
    }
}*/

LPoint* computeRidge(double*** dens, double xMin, int nbX, double yMin, int nbY, double zMin, int nbZ, double dx){
    LPoint* l = listeVideP();
    PointCalcul p;
    double densPrec, densCourante, densSuiv;

    if (nbZ > 1) {
        for (int i = 0 ; i < nbX ; i++) {
            p.pos[0] = indToCoord(i, xMin, dx);
            for (int j = 0 ; j < nbY ; j++) {
                p.pos[1] = indToCoord(j, yMin, dx);
                densPrec = dens[i][j][0];
                densCourante = dens[i][j][1];
                for (int k = 1 ; k < nbZ - 1 ; k++) {
                    densSuiv = dens[i][j][k + 1];
                    if ((densCourante < densPrec) && (densCourante < densSuiv)) {
                        p.pos[2] = indToCoord(k, zMin, dx);
                        p.dens = densCourante;
                        p.idp = k + nbZ * (j +  i * nbY);
                        ajouterP(&l, p);
                    }
                    densPrec = densCourante;
                    densCourante = densSuiv;
                }
            }
            printf("%d / %d\n", (i+1), nbX);
        }
        printf("Scan Z ended.\n");
    }

    if (nbY > 1) {
        for (int i = 0 ; i < nbX ; i++) {
            p.pos[0] = indToCoord(i, xMin, dx);
            for (int k = 0 ; k < nbZ ; k++) {
                p.pos[2] = indToCoord(k, zMin, dx);
                densPrec = dens[i][0][k];
                densCourante = dens[i][1][k];
                for (int j = 1 ; j < nbY - 1 ; j++) {
                    densSuiv = dens[i][j + 1][k];
                    if ((densCourante < densPrec) && (densCourante < densSuiv)) {

                        p.pos[1] = indToCoord(j, yMin, dx);
                        p.dens = densCourante;
                        p.idp = k + nbZ * (j +  i * nbY);
                        ajouterP(&l, p);
                    }
                    densPrec = densCourante;
                    densCourante = densSuiv;
                }
            }
            printf("%d / %d\n", (i+1), nbX);
        }
        printf("Scan Y ended.\n");
    }

    if (nbX > 1) {
        for (int j = 0 ; j < nbY ; j++) {
            p.pos[1] = indToCoord(j, yMin, dx);
            for (int k = 0 ; k < nbZ ; k++) {
                p.pos[2] = indToCoord(k, zMin, dx);
                densPrec = dens[0][j][k];
                densCourante = dens[1][j][k];
                for (int i = 1 ; i < nbX - 1 ; i++) {
                    densSuiv = dens[i + 1][j][k];
                    if ((densCourante < densPrec) && (densCourante < densSuiv)) {
                        p.pos[0] = indToCoord(i, xMin, dx);
                        p.dens = densCourante;
                        p.idp = k + nbZ * (j +  i * nbY);
                        ajouterP(&l, p);
                    }
                    densPrec = densCourante;
                    densCourante = densSuiv;
                }
            }
            printf("%d / %d\n", (j+1), nbY);
        }
        printf("Scan X ended.\n");
    }

    return l;
}
