//
// Created by augustin on 31/07/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "data.h"
#include "typeUnit.h"

#define MAX_LEN 500
#define TROIS_SUR_QUATRE_PI 0.23873241463
#define max(x,y) ((x>y)?(x):(y))
#define min(x,y) ((x>y)?(y):(x))

void giveWord(char* str, char sep, int* i, char* word) {
    int j = *i, l = 0;
    while ((str[j] != '\0') && (str[j] == sep)) {
        j++;
    }
    while ((str[j] != '\0') && (str[j] != sep)) {
        word[l] = str[j];
        j++;
        l++;
    }
    word[l] = '\0';
    *i = j;
}

int comp(char* word1, char* word2) {
    int cmp = 1, i = 0;
    while ((word1[i] != '\0') && (word2[i] != '\0')){
        cmp &= (word1[i] == word2[i]);
        i++;
    }
    return (cmp && (word1[i] == word2[i]));
}

LPart* getData(char *fileName, double *xMin, double *xMax, double *yMin, double *yMax, double *zMin, double *zMax) {
    char str[MAX_LEN] = "";
    FILE *fic;

    *xMin = 1000.;
    *yMin = 1000.;
    *zMin = 1000.;
    *xMax = -1000.;
    *yMax = -1000.;
    *zMax = -1000.;

    fic = fopen(fileName, "r");

    if (fic == NULL) {
        printf("File %s could not be loaded.\n", fileName);
        return 0;
    }
    else {
        /*recovering of the usefull variables (position, mass and tensor)*/
        fgets(str, MAX_LEN, fic);
        fgets(str, MAX_LEN, fic);
        fgets(str, MAX_LEN, fic);
        fgets(str, MAX_LEN, fic);
        printf("%s\n", str);
        int i = 0, j = 0;
        int indPosX = 0, indMass = 0, indTens = 0, indRho = 0;
        char* word = malloc(sizeof(char) * 15);
        while ((indPosX == 0) || (indMass == 0) || (indTens == 0)) {
            j++;
            giveWord(str, ';', &i, word);
            //printf("%d %s", i, word);
            if (comp(word, "Pos.x")) {
                indPosX = j;
            }
            else if (comp(word, "Mass")) {
                indMass = j;
            }
            else if (comp(word, "Qfxx")) {
                indTens = j;
            }
            else if (comp(word, "Rhop")) {
                indRho = j;
            }
        }
        int maxInd = max(max(indPosX, indMass), indTens);
        /*creation of the list*/
        LPart* l = listeVide();
        j = 0;
        while (fgets(str, MAX_LEN, fic) != NULL) {
            j = 1;
            i = 0;
            Particule part;
            do {
                if (j == indPosX) {
                    giveWord(str, ';', &i, word);
                    part.pos[0] = atof(word);
                    *xMin = min(*xMin, part.pos[0]);
                    *xMax = max(*xMax, part.pos[0]);
                    giveWord(str, ';', &i, word);
                    part.pos[1] = atof(word);
                    *yMin = min(*yMin, part.pos[1]);
                    *yMax = max(*yMax, part.pos[1]);
                    giveWord(str, ';', &i, word);
                    part.pos[2] = atof(word);
                    *zMin = min(*zMin, part.pos[2]);
                    *zMax = max(*zMax, part.pos[2]);
                    j += 3;
                }
                else if (j == indMass) {
                    giveWord(str, ';', &i, word);
                    part.mass = atof(word);
                    j++;
                }
                else if (j == indTens) {
                    giveWord(str, ';', &i, word);
                    part.shape.xx = atof(word);
                    giveWord(str, ';', &i, word);
                    part.shape.xy = atof(word);
                    giveWord(str, ';', &i, word);
                    part.shape.xz = atof(word);
                    giveWord(str, ';', &i, word);
                    part.shape.yy = atof(word);
                    giveWord(str, ';', &i, word);
                    part.shape.yz = atof(word);
                    giveWord(str, ';', &i, word);
                    part.shape.zz = atof(word);
                    part.shape.sqrtDet = sqrt(determinantSym(part.shape));
                    //normalizeMat(&(part.shape));
                    j += 6;
                }
                else if (j == indRho) {
                    giveWord(str, ';', &i, word);
                    part.dens = atof(word);
                    j++;
                }
                else {
                    giveWord(str, ';', &i, word);
                    j++;
                }
            } while (j <= maxInd);
            part.h = pow(TROIS_SUR_QUATRE_PI * part.mass / part.dens, 0.3333);
            ajouter(&l, part);
        }
        fclose(fic);
        //afficher(l);
        return l;
    }
}

void saveVTK(char* name, LPoint* l) {
    int len = tailleP(l);
    FILE *fic = fopen(name, "w");
    fprintf(fic, "# vtk DataFile Version 3.0\nData %s\nASCII\nDATASET POLYDATA\nPOINTS %d float\n", name, len);
    LPoint* l1 = l;
    while (!estVideP(l1)) {
        fprintf(fic, "%f %f %f\n", l1->point.pos[0], l1->point.pos[1], l1->point.pos[2]);
        l1 = obtenirListeSuivanteP(l1);
    }

    fprintf(fic, "VERTICES %d %d\n", len, (len * 2));
    for (int i = 0 ; i < len ; i++) {
        fprintf(fic, "1 %d\n", i);
    }

    fprintf(fic, "POINT_DATA %d\nSCALARS Idp unsigned_int\nLOOKUP_TABLE default\n", len);

    //fprintf(fic, "FIELD FieldData 1\nIdp 1 %d unsigned_int\n", len);
    while (!estVideP(l)) {
        fprintf(fic, "%20ld ", l->point.idp);
        l = obtenirListeSuivanteP(l);
    }
    fclose(fic);
}
