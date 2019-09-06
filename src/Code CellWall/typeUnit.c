//
// Created by augustin on 31/07/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "typeUnit.h"

void difference3D(Position3D x1, Position3D x2, Position3D result) {
    result[0] = x1[0] - x2[0];
    result[1] = x1[1] - x2[1];
    result[2] = x1[2] - x2[2];
}

double bilFormSym3D(Position3D vector, BilForm mat){
    return mat.xx * vector[0] * vector[0] + mat.yy * vector[1] * vector[1] + mat.zz * vector[2] * vector[2] + 2 * mat.xy * vector[0] * vector[1] + 2 * mat.xz * vector[0] * vector[2] + 2 * mat.yz * vector[2] * vector[1];
}

double determinantSym(BilForm bil){
    return bil.xx*bil.yy*bil.zz + 2*bil.xy*bil.xz*bil.yz - bil.yy*bil.xz*bil.xz - bil.xx*bil.yz*bil.yz - bil.zz*bil.xy*bil.xy;
}

void normalizeMat(BilForm* pBil){
    double sDet = sqrt(determinantSym(*pBil));
    pBil->xx = pBil->xx / sDet;
    pBil->yy = pBil->yy / sDet;
    pBil->zz = pBil->zz / sDet;
    pBil->xy = pBil->xy / sDet;
    pBil->xz = pBil->xz / sDet;
    pBil->yz = pBil->yz / sDet;
}

void afficherPart(Particule part){
    printf("Pos:%f,%f,%f ; Mass:%f\n\t%.15lf %.15lf %.15lf %.15lf %.15lf %.15lf\n", part.pos[0], part.pos[1], part.pos[2], part.mass, part.shape.xx, part.shape.yy, part.shape.zz, part.shape.xy, part.shape.xz, part.shape.yz);
}

void afficherPoint(PointCalcul point){
    printf("Pos:%f,%f,%f ; Dens:%f\n", point.pos[0], point.pos[1], point.pos[2], point.dens);
}

/*Listes particules*/

LPart* listeVide(){
    return NULL;
}

int estVide(LPart* l){
    return (l == NULL);
}

void ajouter(LPart **l, Particule part){
    LPart *new = malloc(sizeof(LPart));
    new->part = part;
    new->suivant = *l;
    *l = new;
}

Particule obtenirElement(LPart* l){
    return l->part;
}

LPart* obtenirListeSuivante(LPart* l){
    return l->suivant;
}

void fixerListeSuivante(LPart* l, LPart* lSuiv){
    l->suivant = lSuiv;
}

void supprimerTete(LPart** l){
    LPart *temp = *l;
    *l = obtenirListeSuivante(*l);
    free(temp);
}

void supprimer(LPart **l){
    while (!estVide(*l)){
        supprimerTete(l);
    }
}

int taille(LPart* l){
    int i = 0;
    while (!estVide(l)){
        i++;
        l = obtenirListeSuivante(l);
    }
    return i;
}

void afficher(LPart* l){
    while (!estVide(l)){
        afficherPart(l->part);
        l = obtenirListeSuivante(l);
    }
    printf("\n");
}



/*Listes points*/

LPoint* listeVideP(){
    return NULL;
}

int estVideP(LPoint* l){
    return (l == NULL);
}

void ajouterP(LPoint **l, PointCalcul point){
    LPoint *new = malloc(sizeof(LPoint));
    new->point = point;
    new->suivant = *l;
    *l = new;
}

PointCalcul obtenirElementP(LPoint* l){
    return l->point;
}

LPoint* obtenirListeSuivanteP(LPoint* l){
    return l->suivant;
}

void fixerListeSuivanteP(LPoint* l, LPoint* lSuiv){
    l->suivant = lSuiv;
}

void supprimerTeteP(LPoint** l){
    LPoint *temp = *l;
    *l = obtenirListeSuivanteP(*l);
    free(temp);
}

void supprimerP(LPoint **l){
    while (!estVideP(*l)){
        supprimerTeteP(l);
    }
}

int tailleP(LPoint* l){
    int i = 0;
    while (!estVideP(l)){
        i++;
        l = obtenirListeSuivanteP(l);
    }
    return i;
}

void afficherP(LPoint* l){
    while (!estVideP(l)){
        afficherPoint(l->point);
        l = obtenirListeSuivanteP(l);
    }
    printf("\n");
}
