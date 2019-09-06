//
// Created by augustin on 31/07/19.
//

#ifndef CONTINUEDDENSITY_TYPEUNIT_H
#define CONTINUEDDENSITY_TYPEUNIT_H

typedef struct BilForm {
    double xx, yy, zz, xy, xz, yz;
    double sqrtDet;
} BilForm;

typedef double Position3D[3];
typedef double Position2D[2];

typedef struct Particule{
    Position3D pos;
    double mass, dens, h;
    BilForm shape;
} Particule;

typedef struct PointCalcul{
    Position3D pos;
    double dens;
    unsigned long idp;
} PointCalcul;

typedef struct LPart{
    Particule part;
    struct LPart *suivant;
} LPart;

typedef struct LPoint{
    PointCalcul point;
    struct LPoint *suivant;
} LPoint;

void difference3D(Position3D x1, Position3D x2, Position3D result);

double bilFormSym3D(Position3D vector, BilForm mat);
//void normalizeMat(BilForm* pBil);
double determinantSym(BilForm bil);

void afficherPart(Particule part);

LPart* listeVide();
int estVide(LPart* l);
void ajouter(LPart **l, Particule part);
Particule obtenirElement(LPart* l);
LPart* obtenirListeSuivante(LPart* l);
void fixerListeSuivante(LPart* l, LPart* lSuiv);
void supprimerTete(LPart** l);
void supprimer(LPart **l);
int taille(LPart* l);
void afficher(LPart* l);

LPoint* listeVideP();
int estVideP(LPoint* l);
void ajouterP(LPoint **l, PointCalcul part);
PointCalcul obtenirElementP(LPoint* l);
LPoint* obtenirListeSuivanteP(LPoint* l);
void fixerListeSuivanteP(LPoint* l, LPoint* lSuiv);
void supprimerTeteP(LPoint** l);
void supprimerP(LPoint **l);
int tailleP(LPoint* l);
void afficherP(LPoint* l);

#endif //CONTINUEDDENSITY_TYPEUNIT_H
