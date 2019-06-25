#pragma once



#ifndef _GENCASEBIS_T_
#define _GENCASEBIS_T_

#include "JPartDataBi4.h"
#include "JXml.h"
#include "TypesDef.h"
#include "JSph.h"
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

class GenCaseBis_T
{

private:
	JPartDataBi4 *jpd;
	bool test = true;
	bool useGencase;

public:
	GenCaseBis_T();
	~GenCaseBis_T();
	void UseGencase(std::string caseName);
	void Bridge(std::string caseName);
	bool getUseGencase() { return useGencase; }
private:
	int calculNbParticles();
	std::vector<std::string> split(std::string line, char delim);
	void loadCsv(int np, int *idp, double *vol, tdouble3 *pos);
	float loadRhop0();
	double computeRayMax(int np, double *vol);
	void computeMassP(int np, double *vol, float *mp, float *rhop, float rhop0);
	void researchCasePosMaxAndMin(tdouble3 *pos, int np, tdouble3 *posMax, tdouble3 *posMin);
	double computeBorddomain(int np, tdouble3 posMax, tdouble3 posMin);
	void updateXml(std::string caseName, int np, double rMax, double borddomain);
};
#endif
