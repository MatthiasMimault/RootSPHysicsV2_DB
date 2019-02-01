#include "GenCaseBis_T.h"


using namespace std;


//==============================================================================
/// Constructor of objects.
//==============================================================================
GenCaseBis_T::GenCaseBis_T()
{
	
}


//==============================================================================
/// Destructor of objects.
//==============================================================================
GenCaseBis_T::~GenCaseBis_T()
{
}

void GenCaseBis_T::UseGencase(std::string runPath) {
	string directoryXml = runPath + "/Def.xml";
	JXml xml; xml.LoadFile(directoryXml);

	int i;
	((xml.GetNode("case.casedef.constantsdef.useGencase", false))->ToElement())->QueryIntAttribute("value", &i);
	if (i == 1) {
		useGencase = true;
		printf("\n useGencase : true\n");
	}
	else {
		useGencase = false;
		printf("\n useGencase : false\n");
	}

}

//==============================================================================
/// Load the xml file Def and adds the informations about particles
/// Load csv file and create a JPartDataBi4 with particles data
//==============================================================================
void GenCaseBis_T::Bridge(std::string caseName) {
	int *idp;
	tdouble3 *pos;
	tdouble3 posMin;
	tdouble3 posMax;
	tfloat3 *vel;
	double *vol;
	float *mp;
	float *rhop;
	float rhop0;
	double rMax = 0;
	double borddomain = 0;
	int np;

	//calcul nb of particles
	np = calculNbParticles();
	printf("\nnp = %d\n", np);

	idp = (int*)malloc(sizeof(int)*(np));
	pos = (tdouble3*)malloc(sizeof(tdouble3)*(np));
	vel = (tfloat3*)malloc(sizeof(tfloat3)*(np));
	vol = (double*)malloc(sizeof(double)*(np));
	mp = (float*)malloc(sizeof(float)*(np));
	rhop = (float*)malloc(sizeof(float)*(np));
	rhop0 = loadRhop0();

	//load particles id & positions
	loadCsv(np, idp, vol, pos);
	rMax = computeRayMax(np, vol);
	printf("\nray = %1.10f\n", rMax);

	for (int i = 0; i < np; i++)
	{
		vel[i] = TFloat3(0,0,0);
	}
	computeMassP(np, vol, mp, rhop, rhop0);

	/*for (int i = 0; i < np; i++)
	{
		printf("\nx: %f  y: %f  z: %f", vel[i].x, vel[i].y, vel[i].z);
	}*/

	//research CasePosMax
	researchCasePosMaxAndMin(pos, np, &posMax, &posMin);
	printf("\nPosMax = x: %f  y: %f  z: %f\n", posMax.x, posMax.y, posMax.z);
	borddomain = computeBorddomain(np, posMax, posMin);
	printf("\nborddomain = %1.10f\n", borddomain);
	
	//void AddPartData(unsigned npok,const unsigned *idp, const tdouble3 *posd,const tfloat3 *vel,const float *rhop,bool externalpointer=true)
	jpd = new JPartDataBi4();
	jpd->ConfigBasic(0,1,"", "", "",false, NULL, "");
	jpd->ConfigParticles(np, 0, 0, 0, np, posMin, posMax, NULL, NULL);
	jpd->ConfigCtes(0,0,0,rhop0,0,0,0);
	jpd->AddPartInfo((unsigned)0, 0, (unsigned)np, 0, 0, 0, TDouble3(0,0,0), TDouble3(0, 0, 0), 0, 0);
	jpd->AddPartData_T((unsigned)np, idp, pos, vel, rhop, mp);
	
	if (test) jpd->SaveFileCase(caseName);
	else jpd->SaveFileCase(caseName + "0.xml");
	// add particles informations on Xml file
	updateXml(caseName, np, rMax, borddomain);
	
}

//==============================================================================
/// 
//==============================================================================
int GenCaseBis_T::calculNbParticles() {
	string line;
	ifstream file("Data.csv");

	printf("\n------entre ds calculNb------");

	if (file.good())
	{
		printf("\ncalcule np\n");

		getline(file, line); //don't need first line 

		int nbLine = 0;
		while (getline(file, line))
		{
			nbLine++;
		}
		file.close();

		return (nbLine - 4); //-4 because there is 4 description lines at the end of file
	}
	else return 0;
}


//==============================================================================
/// split the line 
//==============================================================================
vector<std::string> GenCaseBis_T::split(std::string line, char delim)
{
	vector<std::string> result(6);
	string current;
	
	//printf("\nentre ds split");

	if (!line.empty())
	{
		size_t length = line.length();
		size_t pointer = 0;
		size_t pointerResult = 0;
		current = string();

		while (pointer < length)
		{
			if (line[pointer] != delim) {
			current += line[pointer];
			}
			else
			{
				result[pointerResult] = current;
				pointerResult++;
				current.clear();
			}
			pointer++;
		}
		result[pointerResult] = current;
		current.clear();
	}
		
	return result;
}


//==============================================================================
/// extract particles position and return a tdouble3 with positions
//==============================================================================
void GenCaseBis_T::loadCsv(int np, int *idp, double *vol, tdouble3 *pos) {
	string line;
	ifstream file("Data.csv");
	vector<std::string> tempo;
	

	printf("\n------entre ds loadCsv------");

	if (file.good())
	{
		printf("\nrempli  pos et idp\n");
		getline(file, line); //don't need first line 

		for (size_t i = 0; i < np; i++)
		{
			getline(file, line);
			tempo = split(line, ',');
			//printf("\nString = %s\nString split = 1: %s 2: %s 3: %s 4: %s 5: %s  6: %s", line.c_str(), tempo[0].c_str(), tempo[1].c_str(), tempo[2].c_str(), tempo[3].c_str(), tempo[4].c_str(), tempo[5].c_str());
			//idp[i] = (int)(::atof(tempo[0].c_str()));
			idp[i] = int(i);
			vol[i] = ::atof(tempo[2].c_str()) * 0.000000000000000001;
			pos[i].x = ::atof(tempo[3].c_str()) * 0.000001;
			pos[i].y = ::atof(tempo[4].c_str()) * 0.000001;
			pos[i].z = ::atof(tempo[5].c_str()) * 0.000001;
		}

		file.close();
	}
}

//==============================================================================
/// read xml file to load rhop0 value
//==============================================================================
float GenCaseBis_T::loadRhop0() {
	string directoryXml = "Def.xml";
	JXml xml; xml.LoadFile(directoryXml);
	float res;
	(((xml.GetNode("case.casedef.constantsdef.rhop0", false))->ToElement())->QueryFloatAttribute("value", &res));
	return res;
}

//==============================================================================
///calcul particles' rayon
//==============================================================================
double GenCaseBis_T::computeRayMax(int np, double *vol) {
	double res = 0;
		
	for ( int i = 0; i < np; i++)
	{
		double ray = pow((vol[i]*3.0/4.0/PI), 1.0/3.0);
		res = ray > res ? ray : res;
	}

	return res;
}

//==============================================================================
///calcul particles' and densoty' particles
//==============================================================================
void GenCaseBis_T::computeMassP(int np, double *vol, float *mp, float *rhop, float rhop0){

	for (size_t i = 0; i < np; i++)
	{
		mp[i] = rhop0 * float(vol[i]);
		rhop[i] = rhop0;
	}

}

//==============================================================================
/// Research the maximum position of all particls
//==============================================================================
void GenCaseBis_T::researchCasePosMaxAndMin(tdouble3 *pos, int np, tdouble3 *posMax, tdouble3 *posMin) {

	tdouble3 &max = TDouble3(0, 0, 0);
	tdouble3 &min = TDouble3(0, 0, 0);

	printf("\n------entre ds researchCasePosMax------");

	for (int i = 0; i < np; i++)
	{
		max = MaxValues(max, pos[i]);
		//printf("\nPosMax = x: %f  y: %f  z: %f", max.x, max.y, max.z);

		min = MinValues(min, pos[i]);
		//printf("\nPosMin = x: %f  y: %f  z: %f", min.x, min.y, min.z);
	}
	*posMax = max;
	*posMin = min;
}

double GenCaseBis_T::computeBorddomain(int np, tdouble3 posMax, tdouble3 posMin) {
	double res = 0;

	double width = abs(posMax.x) > abs(posMin.x) ? abs(posMax.x) : abs(posMin.x);
	double length = abs(posMax.y) > abs(posMin.y) ? abs(posMax.y) : abs(posMin.y);
	double height = abs(posMax.z) > abs(posMin.z) ? abs(posMax.z) : abs(posMin.z);

	res = width > length ? width : length;
	res = height > res ? height : res;

	return res;
}


//==============================================================================
/// Add particles's informations on the Xml file 
//==============================================================================
void GenCaseBis_T::updateXml(std::string caseName, int np, double rMax, double borddomain) {
	string directoryXml = "Def.xml";
	JXml xml; xml.LoadFile(directoryXml);

	TiXmlNode *node = xml.GetNode("case", false); 
	TiXmlElement *ele1 = node->FirstChildElement(); //ele1 point on the "casedef" node
	TiXmlElement *ele2 = ele1->NextSiblingElement(); //ele2 point on the "execution" node
	ele1 = ele1->FirstChildElement(); //ele1 point on the "constantsdef" node

	((xml.GetNode("case.casedef.constantsdef.borddomain", false))->ToElement())->SetDoubleAttribute("value", (borddomain*2));



	//-------------------------------------------
	//feel particles part
	//-------------------------------------------
	TiXmlElement particles("particles");
	particles.SetAttribute("np", np);
	particles.SetAttribute("nb", "0");
	particles.SetAttribute("nbf", "0");
	particles.SetAttribute("mkboundfirst", "11");
	particles.SetAttribute("mkfluidfirst", "1");

	TiXmlElement summary("_summary");
	TiXmlElement fluid("fluid");
	fluid.SetAttribute("count", np);
	char tampon[32];
	sprintf_s(tampon, "0-%d", (np-1));
	fluid.SetAttribute("id", tampon);
	fluid.SetAttribute("mkcount", "1");
	fluid.SetAttribute("mkvalues", "1");

	(&summary)->InsertEndChild(fluid);
	(&particles)->InsertEndChild(summary);

	TiXmlElement fluid2("fluid");
	fluid2.SetAttribute("mkfluid", "0");
	fluid2.SetAttribute("mk", "1");
	fluid2.SetAttribute("begin", "0");
	fluid2.SetAttribute("count", np);

	(&particles)->InsertEndChild(fluid2);
	ele2->InsertEndChild(particles);	//insert particles into ele (execution note)
	//end------------------
	//-------------------------------------------

	//-------------------------------------------
	//feel constants part
	//-------------------------------------------
	TiXmlElement constants("constants");

	TiXmlElement gravity("gravity");
	
	gravity.SetAttribute("x", ((xml.GetNode("case.casedef.constantsdef.gravity", false))->ToElement())->Attribute("x"));
	gravity.SetAttribute("y", ((xml.GetNode("case.casedef.constantsdef.gravity", false))->ToElement())->Attribute("y"));
	gravity.SetAttribute("z", ((xml.GetNode("case.casedef.constantsdef.gravity", false))->ToElement())->Attribute("z"));
	gravity.SetAttribute("units_comment", "m/s^2");
	(&constants)->InsertEndChild(gravity);

	TiXmlElement cflnumber("cflnumber");
	cflnumber.SetAttribute("value", ((xml.GetNode("case.casedef.constantsdef.cflnumber", false))->ToElement())->Attribute("value"));
	(&constants)->InsertEndChild(cflnumber);

	TiXmlElement gamma("gamma");
	gamma.SetAttribute("value", ((xml.GetNode("case.casedef.constantsdef.gamma", false))->ToElement())->Attribute("value"));
	(&constants)->InsertEndChild(gamma);

	TiXmlElement rhop0("rhop0");
	rhop0.SetAttribute("value", ((xml.GetNode("case.casedef.constantsdef.rhop0", false))->ToElement())->Attribute("value"));
	rhop0.SetAttribute("units_comment", ((xml.GetNode("case.casedef.constantsdef.rhop0", false))->ToElement())->Attribute("units_comment"));
	(&constants)->InsertEndChild(rhop0);

	char tampon2[32];
	sprintf_s(tampon2, "%1.16f", rMax);
	TiXmlElement dp("dp");
	dp.SetAttribute("value", tampon2);
	dp.SetAttribute("units_comment", "metres (m)");
	(&constants)->InsertEndChild(dp);


	double d = rMax * 3.5;
	char tampon3[32];
	sprintf_s(tampon3, "%1.16f", d);
	TiXmlElement h("h");
	h.SetAttribute("value", tampon3);
	h.SetAttribute("units_comment", "metres (m)");
	(&constants)->InsertEndChild(h);

	//a calculer
	TiXmlElement b("b");
	b.SetAttribute("value", "1.4285714286E+04");
	b.SetAttribute("units_comment", "Pascal (Pa)");
	(&constants)->InsertEndChild(b);

	//useless
	/*TiXmlElement massbound("massbound");
	massbound.SetAttribute("value", "8.0000000000E+00");
	massbound.SetAttribute("units_comment", "kg");
	(&constants)->InsertEndChild(massbound);

	//useless
	TiXmlElement massfluid("massfluid");
	massfluid.SetAttribute("value", "8.0000000000E+00");
	massfluid.SetAttribute("units_comment", "kg");
	(&constants)->InsertEndChild(massfluid);*/

	ele2->InsertEndChild(constants);//insert constants into ele (execution note)
	//end------------------
	//-------------------------------------------


	TiXmlElement motion("motion");
	ele2->InsertEndChild(motion);//insert motion into ele (execution note)

	if (test) xml.SaveFile(caseName +".xml");//save the xml file
	else xml.SaveFile(caseName + "0.xml");//save the xml file
}

