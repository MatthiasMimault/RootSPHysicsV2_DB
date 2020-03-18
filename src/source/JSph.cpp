//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2017 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics.

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.
*/

/// \file JSph.cpp \brief Implements the class \ref JSph.

#include "JSph.h"
#include "Functions.h"
#include "JSphMk.h"
#include "JSphMotion.h"
#include "JXml.h"
#include "JSpaceCtes.h"
#include "JSpaceEParms.h"
#include "JSpaceParts.h"
#include "JFormatFiles2.h"
#include "JCellDivCpu.h"
#include "JFormatFiles2.h"
#include "JSphDtFixed.h"
#include "JSaveDt.h"
#include "JTimeOut.h"
#include "JSphVisco.h"
#include "JGaugeSystem.h"
#include "JWaveGen.h"
#include "JSphAccInput.h"
#include "JPartDataBi4.h"
#include "JPartOutBi4Save.h"
#include "JPartFloatBi4.h"
#include "JPartsOut.h"
#include "JDamping.h"
#include "JSphInitialize.h"
#include <climits>
#include <string>
#include <iostream>
#include <sstream>

//using namespace std;
using std::string;
using std::ofstream;
using std::endl;
using std::max;

//==============================================================================
/// Constructor.
//==============================================================================
JSph::JSph(bool cpu,bool withmpi):Cpu(cpu),WithMpi(withmpi){
  ClassName="JSph";
  DataBi4=NULL;
  DataOutBi4=NULL;
  DataFloatBi4=NULL;
  PartsOut=NULL;
  Log=NULL;
  ViscoTime=NULL;
  DtFixed=NULL;
  SaveDt=NULL;
  TimeOut=NULL;
  MkInfo=NULL;
  Motion=NULL;
  MotionObjBegin=NULL;
  FtObjs=NULL;
  DemData=NULL;
  GaugeSystem=NULL;
  WaveGen=NULL;
  Damping=NULL;
  AccInput=NULL;
  InitVars();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSph::~JSph(){
  DestructorActive=true;
  delete DataBi4;
  delete DataOutBi4;
  delete DataFloatBi4;
  delete PartsOut;
  delete ViscoTime;
  delete DtFixed;
  delete SaveDt;
  delete TimeOut;
  delete MkInfo;
  delete Motion;
  delete[] MotionObjBegin; MotionObjBegin=NULL;
  AllocMemoryFloating(0);
  delete[] DemData; DemData=NULL;
  delete GaugeSystem;
  delete WaveGen;
  delete Damping;
  delete AccInput;
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSph::InitVars(){
  ClearCfgDomain();
  OutPosCount=OutRhopCount=OutMoveCount=0;
  Simulate2D=false;
  Simulate2DPosY=0;
  Stable=false;
  Psingle=true;
  SvDouble=false;
  RunCode=CalcRunCode();
  RunTimeDate="";
  RunCommand=""; RunPath="";
  CaseName=""; DirCase=""; RunName="";
  DirOut="";
  DirDataOut="";
  FileXml = "";
  AddFileXml_M = "";
  TStep=STEP_None;
  VerletSteps=40;
  TKernel=KERNEL_Wendland;
  Awen=Bwen=Agau=Bgau=0;
  memset(&CubicCte,0,sizeof(StCubicCte));
  TVisco=VISCO_None;
  TDeltaSph=DELTA_None; DeltaSph=0;
  TShifting=SHIFT_None; ShiftCoef=ShiftTFS=0;
  Visco=0; ViscoBoundFactor=1;
  UseDEM=false;  //(DEM)
  DemDtForce=0;  //(DEM)
  delete[] DemData; DemData=NULL;  //(DEM)
  RhopOut=true; RhopOutMin=700; RhopOutMax=1300;
  TimeMax=TimePart=0;
  DtIni=DtMin=0; CoefDtMin=0; DtAllParticles=false;
  PartsOutMax=0;
  NpMinimum=0;
  PartsOutWrn=1; PartsOutTotWrn=10;

  SvData=byte(SDAT_Binx)|byte(SDAT_Info);
  SvRes=false;
  SvTimers=false;
  SvDomainVtk=false;

  H=Hmin=Hmax=hmin=hmax=CteB=Gamma=RhopZero=CFLnumber=0;
  // Matthias
  // # dev
  dev_asph = true;
  typeCase = typeCompression = typeGrowth = typeDivision = typeYoung = 0;
  Dp=0;
  Cs0=0;
  Delta2H=0;
  MassFluid=MassBound=0;
  Gravity=TFloat3(0);
  Dosh=H2=Fourh2=Eta2=0;
  SpsSmag=SpsBlin=0;
  // Matthias
  LocDiv_M = TDouble3(0,0,0);
  VelDiv_M = TDouble3(0,0,0);
  VelDivCoef_M = 0;
  PoreZero = RateBirth_M = Spread_M = 0;
  LambdaMass = 0;
  SizeDivision_M = 0;
  AnisotropyK_M = TFloat3(0);
  AnisotropyG_M = TSymatrix3f(0);

  CasePosMin=CasePosMax=TDouble3(0);
  CaseNp=CaseNbound=CaseNfixed=CaseNmoving=CaseNfloat=CaseNfluid=CaseNpb=0;

  memset(&PeriodicConfig,0,sizeof(StPeriodic));
  PeriActive=0;
  PeriX=PeriY=PeriZ=PeriXY=PeriXZ=PeriYZ=false;
  PeriXinc=PeriYinc=PeriZinc=TDouble3(0);

  PartBeginDir="";
  PartBegin=PartBeginFirst=0;
  PartBeginTimeStep=0;
  PartBeginTotalNp=0;

  MotionTimeMod=0;
  MotionObjCount=0;
  delete[] MotionObjBegin; MotionObjBegin=NULL;

  FtCount=0;
  FtPause=0;

  AllocMemoryFloating(0);

  CellOrder=ORDER_None;
  CellMode=CELLMODE_None;
  Hdiv=0;
  Scell=0;
  MovLimit=0;

  Map_PosMin=Map_PosMax=Map_Size=TDouble3(0);
  Map_Cells=TUint3(0);
  MapRealPosMin=MapRealPosMax=MapRealSize=TDouble3(0);

  DomCelIni=DomCelFin=TUint3(0);
  DomCells=TUint3(0);
  DomPosMin=DomPosMax=DomSize=TDouble3(0);
  DomRealPosMin=DomRealPosMax=TDouble3(0);
  DomCellCode=0;

  NpDynamic=ReuseIds=false;
  TotalNp=0; IdMax=0;

  DtModif=0;
  DtModifWrn=1;
  PartDtMin=DBL_MAX; PartDtMax=-DBL_MAX;

  MaxMemoryCpu=MaxMemoryGpu=MaxParticles=MaxCells=0;

  PartIni=Part=0;
  Nstep=0; PartNstep=-1;
  PartOut=0;

  TimeStepIni=0;
  TimeStep=TimeStepM1=0;
  TimePartNext=0;
}

//==============================================================================
/// Generates a random code to identify the file of the results of the execution.
//==============================================================================
std::string JSph::CalcRunCode()const{
  srand((unsigned)time(NULL));
  const unsigned len=8;
  char code[len+1];
  for(unsigned c=0;c<len;c++){
    char let=char(float(rand())/float(RAND_MAX)*36);
    code[c]=(let<10? let+48: let+87);
  }
  code[len]=0;
  return(code);
}

//==============================================================================
/// Sets the configuration of the domain limits by default.
//==============================================================================
void JSph::ClearCfgDomain(){
  CfgDomainParticlesMin=CfgDomainParticlesMax=TDouble3(DBL_MAX);
  CfgDomainParticlesPrcMin=CfgDomainParticlesPrcMax=TDouble3(DBL_MAX);
  CfgDomainFixedMin=CfgDomainFixedMax=TDouble3(DBL_MAX);
}

//==============================================================================
/// Sets the configuration of the domain limits using given values.
//==============================================================================
void JSph::ConfigDomainFixed(tdouble3 vmin,tdouble3 vmax){
  ClearCfgDomain();
  CfgDomainFixedMin=vmin; CfgDomainFixedMax=vmax;
}

//==============================================================================
/// Sets the configuration of the domain limits using given values.
//==============================================================================
void JSph::ConfigDomainFixedValue(std::string key,double v){
  const char met[]="ConfigDomainFixedValue";
  const string keyend=(key.size()>=4? key.substr(key.size()-4,4): "");
       if(keyend=="Xmin")CfgDomainFixedMin.x=v;
  else if(keyend=="Ymin")CfgDomainFixedMin.y=v;
  else if(keyend=="Zmin")CfgDomainFixedMin.z=v;
  else if(keyend=="Xmax")CfgDomainFixedMax.x=v;
  else if(keyend=="Ymax")CfgDomainFixedMax.y=v;
  else if(keyend=="Zmax")CfgDomainFixedMax.z=v;
  else RunException(met,"Key for limit is invalid.");
}

//==============================================================================
/// Sets the configuration of the domain limits using positions of particles.
//==============================================================================
void JSph::ConfigDomainParticles(tdouble3 vmin,tdouble3 vmax){
  CfgDomainParticlesMin=vmin; CfgDomainParticlesMax=vmax;
}

//==============================================================================
/// Sets the configuration of the domain limits using positions of particles.
//==============================================================================
void JSph::ConfigDomainParticlesValue(std::string key,double v){
  const char met[]="ConfigDomainParticlesValue";
  const string keyend=(key.size()>=4? key.substr(key.size()-4,4): "");
       if(keyend=="Xmin")CfgDomainParticlesMin.x=v;
  else if(keyend=="Ymin")CfgDomainParticlesMin.y=v;
  else if(keyend=="Zmin")CfgDomainParticlesMin.z=v;
  else if(keyend=="Xmax")CfgDomainParticlesMax.x=v;
  else if(keyend=="Ymax")CfgDomainParticlesMax.y=v;
  else if(keyend=="Zmax")CfgDomainParticlesMax.z=v;
  else RunException(met,"Key for limit is invalid.");
}

//==============================================================================
/// Sets the configuration of the domain limits using positions plus a percentage.
//==============================================================================
void JSph::ConfigDomainParticlesPrc(tdouble3 vmin,tdouble3 vmax){
  CfgDomainParticlesPrcMin=vmin; CfgDomainParticlesPrcMax=vmax;
}

//==============================================================================
/// Sets the configuration of the domain limits using positions plus a percentage.
//==============================================================================
void JSph::ConfigDomainParticlesPrcValue(std::string key,double v){
  const char met[]="ConfigDomainParticlesValue";
  const string keyend=(key.size()>=4? key.substr(key.size()-4,4): "");
       if(keyend=="Xmin")CfgDomainParticlesPrcMin.x=v;
  else if(keyend=="Ymin")CfgDomainParticlesPrcMin.y=v;
  else if(keyend=="Zmin")CfgDomainParticlesPrcMin.z=v;
  else if(keyend=="Xmax")CfgDomainParticlesPrcMax.x=v;
  else if(keyend=="Ymax")CfgDomainParticlesPrcMax.y=v;
  else if(keyend=="Zmax")CfgDomainParticlesPrcMax.z=v;
  else RunException(met,"Key for limit is invalid.");
}

//==============================================================================
/// Allocates memory of floating objectcs.
//==============================================================================
void JSph::AllocMemoryFloating(unsigned ftcount){
  delete[] FtObjs; FtObjs=NULL;
  if(ftcount)FtObjs=new StFloatingData[ftcount];
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
llong JSph::GetAllocMemoryCpu()const{
  //-Allocated in AllocMemoryCase().
  llong s=0;
  //-Allocated in AllocMemoryFloating().
  if(FtObjs)s+=sizeof(StFloatingData)*FtCount;
  //-Allocated in other objects.
  if(PartsOut)s+=PartsOut->GetAllocMemory();
  if(ViscoTime)s+=ViscoTime->GetAllocMemory();
  if(DtFixed)s+=DtFixed->GetAllocMemory();
  if(AccInput)s+=AccInput->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Loads the configuration of the execution.
//==============================================================================
void JSph::LoadConfig(const JCfgRun *cfg){
  const char* met="LoadConfig";
  TimerTot.Start();
  Stable=cfg->Stable;
  Psingle=true; SvDouble=false; //-Options by default.
  RunCommand=cfg->RunCommand;
  RunPath=cfg->RunPath;
  DirOut=fun::GetDirWithSlash(cfg->DirOut);
  DirDataOut=(!cfg->DirDataOut.empty()? fun::GetDirWithSlash(DirOut+cfg->DirDataOut): DirOut);
  CaseName=cfg->CaseName;
  DirCase=fun::GetDirWithSlash(fun::GetDirParent(CaseName));
  CaseName=CaseName.substr(DirCase.length());
  if(!CaseName.length())RunException(met,"Name of the case for execution was not indicated.");
  RunName=(cfg->RunName.length()? cfg->RunName: CaseName);
  FileXml=DirCase+CaseName+".xml";
  //Log->Printf("FileXml=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, FileXml), 3).c_str());
  //Log->Printf("DirAddXml_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirAddXml_M), 3).c_str());
  //Log->Printf("AddFileXml_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, AddFileXml_M), 3).c_str());
  PartBeginDir=cfg->PartBeginDir; PartBegin=cfg->PartBegin; PartBeginFirst=cfg->PartBeginFirst;

  //-Output options:
  CsvSepComa=cfg->CsvSepComa;
  SvData=byte(SDAT_None);
  if(cfg->Sv_Csv&&!WithMpi)SvData|=byte(SDAT_Csv);
  if(cfg->Sv_Binx)SvData|=byte(SDAT_Binx);
  if(cfg->Sv_Info)SvData|=byte(SDAT_Info);
  if(cfg->Sv_Vtk)SvData|=byte(SDAT_Vtk);

  SvRes=cfg->SvRes;
  SvTimers=cfg->SvTimers;
  SvDomainVtk=cfg->SvDomainVtk;

  printf("\n");
  RunTimeDate=fun::GetDateTime();
  Log->Printf("[Initialising %s  %s]",ClassName.c_str(),RunTimeDate.c_str());

  Log->Printf("ProgramFile=\"%s\"",fun::GetPathLevels(fun::GetCanonicalPath(RunPath,RunCommand),3).c_str());
  Log->Printf("ExecutionDir=\"%s\"",fun::GetPathLevels(RunPath,3).c_str());
  Log->Printf("XmlFile=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, FileXml), 3).c_str());
  //Log->Printf("AddXmlFile_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, AddFileXml_M), 3).c_str());
  Log->Printf("OutputDir=\"%s\"",fun::GetPathLevels(fun::GetCanonicalPath(RunPath,DirOut),3).c_str());
  Log->Printf("OutputDataDir=\"%s\"",fun::GetPathLevels(fun::GetCanonicalPath(RunPath,DirDataOut),3).c_str());

  if(PartBegin){
    Log->Print(fun::VarStr("PartBegin",PartBegin));
    Log->Print(fun::VarStr("PartBeginDir",PartBeginDir));
    Log->Print(fun::VarStr("PartBeginFirst",PartBeginFirst));
  }

  LoadCaseConfig();

  //-Aplies configuration using command line.
  if(cfg->PosDouble==0){      Psingle=true;  SvDouble=false; }
  else if(cfg->PosDouble==1){ Psingle=false; SvDouble=false; }
  else if(cfg->PosDouble==2){ Psingle=false; SvDouble=true;  }
  if(cfg->TStep)TStep=cfg->TStep;
  if(cfg->VerletSteps>=0)VerletSteps=cfg->VerletSteps;
  if(cfg->TKernel)TKernel=cfg->TKernel;
  if(cfg->TVisco){ TVisco=cfg->TVisco; Visco=cfg->Visco; }
  if(cfg->ViscoBoundFactor>=0)ViscoBoundFactor=cfg->ViscoBoundFactor;
  if(cfg->DeltaSph>=0){
    DeltaSph=cfg->DeltaSph;
    TDeltaSph=(DeltaSph? DELTA_Dynamic: DELTA_None);
  }
  if(TDeltaSph==DELTA_Dynamic && Cpu)TDeltaSph=DELTA_DynamicExt; //-It is necessary because the interaction is divided in two steps: fluid-fluid/float and fluid-bound.

  // #Shift
  if(cfg->Shifting>=0){
    switch(cfg->Shifting){
      case 0:  TShifting=SHIFT_None;     break;
      case 1:  TShifting=SHIFT_NoBound;  break;
      case 2:  TShifting=SHIFT_NoFixed;  break;
      case 3:  TShifting=SHIFT_Full;     break;
      default: RunException(met,"Shifting mode is not valid.");
    }
    if(TShifting!=SHIFT_None){
      ShiftCoef=-2; ShiftTFS=0;
    }
    else ShiftCoef=ShiftTFS=0;
  }

  if(cfg->FtPause>=0)FtPause=cfg->FtPause;
  if(cfg->TimeMax>0)TimeMax=cfg->TimeMax;
  //-Configuration of JTimeOut with TimePart.
  TimeOut=new JTimeOut();
  if(cfg->TimePart>=0){
    TimePart=cfg->TimePart;
    TimeOut->Config(TimePart);
  }
  else TimeOut->Config(FileXml,"case.execution.special.timeout",TimePart);

  CellOrder=cfg->CellOrder;
  CellMode=cfg->CellMode;
  if(cfg->DomainMode==1){
    ConfigDomainParticles(cfg->DomainParticlesMin,cfg->DomainParticlesMax);
    ConfigDomainParticlesPrc(cfg->DomainParticlesPrcMin,cfg->DomainParticlesPrcMax);
  }
  else if(cfg->DomainMode==2)ConfigDomainFixed(cfg->DomainFixedMin,cfg->DomainFixedMax);
  if(cfg->RhopOutModif){
    RhopOutMin=cfg->RhopOutMin; RhopOutMax=cfg->RhopOutMax;
  }
  RhopOut=(RhopOutMin<RhopOutMax);
  if(!RhopOut){ RhopOutMin=-FLT_MAX; RhopOutMax=FLT_MAX; }
}

//==============================================================================
/// Loads the configuration of the execution and modify the xml.
//==============================================================================
void JSph::LoadConfig_Mixed_M(const JCfgRun* cfg) {
	const char* met = "LoadConfig";
	TimerTot.Start();
	Stable = cfg->Stable;
	Psingle = true; SvDouble = false; //-Options by default.
	RunCommand = cfg->RunCommand;
	RunPath = cfg->RunPath;
	DirOut = fun::GetDirWithSlash(cfg->DirOut);
	DirDataOut = (!cfg->DirDataOut.empty() ? fun::GetDirWithSlash(DirOut + cfg->DirDataOut) : DirOut);
	CaseName = cfg->CaseName;
	DirCase = fun::GetDirWithSlash(fun::GetDirParent(CaseName));
	CaseName = CaseName.substr(DirCase.length());
	if (!CaseName.length())RunException(met, "Name of the case for execution was not indicated.");
	RunName = (cfg->RunName.length() ? cfg->RunName : CaseName);
	FileXml = DirCase + CaseName + ".xml";
	//Log->Printf("FileXml=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, FileXml), 3).c_str());
	//Log->Printf("DirAddXml_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirAddXml_M), 3).c_str());
	//Log->Printf("AddFileXml_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, AddFileXml_M), 3).c_str());
	PartBeginDir = cfg->PartBeginDir; PartBegin = cfg->PartBegin; PartBeginFirst = cfg->PartBeginFirst;

	//-Output options:
	CsvSepComa = cfg->CsvSepComa;
	SvData = byte(SDAT_None);
	if (cfg->Sv_Csv && !WithMpi)SvData |= byte(SDAT_Csv);
	if (cfg->Sv_Binx)SvData |= byte(SDAT_Binx);
	if (cfg->Sv_Info)SvData |= byte(SDAT_Info);
	if (cfg->Sv_Vtk)SvData |= byte(SDAT_Vtk);

	SvRes = cfg->SvRes;
	SvTimers = cfg->SvTimers;
	SvDomainVtk = cfg->SvDomainVtk;

	printf("\n");
	RunTimeDate = fun::GetDateTime();
	Log->Printf("[Initialising %s  %s]", ClassName.c_str(), RunTimeDate.c_str());

	Log->Printf("ProgramFile=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, RunCommand), 3).c_str());
	Log->Printf("ExecutionDir=\"%s\"", fun::GetPathLevels(RunPath, 3).c_str());
	Log->Printf("XmlFile=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, FileXml), 3).c_str());
	//Log->Printf("AddXmlFile_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, AddFileXml_M), 3).c_str());
	Log->Printf("OutputDir=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirOut), 3).c_str());
	Log->Printf("OutputDataDir=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirDataOut), 3).c_str());

	if (PartBegin) {
		Log->Print(fun::VarStr("PartBegin", PartBegin));
		Log->Print(fun::VarStr("PartBeginDir", PartBeginDir));
		Log->Print(fun::VarStr("PartBeginFirst", PartBeginFirst));
	}

	// Load and update case
	// #XMLUpdate
	UpdateCaseConfig_Mixed_M();
	LoadCaseConfig();
	

	//-Aplies configuration using command line.
	if (cfg->PosDouble == 0) { Psingle = true;  SvDouble = false; }
	else if (cfg->PosDouble == 1) { Psingle = false; SvDouble = false; }
	else if (cfg->PosDouble == 2) { Psingle = false; SvDouble = true; }
	if (cfg->TStep)TStep = cfg->TStep;
	if (cfg->VerletSteps >= 0)VerletSteps = cfg->VerletSteps;
	if (cfg->TKernel)TKernel = cfg->TKernel;
	if (cfg->TVisco) { TVisco = cfg->TVisco; Visco = cfg->Visco; }
	if (cfg->ViscoBoundFactor >= 0)ViscoBoundFactor = cfg->ViscoBoundFactor;
	if (cfg->DeltaSph >= 0) {
		DeltaSph = cfg->DeltaSph;
		TDeltaSph = (DeltaSph ? DELTA_Dynamic : DELTA_None);
	}
	if (TDeltaSph == DELTA_Dynamic && Cpu)TDeltaSph = DELTA_DynamicExt; //-It is necessary because the interaction is divided in two steps: fluid-fluid/float and fluid-bound.

	// #Shift
	if (cfg->Shifting >= 0) {
		switch (cfg->Shifting) {
		case 0:  TShifting = SHIFT_None;     break;
		case 1:  TShifting = SHIFT_NoBound;  break;
		case 2:  TShifting = SHIFT_NoFixed;  break;
		case 3:  TShifting = SHIFT_Full;     break;
		default: RunException(met, "Shifting mode is not valid.");
		}
		if (TShifting != SHIFT_None) {
			ShiftCoef = -2; ShiftTFS = 0;
		}
		else ShiftCoef = ShiftTFS = 0;
	}

	if (cfg->FtPause >= 0)FtPause = cfg->FtPause;
	if (cfg->TimeMax > 0)TimeMax = cfg->TimeMax;
	//-Configuration of JTimeOut with TimePart.
	TimeOut = new JTimeOut();
	if (cfg->TimePart >= 0) {
		TimePart = cfg->TimePart;
		TimeOut->Config(TimePart);
	}
	else TimeOut->Config(FileXml, "case.execution.special.timeout", TimePart);

	CellOrder = cfg->CellOrder;
	CellMode = cfg->CellMode;
	if (cfg->DomainMode == 1) {
		ConfigDomainParticles(cfg->DomainParticlesMin, cfg->DomainParticlesMax);
		ConfigDomainParticlesPrc(cfg->DomainParticlesPrcMin, cfg->DomainParticlesPrcMax);
	}
	else if (cfg->DomainMode == 2)ConfigDomainFixed(cfg->DomainFixedMin, cfg->DomainFixedMax);
	if (cfg->RhopOutModif) {
		RhopOutMin = cfg->RhopOutMin; RhopOutMax = cfg->RhopOutMax;
	}
	RhopOut = (RhopOutMin < RhopOutMax);
	if (!RhopOut) { RhopOutMin = -FLT_MAX; RhopOutMax = FLT_MAX; }
}

//==============================================================================
/// Loads the configuration of the execution and modify the xml - Unified version
//==============================================================================
void JSph::LoadConfig_Uni_M(const JCfgRun* cfg) {
	const char* met = "LoadConfig";
	TimerTot.Start();
	Stable = cfg->Stable;
	Psingle = true; SvDouble = false; //-Options by default.
	RunCommand = cfg->RunCommand;
	RunPath = cfg->RunPath;
	DirOut = fun::GetDirWithSlash(cfg->DirOut);
	DirDataOut = (!cfg->DirDataOut.empty() ? fun::GetDirWithSlash(DirOut + cfg->DirDataOut) : DirOut);
	CaseName = cfg->CaseName;
	DirCase = fun::GetDirWithSlash(fun::GetDirParent(CaseName));
	CaseName = CaseName.substr(DirCase.length());
	if (!CaseName.length())RunException(met, "Name of the case for execution was not indicated.");
	RunName = (cfg->RunName.length() ? cfg->RunName : CaseName);
	FileXml = DirCase + CaseName + ".xml";
	PartBeginDir = cfg->PartBeginDir; PartBegin = cfg->PartBegin; PartBeginFirst = cfg->PartBeginFirst;

	//-Output options:
	CsvSepComa = cfg->CsvSepComa;
	SvData = byte(SDAT_None);
	if (cfg->Sv_Csv && !WithMpi)SvData |= byte(SDAT_Csv);
	if (cfg->Sv_Binx)SvData |= byte(SDAT_Binx);
	if (cfg->Sv_Info)SvData |= byte(SDAT_Info);
	if (cfg->Sv_Vtk)SvData |= byte(SDAT_Vtk);

	SvRes = cfg->SvRes;
	SvTimers = cfg->SvTimers;
	SvDomainVtk = cfg->SvDomainVtk;

	printf("\n");
	RunTimeDate = fun::GetDateTime();
	Log->Printf("[Initialising %s  %s]", ClassName.c_str(), RunTimeDate.c_str());

	Log->Printf("ProgramFile=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, RunCommand), 3).c_str());
	Log->Printf("ExecutionDir=\"%s\"", fun::GetPathLevels(RunPath, 3).c_str());
	Log->Printf("XmlFile=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, FileXml), 3).c_str());
	Log->Printf("OutputDir=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirOut), 3).c_str());
	Log->Printf("OutputDataDir=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirDataOut), 3).c_str());

	if (PartBegin) {
		Log->Print(fun::VarStr("PartBegin", PartBegin));
		Log->Print(fun::VarStr("PartBeginDir", PartBeginDir));
		Log->Print(fun::VarStr("PartBeginFirst", PartBeginFirst));
	}

	// Load and update case
	// #XMLUpdate
	// #Unified - check type case
	JXml xml; xml.LoadFile(FileXml);
	JSpaceCtes ctes;     
	ctes.LoadAddXmlRun_M(&xml, "case.casedef.constantsdef");
	typeCase = ctes.GetCase();
	if (typeCase == 1) UpdateCaseConfig_Mixed_M();
	LoadCaseConfig();


	//-Aplies configuration using command line.
	if (cfg->PosDouble == 0) { Psingle = true;  SvDouble = false; }
	else if (cfg->PosDouble == 1) { Psingle = false; SvDouble = false; }
	else if (cfg->PosDouble == 2) { Psingle = false; SvDouble = true; }
	if (cfg->TStep)TStep = cfg->TStep;
	if (cfg->VerletSteps >= 0)VerletSteps = cfg->VerletSteps;
	if (cfg->TKernel)TKernel = cfg->TKernel;
	if (cfg->TVisco) { TVisco = cfg->TVisco; Visco = cfg->Visco; }
	if (cfg->ViscoBoundFactor >= 0)ViscoBoundFactor = cfg->ViscoBoundFactor;
	if (cfg->DeltaSph >= 0) {
		DeltaSph = cfg->DeltaSph;
		TDeltaSph = (DeltaSph ? DELTA_Dynamic : DELTA_None);
	}
	if (TDeltaSph == DELTA_Dynamic && Cpu)TDeltaSph = DELTA_DynamicExt; //-It is necessary because the interaction is divided in two steps: fluid-fluid/float and fluid-bound.

	// #Shift
	if (cfg->Shifting >= 0) {
		switch (cfg->Shifting) {
		case 0:  TShifting = SHIFT_None;     break;
		case 1:  TShifting = SHIFT_NoBound;  break;
		case 2:  TShifting = SHIFT_NoFixed;  break;
		case 3:  TShifting = SHIFT_Full;     break;
		default: RunException(met, "Shifting mode is not valid.");
		}
		if (TShifting != SHIFT_None) {
			ShiftCoef = -2; ShiftTFS = 0;
		}
		else ShiftCoef = ShiftTFS = 0;
	}

	if (cfg->FtPause >= 0)FtPause = cfg->FtPause;
	if (cfg->TimeMax > 0)TimeMax = cfg->TimeMax;
	//-Configuration of JTimeOut with TimePart.
	TimeOut = new JTimeOut();
	if (cfg->TimePart >= 0) {
		TimePart = cfg->TimePart;
		TimeOut->Config(TimePart);
	}
	else TimeOut->Config(FileXml, "case.execution.special.timeout", TimePart);

	CellOrder = cfg->CellOrder;
	CellMode = cfg->CellMode;
	if (cfg->DomainMode == 1) {
		ConfigDomainParticles(cfg->DomainParticlesMin, cfg->DomainParticlesMax);
		ConfigDomainParticlesPrc(cfg->DomainParticlesPrcMin, cfg->DomainParticlesPrcMax);
	}
	else if (cfg->DomainMode == 2)ConfigDomainFixed(cfg->DomainFixedMin, cfg->DomainFixedMax);
	if (cfg->RhopOutModif) {
		RhopOutMin = cfg->RhopOutMin; RhopOutMax = cfg->RhopOutMax;
	}
	RhopOut = (RhopOutMin < RhopOutMax);
	if (!RhopOut) { RhopOutMin = -FLT_MAX; RhopOutMax = FLT_MAX; }
}

//==============================================================================
/// Loads the configuration of the execution.
//==============================================================================
void JSph::LoadConfig_T(const JCfgRun *cfg) {
	const char* met = "LoadConfig";
	TimerTot.Start();
	Stable = cfg->Stable;
	Psingle = true; SvDouble = false; //-Options by default.
	RunCommand = cfg->RunCommand;
	RunPath = cfg->RunPath;
	DirOut = fun::GetDirWithSlash(cfg->DirOut);
	DirDataOut = (!cfg->DirDataOut.empty() ? fun::GetDirWithSlash(DirOut + cfg->DirDataOut) : DirOut);
	CaseName = cfg->CaseName;
	DirCase = fun::GetDirWithSlash(fun::GetDirParent(CaseName));
	CaseName = CaseName.substr(DirCase.length());
	if (!CaseName.length())RunException(met, "Name of the case for execution was not indicated.");
	RunName = (cfg->RunName.length() ? cfg->RunName : CaseName);
	FileXml = DirCase + CaseName + ".xml";
	//Log->Printf("FileXml=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, FileXml), 3).c_str());
	//Log->Printf("DirAddXml_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirAddXml_M), 3).c_str());
	//Log->Printf("AddFileXml_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, AddFileXml_M), 3).c_str());
	PartBeginDir = cfg->PartBeginDir; PartBegin = cfg->PartBegin; PartBeginFirst = cfg->PartBeginFirst;

	//-Output options:
	CsvSepComa = cfg->CsvSepComa;
	SvData = byte(SDAT_None);
	if (cfg->Sv_Csv && !WithMpi)SvData |= byte(SDAT_Csv);
	if (cfg->Sv_Binx)SvData |= byte(SDAT_Binx);
	if (cfg->Sv_Info)SvData |= byte(SDAT_Info);
	if (cfg->Sv_Vtk)SvData |= byte(SDAT_Vtk);

	SvRes = cfg->SvRes;
	SvTimers = cfg->SvTimers;
	SvDomainVtk = cfg->SvDomainVtk;

	printf("\n");
	RunTimeDate = fun::GetDateTime();
	Log->Printf("[Initialising %s  %s]", ClassName.c_str(), RunTimeDate.c_str());

	Log->Printf("ProgramFile=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, RunCommand), 3).c_str());
	Log->Printf("ExecutionDir=\"%s\"", fun::GetPathLevels(RunPath, 3).c_str());
	Log->Printf("XmlFile=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, FileXml), 3).c_str());
	//Log->Printf("AddXmlFile_M=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, AddFileXml_M), 3).c_str());
	Log->Printf("OutputDir=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirOut), 3).c_str());
	Log->Printf("OutputDataDir=\"%s\"", fun::GetPathLevels(fun::GetCanonicalPath(RunPath, DirDataOut), 3).c_str());

	if (PartBegin) {
		Log->Print(fun::VarStr("PartBegin", PartBegin));
		Log->Print(fun::VarStr("PartBeginDir", PartBeginDir));
		Log->Print(fun::VarStr("PartBeginFirst", PartBeginFirst));
	}

	LoadCaseConfig_T();

	//-Aplies configuration using command line.
	if (cfg->PosDouble == 0) { Psingle = true;  SvDouble = false; }
	else if (cfg->PosDouble == 1) { Psingle = false; SvDouble = false; }
	else if (cfg->PosDouble == 2) { Psingle = false; SvDouble = true; }
	if (cfg->TStep)TStep = cfg->TStep;
	if (cfg->VerletSteps >= 0)VerletSteps = cfg->VerletSteps;
	if (cfg->TKernel)TKernel = cfg->TKernel;
	if (cfg->TVisco) { TVisco = cfg->TVisco; Visco = cfg->Visco; }
	if (cfg->ViscoBoundFactor >= 0)ViscoBoundFactor = cfg->ViscoBoundFactor;
	if (cfg->DeltaSph >= 0) {
		DeltaSph = cfg->DeltaSph;
		TDeltaSph = (DeltaSph ? DELTA_Dynamic : DELTA_None);
	}
	if (TDeltaSph == DELTA_Dynamic && Cpu)TDeltaSph = DELTA_DynamicExt; //-It is necessary because the interaction is divided in two steps: fluid-fluid/float and fluid-bound.

	if (cfg->Shifting >= 0) {
		switch (cfg->Shifting) {
		case 0:  TShifting = SHIFT_None;     break;
		case 1:  TShifting = SHIFT_NoBound;  break;
		case 2:  TShifting = SHIFT_NoFixed;  break;
		case 3:  TShifting = SHIFT_Full;     break;
		default: RunException(met, "Shifting mode is not valid.");
		}
		if (TShifting != SHIFT_None) {
			ShiftCoef = -2; ShiftTFS = 0;
		}
		else ShiftCoef = ShiftTFS = 0;
	}

	if (cfg->FtPause >= 0)FtPause = cfg->FtPause;
	if (cfg->TimeMax>0)TimeMax = cfg->TimeMax;
	//-Configuration of JTimeOut with TimePart.
	TimeOut = new JTimeOut();
	if (cfg->TimePart >= 0) {
		TimePart = cfg->TimePart;
		TimeOut->Config(TimePart);
	}
	else TimeOut->Config(FileXml, "case.execution.special.timeout", TimePart);

	CellOrder = cfg->CellOrder;
	CellMode = cfg->CellMode;
	if (cfg->DomainMode == 1) {
		ConfigDomainParticles(cfg->DomainParticlesMin, cfg->DomainParticlesMax);
		ConfigDomainParticlesPrc(cfg->DomainParticlesPrcMin, cfg->DomainParticlesPrcMax);
	}
	else if (cfg->DomainMode == 2)ConfigDomainFixed(cfg->DomainFixedMin, cfg->DomainFixedMax);
	if (cfg->RhopOutModif) {
		RhopOutMin = cfg->RhopOutMin; RhopOutMax = cfg->RhopOutMax;
	}
	RhopOut = (RhopOutMin<RhopOutMax);
	if (!RhopOut) { RhopOutMin = -FLT_MAX; RhopOutMax = FLT_MAX; }
}

//==============================================================================
/// Loads the case configuration to be executed.
//==============================================================================
void JSph::LoadCaseConfig(){
  const char* met="LoadCaseConfig";
  if(!fun::FileExists(FileXml))RunException(met,"Case configuration was not found.",FileXml);
  JXml xml; xml.LoadFile(FileXml);
  JSpaceCtes ctes;     ctes.LoadXmlRun(&xml, "case.execution.constants");
  //ctes.LoadAddXmlRun_M(&addXml, "case.constantsdef");
  ctes.LoadAddXmlRun_M(&xml, "case.casedef.constantsdef");
  JSpaceEParms eparms; eparms.LoadXml(&xml,"case.execution.parameters");
  JSpaceParts parts;   parts.LoadXml(&xml,"case.execution.particles");

  //-Execution parameters.
  switch(eparms.GetValueInt("PosDouble",true,0)){
    case 0:  Psingle=true;  SvDouble=false;  break;
    case 1:  Psingle=false; SvDouble=false;  break;
    case 2:  Psingle=false; SvDouble=true;   break;
    default: RunException(met,"PosDouble value is not valid.");
  }
  switch(eparms.GetValueInt("RigidAlgorithm",true,1)){ //(DEM)
    case 1:  UseDEM=false;  break;
    case 2:  UseDEM=true;   break;
    default: RunException(met,"Rigid algorithm is not valid.");
  }
  switch(eparms.GetValueInt("StepAlgorithm",true,1)){
    case 1:  TStep=STEP_Verlet;      break;
	case 2:  TStep = STEP_Symplectic;  break;
	case 3:  TStep = STEP_Euler;  break;
    default: RunException(met,"Step algorithm is not valid.");
  }
  VerletSteps=eparms.GetValueInt("VerletSteps",true,40);
  switch(eparms.GetValueInt("Kernel",true,2)){
    case 1:  TKernel=KERNEL_Cubic;     break;
    case 2:  TKernel=KERNEL_Wendland;  break;
    case 3:  TKernel=KERNEL_Gaussian;  break;
    default: RunException(met,"Kernel choice is not valid.");
  }
  switch(eparms.GetValueInt("ViscoTreatment",true,1)){
    case 1:  TVisco=VISCO_Artificial;  break;
    case 2:  TVisco=VISCO_LaminarSPS;  break;
    default: RunException(met,"Viscosity treatment is not valid.");
  }
  Visco=eparms.GetValueFloat("Visco");
  ViscoBoundFactor=eparms.GetValueFloat("ViscoBoundFactor",true,1.f);
  string filevisco=eparms.GetValueStr("ViscoTime",true);
  if(!filevisco.empty()){
    ViscoTime=new JSphVisco();
    ViscoTime->LoadFile(DirCase+filevisco);
  }
  DeltaSph=eparms.GetValueFloat("DeltaSPH",true,0);
  TDeltaSph=(DeltaSph? DELTA_Dynamic: DELTA_None);

  switch(eparms.GetValueInt("Shifting",true,0)){
    case 0:  TShifting=SHIFT_None;     break;
    case 1:  TShifting=SHIFT_NoBound;  break;
    case 2:  TShifting=SHIFT_NoFixed;  break;
    case 3:  TShifting=SHIFT_Full;     break;
    default: RunException(met,"Shifting mode is not valid.");
  }
  if(TShifting!=SHIFT_None){
    ShiftCoef=eparms.GetValueFloat("ShiftCoef",true,-2);
    if(ShiftCoef==0)TShifting=SHIFT_None;
    else ShiftTFS=eparms.GetValueFloat("ShiftTFS",true,0);
  }

  FtPause=eparms.GetValueFloat("FtPause",true,0);
  TimeMax=eparms.GetValueDouble("TimeMax");
  TimePart=eparms.GetValueDouble("TimeOut");

  DtIni=eparms.GetValueDouble("DtIni",true,0);
  DtMin=eparms.GetValueDouble("DtMin",true,0);
  CoefDtMin=eparms.GetValueFloat("CoefDtMin",true,0.05f);
  DtAllParticles=(eparms.GetValueInt("DtAllParticles",true,0)==1);

  string filedtfixed=eparms.GetValueStr("DtFixed",true);
  if(!filedtfixed.empty()){
    DtFixed=new JSphDtFixed();
    DtFixed->LoadFile(DirCase+filedtfixed);
  }
  if(eparms.Exists("RhopOutMin"))RhopOutMin=eparms.GetValueFloat("RhopOutMin");
  if(eparms.Exists("RhopOutMax"))RhopOutMax=eparms.GetValueFloat("RhopOutMax");
  PartsOutMax=eparms.GetValueFloat("PartsOutMax",true,1);

  //-Configuration of periodic boundaries.
  if(eparms.Exists("XPeriodicIncY")){ PeriXinc.y=eparms.GetValueDouble("XPeriodicIncY"); PeriX=true; }
  if(eparms.Exists("XPeriodicIncZ")){ PeriXinc.z=eparms.GetValueDouble("XPeriodicIncZ"); PeriX=true; }
  if(eparms.Exists("YPeriodicIncX")){ PeriYinc.x=eparms.GetValueDouble("YPeriodicIncX"); PeriY=true; }
  if(eparms.Exists("YPeriodicIncZ")){ PeriYinc.z=eparms.GetValueDouble("YPeriodicIncZ"); PeriY=true; }
  if(eparms.Exists("ZPeriodicIncX")){ PeriZinc.x=eparms.GetValueDouble("ZPeriodicIncX"); PeriZ=true; }
  if(eparms.Exists("ZPeriodicIncY")){ PeriZinc.y=eparms.GetValueDouble("ZPeriodicIncY"); PeriZ=true; }
  if(eparms.Exists("XYPeriodic")){ PeriXY=PeriX=PeriY=true; PeriXZ=PeriYZ=false; PeriXinc=PeriYinc=TDouble3(0); }
  if(eparms.Exists("XZPeriodic")){ PeriXZ=PeriX=PeriZ=true; PeriXY=PeriYZ=false; PeriXinc=PeriZinc=TDouble3(0); }
  if(eparms.Exists("YZPeriodic")){ PeriYZ=PeriY=PeriZ=true; PeriXY=PeriXZ=false; PeriYinc=PeriZinc=TDouble3(0); }
  PeriActive=(PeriX? 1: 0)+(PeriY? 2: 0)+(PeriZ? 4: 0);

  //-Configuration of domain size.
  float incz=eparms.GetValueFloat("IncZ",true,0.f);
  if(incz){
    ClearCfgDomain();
    CfgDomainParticlesPrcMax.z=incz;
  }
  string key;
  if(eparms.Exists(key="DomainParticles"))ConfigDomainParticles(TDouble3(eparms.GetValueNumDouble(key,0),eparms.GetValueNumDouble(key,1),eparms.GetValueNumDouble(key,2)),TDouble3(eparms.GetValueNumDouble(key,3),eparms.GetValueNumDouble(key,4),eparms.GetValueNumDouble(key,5)));
  if(eparms.Exists(key="DomainParticlesXmin"))ConfigDomainParticlesValue(key,-eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesYmin"))ConfigDomainParticlesValue(key,-eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesZmin"))ConfigDomainParticlesValue(key,-eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesXmax"))ConfigDomainParticlesValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesYmax"))ConfigDomainParticlesValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesZmax"))ConfigDomainParticlesValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesPrc"))ConfigDomainParticlesPrc(TDouble3(eparms.GetValueNumDouble(key,0),eparms.GetValueNumDouble(key,1),eparms.GetValueNumDouble(key,2)),TDouble3(eparms.GetValueNumDouble(key,3),eparms.GetValueNumDouble(key,4),eparms.GetValueNumDouble(key,5)));
  if(eparms.Exists(key="DomainParticlesPrcXmin"))ConfigDomainParticlesPrcValue(key,-eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesPrcYmin"))ConfigDomainParticlesPrcValue(key,-eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesPrcZmin"))ConfigDomainParticlesPrcValue(key,-eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesPrcXmax"))ConfigDomainParticlesPrcValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesPrcYmax"))ConfigDomainParticlesPrcValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainParticlesPrcZmax"))ConfigDomainParticlesPrcValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainFixed"))ConfigDomainFixed(TDouble3(eparms.GetValueNumDouble(key,0),eparms.GetValueNumDouble(key,1),eparms.GetValueNumDouble(key,2)),TDouble3(eparms.GetValueNumDouble(key,3),eparms.GetValueNumDouble(key,4),eparms.GetValueNumDouble(key,5)));
  if(eparms.Exists(key="DomainFixedXmin"))ConfigDomainFixedValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainFixedYmin"))ConfigDomainFixedValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainFixedZmin"))ConfigDomainFixedValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainFixedXmax"))ConfigDomainFixedValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainFixedYmax"))ConfigDomainFixedValue(key,eparms.GetValueDouble(key));
  if(eparms.Exists(key="DomainFixedZmax"))ConfigDomainFixedValue(key,eparms.GetValueDouble(key));

  //-Predefined constantes.
  if(ctes.GetEps()!=0)Log->PrintWarning("Eps value is not used (this correction is deprecated).");
  H=(float)ctes.GetH();
  Gamma=(float)ctes.GetGamma();
  RhopZero=(float)ctes.GetRhop0();
  CFLnumber=(float)ctes.GetCFLnumber();
  Dp=ctes.GetDp();
  Gravity=ToTFloat3(ctes.GetGravity());
  MassFluid=(float)ctes.GetMassFluid();
  MassBound=(float)ctes.GetMassBound();
  //Matthias
  // Simulation #choices markers
  //typeCase = ctes.GetCase();
  typeCompression = ctes.GetComp();
  typeDivision = ctes.GetDiv();
  typeGrowth = ctes.GetGrow();
  typeYoung = ctes.GetYoung();

  // Activation des conditions de bord
  PlanMirror = (float)ctes.GetPlanMirror();

  // Extension domain
  BordDomain = (float)ctes.GetBordDomain();
  // Solid anisotropic
  Ex = (float)ctes.GetYoungX();
  Ey = (float)ctes.GetYoungY();
  nuxy = (float)ctes.GetPoissonXY();
  nuyz = (float)ctes.GetPoissonYZ();
  Gf = (float)ctes.GetShear();

  //#Constants
  // Pore
  PoreZero = (float)ctes.GetPoreZero();
  // Mass
  LambdaMass = (float)ctes.GetLambdaMass();
  // Cell division
  SizeDivision_M = (float)ctes.GetSizeDivision();
  LocDiv_M = (tdouble3)ctes.GetLocalDivision();
  VelDivCoef_M = (float)ctes.GetVelocityDivisionCoef();
  Spread_M = (float)ctes.GetSpreadDivision();
  // Anisotropy
  AnisotropyK_M = ToTFloat3(ctes.GetAnisotropyK());
  AnisotropyG_M = ctes.GetAnisotropyG();

  Hmin = (float)ctes.GetHmin();
  Hmax = (float)ctes.GetHmax();

  //-Particle data.
  CaseNp=parts.Count();
  CaseNfixed=parts.Count(PT_Fixed);
  CaseNmoving=parts.Count(PT_Moving);
  CaseNfloat=parts.Count(PT_Floating);
  CaseNfluid=parts.Count(PT_Fluid);
  CaseNbound=CaseNp-CaseNfluid;
  CaseNpb=CaseNbound-CaseNfloat;

  NpDynamic=ReuseIds=false;
  TotalNp=CaseNp; IdMax=CaseNp-1;

  //-Loads and configures MK of particles.
  MkInfo=new JSphMk();
  MkInfo->Config(&parts);

  //-Configuration of GaugeSystem.
  GaugeSystem=new JGaugeSystem(Cpu,Log);

  //-Configuration of WaveGen.
  if(xml.GetNode("case.execution.special.wavepaddles",false)){
    bool useomp=false,usegpu=false;
    #ifdef OMP_USE_WAVEGEN
      useomp=(omp_get_max_threads()>1);
    #endif
    #ifdef _WITHGPU
      usegpu=!Cpu;
    #endif
    WaveGen=new JWaveGen(useomp,usegpu,Log,DirCase,&xml,"case.execution.special.wavepaddles");
  }

  //-Configuration of AccInput.
  if(xml.GetNode("case.execution.special.accinputs",false)){
    AccInput=new JSphAccInput(Log,DirCase,&xml,"case.execution.special.accinputs");
  }

  //-Loads and configures MOTION.
  MotionObjCount=parts.CountBlocks(PT_Moving);
  if(MotionObjCount){
    if(MotionObjCount>CODE_MKRANGEMAX)RunException(met,"The number of mobile objects exceeds the maximum.");
    //-Prepares memory.
    MotionObjBegin=new unsigned[MotionObjCount+1];
    memset(MotionObjBegin,0,sizeof(unsigned)*(MotionObjCount+1));
    //-Loads configuration.
    unsigned cmot=0;
    for(unsigned c=0;c<parts.CountBlocks();c++){
      const JSpacePartBlock &block=parts.GetBlock(c);
      if(block.Type==PT_Moving){
        if(cmot>=MotionObjCount)RunException(met,"The number of mobile objects exceeds the expected maximum.");
        //:printf("block[%2d]=%d -> %d\n",c,block.GetBegin(),block.GetCount());
        MotionObjBegin[cmot]=block.GetBegin();
        MotionObjBegin[cmot+1]=MotionObjBegin[cmot]+block.GetCount();
        if(WaveGen)WaveGen->ConfigPaddle(block.GetMkType(),cmot,block.GetBegin(),block.GetCount());
        cmot++;
      }
    }
    if(cmot!=MotionObjCount)RunException(met,"The number of mobile objects is invalid.");
  }

  if(MotionObjCount){
    Motion=new JSphMotion();
    if(int(MotionObjCount)<Motion->Init(&xml,"case.execution.motion",DirCase))RunException(met,"The number of mobile objects is lower than expected.");
  }

  //-Configuration of damping zones.
  if(xml.GetNode("case.execution.special.damping",false)){
    Damping=new JDamping(Log);
    Damping->LoadXml(&xml,"case.execution.special.damping");
  }

  //-Loads floating objects.
  FtCount=parts.CountBlocks(PT_Floating);
  if(FtCount){
    if(FtCount>CODE_MKRANGEMAX)RunException(met,"The number of floating objects exceeds the maximum.");
    AllocMemoryFloating(FtCount);
    unsigned cobj=0;
    for(unsigned c=0;c<parts.CountBlocks()&&cobj<FtCount;c++){
      const JSpacePartBlock &block=parts.GetBlock(c);
      if(block.Type==PT_Floating){
        const JSpacePartBlock_Floating &fblock=(const JSpacePartBlock_Floating &)block;
        StFloatingData* fobj=FtObjs+cobj;
        fobj->mkbound=fblock.GetMkType();
        fobj->begin=fblock.GetBegin();
        fobj->count=fblock.GetCount();
        fobj->mass=(float)fblock.GetMassbody();
        fobj->massp=fobj->mass/fobj->count;
        fobj->radius=0;
        fobj->center=fblock.GetCenter();
        fobj->angles=TFloat3(0);
        fobj->fvel=ToTFloat3(fblock.GetVelini());
        fobj->fomega=ToTFloat3(fblock.GetOmegaini());
        fobj->inertiaini=ToTMatrix3f(fblock.GetInertia());
        cobj++;
      }
    }
  }
  else UseDEM=false;

  //-Loads DEM data for the objects. (DEM)
  if(UseDEM){
    DemData=new StDemData[DemDataSize];
    memset(DemData,0,sizeof(StDemData)*DemDataSize);
    for(unsigned c=0;c<parts.CountBlocks();c++){
      const JSpacePartBlock &block=parts.GetBlock(c);
      if(block.Type!=PT_Fluid){
        const unsigned cmk=MkInfo->GetMkBlockByMkBound(block.GetMkType());
        if(cmk>=MkInfo->Size())RunException(met,fun::PrintStr("Error loading DEM objects. Mkbound=%u is unknown.",block.GetMkType()));
        const unsigned tav=CODE_GetTypeAndValue(MkInfo->Mkblock(cmk)->Code);
        //:Log->Printf("___> tav[%u]:%u",cmk,tav);
        if(block.Type==PT_Floating){
          const JSpacePartBlock_Floating &fblock=(const JSpacePartBlock_Floating &)block;
          DemData[tav].mass=(float)fblock.GetMassbody();
          DemData[tav].massp=(float)(fblock.GetMassbody()/fblock.GetCount());
        }
        else DemData[tav].massp=MassBound;
        if(!block.ExistsSubValue("Young_Modulus","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of Young_Modulus is invalid.",block.GetMk()));
        if(!block.ExistsSubValue("PoissonRatio","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of PoissonRatio is invalid.",block.GetMk()));
        if(!block.ExistsSubValue("Kfric","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of Kfric is invalid.",block.GetMk()));
        if(!block.ExistsSubValue("Restitution_Coefficient","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of Restitution_Coefficient is invalid.",block.GetMk()));
        DemData[tav].young=block.GetSubValueFloat("Young_Modulus","value",true,0);
        DemData[tav].poisson=block.GetSubValueFloat("PoissonRatio","value",true,0);
        DemData[tav].tau=(DemData[tav].young? (1-DemData[tav].poisson*DemData[tav].poisson)/DemData[tav].young: 0);
        DemData[tav].kfric=block.GetSubValueFloat("Kfric","value",true,0);
        DemData[tav].restitu=block.GetSubValueFloat("Restitution_Coefficient","value",true,0);
        if(block.ExistsValue("Restitution_Coefficient_User"))DemData[tav].restitu=block.GetValueFloat("Restitution_Coefficient_User");
      }
    }
  }

  NpMinimum=CaseNp-unsigned(PartsOutMax*CaseNfluid);
  Log->Print("**Basic case configuration is loaded");
}

//==============================================================================
/// Loads the case configuration to be executed.
//==============================================================================
void JSph::LoadCaseConfig_T() {
	const char* met = "LoadCaseConfig";
	if (!fun::FileExists(FileXml))RunException(met, "Case configuration was not found.", FileXml);
	JXml xml; xml.LoadFile(FileXml);
	JSpaceCtes ctes;     ctes.LoadXmlRun_T(&xml, "case.execution.constants");
	//ctes.LoadAddXmlRun_M(&addXml, "case.constantsdef");
	ctes.LoadAddXmlRun_M(&xml, "case.casedef.constantsdef");
	JSpaceEParms eparms; eparms.LoadXml(&xml, "case.execution.parameters");
	JSpaceParts parts;   parts.LoadXml(&xml, "case.execution.particles");

	//-Execution parameters.
	switch (eparms.GetValueInt("PosDouble", true, 0)) {
	case 0:  Psingle = true;  SvDouble = false;  break;
	case 1:  Psingle = false; SvDouble = false;  break;
	case 2:  Psingle = false; SvDouble = true;   break;
	default: RunException(met, "PosDouble value is not valid.");
	}
	switch (eparms.GetValueInt("RigidAlgorithm", true, 1)) { //(DEM)
	case 1:  UseDEM = false;  break;
	case 2:  UseDEM = true;   break;
	default: RunException(met, "Rigid algorithm is not valid.");
	}
	switch (eparms.GetValueInt("StepAlgorithm", true, 1)) {
	case 1:  TStep = STEP_Verlet;      break;
	case 2:  TStep = STEP_Symplectic;  break;
	case 3:  TStep = STEP_Euler;  break;
	default: RunException(met, "Step algorithm is not valid.");
	}
	VerletSteps = eparms.GetValueInt("VerletSteps", true, 40);
	switch (eparms.GetValueInt("Kernel", true, 2)) {
	case 1:  TKernel = KERNEL_Cubic;     break;
	case 2:  TKernel = KERNEL_Wendland;  break;
	case 3:  TKernel = KERNEL_Gaussian;  break;
	default: RunException(met, "Kernel choice is not valid.");
	}
	switch (eparms.GetValueInt("ViscoTreatment", true, 1)) {
	case 1:  TVisco = VISCO_Artificial;  break;
	case 2:  TVisco = VISCO_LaminarSPS;  break;
	default: RunException(met, "Viscosity treatment is not valid.");
	}
	Visco = eparms.GetValueFloat("Visco");
	ViscoBoundFactor = eparms.GetValueFloat("ViscoBoundFactor", true, 1.f);
	string filevisco = eparms.GetValueStr("ViscoTime", true);
	if (!filevisco.empty()) {
		ViscoTime = new JSphVisco();
		ViscoTime->LoadFile(DirCase + filevisco);
	}
	DeltaSph = eparms.GetValueFloat("DeltaSPH", true, 0);
	TDeltaSph = (DeltaSph ? DELTA_Dynamic : DELTA_None);

	switch (eparms.GetValueInt("Shifting", true, 0)) {
	case 0:  TShifting = SHIFT_None;     break;
	case 1:  TShifting = SHIFT_NoBound;  break;
	case 2:  TShifting = SHIFT_NoFixed;  break;
	case 3:  TShifting = SHIFT_Full;     break;
	default: RunException(met, "Shifting mode is not valid.");
	}
	if (TShifting != SHIFT_None) {
		ShiftCoef = eparms.GetValueFloat("ShiftCoef", true, -2);
		if (ShiftCoef == 0)TShifting = SHIFT_None;
		else ShiftTFS = eparms.GetValueFloat("ShiftTFS", true, 0);
	}

	FtPause = eparms.GetValueFloat("FtPause", true, 0);
	TimeMax = eparms.GetValueDouble("TimeMax");
	TimePart = eparms.GetValueDouble("TimeOut");

	DtIni = eparms.GetValueDouble("DtIni", true, 0);
	DtMin = eparms.GetValueDouble("DtMin", true, 0);
	CoefDtMin = eparms.GetValueFloat("CoefDtMin", true, 0.05f);
	DtAllParticles = (eparms.GetValueInt("DtAllParticles", true, 0) == 1);

	string filedtfixed = eparms.GetValueStr("DtFixed", true);
	if (!filedtfixed.empty()) {
		DtFixed = new JSphDtFixed();
		DtFixed->LoadFile(DirCase + filedtfixed);
	}
	if (eparms.Exists("RhopOutMin"))RhopOutMin = eparms.GetValueFloat("RhopOutMin");
	if (eparms.Exists("RhopOutMax"))RhopOutMax = eparms.GetValueFloat("RhopOutMax");
	PartsOutMax = eparms.GetValueFloat("PartsOutMax", true, 1);

	//-Configuration of periodic boundaries.
	if (eparms.Exists("XPeriodicIncY")) { PeriXinc.y = eparms.GetValueDouble("XPeriodicIncY"); PeriX = true; }
	if (eparms.Exists("XPeriodicIncZ")) { PeriXinc.z = eparms.GetValueDouble("XPeriodicIncZ"); PeriX = true; }
	if (eparms.Exists("YPeriodicIncX")) { PeriYinc.x = eparms.GetValueDouble("YPeriodicIncX"); PeriY = true; }
	if (eparms.Exists("YPeriodicIncZ")) { PeriYinc.z = eparms.GetValueDouble("YPeriodicIncZ"); PeriY = true; }
	if (eparms.Exists("ZPeriodicIncX")) { PeriZinc.x = eparms.GetValueDouble("ZPeriodicIncX"); PeriZ = true; }
	if (eparms.Exists("ZPeriodicIncY")) { PeriZinc.y = eparms.GetValueDouble("ZPeriodicIncY"); PeriZ = true; }
	if (eparms.Exists("XYPeriodic")) { PeriXY = PeriX = PeriY = true; PeriXZ = PeriYZ = false; PeriXinc = PeriYinc = TDouble3(0); }
	if (eparms.Exists("XZPeriodic")) { PeriXZ = PeriX = PeriZ = true; PeriXY = PeriYZ = false; PeriXinc = PeriZinc = TDouble3(0); }
	if (eparms.Exists("YZPeriodic")) { PeriYZ = PeriY = PeriZ = true; PeriXY = PeriXZ = false; PeriYinc = PeriZinc = TDouble3(0); }
	PeriActive = (PeriX ? 1 : 0) + (PeriY ? 2 : 0) + (PeriZ ? 4 : 0);

	//-Configuration of domain size.
	float incz = eparms.GetValueFloat("IncZ", true, 0.f);
	if (incz) {
		ClearCfgDomain();
		CfgDomainParticlesPrcMax.z = incz;
	}
	string key;
	if (eparms.Exists(key = "DomainParticles"))ConfigDomainParticles(TDouble3(eparms.GetValueNumDouble(key, 0), eparms.GetValueNumDouble(key, 1), eparms.GetValueNumDouble(key, 2)), TDouble3(eparms.GetValueNumDouble(key, 3), eparms.GetValueNumDouble(key, 4), eparms.GetValueNumDouble(key, 5)));
	if (eparms.Exists(key = "DomainParticlesXmin"))ConfigDomainParticlesValue(key, -eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesYmin"))ConfigDomainParticlesValue(key, -eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesZmin"))ConfigDomainParticlesValue(key, -eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesXmax"))ConfigDomainParticlesValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesYmax"))ConfigDomainParticlesValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesZmax"))ConfigDomainParticlesValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesPrc"))ConfigDomainParticlesPrc(TDouble3(eparms.GetValueNumDouble(key, 0), eparms.GetValueNumDouble(key, 1), eparms.GetValueNumDouble(key, 2)), TDouble3(eparms.GetValueNumDouble(key, 3), eparms.GetValueNumDouble(key, 4), eparms.GetValueNumDouble(key, 5)));
	if (eparms.Exists(key = "DomainParticlesPrcXmin"))ConfigDomainParticlesPrcValue(key, -eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesPrcYmin"))ConfigDomainParticlesPrcValue(key, -eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesPrcZmin"))ConfigDomainParticlesPrcValue(key, -eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesPrcXmax"))ConfigDomainParticlesPrcValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesPrcYmax"))ConfigDomainParticlesPrcValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainParticlesPrcZmax"))ConfigDomainParticlesPrcValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainFixed"))ConfigDomainFixed(TDouble3(eparms.GetValueNumDouble(key, 0), eparms.GetValueNumDouble(key, 1), eparms.GetValueNumDouble(key, 2)), TDouble3(eparms.GetValueNumDouble(key, 3), eparms.GetValueNumDouble(key, 4), eparms.GetValueNumDouble(key, 5)));
	if (eparms.Exists(key = "DomainFixedXmin"))ConfigDomainFixedValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainFixedYmin"))ConfigDomainFixedValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainFixedZmin"))ConfigDomainFixedValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainFixedXmax"))ConfigDomainFixedValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainFixedYmax"))ConfigDomainFixedValue(key, eparms.GetValueDouble(key));
	if (eparms.Exists(key = "DomainFixedZmax"))ConfigDomainFixedValue(key, eparms.GetValueDouble(key));

	//-Predefined constantes.
	if (ctes.GetEps() != 0)Log->PrintWarning("Eps value is not used (this correction is deprecated).");
	H = (float)ctes.GetH();
	//Hmin = (float)0.5 * SizeDivision_M * ctes.GetH();
	//Hmax=(float)ctes.GetHmax();
	//CteB = (float)ctes.GetB();
	Gamma = (float)ctes.GetGamma();
	RhopZero = (float)ctes.GetRhop0();
	CFLnumber = (float)ctes.GetCFLnumber();
	Dp = ctes.GetDp();
	Gravity = ToTFloat3(ctes.GetGravity());
	MassFluid = (float)ctes.GetMassFluid();
	MassBound = (float)ctes.GetMassBound();
	//Matthias
  // Extension domain
	BordDomain = (float)ctes.GetBordDomain();
	// Solid anisotropic
	Ex = (float)ctes.GetYoungX();
	Ey = (float)ctes.GetYoungY();
	nuxy = (float)ctes.GetPoissonXY();
	nuyz = (float)ctes.GetPoissonYZ();
	Gf = (float)ctes.GetShear();

	// Pore
	PoreZero = (float)ctes.GetPoreZero();
	// Mass
	LambdaMass = (float)ctes.GetLambdaMass();
	// Cell division
	SizeDivision_M = (float)ctes.GetSizeDivision();
	LocDiv_M = (tdouble3)ctes.GetLocalDivision();
	VelDivCoef_M = (float)ctes.GetVelocityDivisionCoef();
	Spread_M = (float)ctes.GetSpreadDivision();
	// Anisotropy
	AnisotropyK_M = ToTFloat3(ctes.GetAnisotropyK());
	AnisotropyG_M = ctes.GetAnisotropyG();

	//-Particle data.
	CaseNp = parts.Count();
	CaseNfixed = parts.Count(PT_Fixed);
	CaseNmoving = parts.Count(PT_Moving);
	CaseNfloat = parts.Count(PT_Floating);
	CaseNfluid = parts.Count(PT_Fluid);
	CaseNbound = CaseNp - CaseNfluid;
	CaseNpb = CaseNbound - CaseNfloat;

	NpDynamic = ReuseIds = false;
	TotalNp = CaseNp; IdMax = CaseNp - 1;

	//-Loads and configures MK of particles.
	MkInfo = new JSphMk();
	MkInfo->Config(&parts);

	//-Configuration of GaugeSystem.
	GaugeSystem = new JGaugeSystem(Cpu, Log);

	//-Configuration of WaveGen.
	if (xml.GetNode("case.execution.special.wavepaddles", false)) {
		bool useomp = false, usegpu = false;
#ifdef OMP_USE_WAVEGEN
		useomp = (omp_get_max_threads()>1);
#endif
#ifdef _WITHGPU
		usegpu = !Cpu;
#endif
		WaveGen = new JWaveGen(useomp, usegpu, Log, DirCase, &xml, "case.execution.special.wavepaddles");
	}

	//-Configuration of AccInput.
	if (xml.GetNode("case.execution.special.accinputs", false)) {
		AccInput = new JSphAccInput(Log, DirCase, &xml, "case.execution.special.accinputs");
	}

	//-Loads and configures MOTION.
	MotionObjCount = parts.CountBlocks(PT_Moving);
	if (MotionObjCount) {
		if (MotionObjCount>CODE_MKRANGEMAX)RunException(met, "The number of mobile objects exceeds the maximum.");
		//-Prepares memory.
		MotionObjBegin = new unsigned[MotionObjCount + 1];
		memset(MotionObjBegin, 0, sizeof(unsigned)*(MotionObjCount + 1));
		//-Loads configuration.
		unsigned cmot = 0;
		for (unsigned c = 0; c<parts.CountBlocks(); c++) {
			const JSpacePartBlock &block = parts.GetBlock(c);
			if (block.Type == PT_Moving) {
				if (cmot >= MotionObjCount)RunException(met, "The number of mobile objects exceeds the expected maximum.");
				//:printf("block[%2d]=%d -> %d\n",c,block.GetBegin(),block.GetCount());
				MotionObjBegin[cmot] = block.GetBegin();
				MotionObjBegin[cmot + 1] = MotionObjBegin[cmot] + block.GetCount();
				if (WaveGen)WaveGen->ConfigPaddle(block.GetMkType(), cmot, block.GetBegin(), block.GetCount());
				cmot++;
			}
		}
		if (cmot != MotionObjCount)RunException(met, "The number of mobile objects is invalid.");
	}

	if (MotionObjCount) {
		Motion = new JSphMotion();
		if (int(MotionObjCount)<Motion->Init(&xml, "case.execution.motion", DirCase))RunException(met, "The number of mobile objects is lower than expected.");
	}

	//-Configuration of damping zones.
	if (xml.GetNode("case.execution.special.damping", false)) {
		Damping = new JDamping(Log);
		Damping->LoadXml(&xml, "case.execution.special.damping");
	}

	//-Loads floating objects.
	FtCount = parts.CountBlocks(PT_Floating);
	if (FtCount) {
		if (FtCount>CODE_MKRANGEMAX)RunException(met, "The number of floating objects exceeds the maximum.");
		AllocMemoryFloating(FtCount);
		unsigned cobj = 0;
		for (unsigned c = 0; c<parts.CountBlocks() && cobj<FtCount; c++) {
			const JSpacePartBlock &block = parts.GetBlock(c);
			if (block.Type == PT_Floating) {
				const JSpacePartBlock_Floating &fblock = (const JSpacePartBlock_Floating &)block;
				StFloatingData* fobj = FtObjs + cobj;
				fobj->mkbound = fblock.GetMkType();
				fobj->begin = fblock.GetBegin();
				fobj->count = fblock.GetCount();
				fobj->mass = (float)fblock.GetMassbody();
				fobj->massp = fobj->mass / fobj->count;
				fobj->radius = 0;
				fobj->center = fblock.GetCenter();
				fobj->angles = TFloat3(0);
				fobj->fvel = ToTFloat3(fblock.GetVelini());
				fobj->fomega = ToTFloat3(fblock.GetOmegaini());
				fobj->inertiaini = ToTMatrix3f(fblock.GetInertia());
				cobj++;
			}
		}
	}
	else UseDEM = false;

	//-Loads DEM data for the objects. (DEM)
	if (UseDEM) {
		DemData = new StDemData[DemDataSize];
		memset(DemData, 0, sizeof(StDemData)*DemDataSize);
		for (unsigned c = 0; c<parts.CountBlocks(); c++) {
			const JSpacePartBlock &block = parts.GetBlock(c);
			if (block.Type != PT_Fluid) {
				const unsigned cmk = MkInfo->GetMkBlockByMkBound(block.GetMkType());
				if (cmk >= MkInfo->Size())RunException(met, fun::PrintStr("Error loading DEM objects. Mkbound=%u is unknown.", block.GetMkType()));
				const unsigned tav = CODE_GetTypeAndValue(MkInfo->Mkblock(cmk)->Code);
				//:Log->Printf("___> tav[%u]:%u",cmk,tav);
				if (block.Type == PT_Floating) {
					const JSpacePartBlock_Floating &fblock = (const JSpacePartBlock_Floating &)block;
					DemData[tav].mass = (float)fblock.GetMassbody();
					DemData[tav].massp = (float)(fblock.GetMassbody() / fblock.GetCount());
				}
				else DemData[tav].massp = MassBound;
				if (!block.ExistsSubValue("Young_Modulus", "value"))RunException(met, fun::PrintStr("Object mk=%u - Value of Young_Modulus is invalid.", block.GetMk()));
				if (!block.ExistsSubValue("PoissonRatio", "value"))RunException(met, fun::PrintStr("Object mk=%u - Value of PoissonRatio is invalid.", block.GetMk()));
				if (!block.ExistsSubValue("Kfric", "value"))RunException(met, fun::PrintStr("Object mk=%u - Value of Kfric is invalid.", block.GetMk()));
				if (!block.ExistsSubValue("Restitution_Coefficient", "value"))RunException(met, fun::PrintStr("Object mk=%u - Value of Restitution_Coefficient is invalid.", block.GetMk()));
				DemData[tav].young = block.GetSubValueFloat("Young_Modulus", "value", true, 0);
				DemData[tav].poisson = block.GetSubValueFloat("PoissonRatio", "value", true, 0);
				DemData[tav].tau = (DemData[tav].young ? (1 - DemData[tav].poisson*DemData[tav].poisson) / DemData[tav].young : 0);
				DemData[tav].kfric = block.GetSubValueFloat("Kfric", "value", true, 0);
				DemData[tav].restitu = block.GetSubValueFloat("Restitution_Coefficient", "value", true, 0);
				if (block.ExistsValue("Restitution_Coefficient_User"))DemData[tav].restitu = block.GetValueFloat("Restitution_Coefficient_User");
			}
		}
	}

	NpMinimum = CaseNp - unsigned(PartsOutMax*CaseNfluid);
	Log->Print("**Basic case configuration is loaded");
}

//==============================================================================
/// Once the case is load, the xml file should update with the info from the real data - Matthias
//==============================================================================
void JSph::UpdateCaseConfig_Mixed_M() {
	string directoryXml = "Def.xml";
	JXml xml; xml.LoadFile(FileXml);
// #xml #updateXml

	// Read csv 1
	std::vector<string> row;
	string line, word;
	Datacsvname = (((xml.GetNode("case.casedef.dataloader.file", false))->ToElement())->Attribute("name"));
	//int np;
	
	// Initialisation
	if (!xml.ExistsAttribute((xml.GetNode("case.execution.particles._summary.root", true))->ToElement(), "loaded")) {
		((xml.GetNode("case.execution.particles._summary.root", true))->ToElement())->SetAttribute("loaded", 1);

		// Real data
		// >>> Read XML and get namefile
		std::ifstream file(Datacsvname+".csv");
		int np = (int)count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') - 5; // remove 4 non particle related lines


		//TiXmlNode* node = xml.GetNode("case", false);
		int res;
		(((xml.GetNode("case.execution.particles._summary.fixed", false))->ToElement())->QueryIntAttribute("count", &res));

		// Modify particles node
		TiXmlNode* particles = xml.GetNode("case.execution.particles", false);
		int np_temp;
		(particles->ToElement())->QueryIntAttribute("np", &np_temp);
		(particles->ToElement())->RemoveAttribute("np");
		(particles->ToElement())->SetAttribute("np", np_temp + np); // new number of ptcs

		// V2 fluid _summary
		/*TiXmlElement fluid_summary("fluid");
		JXml::AddAttribute(&fluid_summary, "count", 1);
		JXml::AddAttribute(&fluid_summary, "id", "1179-1179"); // wrong value
		JXml::AddAttribute(&fluid_summary, "mkcount", 1);
		JXml::AddAttribute(&fluid_summary, "mkvalues", 1);

		TiXmlElement fluid("fluid");
		JXml::AddAttribute(&fluid, "mkfluid", "0");
		JXml::AddAttribute(&fluid, "mk", 1);
		JXml::AddAttribute(&fluid, "begin", 1179);
		JXml::AddAttribute(&fluid, "count", 1);*/

		// V3 fluid
		if (!xml.ExistsAttribute((xml.GetNode("case.execution.particles._summary.fluid", true))->ToElement(), "count")) {
			TiXmlElement fluid_summary("fluid");
			JXml::AddAttribute(&fluid_summary, "count", np);
			string s = std::to_string(res) + "-" + std::to_string(res + np);
			JXml::AddAttribute(&fluid_summary, "id", s); // wrong value
			JXml::AddAttribute(&fluid_summary, "mkcount", 1);
			JXml::AddAttribute(&fluid_summary, "mkvalues", 1);

			TiXmlElement fluid("fluid");
			JXml::AddAttribute(&fluid, "mkfluid", "0");
			JXml::AddAttribute(&fluid, "mk", 1);
			JXml::AddAttribute(&fluid, "begin", res);
			JXml::AddAttribute(&fluid, "count", np);

			// Save / update XML
			xml.GetNode("case.execution.particles._summary", true)->InsertEndChild(fluid_summary);
			xml.GetNode("case.execution.particles", true)->InsertEndChild(fluid);
		}
		else {
			TiXmlNode* fluid_summary = xml.GetNode("case.execution.particles._summary.fluid", false);
			int mkcounter_fluid = xml.ExistsAttribute(fluid_summary->ToElement(), "mkcount");
			(fluid_summary->ToElement())->SetAttribute("count", np); // new number of ptcs
			string s = std::to_string(res) + "-" + std::to_string(res + np);
			(fluid_summary->ToElement())->SetAttribute("id", s.c_str()); // new number of ptcs
			(fluid_summary->ToElement())->SetAttribute("mkcount", mkcounter_fluid+1); // new number of ptcs
			string s2 = std::to_string(1) + "-" + std::to_string(mkcounter_fluid + 1);
			(fluid_summary->ToElement())->SetAttribute("count", s2.c_str()); // new number of ptcs

		}

		if (false) xml.SaveFile(FileXml + "XXXMMMLLL.xml");//save the xml file
		else xml.SaveFile(FileXml);//save the xml file
	}
}

//==============================================================================
/// Shows coefficients used for DEM objects.
//==============================================================================
void JSph::VisuDemCoefficients()const{
  //-Gets info for each block of particles.
  Log->Printf("Coefficients for DEM:");
  for(unsigned c=0;c<MkInfo->Size();c++){
    const JSphMkBlock *pmk=MkInfo->Mkblock(c);
    const typecode code=pmk->Code;
    const typecode type=CODE_GetType(code);
    const unsigned tav=CODE_GetTypeAndValue(code);
    if(type==CODE_TYPE_FIXED || type==CODE_TYPE_MOVING || type==CODE_TYPE_FLOATING){
      Log->Printf("  Object %s  mkbound:%u  mk:%u",(type==CODE_TYPE_FIXED? "Fixed": (type==CODE_TYPE_MOVING? "Moving": "Floating")),pmk->MkType,pmk->Mk);
      Log->Printf("    Young_Modulus: %g",DemData[tav].young);
      Log->Printf("    PoissonRatio.: %g",DemData[tav].poisson);
      Log->Printf("    Kfric........: %g",DemData[tav].kfric);
      Log->Printf("    Restitution..: %g",DemData[tav].restitu);
    }
  }
}

//==============================================================================
/// Loads the code of a particle group and flags the last "nout"
/// particles as excluded.
///
/// Carga el codigo de grupo de las particulas y marca las nout ultimas
/// particulas como excluidas.
//==============================================================================
void JSph::LoadCodeParticles(unsigned np,const unsigned *idp,typecode *code)const{
  const char met[]="LoadCodeParticles";
  // # printf Debug GetMkBy Id
  printf("LoadCodeParticles\n");
  //-Assigns code to each group of particles.
  for(unsigned p=0;p<np;p++)code[p]=MkInfo->GetCodeById(idp[p]);
}

//==============================================================================
/// Sets DBL_MAX values by indicated values.
//==============================================================================
void JSph::PrepareCfgDomainValues(tdouble3 &v,tdouble3 vdef)const{
  if(v.x==DBL_MAX)v.x=vdef.x;
  if(v.y==DBL_MAX)v.y=vdef.y;
  if(v.z==DBL_MAX)v.z=vdef.z;
}

//==============================================================================
/// Resizes limits of the map according to case configuration.
//==============================================================================
void JSph::ResizeMapLimits(){
  Log->Print(string("MapRealPos(border)=")+fun::Double3gRangeStr(MapRealPosMin,MapRealPosMax));
  tdouble3 dmin=MapRealPosMin,dmax=MapRealPosMax;
  //-Sets Y configuration when it is a 2-D simulation.
  if(Simulate2D){
    CfgDomainParticlesMin.y=CfgDomainParticlesMax.y=DBL_MAX;
    CfgDomainParticlesPrcMin.y=CfgDomainParticlesPrcMax.y=DBL_MAX;
    CfgDomainFixedMin.y=CfgDomainFixedMax.y=DBL_MAX;
  }
  //-Configuration according particles domain.
  PrepareCfgDomainValues(CfgDomainParticlesMin);
  PrepareCfgDomainValues(CfgDomainParticlesMax);
  PrepareCfgDomainValues(CfgDomainParticlesPrcMin);
  PrepareCfgDomainValues(CfgDomainParticlesPrcMax);
  const tdouble3 dif=dmax-dmin;
  dmin=dmin-dif*CfgDomainParticlesPrcMin;
  dmax=dmax+dif*CfgDomainParticlesPrcMax;
  dmin=dmin-CfgDomainParticlesMin;
  dmax=dmax+CfgDomainParticlesMax;
  //-Fixed domain configuration.
  PrepareCfgDomainValues(CfgDomainFixedMin,dmin);
  PrepareCfgDomainValues(CfgDomainFixedMax,dmax);
  dmin=CfgDomainFixedMin;
  dmax=CfgDomainFixedMax;
  //-Checks domain limits.
  if(dmin.x>MapRealPosMin.x||dmin.y>MapRealPosMin.y||dmin.z>MapRealPosMin.z||dmax.x<MapRealPosMax.x||dmax.y<MapRealPosMax.y||dmax.z<MapRealPosMax.z)
    RunException("ResizeMapLimits",fun::PrintStr("Domain limits %s are not valid.",fun::Double3gRangeStr(dmin,dmax).c_str()));
  //-Periodic domain configuration.
  if(!PeriX){ MapRealPosMin.x=dmin.x; MapRealPosMax.x=dmax.x; }
  if(!PeriY){ MapRealPosMin.y=dmin.y; MapRealPosMax.y=dmax.y; }
  if(!PeriZ){ MapRealPosMin.z=dmin.z; MapRealPosMax.z=dmax.z; }
}

//==============================================================================
/// Configures value of constants.
//==============================================================================
void JSph::ConfigConstants(bool simulate2d){
  const char* met="ConfigConstants";

  // Matthias - Solid mechanics #constants
  const float  nf = Ey / Ex;

  if (Simulate2D) {
	  const float Delta = 1.0f / (1.0f - nuxy * nuxy * nf);
	  C1 = Delta * Ex; C12 = 0.0f; C13 = Delta * nuxy * Ey;
	  C2 = 0.0f; C23 = 0.0f; C3 = Delta * Ey;	  
	  
	  C4 = 0.0f; C5 = Gf; C6 = 0.0f;

	  //#S
	  S1 = 1 / Ex;		S12 = 0.0f; S13 = -nuxy / Ex;
	  S21 = 0.0f;		S2 = 0.0f;	S23 = 0.0f;
	  S31 = -nuxy / Ex; S32 = 0.0f; S3 = 1 / Ey;
	  Kani = 1 / (S1 + S12 + S13 + S21 + S2 + S23 + S31 + S32 + S3);

  }
  else {
	  const float Delta = nf * Ex / (1.0f - nuyz - 2.0f*nf*nuxy*nuxy);
	  C1 = Delta * (1.0f - nuyz) / nf;
	  C2 = C3 = Delta * (1.0f - nf * nuxy*nuxy) / (1.0f + nuyz);
	  C12 = C13 = Delta * nuxy;
	  C23 = Delta * (nuyz + nf * nuxy*nuxy) / (1.0f + nuyz);

	  C4 = Ey / (2.0f + 2.0f*nuxy); C5 = Gf; C6 = Gf;

	  S1 = 1 / Ex; S12 = -nuxy / Ex; S13 = -nuxy / Ex;
	  S21 = -nuxy / Ex; S2 = 1 / Ey; S23 = -nuyz / Ey;
	  S31 = -nuxy / Ex; S32 = -nuyz / Ey; S3 = 1 / Ey;
	  Kani = 1 / (S1 + S12 + S13 + S21 + S2 + S23 + S31 + S32 + S3);
  }


  printf("////\n");
  printf("C1 = %.3f, C12 = %.3f, C13 = %.3f\n", C1, C12, C13);
  printf("C12 = %.3f, C2 = %.3f, C23 = %.3f\n", C12, C2, C23);
  printf("C13 = %.3f, C23 = %.3f, C3 = %.3f\n", C13, C23, C3);
  printf("S1 = %.8f, S12 = %.8f, S13 = %.8f\n", S1, S12, S13);
  printf("S12 = %.8f, S2 = %.8f, S23 = %.8f\n", S12, S2, S23);
  printf("S13 = %.8f, S23 = %.8f, S3 = %.8f\n", S13, S23, S3);
  printf("Kani = %.8f\n", Kani);

  // New B for anisotropy
  hmin = Hmin * (float)Dp * (simulate2d ? sqrt(2.0f) : sqrt(3.0f));
  hmax = Hmax * (float)Dp * (simulate2d ? sqrt(2.0f) : sqrt(3.0f));

  //-Computation of constants.
  const double h=hmax;
  Delta2H=float(h*2*DeltaSph);

  // Cs0 version originale
  Cs0=10*sqrt(double(Gamma)*double(max(CalcK(0.0), CalcK(1.5) )/Gamma)/double(RhopZero)); 

  // Old anisotropic versions of Cs0 (vec3) removed - Matthias

  if(!DtIni)DtIni=hmin/Cs0;
  if(!DtMin)DtMin=(hmin/Cs0)*CoefDtMin;
  Dosh=float(h*2);
  H2=float(h*h);
  Fourh2=float(h*h*4);
  Eta2=float((h*0.1)*(h*0.1));
  if(simulate2d){
    if(TKernel==KERNEL_Wendland){
      Awen=float(0.557/(h*h));
      Bwen=float(-2.7852/(h*h*h));
    }
    else if(TKernel==KERNEL_Gaussian){
      const double a1=4./PI;
      const double a2=a1/(h*h);
      const double aa=a1/(h*h*h);
      Agau=float(a2);
      Bgau=float(-8.*aa);
    }
    else if(TKernel==KERNEL_Cubic){
      const double a1=10./(PI*7.);
      const double a2=a1/(h*h);
      const double aa=a1/(h*h*h);
      const double deltap=1./1.5;
      const double wdeltap=a2*(1.-1.5*deltap*deltap+0.75*deltap*deltap*deltap);
      CubicCte.od_wdeltap=float(1./wdeltap);
      CubicCte.a1=float(a1);
      CubicCte.a2=float(a2);
      CubicCte.aa=float(aa);
      CubicCte.a24=float(0.25*a2);
      CubicCte.c1=float(-3.*aa);
      CubicCte.d1=float(9.*aa/4.);
      CubicCte.c2=float(-3.*aa/4.);
    }
  }
  else{
    if(TKernel==KERNEL_Wendland){
      Awen=float(0.41778/(h*h*h));
      Bwen=float(-2.08891/(h*h*h*h));
    }
    else if(TKernel==KERNEL_Gaussian){
      const double a1=8./5.5683;
      const double a2=a1/(h*h*h);
      const double aa=a1/(h*h*h*h);
      Agau=float(a2);
      Bgau=float(-8.*aa);
    }
    else if(TKernel==KERNEL_Cubic){
      const double a1=1./PI;
      const double a2=a1/(h*h*h);
      const double aa=a1/(h*h*h*h);
      const double deltap=1./1.5;
      const double wdeltap=a2*(1.-1.5*deltap*deltap+0.75*deltap*deltap*deltap);
      CubicCte.od_wdeltap=float(1./wdeltap);
      CubicCte.a1=float(a1);
      CubicCte.a2=float(a2);
      CubicCte.aa=float(aa);
      CubicCte.a24=float(0.25*a2);
      CubicCte.c1=float(-3.*aa);
      CubicCte.d1=float(9.*aa/4.);
      CubicCte.c2=float(-3.*aa/4.);
    }
  }
  //-Constants for Laminar viscosity + SPS turbulence model.
  if(TVisco==VISCO_LaminarSPS){
    double dp_sps=(Simulate2D? sqrt(Dp*Dp*2.)/2.: sqrt(Dp*Dp*3.)/3.);
    SpsSmag=float(pow((0.12*dp_sps),2));
    SpsBlin=float((2./3.)*0.0066*dp_sps*dp_sps);
  }
  VisuConfig();
}

//=======================
// Calculate K value (sigmoid between isotropic and anisotropic behaviour)
// #CalcK  #KS
//=======================
float JSph::CalcK(double x) {
	float K;
	// #MdYoung #Gradual #young
	//int typeMdYoung = 0;
	float theta = 1.0f; // Theta constant
	//const float theta = 2.0f-float(x); // Theta linear
	switch (typeYoung){
		case 1: {
			theta = SigmoidGrowth(float(x)); // Theta sigmoid
			break;
			break;
		}
		case 2: {
			theta = CircleYoung(float(x)); // Circle shape theta
			break;
		}
		case 3: {
			theta = 0.0f; // FullI
			break;
		}
		default: {
			theta = 1.0f; // FullA
			break;
		}
	}
	const float E = theta * Ey + (1.0f - theta) * Ex;
	const float G = theta * Gf + (1.0f - theta) * Ex * 0.5f * (1 + nuxy);
	const float nu = theta * nuyz + (1.0f - theta) * nuxy;
	const float  nf = E / Ex;

	if (Simulate2D) {
		const float KS1 = 1 / Ex;
		const float KS12 = 0.0f; 
		const float KS13 = -nuxy / Ex;
		const float KS21 = 0.0f;		
		const float KS2 = 0.0f;	
		const float KS23 = 0.0f;
		const float KS31 = -nuxy / Ex; 
		const float KS32 = 0.0f; 
		const float KS3 = 1 / E;

		K = 1 / (KS1 + KS12 + KS13 + KS21 + KS2 + KS23 + KS31 + KS32 + KS3);

	}
	else {
		const float KS1 = 1 / Ex; 
		const float KS12 = -nuxy / Ex; 
		const float KS13 = -nuxy / Ex;
		const float KS21 = -nuxy / Ex;
		const float KS2 = 1 / E; 
		const float KS23 = -nu / E;
		const float KS31 = -nuxy / Ex;
		const float KS32 = -nu / E; 
		const float KS3 = 1 / E;

		K = 1 / (KS1 + KS12 + KS13 + KS21 + KS2 + KS23 + KS31 + KS32 + KS3);
	}

	return K;
}

float JSph::SigmoidGrowth(double x) const {
	float x0 = 0.15f;
	float k = 15.0f;
	return 1.0f / (1.0f + exp(-k * (float(x) - x0)));
}

float JSph::CircleYoung(float x) const {
	const float radius = 0.5f;
	//float c = 2.0f * (radius - sqrt(pow(radius, 2) - pow(x - radius, 2)));
	//if (x < radius) return 2.0f * (radius - sqrt(pow(radius, 2) - pow(x - radius, 2)));
	if (x < 0.0f) return 0.0f;
	else if (x < radius) return 2.0f * (sqrt(pow(radius, 2) - pow(radius - x, 2)));
	else return 1.0f;
}

//==============================================================================
/// Prints out configuration of the case.
//==============================================================================
void JSph::VisuConfig()const{
  const char* met="VisuConfig";
  Log->Print(Simulate2D? "**2D-Simulation parameters:": "**3D-Simulation parameters:");
  Log->Print(fun::VarStr("CaseName",CaseName));
  Log->Print(fun::VarStr("RunName",RunName));
  if(Simulate2D)Log->Print(fun::VarStr("Simulate2DPosY",Simulate2DPosY));
  Log->Print(fun::VarStr("PosDouble",GetPosDoubleName(Psingle,SvDouble)));
  Log->Print(fun::VarStr("SvTimers",SvTimers));
  Log->Print(fun::VarStr("StepAlgorithm",GetStepName(TStep)));
  if(TStep==STEP_None)RunException(met,"StepAlgorithm value is invalid.");
  if(TStep==STEP_Verlet)Log->Print(fun::VarStr("VerletSteps",VerletSteps));
  Log->Print(fun::VarStr("Kernel",GetKernelName(TKernel)));
  Log->Print(fun::VarStr("Viscosity",GetViscoName(TVisco)));
  Log->Print(fun::VarStr("Visco",Visco));
  Log->Print(fun::VarStr("ViscoBoundFactor",ViscoBoundFactor));
  if(ViscoTime)Log->Print(fun::VarStr("ViscoTime",ViscoTime->GetFile()));
  Log->Print(fun::VarStr("DeltaSph",GetDeltaSphName(TDeltaSph)));
  if(TDeltaSph!=DELTA_None)Log->Print(fun::VarStr("DeltaSphValue",DeltaSph));
  Log->Print(fun::VarStr("Shifting",GetShiftingName(TShifting)));
  if(TShifting!=SHIFT_None){
    Log->Print(fun::VarStr("ShiftCoef",ShiftCoef));
    if(ShiftTFS)Log->Print(fun::VarStr("ShiftTFS",ShiftTFS));
  }
  Log->Print(fun::VarStr("RigidAlgorithm",(!FtCount? "None": (UseDEM? "SPH+DEM": "SPH"))));
  Log->Print(fun::VarStr("FloatingCount",FtCount));
  if(FtCount)Log->Print(fun::VarStr("FtPause",FtPause));
  Log->Print(fun::VarStr("CaseNp",CaseNp));
  Log->Print(fun::VarStr("CaseNbound",CaseNbound));
  Log->Print(fun::VarStr("CaseNfixed",CaseNfixed));
  Log->Print(fun::VarStr("CaseNmoving",CaseNmoving));
  Log->Print(fun::VarStr("CaseNfloat",CaseNfloat));
  Log->Print(fun::VarStr("CaseNfluid",CaseNfluid));
  Log->Print(fun::VarStr("PeriodicActive",PeriActive));
  if(PeriXY)Log->Print(fun::VarStr("PeriodicXY",PeriXY));
  if(PeriXZ)Log->Print(fun::VarStr("PeriodicXZ",PeriXZ));
  if(PeriYZ)Log->Print(fun::VarStr("PeriodicYZ",PeriYZ));
  if(PeriX)Log->Print(fun::VarStr("PeriodicXinc",PeriXinc));
  if(PeriY)Log->Print(fun::VarStr("PeriodicYinc",PeriYinc));
  if(PeriZ)Log->Print(fun::VarStr("PeriodicZinc",PeriZinc));
  Log->Print(fun::VarStr("Dx",Dp));
  Log->Print(fun::VarStr("H",H));
  Log->Print(fun::VarStr("CoefficientH",H/(Dp*sqrt(Simulate2D? 2.f: 3.f))));
  //Log->Print(fun::VarStr("CteB", CteB));
  // Matthias
  Log->Print(fun::VarStr("Gamma",Gamma));
  Log->Print(fun::VarStr("RhopZero",RhopZero));
  Log->Print(fun::VarStr("Cs0",Cs0));
  Log->Print(fun::VarStr("CFLnumber", CFLnumber));
  Log->Print(fun::VarStr("PlanMirror - X", PlanMirror));
  Log->Print(fun::VarStr("DtIni",DtIni));
  Log->Print(fun::VarStr("DtMin",DtMin));
  Log->Print(fun::VarStr("DtAllParticles",DtAllParticles));
  if(DtFixed)Log->Print(fun::VarStr("DtFixed",DtFixed->GetFile()));
  Log->Print(fun::VarStr("MassFluid",MassFluid));
  Log->Print(fun::VarStr("MassBound",MassBound));
  // Solid - anisotropic
  Log->Print("SolidVariables");
  /*  Log->Print(fun::VarStr("Young modulus", K));
  Log->Print(fun::VarStr("Shear modulus", Mu));*/
  Log->Print(fun::VarStr("Young modulus x", Ex));
  Log->Print(fun::VarStr("Young modulus y", Ey));
  Log->Print(fun::VarStr("Shear modulus", Gf));
  Log->Print(fun::VarStr("Poisson modulus xy", nuxy));
  Log->Print(fun::VarStr("Poisson modulus yz", nuyz));


  if(TKernel==KERNEL_Wendland){
    Log->Print(fun::VarStr("Awen (Wendland)",Awen));
    Log->Print(fun::VarStr("Bwen (Wendland)",Bwen));
  }
  else if(TKernel==KERNEL_Gaussian){
    Log->Print(fun::VarStr("Agau (Gaussian)",Agau));
    Log->Print(fun::VarStr("Bgau (Gaussian)",Bgau));
  }
  else if(TKernel==KERNEL_Cubic){
    Log->Print(fun::VarStr("CubicCte.a1",CubicCte.a1));
    Log->Print(fun::VarStr("CubicCte.aa",CubicCte.aa));
    Log->Print(fun::VarStr("CubicCte.a24",CubicCte.a24));
    Log->Print(fun::VarStr("CubicCte.c1",CubicCte.c1));
    Log->Print(fun::VarStr("CubicCte.c2",CubicCte.c2));
    Log->Print(fun::VarStr("CubicCte.d1",CubicCte.d1));
    Log->Print(fun::VarStr("CubicCte.od_wdeltap",CubicCte.od_wdeltap));
  }
  if(TVisco==VISCO_LaminarSPS){
    Log->Print(fun::VarStr("SpsSmag",SpsSmag));
    Log->Print(fun::VarStr("SpsBlin",SpsBlin));
  }
  if(UseDEM)VisuDemCoefficients();
  if(CaseNfloat)Log->Print(fun::VarStr("FtPause",FtPause));
  Log->Print(fun::VarStr("TimeMax",TimeMax));
  Log->Print(fun::VarStr("TimePart",TimePart));
  Log->Print(fun::VarStr("Gravity",Gravity));
  Log->Print(fun::VarStr("NpMinimum",NpMinimum));
  Log->Print(fun::VarStr("RhopOut",RhopOut));
  if(RhopOut){
    Log->Print(fun::VarStr("RhopOutMin",RhopOutMin));
    Log->Print(fun::VarStr("RhopOutMax",RhopOutMax));
  }
  //if(CteB==0)RunException(met,"Constant \'b\' cannot be zero.\n\'b\' is zero when fluid height is zero (or fluid particles were not created)");
}

//==============================================================================
/// Shows particle and MK blocks summary.
//==============================================================================
void JSph::VisuParticleSummary()const{
  JXml xml; xml.LoadFile(FileXml);
  JSpaceParts parts;
  parts.LoadXml(&xml,"case.execution.particles");
  std::vector<std::string> summary;
  parts.GetParticleSummary(summary);
  Log->Print(summary);
  Log->Print(" ");
}

//==============================================================================
/// Computes cell particles and checks if there are more particles
/// excluded than expected.
///
/// Calcula celda de las particulas y comprueba que no existan mas particulas
/// excluidas de las previstas.
//==============================================================================
void JSph::LoadDcellParticles(unsigned n,const typecode *code,const tdouble3 *pos,unsigned *dcell)const{
  const char met[]="LoadDcellParticles";
  for(unsigned p=0;p<n;p++){
    typecode codeout=CODE_GetSpecialValue(code[p]);
    if(codeout<CODE_OUTIGNORE){
      const tdouble3 ps=pos[p];
      if(ps>=DomRealPosMin && ps<DomRealPosMax){//-Particle in.
        const double dx=ps.x-DomPosMin.x;
        const double dy=ps.y-DomPosMin.y;
        const double dz=ps.z-DomPosMin.z;
        unsigned cx=unsigned(dx/Scell),cy=unsigned(dy/Scell),cz=unsigned(dz/Scell);
        dcell[p]=PC__Cell(DomCellCode,cx,cy,cz);
      }
      else{//-Particle out.
        RunException(met,"Found new particles out."); //-There can not be new particles excluded. | No puede haber nuevas particulas excluidas.
        dcell[p]=PC__CodeOut;
      }
    }
    else dcell[p]=PC__CodeOut;
  }
}

//==============================================================================
/// Initializes data of particles according XML configuration.
///
/// Inicializa datos de las particulas a partir de la configuracion en el XML.
//==============================================================================
void JSph::RunInitialize(unsigned np,unsigned npb,const tdouble3 *pos,const unsigned *idp,const typecode *code,tfloat4 *velrhop){
  const char met[]="RunInitialize";
  JSphInitialize init(FileXml);
  if(init.Count()){
    //-Creates array with mktype value.
    word *mktype=new word[np];
    for(unsigned p=0;p<np;p++){
      const unsigned cmk=MkInfo->GetMkBlockByCode(code[p]);
      mktype[p]=(cmk<MkInfo->Size()? word(MkInfo->Mkblock(cmk)->MkType): USHRT_MAX);
    }
    init.Run(np,npb,pos,idp,mktype,velrhop);
    init.GetConfig(InitializeInfo);
    //-Frees memory.
    delete[] mktype; mktype=NULL;
  }
}

//==============================================================================
/// Configures CellOrder and adjusts order of components in data.
/// Configura CellOrder y ajusta orden de componentes en datos.
//==============================================================================
void JSph::ConfigCellOrder(TpCellOrder order,unsigned np,tdouble3* pos,tfloat4* velrhop){
  //-Stores periodic configuration in PeriodicConfig.
  //-Guarda configuracion periodica en PeriodicConfig.
  PeriodicConfig.PeriActive=PeriActive;
  PeriodicConfig.PeriX=PeriX;
  PeriodicConfig.PeriY=PeriY;
  PeriodicConfig.PeriZ=PeriZ;
  PeriodicConfig.PeriXY=PeriXY;
  PeriodicConfig.PeriXZ=PeriXZ;
  PeriodicConfig.PeriYZ=PeriYZ;
  PeriodicConfig.PeriXinc=PeriXinc;
  PeriodicConfig.PeriYinc=PeriYinc;
  PeriodicConfig.PeriZinc=PeriZinc;
  //-Applies CellOrder.
  CellOrder=order;
  if(CellOrder==ORDER_None)CellOrder=ORDER_XYZ;
  if(Simulate2D&&CellOrder!=ORDER_XYZ&&CellOrder!=ORDER_ZYX)RunException("ConfigCellOrder","In 2D simulations the value of CellOrder must be XYZ or ZYX.");
  Log->Print(fun::VarStr("CellOrder",string(GetNameCellOrder(CellOrder))));
  if(CellOrder!=ORDER_XYZ){
    //-Modifies initial particle data.
    OrderCodeData(CellOrder,np,pos);
    OrderCodeData(CellOrder,np,velrhop);
    //-Modifies other constants.
    Gravity=OrderCodeValue(CellOrder,Gravity);
    MapRealPosMin=OrderCodeValue(CellOrder,MapRealPosMin);
    MapRealPosMax=OrderCodeValue(CellOrder,MapRealPosMax);
    MapRealSize=OrderCodeValue(CellOrder,MapRealSize);
    Map_PosMin=OrderCodeValue(CellOrder,Map_PosMin);
    Map_PosMax=OrderCodeValue(CellOrder,Map_PosMax);
    Map_Size=OrderCodeValue(CellOrder,Map_Size);
    //-Modifies periodic configuration.
    bool perix=PeriX,periy=PeriY,periz=PeriZ;
    bool perixy=PeriXY,perixz=PeriXZ,periyz=PeriYZ;
    tdouble3 perixinc=PeriXinc,periyinc=PeriYinc,perizinc=PeriZinc;
    tuint3 v={1,2,3};
    v=OrderCode(v);
    if(v.x==2){ PeriX=periy; PeriXinc=OrderCode(periyinc); }
    if(v.x==3){ PeriX=periz; PeriXinc=OrderCode(perizinc); }
    if(v.y==1){ PeriY=perix; PeriYinc=OrderCode(perixinc); }
    if(v.y==3){ PeriY=periz; PeriYinc=OrderCode(perizinc); }
    if(v.z==1){ PeriZ=perix; PeriZinc=OrderCode(perixinc); }
    if(v.z==2){ PeriZ=periy; PeriZinc=OrderCode(periyinc); }
    if(perixy){
      PeriXY=(CellOrder==ORDER_XYZ||CellOrder==ORDER_YXZ);
      PeriXZ=(CellOrder==ORDER_XZY||CellOrder==ORDER_YZX);
      PeriYZ=(CellOrder==ORDER_ZXY||CellOrder==ORDER_ZYX);
    }
    if(perixz){
      PeriXY=(CellOrder==ORDER_XZY||CellOrder==ORDER_ZXY);
      PeriXZ=(CellOrder==ORDER_XYZ||CellOrder==ORDER_ZYX);
      PeriYZ=(CellOrder==ORDER_YXZ||CellOrder==ORDER_YZX);
    }
    if(periyz){
      PeriXY=(CellOrder==ORDER_YZX||CellOrder==ORDER_ZYX);
      PeriXZ=(CellOrder==ORDER_YXZ||CellOrder==ORDER_ZXY);
      PeriYZ=(CellOrder==ORDER_XYZ||CellOrder==ORDER_XZY);
    }
  }
  PeriActive=(PeriX? 1: 0)+(PeriY? 2: 0)+(PeriZ? 4: 0);
}

//==============================================================================
/// Converts pos[] and vel[] to the original dimension order.
/// Convierte pos[] y vel[] al orden dimensional original.
//==============================================================================
void JSph::DecodeCellOrder(unsigned np,tdouble3 *pos,tfloat3 *vel)const{
  if(CellOrder!=ORDER_XYZ){
    OrderDecodeData(CellOrder,np,pos);
    OrderDecodeData(CellOrder,np,vel);
  }
}

//==============================================================================
/// Modifies order of components of an array of type tfloat3.
/// Modifica orden de componentes de un array de tipo tfloat3.
//==============================================================================
void JSph::OrderCodeData(TpCellOrder order,unsigned n,tfloat3 *v){
  if(order==ORDER_XZY)for(unsigned c=0;c<n;c++)v[c]=ReOrderXZY(v[c]);
  if(order==ORDER_YXZ)for(unsigned c=0;c<n;c++)v[c]=ReOrderYXZ(v[c]);
  if(order==ORDER_YZX)for(unsigned c=0;c<n;c++)v[c]=ReOrderYZX(v[c]);
  if(order==ORDER_ZXY)for(unsigned c=0;c<n;c++)v[c]=ReOrderZXY(v[c]);
  if(order==ORDER_ZYX)for(unsigned c=0;c<n;c++)v[c]=ReOrderZYX(v[c]);
}

//==============================================================================
/// Modifies order of components of an array of type tdouble3.
/// Modifica orden de componentes de un array de tipo tdouble3.
//==============================================================================
void JSph::OrderCodeData(TpCellOrder order,unsigned n,tdouble3 *v){
  if(order==ORDER_XZY)for(unsigned c=0;c<n;c++)v[c]=ReOrderXZY(v[c]);
  if(order==ORDER_YXZ)for(unsigned c=0;c<n;c++)v[c]=ReOrderYXZ(v[c]);
  if(order==ORDER_YZX)for(unsigned c=0;c<n;c++)v[c]=ReOrderYZX(v[c]);
  if(order==ORDER_ZXY)for(unsigned c=0;c<n;c++)v[c]=ReOrderZXY(v[c]);
  if(order==ORDER_ZYX)for(unsigned c=0;c<n;c++)v[c]=ReOrderZYX(v[c]);
}

//==============================================================================
/// Modifies order of components of an array of type tfloat4.
/// Modifica orden de componentes de un array de tipo tfloat4.
//==============================================================================
void JSph::OrderCodeData(TpCellOrder order,unsigned n,tfloat4 *v){
  if(order==ORDER_XZY)for(unsigned c=0;c<n;c++)v[c]=ReOrderXZY(v[c]);
  if(order==ORDER_YXZ)for(unsigned c=0;c<n;c++)v[c]=ReOrderYXZ(v[c]);
  if(order==ORDER_YZX)for(unsigned c=0;c<n;c++)v[c]=ReOrderYZX(v[c]);
  if(order==ORDER_ZXY)for(unsigned c=0;c<n;c++)v[c]=ReOrderZXY(v[c]);
  if(order==ORDER_ZYX)for(unsigned c=0;c<n;c++)v[c]=ReOrderZYX(v[c]);
}

//==============================================================================
/// Configures cell division.
//==============================================================================
void JSph::ConfigCellDivision(){
  if(CellMode!=CELLMODE_2H && CellMode!=CELLMODE_H)RunException("ConfigCellDivision","The CellMode is invalid.");
  Hdiv=(CellMode==CELLMODE_2H? 1: 2);
  Scell=Dosh/Hdiv;
  MovLimit=Scell*0.9f;
  Map_Cells=TUint3(unsigned(ceil(Map_Size.x/Scell)),unsigned(ceil(Map_Size.y/Scell)),unsigned(ceil(Map_Size.z/Scell)));
  //-Prints configuration.
  Log->Print(fun::VarStr("CellMode",string(GetNameCellMode(CellMode))));
  Log->Print(fun::VarStr("Hdiv",Hdiv));
  Log->Print(string("MapCells=(")+fun::Uint3Str(OrderDecode(Map_Cells))+")");
  //-Creates VTK file with map cells.
  printf("\n---SaveMapCellsVtkSize--- %d", SaveMapCellsVtkSize());
  if(SaveMapCellsVtkSize()<1024*1024*10)SaveMapCellsVtk(Scell);
  else Log->PrintWarning("File CfgInit_MapCells.vtk was not created because number of cells is too high.");
}

//==============================================================================
/// Sets local domain of simulation within Map_Cells and computes DomCellCode.
/// Establece dominio local de simulacion dentro de Map_Cells y calcula DomCellCode.
//==============================================================================
void JSph::SelecDomain(tuint3 celini,tuint3 celfin){
  const char met[]="SelecDomain";
  DomCelIni=celini;
  DomCelFin=celfin;
  DomCells=DomCelFin-DomCelIni;
  if(DomCelIni.x>=Map_Cells.x || DomCelIni.y>=Map_Cells.y || DomCelIni.z>=Map_Cells.z )RunException(met,"DomCelIni is invalid.");
  if(DomCelFin.x>Map_Cells.x || DomCelFin.y>Map_Cells.y || DomCelFin.z>Map_Cells.z )RunException(met,"DomCelFin is invalid.");
  if(DomCells.x<1 || DomCells.y<1 || DomCells.z<1 )RunException(met,"The domain of cells is invalid.");
  //-Computes local domain limits.
  DomPosMin.x=Map_PosMin.x+(DomCelIni.x*Scell);
  DomPosMin.y=Map_PosMin.y+(DomCelIni.y*Scell);
  DomPosMin.z=Map_PosMin.z+(DomCelIni.z*Scell);
  DomPosMax.x=Map_PosMin.x+(DomCelFin.x*Scell);
  DomPosMax.y=Map_PosMin.y+(DomCelFin.y*Scell);
  DomPosMax.z=Map_PosMin.z+(DomCelFin.z*Scell);
  //-Adjusts final limits.
  if(DomPosMax.x>Map_PosMax.x)DomPosMax.x=Map_PosMax.x;
  if(DomPosMax.y>Map_PosMax.y)DomPosMax.y=Map_PosMax.y;
  if(DomPosMax.z>Map_PosMax.z)DomPosMax.z=Map_PosMax.z;
  //-Computes actual limits of local domain.
  //-Calcula limites reales del dominio local.
  DomRealPosMin=DomPosMin;
  DomRealPosMax=DomPosMax;
  if(DomRealPosMax.x>MapRealPosMax.x)DomRealPosMax.x=MapRealPosMax.x;
  if(DomRealPosMax.y>MapRealPosMax.y)DomRealPosMax.y=MapRealPosMax.y;
  if(DomRealPosMax.z>MapRealPosMax.z)DomRealPosMax.z=MapRealPosMax.z;
  if(DomRealPosMin.x<MapRealPosMin.x)DomRealPosMin.x=MapRealPosMin.x;
  if(DomRealPosMin.y<MapRealPosMin.y)DomRealPosMin.y=MapRealPosMin.y;
  if(DomRealPosMin.z<MapRealPosMin.z)DomRealPosMin.z=MapRealPosMin.z;
  //-Computes cofification of cells for the selected domain.
  //-Calcula codificacion de celdas para el dominio seleccionado.
  DomCellCode=CalcCellCode(DomCells+TUint3(1));
  if(!DomCellCode)RunException(met,string("Failed to select a valid CellCode for ")+fun::UintStr(DomCells.x)+"x"+fun::UintStr(DomCells.y)+"x"+fun::UintStr(DomCells.z)+" cells (CellMode="+GetNameCellMode(CellMode)+").");
  //-Prints configurantion.
  Log->Print(string("DomCells=(")+fun::Uint3Str(OrderDecode(DomCells))+")");
  Log->Print(fun::VarStr("DomCellCode",fun::UintStr(PC__GetSx(DomCellCode))+"_"+fun::UintStr(PC__GetSy(DomCellCode))+"_"+fun::UintStr(PC__GetSz(DomCellCode))));
}

//==============================================================================
/// Selects an adequate code for cell configuration.
/// Selecciona un codigo adecuado para la codificion de celda.
//==============================================================================
unsigned JSph::CalcCellCode(tuint3 ncells){
  unsigned sxmin=2; for(;ncells.x>>sxmin;sxmin++);
  unsigned symin=2; for(;ncells.y>>symin;symin++);
  unsigned szmin=2; for(;ncells.z>>szmin;szmin++);
  unsigned smin=sxmin+symin+szmin;
  unsigned ccode=0;
  if(smin<=32){
    unsigned sx=sxmin,sy=symin,sz=szmin;
    unsigned rest=32-smin;
    while(rest){
      if(rest){ sx++; rest--; }
      if(rest){ sy++; rest--; }
      if(rest){ sz++; rest--; }
    }
    ccode=PC__GetCode(sx,sy,sz);
  }
  return(ccode);
}

//==============================================================================
/// Computes maximum distance between particles and center of floating.
/// Calcula distancia maxima entre particulas y centro de cada floating.
//==============================================================================
void JSph::CalcFloatingRadius(unsigned np,const tdouble3 *pos,const unsigned *idp){
  const char met[]="CalcFloatingsRadius";
  const float overradius=1.2f; //-Percentage of ration increase. | Porcentaje de incremento de radio.
  unsigned *ridp=new unsigned[CaseNfloat];
  //-Assigns values UINT_MAX.
  memset(ridp,255,sizeof(unsigned)*CaseNfloat);
  //-Computes position according to id assuming that all particles are not periodic.
  //-Calcula posicion segun id suponiendo que todas las particulas son normales (no periodicas).
  const unsigned idini=CaseNpb,idfin=CaseNpb+CaseNfloat;
  for(unsigned p=0;p<np;p++){
    const unsigned id=idp[p];
    if(idini<=id && id<idfin)ridp[id-idini]=p;
  }
  //-Checks that all floating particles are located.
  //-Comprueba que todas las particulas floating estan localizadas.
  for(unsigned fp=0;fp<CaseNfloat;fp++){
    if(ridp[fp]==UINT_MAX)RunException(met,"There are floating particles not found.");
  }
  //-Calculates maximum distance between particles and center of the floating (all are valid).
  //-Calcula distancia maxima entre particulas y centro de floating (todas son validas).
  float radiusmax=0;
  for(unsigned cf=0;cf<FtCount;cf++){
    StFloatingData *fobj=FtObjs+cf;
    const unsigned fpini=fobj->begin-CaseNpb;
    const unsigned fpfin=fpini+fobj->count;
    const tdouble3 fcen=fobj->center;
    double r2max=0;
    for(unsigned fp=fpini;fp<fpfin;fp++){
      const int p=ridp[fp];
      const double dx=fcen.x-pos[p].x,dy=fcen.y-pos[p].y,dz=fcen.z-pos[p].z;
      double r2=dx*dx+dy*dy+dz*dz;
      if(r2max<r2)r2max=r2;
    }
    fobj->radius=float(sqrt(r2max)*overradius);
    if(radiusmax<fobj->radius)radiusmax=fobj->radius;
  }
  //-Deallocate of memory.
  delete[] ridp; ridp=NULL;
  //-Checks maximum radius < dimensions of the periodic domain.
  //-Comprueba que el radio maximo sea menor que las dimensiones del dominio periodico.
  if(PeriX && fabs(PeriXinc.x)<=radiusmax)RunException(met,fun::PrintStr("The floating radius (%g) is too large for periodic distance in X (%g).",radiusmax,abs(PeriXinc.x)));
  if(PeriY && fabs(PeriYinc.y)<=radiusmax)RunException(met,fun::PrintStr("The floating radius (%g) is too large for periodic distance in Y (%g).",radiusmax,abs(PeriYinc.y)));
  if(PeriZ && fabs(PeriZinc.z)<=radiusmax)RunException(met,fun::PrintStr("The floating radius (%g) is too large for periodic distance in Z (%g).",radiusmax,abs(PeriZinc.z)));
}

//==============================================================================
/// Returns the corrected position after applying periodic conditions.
/// Devuelve la posicion corregida tras aplicar condiciones periodicas.
//==============================================================================
tdouble3 JSph::UpdatePeriodicPos(tdouble3 ps)const{
  double dx=ps.x-MapRealPosMin.x;
  double dy=ps.y-MapRealPosMin.y;
  double dz=ps.z-MapRealPosMin.z;
  const bool out=(dx!=dx || dy!=dy || dz!=dz || dx<0 || dy<0 || dz<0 || dx>=MapRealSize.x || dy>=MapRealSize.y || dz>=MapRealSize.z);
  //-Adjusts position according to periodic conditions and checks again domain limtis.
  //-Ajusta posicion segun condiciones periodicas y vuelve a comprobar los limites del dominio.
  if(PeriActive && out){
    bool xperi=((PeriActive&1)!=0),yperi=((PeriActive&2)!=0),zperi=((PeriActive&4)!=0);
    if(xperi){
      if(dx<0)             { dx-=PeriXinc.x; dy-=PeriXinc.y; dz-=PeriXinc.z; }
      if(dx>=MapRealSize.x){ dx+=PeriXinc.x; dy+=PeriXinc.y; dz+=PeriXinc.z; }
    }
    if(yperi){
      if(dy<0)             { dx-=PeriYinc.x; dy-=PeriYinc.y; dz-=PeriYinc.z; }
      if(dy>=MapRealSize.y){ dx+=PeriYinc.x; dy+=PeriYinc.y; dz+=PeriYinc.z; }
    }
    if(zperi){
      if(dz<0)             { dx-=PeriZinc.x; dy-=PeriZinc.y; dz-=PeriZinc.z; }
      if(dz>=MapRealSize.z){ dx+=PeriZinc.x; dy+=PeriZinc.y; dz+=PeriZinc.z; }
    }
    ps=TDouble3(dx,dy,dz)+MapRealPosMin;
  }
  return(ps);
}

//==============================================================================
/// Display a message with reserved memory for the basic data of particles.
/// Muestra un mensaje con la memoria reservada para los datos basicos de las particulas.
//==============================================================================
void JSph::PrintSizeNp(unsigned np,llong size)const{
  Log->Printf("**Requested %s memory for %u particles: %.1f MB.",(Cpu? "cpu": "gpu"),np,double(size)/(1024*1024));
}

//==============================================================================
/// Display headers of PARTs
/// Visualiza cabeceras de PARTs.
//==============================================================================
void JSph::PrintHeadPart(){
  Log->Print("PART       PartTime      TotalSteps    Steps    Time/Sec   Finish time        ");
  Log->Print("=========  ============  ============  =======  =========  ===================");
  fflush(stdout);
}

//==============================================================================
/// Sets configuration for recordering of particles.
/// Establece configuracion para grabacion de particulas.
//==============================================================================
void JSph::ConfigSaveData(unsigned piece,unsigned pieces,std::string div){
  const char met[]="ConfigSaveData";
  //-Configures object to store particles and information.
  //-Configura objeto para grabacion de particulas e informacion.
  // Matthias - Might be a problem in 3D anisotropy
  if(SvData&SDAT_Info || SvData&SDAT_Binx){
    DataBi4=new JPartDataBi4();
    DataBi4->ConfigBasic(piece,pieces,RunCode,AppName,CaseName,Simulate2D,Simulate2DPosY,DirDataOut);
    DataBi4->ConfigParticles(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid,CasePosMin,CasePosMax,NpDynamic,ReuseIds);
    //DataBi4->ConfigCtes(Dp,H,CteB,RhopZero,Gamma,MassBound,MassFluid); #Gradual #Young
    DataBi4->ConfigCtes(Dp,H, max(CalcK(0.0),CalcK(1.5))/Gamma,RhopZero,Gamma,MassBound,MassFluid);
    DataBi4->ConfigSimMap(OrderDecode(MapRealPosMin),OrderDecode(MapRealPosMax));
    JPartDataBi4::TpPeri tperi=JPartDataBi4::PERI_None;
    if(PeriodicConfig.PeriActive){
      if(PeriodicConfig.PeriXY)tperi=JPartDataBi4::PERI_XY;
      else if(PeriodicConfig.PeriXZ)tperi=JPartDataBi4::PERI_XZ;
      else if(PeriodicConfig.PeriYZ)tperi=JPartDataBi4::PERI_YZ;
      else if(PeriodicConfig.PeriX)tperi=JPartDataBi4::PERI_X;
      else if(PeriodicConfig.PeriY)tperi=JPartDataBi4::PERI_Y;
      else if(PeriodicConfig.PeriZ)tperi=JPartDataBi4::PERI_Z;
      else RunException(met,"The periodic configuration is invalid.");
    }
    DataBi4->ConfigSimPeri(tperi,PeriodicConfig.PeriXinc,PeriodicConfig.PeriYinc,PeriodicConfig.PeriZinc);
    if(div.empty())DataBi4->ConfigSimDiv(JPartDataBi4::DIV_None);
    else if(div=="X")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_X);
    else if(div=="Y")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_Y);
    else if(div=="Z")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_Z);
    else RunException(met,"The division configuration is invalid.");
    if(SvData&SDAT_Binx)Log->AddFileInfo(DirDataOut+"Part_????.bi4","Binary file with particle data in different instants.");
    if(SvData&SDAT_Binx)Log->AddFileInfo(DirDataOut+"PartInfo.ibi4","Binary file with execution information for each instant (input for PartInfo program).");
  }
  //-Configures object to store excluded particles.
  //-Configura objeto para grabacion de particulas excluidas.
  if(SvData&SDAT_Binx){
    DataOutBi4=new JPartOutBi4Save();
    DataOutBi4->ConfigBasic(piece,pieces,RunCode,AppName,Simulate2D,DirDataOut);
    DataOutBi4->ConfigParticles(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid);
    DataOutBi4->ConfigLimits(OrderDecode(MapRealPosMin),OrderDecode(MapRealPosMax),(RhopOut? RhopOutMin: 0),(RhopOut? RhopOutMax: 0));
    DataOutBi4->SaveInitial();
    Log->AddFileInfo(DirDataOut+"PartOut_???.obi4","Binary file with particles excluded during simulation (input for PartVtkOut program).");
  }
  //-Configures object to store data of floatings.
  //-Configura objeto para grabacion de datos de floatings.
  if(SvData&SDAT_Binx && FtCount){
    DataFloatBi4=new JPartFloatBi4Save();
    DataFloatBi4->Config(AppName,DirDataOut,FtCount);
    for(unsigned cf=0;cf<FtCount;cf++)DataFloatBi4->AddHeadData(cf,FtObjs[cf].mkbound,FtObjs[cf].begin,FtObjs[cf].count,FtObjs[cf].mass,FtObjs[cf].radius);
    DataFloatBi4->SaveInitial();
    Log->AddFileInfo(DirDataOut+"PartFloat.fbi4","Binary file with floating body information for each instant (input for FloatingInfo program).");
  }
  //-Creates object to store excluded particles until recordering.
  //-Crea objeto para almacenar las particulas excluidas hasta su grabacion.
  PartsOut=new JPartsOut();
}

//==============================================================================
/// Stores new excluded particles until recordering next PART.
/// Almacena nuevas particulas excluidas hasta la grabacion del proximo PART.
//==============================================================================
void JSph::AddParticlesOut(unsigned nout,const unsigned *idp,const tdouble3 *pos
  ,const tfloat3 *vel,const float *rhop,const typecode *code)
{
  PartsOut->AddParticles(nout,idp,pos,vel,rhop,code);
}

//==============================================================================
/// Manages excluded particles fixed, moving and floating before aborting the execution.
/// Gestiona particulas excluidas fixed, moving y floating antes de abortar la ejecucion.
//==============================================================================
void JSph::AbortBoundOut(unsigned nout,const unsigned *idp,const tdouble3 *pos,const tfloat3 *vel,const float *rhop,const typecode *code){
  //-Prepares data of excluded boundary particles.
  byte* type=new byte[nout];
  byte* motive=new byte[nout];
  unsigned outfixed=0,outmoving=0,outfloat=0;
  unsigned outpos=0,outrhop=0,outmove=0;
  for(unsigned p=0;p<nout;p++){
    //-Checks type of particle.
    switch(CODE_GetType(code[p])){
    case CODE_TYPE_FIXED:     type[p]=0;  outfixed++;   break;
    case CODE_TYPE_MOVING:    type[p]=1;  outmoving++;  break;
    case CODE_TYPE_FLOATING:  type[p]=2;  outfloat++;   break;
    default:                  type[p]=99;               break;
    }
    //-Checks reason for exclusion.
    switch(CODE_GetSpecialValue(code[p])){
    case CODE_OUTPOS:   motive[p]=1; outpos++;   break;
    case CODE_OUTRHOP:  motive[p]=2; outrhop++;  break;
    case CODE_OUTMOVE:  motive[p]=3; outmove++;  break;
    default:            motive[p]=0;             break;
    }
  }
  //-Shows excluded particles information.
  Log->Print(" ");
  Log->Print("*** ERROR: Some boundary particle was excluded. ***");
  Log->Printf("TimeStep: %f  (Nstep: %u)",TimeStep,Nstep);
  unsigned npunknown=nout-outfixed-outmoving-outfloat;
  if(!npunknown)Log->Printf("Total boundary: %u  (fixed=%u  moving=%u  floating=%u)",nout,outfixed,outmoving,outfloat);
  else Log->Printf("Total boundary: %u  (fixed=%u  moving=%u  floating=%u  UNKNOWN=%u)",nout,outfixed,outmoving,outfloat,npunknown);
  npunknown=nout-outpos-outrhop-outmove;
  if(!npunknown)Log->Printf("Excluded for: position=%u  rhop=%u  velocity=%u",outpos,outrhop,outmove);
  else Log->Printf("Excluded for: position=%u  rhop=%u  velocity=%u  UNKNOWN=%u",outpos,outrhop,outmove,npunknown);
  Log->Print(" ");
  //-Creates VTK file.
  std::vector<JFormatFiles2::StScalarData> fields;
  fields.push_back(JFormatFiles2::DefineField("Idp"   ,JFormatFiles2::UInt32 ,1,idp));
  fields.push_back(JFormatFiles2::DefineField("Vel"   ,JFormatFiles2::Float32,3,vel));
  fields.push_back(JFormatFiles2::DefineField("Rhop"  ,JFormatFiles2::Float32,1,rhop));
  fields.push_back(JFormatFiles2::DefineField("Type"  ,JFormatFiles2::UChar8 ,1,type));
  fields.push_back(JFormatFiles2::DefineField("Motive",JFormatFiles2::UChar8 ,1,motive));
  const string file=DirOut+"Error_BoundaryOut.vtk";
  Log->AddFileInfo(file,"Saves the excluded boundary particles.");
  JFormatFiles2::SaveVtk(file,nout,pos,fields);
  //-Aborts execution.
  RunException("AbortBoundOut","Fixed, moving or floating particles were excluded. Checks VTK file Error_BoundaryOut.vtk with excluded particles.");
}

//==============================================================================
/// Returns dynamic memory pointer with data transformed in tfloat3.
/// THE POINTER MUST BE RELEASED AFTER USING IT.
///
/// Devuelve puntero de memoria dinamica con los datos transformados en tfloat3.
/// EL PUNTERO DEBE SER LIBERADO DESPUES DE USARLO.
//==============================================================================
tfloat3* JSph::GetPointerDataFloat3(unsigned n,const tdouble3* v)const{
  tfloat3* v2=new tfloat3[n];
  for(unsigned c=0;c<n;c++)v2[c]=ToTFloat3(v[c]);
  return(v2);
}

//==============================================================================
/// Stores files of particle data.
/// Graba los ficheros de datos de particulas.
//==============================================================================
void JSph::SavePartData(unsigned npok, unsigned nout, const unsigned *idp, const tdouble3 *pos, const tfloat3 *vel, const float *rhop
	, unsigned ndom, const tdouble3 *vdom, const StInfoPartPlus *infoplus) {
	//-Stores particle data and/or information in bi4 format.
	//-Graba datos de particulas y/o informacion en formato bi4.
	if (DataBi4) {
		tfloat3* posf3 = NULL;
		TimerPart.Stop();
		JBinaryData* bdpart = DataBi4->AddPartInfo(Part, TimeStep, npok, nout, Nstep, TimerPart.GetElapsedTimeD() / 1000., vdom[0], vdom[1], TotalNp);
		if (infoplus && SvData&SDAT_Info) {
			bdpart->SetvDouble("dtmean", (!Nstep ? 0 : (TimeStep - TimeStepM1) / (Nstep - PartNstep)));
			bdpart->SetvDouble("dtmin", (!Nstep ? 0 : PartDtMin));
			bdpart->SetvDouble("dtmax", (!Nstep ? 0 : PartDtMax));
			if (DtFixed)bdpart->SetvDouble("dterror", DtFixed->GetDtError(true));
			bdpart->SetvDouble("timesim", infoplus->timesim);
			bdpart->SetvUint("nct", infoplus->nct);
			bdpart->SetvUint("npbin", infoplus->npbin);
			bdpart->SetvUint("npbout", infoplus->npbout);
			bdpart->SetvUint("npf", infoplus->npf);
			bdpart->SetvUint("npbper", infoplus->npbper);
			bdpart->SetvUint("npfper", infoplus->npfper);
			bdpart->SetvUint("newnp", infoplus->newnp);
			bdpart->SetvLlong("cpualloc", infoplus->memorycpualloc);
			if (infoplus->gpudata) {
				bdpart->SetvLlong("nctalloc", infoplus->memorynctalloc);
				bdpart->SetvLlong("nctused", infoplus->memorynctused);
				bdpart->SetvLlong("npalloc", infoplus->memorynpalloc);
				bdpart->SetvLlong("npused", infoplus->memorynpused);
			}
		}
		if (SvData&SDAT_Binx) {
			if (SvDouble)DataBi4->AddPartData(npok, idp, pos, vel, rhop);
			else {
				posf3 = GetPointerDataFloat3(npok, pos);
				DataBi4->AddPartData(npok, idp, posf3, vel, rhop);
			}
			float *press = NULL;
			if (0) {//-Example saving a new array (Pressure) in files BI4.
				press = new float[npok];
				for (unsigned p = 0; p<npok; p++)press[p] = (idp[p] >= CaseNbound ? CalcK((2.0-pos[p].x))/Gamma * (pow(rhop[p] / RhopZero, Gamma) - 1.0f) : 0.f);
				DataBi4->AddPartData("Pressure", npok, press);
			}
			DataBi4->SaveFilePart();
			delete[] press; press = NULL;//-Memory must to be deallocated after saving file because DataBi4 uses this memory space.
		}
		if (SvData&SDAT_Info)DataBi4->SaveFileInfo();
		delete[] posf3;
	}

	//-Stores VTK nd/or CSV files.
	if ((SvData&SDAT_Csv) || (SvData&SDAT_Vtk)) {
		//-Generates array with posf3 and type of particle.
		tfloat3* posf3 = GetPointerDataFloat3(npok, pos);
		byte *type = new byte[npok];
		for (unsigned p = 0; p<npok; p++) {
			const unsigned id = idp[p];
			type[p] = (id >= CaseNbound ? 3 : (id<CaseNfixed ? 0 : (id<CaseNpb ? 1 : 2)));
		}
		//-Defines fields to be stored.
		JFormatFiles2::StScalarData fields[8];
		unsigned nfields = 0;
		if (idp) { fields[nfields] = JFormatFiles2::DefineField("Idp", JFormatFiles2::UInt32, 1, idp);   nfields++; }
		if (vel) { fields[nfields] = JFormatFiles2::DefineField("Vel", JFormatFiles2::Float32, 3, vel);   nfields++; }
		if (rhop) { fields[nfields] = JFormatFiles2::DefineField("Rhop", JFormatFiles2::Float32, 1, rhop);  nfields++; }
		if (type) { fields[nfields] = JFormatFiles2::DefineField("Type", JFormatFiles2::UChar8, 1, type);  nfields++; }
		if (SvData&SDAT_Vtk)JFormatFiles2::SaveVtk(DirDataOut + fun::FileNameSec("PartVtk.vtk", Part), npok, posf3, nfields, fields);
		if (SvData&SDAT_Csv)JFormatFiles2::SaveCsv(DirDataOut + fun::FileNameSec("PartCsv.csv", Part), CsvSepComa, npok, posf3, nfields, fields);
		//-Deallocate of memory.
		delete[] posf3;
		delete[] type;
	}

	//-Stores data of excluded particles.
	if (DataOutBi4 && PartsOut->GetCount()) {
		DataOutBi4->SavePartOut(SvDouble, Part, TimeStep, PartsOut->GetCount(), PartsOut->GetIdpOut(), NULL, PartsOut->GetPosOut(), PartsOut->GetVelOut(), PartsOut->GetRhopOut(), PartsOut->GetMotiveOut());
	}

	//-Stores data of floating bodies.
	if (DataFloatBi4) {
		if (CellOrder == ORDER_XYZ)for (unsigned cf = 0; cf<FtCount; cf++)DataFloatBi4->AddPartData(cf, FtObjs[cf].center, FtObjs[cf].fvel, FtObjs[cf].fomega);
		else                    for (unsigned cf = 0; cf<FtCount; cf++)DataFloatBi4->AddPartData(cf, OrderDecodeValue(CellOrder, FtObjs[cf].center), OrderDecodeValue(CellOrder, FtObjs[cf].fvel), OrderDecodeValue(CellOrder, FtObjs[cf].fomega));
		DataFloatBi4->SavePartFloat(Part, TimeStep, (UseDEM ? DemDtForce : 0));
	}

	//-Empties stock of excluded particles.
	//-Vacia almacen de particulas excluidas.
	PartsOut->Clear();
}


//==============================================================================
/// Generates data output files.
/// Genera los ficheros de salida de datos.
//==============================================================================
void JSph::SaveData(unsigned npok,const unsigned *idp,const tdouble3 *pos,const tfloat3 *vel,const float *rhop
  ,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus)
{
  const char met[]="SaveData";
  string suffixpartx=fun::PrintStr("_%04d",Part);

  //-Counts new excluded particles.
  //-Contabiliza nuevas particulas excluidas.
  const unsigned noutpos=PartsOut->GetOutPosCount(),noutrhop=PartsOut->GetOutRhopCount(),noutmove=PartsOut->GetOutMoveCount();
  const unsigned nout=noutpos+noutrhop+noutmove;
  if(nout!=PartsOut->GetCount())RunException(met,"Excluded particles with unknown reason.");
  AddOutCount(noutpos,noutrhop,noutmove);

  //-Stores data files of particles.
  SavePartData(npok,nout,idp,pos,vel,rhop,ndom,vdom,infoplus);

  //-Reinitialises limits of dt. | Reinicia limites de dt.
  PartDtMin=DBL_MAX; PartDtMax=-DBL_MAX;

  //-Computation of time.
  if(Part>PartIni||Nstep){
    TimerPart.Stop();
    double tpart=TimerPart.GetElapsedTimeD()/1000;
    double tseg=tpart/(TimeStep-TimeStepM1);
    TimerSim.Stop();
    double tcalc=TimerSim.GetElapsedTimeD()/1000;
    double tleft=(tcalc/(TimeStep-TimeStepIni))*(TimeMax-TimeStep);
    Log->Printf("Part%s  %12.6f  %12d  %7d  %9.2f  %14s",suffixpartx.c_str(),TimeStep,(Nstep+1),Nstep-PartNstep,tseg,fun::GetDateTimeAfter(int(tleft)).c_str());
  }
  else Log->Printf("Part%s        %u particles successfully stored",suffixpartx.c_str(),npok);

  //-Shows info of the excluded particles.
  if(nout){
    PartOut+=nout;
    Log->Printf("  Particles out: %u  (total: %u)",nout,PartOut);
  }

  //-Cheks number of excluded particles.
  if(nout){
    //-Cheks number of excluded particles in one PART.
    if(nout>=float(infoplus->npf)*(float(PartsOutWrn)/100.f)){
      Log->PrintfWarning("More than %d%% of current fluid particles were excluded in one PART (t:%g, nstep:%u)",PartsOutWrn,TimeStep,Nstep);
      if(PartsOutWrn==1)PartsOutWrn=2;
      else if(PartsOutWrn==2)PartsOutWrn=5;
      else if(PartsOutWrn==5)PartsOutWrn=10;
      else PartsOutWrn+=10;
    }
    //-Cheks number of total excluded particles.
    const unsigned noutt=GetOutPosCount()+GetOutRhopCount()+GetOutMoveCount();
    if(PartsOutTotWrn<100 && noutt>=float(TotalNp)*(float(PartsOutTotWrn)/100.f)){
      Log->PrintfWarning("More than %d%% of particles were excluded (t:%g, nstep:%u)",PartsOutTotWrn,TimeStep,Nstep);
      PartsOutTotWrn+=10;
    }
  }

  if(SvDomainVtk)SaveDomainVtk(ndom,vdom);
  if(SaveDt)SaveDt->SaveData();
  if(GaugeSystem)GaugeSystem->SaveResults(Part);
}



////////////////////////////////////////////////////
// SavePartData update 1: add tflaot3 deformation
////////////////////////////////////////////////////
void JSph::SavePartData_M1(unsigned npok, unsigned nout, const unsigned* idp, const tdouble3* pos, const tfloat3* vel
	, const float* rhop, const float* pore, const float* press, const float* massp, const tsymatrix3f* qfp, const float* nabvx
	, const float* vonMises, const float* grVelSave, const unsigned* cellOSpr, tfloat3* gradvel, unsigned ndom, const tdouble3* vdom, const StInfoPartPlus* infoplus) {
	//-Stores particle data and/or information in bi4 format.
	//-Graba datos de particulas y/o informacion en formato bi4.

	if (DataBi4) {
		tfloat3* posf3 = NULL;
		TimerPart.Stop();
		JBinaryData* bdpart = DataBi4->AddPartInfo(Part, TimeStep, npok, nout, Nstep, TimerPart.GetElapsedTimeD() / 1000., vdom[0], vdom[1], TotalNp);
		if (infoplus && SvData & SDAT_Info) {
			bdpart->SetvDouble("dtmean", (!Nstep ? 0 : (TimeStep - TimeStepM1) / (Nstep - PartNstep)));
			bdpart->SetvDouble("dtmin", (!Nstep ? 0 : PartDtMin));
			bdpart->SetvDouble("dtmax", (!Nstep ? 0 : PartDtMax));
			if (DtFixed)bdpart->SetvDouble("dterror", DtFixed->GetDtError(true));
			bdpart->SetvDouble("timesim", infoplus->timesim);
			bdpart->SetvUint("nct", infoplus->nct);
			bdpart->SetvUint("npbin", infoplus->npbin);
			bdpart->SetvUint("npbout", infoplus->npbout);
			bdpart->SetvUint("npf", infoplus->npf);
			bdpart->SetvUint("npbper", infoplus->npbper);
			bdpart->SetvUint("npfper", infoplus->npfper);
			bdpart->SetvLlong("cpualloc", infoplus->memorycpualloc);
			if (infoplus->gpudata) {
				bdpart->SetvLlong("nctalloc", infoplus->memorynctalloc);
				bdpart->SetvLlong("nctused", infoplus->memorynctused);
				bdpart->SetvLlong("npalloc", infoplus->memorynpalloc);
				bdpart->SetvLlong("npused", infoplus->memorynpused);
			}
		}
		if (SvData & SDAT_Binx) {
			if (SvDouble)DataBi4->AddPartData(npok, idp, pos, vel, rhop);
			else {
				posf3 = GetPointerDataFloat3(npok, pos);
				DataBi4->AddPartData(npok, idp, posf3, vel, rhop);
			}
			// Press
			float* pressp = NULL;
			pressp = new float[npok];
			for (unsigned p = 0; p < npok; p++) pressp[p] = press[p];
			DataBi4->AddPartData("Press", npok, pressp);

			// Mass
			float* mass = NULL;
			mass = new float[npok];
			for (unsigned p = 0; p < npok; p++) mass[p] = massp[p];
			DataBi4->AddPartData("Mass", npok, mass);

			// Nabla vx
			float* nvx = NULL;
			nvx = new float[npok];
			for (unsigned p = 0; p < npok; p++) nvx[p] = nabvx[p];
			DataBi4->AddPartData("NabVx", npok, nvx);

			// Von Mises
			float* vM3D = NULL;
			vM3D = new float[npok];
			for (unsigned p = 0; p < npok; p++) vM3D[p] = vonMises[p];
			DataBi4->AddPartData("VonMises3D", npok, vM3D);

			// GradVelSave
			float* grVS = NULL;
			grVS = new float[npok];
			for (unsigned p = 0; p < npok; p++) grVS[p] = grVelSave[p];
			DataBi4->AddPartData("GradVel", npok, grVS);

			// CellOffSpring
			unsigned* cOS = NULL;
			cOS = new unsigned[npok];
			for (unsigned p = 0; p < npok; p++) cOS[p] = cellOSpr[p];
			DataBi4->AddPartData("CellOffSpring", npok, cOS);

			tfloat3* gr = NULL;
			gr = new tfloat3[npok];
			for (unsigned p = 0; p < npok; p++) gr[p] = gradvel[p];
			DataBi4->AddPartData("StrainDot", npok, gr);

			/*// Quadratic form -- Blocked formulation since PartVtk does not seem to read tsymatrix
			tsymatrix3f *qf = NULL;
			qf = new tsymatrix3f[npok];
			for (unsigned p = 0; p < npok; p++) qf[p] = qfp[p];
			DataBi4->AddPartData("Qf", npok, qf);*/
			// Quadratic form -- term to term formulation (Voigt notation)
			float* qfxx = NULL;
			float* qfyy = NULL;
			float* qfzz = NULL;
			float* qfyz = NULL;
			float* qfxz = NULL;
			float* qfxy = NULL;
			qfxx = new float[npok];
			qfyy = new float[npok];
			qfzz = new float[npok];
			qfyz = new float[npok];
			qfxz = new float[npok];
			qfxy = new float[npok];
			for (unsigned p = 0; p < npok; p++) {
				qfxx[p] = qfp[p].xx;
				qfyy[p] = qfp[p].yy;
				qfzz[p] = qfp[p].zz;
				qfyz[p] = qfp[p].yz;
				qfxz[p] = qfp[p].xz;
				qfxy[p] = qfp[p].xy;
			}
			DataBi4->AddPartData("Qfxx", npok, qfxx);
			DataBi4->AddPartData("Qfyy", npok, qfyy);
			DataBi4->AddPartData("Qfzz", npok, qfzz);
			DataBi4->AddPartData("Qfyz", npok, qfyz);
			DataBi4->AddPartData("Qfxz", npok, qfxz);
			DataBi4->AddPartData("Qfxy", npok, qfxy);


			/*tmatrix3f* tensor = NULL;
			tensor = new tmatrix3f[npok];
			for (unsigned p = 0; p < npok; p++) {
				tensor[p].a11 = qfp[p].xx;
				tensor[p].a12 = qfp[p].xy;
				tensor[p].a13 = qfp[p].xz;

				tensor[p].a21 = qfp[p].xy;
				tensor[p].a22 = qfp[p].yy;
				tensor[p].a23 = qfp[p].yz;

				tensor[p].a31 = qfp[p].xz;
				tensor[p].a32 = qfp[p].yz;
				tensor[p].a33 = qfp[p].zz;
			}
			DataBi4->AddPartData("Shape", npok, tensor);*/

			/*tfloat3* tensorAxes = NULL;
			tfloat3* tensorDiag = NULL; // x <- xy ; y <- yz ; z <- xz
			if (qfp) {
				tensorAxes = new tfloat3[npok];
				tensorDiag = new tfloat3[npok];
				for (unsigned p = 0; p < npok; p++) {
					tensorAxes[p].x = qfp[p].xx;
					tensorAxes[p].y = qfp[p].yy;
					tensorAxes[p].z = qfp[p].zz;

					tensorDiag[p].x = qfp[p].xy;
					tensorDiag[p].y = qfp[p].yz;
					tensorDiag[p].z = qfp[p].xz;
				}
			}
			DataBi4->AddPartData("TensorAxes", npok, tensorAxes);
			DataBi4->AddPartData("TensorDiagAxes", npok, tensorDiag);*/

			DataBi4->SaveFilePart();
			//delete[] tensor; tensor = NULL;
			// Cleaning remains: fix 17/12
			delete[] qfxx; qfxx = NULL;
			delete[] qfyy; qfyy = NULL;
			delete[] qfzz; qfzz = NULL;
			delete[] qfyz; qfyz = NULL;
			delete[] qfxz; qfxz = NULL;
			delete[] qfxy; qfxy = NULL;
			delete[] vM3D; vM3D = NULL;
			delete[] mass; mass = NULL;
			delete[] nvx; nvx = NULL;
			delete[] grVS; grVS = NULL;
			delete[] cOS; cOS = NULL;
			delete[] gr; gr = NULL;
			delete[] pressp; pressp = NULL;//-Memory must to be deallocated after saving file because DataBi4 uses this memory space.
										   //delete[] gradvelSave; gradvelSave = NULL;	

		}
		if (SvData & SDAT_Info)DataBi4->SaveFileInfo();
		delete[] posf3;
	}

	//-Graba ficheros VKT y/o CSV.
	//-Stores VTK nd/or CSV files.
	if ((SvData & SDAT_Csv) || (SvData & SDAT_Vtk)) {
		//-Genera array con posf3 y tipo de particula.
		//-Generates array with posf3 and type of particle.
		tfloat3* posf3 = GetPointerDataFloat3(npok, pos);
		byte* type = new byte[npok];
		for (unsigned p = 0; p < npok; p++) {
			const unsigned id = idp[p];
			type[p] = (id >= CaseNbound ? 3 : (id < CaseNfixed ? 0 : (id < CaseNpb ? 1 : 2)));
		}

		// Generate coeffs for csv thanks to symetric matrix -- Augustin
		tfloat3* tensorAxes = NULL;
		tfloat3* tensorDiag = NULL; // x <- xy ; y <- yz ; z <- xz
		if (qfp) {
			tensorAxes = new tfloat3[npok];
			tensorDiag = new tfloat3[npok];
			for (unsigned p = 0; p < npok; p++) {
				tensorAxes[p].x = qfp[p].xx;
				tensorAxes[p].y = qfp[p].yy;
				tensorAxes[p].z = qfp[p].zz;

				tensorDiag[p].x = qfp[p].xy;
				tensorDiag[p].y = qfp[p].yz;
				tensorDiag[p].z = qfp[p].xz;
			}
		}

		//-Define campos a grabar.
		//-Defines fields to be stored.
		JFormatFiles2::StScalarData fields[16];
		unsigned nfields = 0;
		if (idp) { fields[nfields] = JFormatFiles2::DefineField("Idp", JFormatFiles2::UInt32, 1, idp);   nfields++; }
		if (vel) { fields[nfields] = JFormatFiles2::DefineField("Vel", JFormatFiles2::Float32, 3, vel);   nfields++; }
		if (rhop) { fields[nfields] = JFormatFiles2::DefineField("Rhop", JFormatFiles2::Float32, 1, rhop);  nfields++; }
		if (pore) { fields[nfields] = JFormatFiles2::DefineField("Porep", JFormatFiles2::Float32, 1, pore);  nfields++; }
		if (massp) { fields[nfields] = JFormatFiles2::DefineField("Massp", JFormatFiles2::Float32, 1, massp);  nfields++; }
		if (press) { fields[nfields] = JFormatFiles2::DefineField("Pressp", JFormatFiles2::Float32, 1, press);  nfields++; }
		// Augustin
		if (qfp) {
			fields[nfields] = JFormatFiles2::DefineField("TensorAxes", JFormatFiles2::Float32, 3, tensorAxes);  nfields++;
			fields[nfields] = JFormatFiles2::DefineField("TensorDiagAxes", JFormatFiles2::Float32, 3, tensorDiag);  nfields++;
		}
		if (vonMises) { fields[nfields] = JFormatFiles2::DefineField("VonMises3D", JFormatFiles2::Float32, 1, vonMises);  nfields++; }
		if (grVelSave) { fields[nfields] = JFormatFiles2::DefineField("GradVel", JFormatFiles2::Float32, 1, grVelSave);  nfields++; }
		if (cellOSpr) { fields[nfields] = JFormatFiles2::DefineField("CellOffSpring", JFormatFiles2::UInt32, 1, cellOSpr);  nfields++; }
		if (gradvel) { fields[nfields] = JFormatFiles2::DefineField("StrainDot", JFormatFiles2::Float32, 3, gradvel);   nfields++; }
		if (type) { fields[nfields] = JFormatFiles2::DefineField("Type", JFormatFiles2::UChar8, 1, type);  nfields++; }
		if (SvData & SDAT_Vtk)JFormatFiles2::SaveVtk(DirDataOut + fun::FileNameSec("PartVtk.vtk", Part), npok, posf3, nfields, fields);
		//if (SvData&SDAT_Csv)JFormatFiles2::SaveCsv(DirDataOut + fun::FileNameSec("PartCsv.csv", Part), CsvSepComa, npok, posf3, nfields, fields);
		//-libera memoria.
		//-release of memory.
		delete[] posf3;
		delete[] type;
		if (qfp) {
			delete[] tensorAxes;
			delete[] tensorDiag;
		}
	}

	//-Graba datos de particulas excluidas.
	//-Stores data of excluded particles.
	if (DataOutBi4 && PartsOut->GetCount()) {
		DataOutBi4->SavePartOut(SvDouble, Part, TimeStep, PartsOut->GetCount(), PartsOut->GetIdpOut(), NULL, PartsOut->GetPosOut(), PartsOut->GetVelOut(), PartsOut->GetRhopOut(), PartsOut->GetMotiveOut());
	}

	//-Graba datos de floatings.
	//-Stores data of floatings.
	if (DataFloatBi4) {
		if (CellOrder == ORDER_XYZ)for (unsigned cf = 0; cf < FtCount; cf++)DataFloatBi4->AddPartData(cf, FtObjs[cf].center, FtObjs[cf].fvel, FtObjs[cf].fomega);
		else                    for (unsigned cf = 0; cf < FtCount; cf++)DataFloatBi4->AddPartData(cf, OrderDecodeValue(CellOrder, FtObjs[cf].center), OrderDecodeValue(CellOrder, FtObjs[cf].fvel), OrderDecodeValue(CellOrder, FtObjs[cf].fomega));
		DataFloatBi4->SavePartFloat(Part, TimeStep, (UseDEM ? DemDtForce : 0));
	}

	//-Vacia almacen de particulas excluidas.
	//-Empties stock of excluded particles.
	PartsOut->Clear();
}

////////////////////////////////////////////////////
// SavePartData update 1: add tflaot3 deformation, -float NabVx
////////////////////////////////////////////////////
void JSph::SavePartData11_M(unsigned npok, unsigned nout, const unsigned* idp, const tdouble3* pos, const tfloat3* vel
	, const float* rhop, const float* pore, const float* press, const float* massp, const tsymatrix3f* qfp
	, const float* vonMises, const float* grVelSave, const unsigned* cellOSpr, tfloat3* gradvel, unsigned ndom, const tdouble3* vdom, const StInfoPartPlus* infoplus) {
	//-Stores particle data and/or information in bi4 format.
	//-Graba datos de particulas y/o informacion en formato bi4.

	if (DataBi4) {
		tfloat3* posf3 = NULL;
		TimerPart.Stop();
		JBinaryData* bdpart = DataBi4->AddPartInfo(Part, TimeStep, npok, nout, Nstep, TimerPart.GetElapsedTimeD() / 1000., vdom[0], vdom[1], TotalNp);
		if (infoplus && SvData & SDAT_Info) {
			bdpart->SetvDouble("dtmean", (!Nstep ? 0 : (TimeStep - TimeStepM1) / (Nstep - PartNstep)));
			bdpart->SetvDouble("dtmin", (!Nstep ? 0 : PartDtMin));
			bdpart->SetvDouble("dtmax", (!Nstep ? 0 : PartDtMax));
			if (DtFixed)bdpart->SetvDouble("dterror", DtFixed->GetDtError(true));
			bdpart->SetvDouble("timesim", infoplus->timesim);
			bdpart->SetvUint("nct", infoplus->nct);
			bdpart->SetvUint("npbin", infoplus->npbin);
			bdpart->SetvUint("npbout", infoplus->npbout);
			bdpart->SetvUint("npf", infoplus->npf);
			bdpart->SetvUint("npbper", infoplus->npbper);
			bdpart->SetvUint("npfper", infoplus->npfper);
			bdpart->SetvLlong("cpualloc", infoplus->memorycpualloc);
			if (infoplus->gpudata) {
				bdpart->SetvLlong("nctalloc", infoplus->memorynctalloc);
				bdpart->SetvLlong("nctused", infoplus->memorynctused);
				bdpart->SetvLlong("npalloc", infoplus->memorynpalloc);
				bdpart->SetvLlong("npused", infoplus->memorynpused);
			}
		}
		if (SvData & SDAT_Binx) {
			if (SvDouble)DataBi4->AddPartData(npok, idp, pos, vel, rhop);
			else {
				posf3 = GetPointerDataFloat3(npok, pos);
				DataBi4->AddPartData(npok, idp, posf3, vel, rhop);
			}
			// Press
			float* pressp = NULL;
			pressp = new float[npok];
			for (unsigned p = 0; p < npok; p++) pressp[p] = press[p];
			DataBi4->AddPartData("Press", npok, pressp);

			// Mass
			float* mass = NULL;
			mass = new float[npok];
			for (unsigned p = 0; p < npok; p++) mass[p] = massp[p];
			DataBi4->AddPartData("Mass", npok, mass);

			// Von Mises
			float* vM3D = NULL;
			vM3D = new float[npok];
			for (unsigned p = 0; p < npok; p++) vM3D[p] = vonMises[p];
			DataBi4->AddPartData("VonMises3D", npok, vM3D);

			// GradVelSave
			float* grVS = NULL;
			grVS = new float[npok];
			for (unsigned p = 0; p < npok; p++) grVS[p] = grVelSave[p];
			DataBi4->AddPartData("GradVel", npok, grVS);

			// CellOffSpring
			unsigned* cOS = NULL;
			cOS = new unsigned[npok];
			for (unsigned p = 0; p < npok; p++) cOS[p] = cellOSpr[p];
			DataBi4->AddPartData("CellOffSpring", npok, cOS);

			tfloat3* gr = NULL;
			gr = new tfloat3[npok];
			for (unsigned p = 0; p < npok; p++) gr[p] = gradvel[p];
			DataBi4->AddPartData("StrainDot", npok, gr);

			/*// Quadratic form -- Blocked formulation since PartVtk does not seem to read tsymatrix
			tsymatrix3f *qf = NULL;
			qf = new tsymatrix3f[npok];
			for (unsigned p = 0; p < npok; p++) qf[p] = qfp[p];
			DataBi4->AddPartData("Qf", npok, qf);*/
			// Quadratic form -- term to term formulation (Voigt notation)
			float* qfxx = NULL;
			float* qfyy = NULL;
			float* qfzz = NULL;
			float* qfyz = NULL;
			float* qfxz = NULL;
			float* qfxy = NULL;
			qfxx = new float[npok];
			qfyy = new float[npok];
			qfzz = new float[npok];
			qfyz = new float[npok];
			qfxz = new float[npok];
			qfxy = new float[npok];
			for (unsigned p = 0; p < npok; p++) {
				qfxx[p] = qfp[p].xx;
				qfyy[p] = qfp[p].yy;
				qfzz[p] = qfp[p].zz;
				qfyz[p] = qfp[p].yz;
				qfxz[p] = qfp[p].xz;
				qfxy[p] = qfp[p].xy;
			}
			DataBi4->AddPartData("Qfxx", npok, qfxx);
			DataBi4->AddPartData("Qfyy", npok, qfyy);
			DataBi4->AddPartData("Qfzz", npok, qfzz);
			DataBi4->AddPartData("Qfyz", npok, qfyz);
			DataBi4->AddPartData("Qfxz", npok, qfxz);
			DataBi4->AddPartData("Qfxy", npok, qfxy);

			DataBi4->SaveFilePart();
			//delete[] tensor; tensor = NULL;
			// Cleaning remains: fix 17/12
			delete[] qfxx; qfxx = NULL;
			delete[] qfyy; qfyy = NULL;
			delete[] qfzz; qfzz = NULL;
			delete[] qfyz; qfyz = NULL;
			delete[] qfxz; qfxz = NULL;
			delete[] qfxy; qfxy = NULL;
			delete[] vM3D; vM3D = NULL;
			delete[] mass; mass = NULL;
			delete[] grVS; grVS = NULL;
			delete[] cOS; cOS = NULL;
			delete[] gr; gr = NULL;
			delete[] pressp; pressp = NULL;//-Memory must to be deallocated after saving file because DataBi4 uses this memory space.
										   //delete[] gradvelSave; gradvelSave = NULL;	
		}
		if (SvData & SDAT_Info)DataBi4->SaveFileInfo();
		delete[] posf3;
	}

	//-Graba ficheros VKT y/o CSV.
	//-Stores VTK nd/or CSV files.
	if ((SvData & SDAT_Csv) || (SvData & SDAT_Vtk)) {
		//-Genera array con posf3 y tipo de particula.
		//-Generates array with posf3 and type of particle.
		tfloat3* posf3 = GetPointerDataFloat3(npok, pos);
		byte* type = new byte[npok];
		for (unsigned p = 0; p < npok; p++) {
			const unsigned id = idp[p];
			type[p] = (id >= CaseNbound ? 3 : (id < CaseNfixed ? 0 : (id < CaseNpb ? 1 : 2)));
		}

		// Generate coeffs for csv thanks to symetric matrix -- Augustin
		tfloat3* tensorAxes = NULL;
		tfloat3* tensorDiag = NULL; // x <- xy ; y <- yz ; z <- xz
		if (qfp) {
			tensorAxes = new tfloat3[npok];
			tensorDiag = new tfloat3[npok];
			for (unsigned p = 0; p < npok; p++) {
				tensorAxes[p].x = qfp[p].xx;
				tensorAxes[p].y = qfp[p].yy;
				tensorAxes[p].z = qfp[p].zz;

				tensorDiag[p].x = qfp[p].xy;
				tensorDiag[p].y = qfp[p].yz;
				tensorDiag[p].z = qfp[p].xz;
			}
		}

		//-Define campos a grabar.
		//-Defines fields to be stored.
		JFormatFiles2::StScalarData fields[16];
		unsigned nfields = 0;
		if (idp) { fields[nfields] = JFormatFiles2::DefineField("Idp", JFormatFiles2::UInt32, 1, idp);   nfields++; }
		if (vel) { fields[nfields] = JFormatFiles2::DefineField("Vel", JFormatFiles2::Float32, 3, vel);   nfields++; }
		if (rhop) { fields[nfields] = JFormatFiles2::DefineField("Rhop", JFormatFiles2::Float32, 1, rhop);  nfields++; }
		if (pore) { fields[nfields] = JFormatFiles2::DefineField("Porep", JFormatFiles2::Float32, 1, pore);  nfields++; }
		if (massp) { fields[nfields] = JFormatFiles2::DefineField("Massp", JFormatFiles2::Float32, 1, massp);  nfields++; }
		if (press) { fields[nfields] = JFormatFiles2::DefineField("Pressp", JFormatFiles2::Float32, 1, press);  nfields++; }
		// Augustin
		if (qfp) {
			fields[nfields] = JFormatFiles2::DefineField("TensorAxes", JFormatFiles2::Float32, 3, tensorAxes);  nfields++;
			fields[nfields] = JFormatFiles2::DefineField("TensorDiagAxes", JFormatFiles2::Float32, 3, tensorDiag);  nfields++;
		}
		if (vonMises) { fields[nfields] = JFormatFiles2::DefineField("VonMises3D", JFormatFiles2::Float32, 1, vonMises);  nfields++; }
		if (grVelSave) { fields[nfields] = JFormatFiles2::DefineField("GradVel", JFormatFiles2::Float32, 1, grVelSave);  nfields++; }
		if (cellOSpr) { fields[nfields] = JFormatFiles2::DefineField("CellOffSpring", JFormatFiles2::UInt32, 1, cellOSpr);  nfields++; }
		if (gradvel) { fields[nfields] = JFormatFiles2::DefineField("StrainDot", JFormatFiles2::Float32, 3, gradvel);   nfields++; }
		if (type) { fields[nfields] = JFormatFiles2::DefineField("Type", JFormatFiles2::UChar8, 1, type);  nfields++; }
		if (SvData & SDAT_Vtk)JFormatFiles2::SaveVtk(DirDataOut + fun::FileNameSec("PartVtk.vtk", Part), npok, posf3, nfields, fields);
		//if (SvData&SDAT_Csv)JFormatFiles2::SaveCsv(DirDataOut + fun::FileNameSec("PartCsv.csv", Part), CsvSepComa, npok, posf3, nfields, fields);
		//-libera memoria.
		//-release of memory.
		delete[] posf3;
		delete[] type;
		if (qfp) {
			delete[] tensorAxes;
			delete[] tensorDiag;
		}
	}

	//-Graba datos de particulas excluidas.
	//-Stores data of excluded particles.
	if (DataOutBi4 && PartsOut->GetCount()) {
		DataOutBi4->SavePartOut(SvDouble, Part, TimeStep, PartsOut->GetCount(), PartsOut->GetIdpOut(), NULL, PartsOut->GetPosOut(), PartsOut->GetVelOut(), PartsOut->GetRhopOut(), PartsOut->GetMotiveOut());
	}

	//-Graba datos de floatings.
	//-Stores data of floatings.
	if (DataFloatBi4) {
		if (CellOrder == ORDER_XYZ)for (unsigned cf = 0; cf < FtCount; cf++)DataFloatBi4->AddPartData(cf, FtObjs[cf].center, FtObjs[cf].fvel, FtObjs[cf].fomega);
		else                    for (unsigned cf = 0; cf < FtCount; cf++)DataFloatBi4->AddPartData(cf, OrderDecodeValue(CellOrder, FtObjs[cf].center), OrderDecodeValue(CellOrder, FtObjs[cf].fvel), OrderDecodeValue(CellOrder, FtObjs[cf].fomega));
		DataFloatBi4->SavePartFloat(Part, TimeStep, (UseDEM ? DemDtForce : 0));
	}

	//-Vacia almacen de particulas excluidas.
	//-Empties stock of excluded particles.
	PartsOut->Clear();
}


///////////////////////////
// SaveData update 1: add float3 deformation
///////////////////////////
void JSph::SaveData_M1(unsigned npok, const unsigned* idp, const tdouble3* pos, const tfloat3* vel, const float* rhop, const float* pore
	, const float* press, const float* mass, const tsymatrix3f* qf, const float* nabvx, const float* vonMises
	, const float* gradVelSav, unsigned* cellOSpr, tfloat3* gradvel, unsigned ndom, const tdouble3* vdom, const StInfoPartPlus* infoplus)
{
	string suffixpartx = fun::PrintStr("_%04d", Part);

	//-Contabiliza nuevas particulas excluidas.
	//-Counts new excluded particles.
	const unsigned noutpos = PartsOut->GetOutPosCount(), noutrhop = PartsOut->GetOutRhopCount(), noutmove = PartsOut->GetOutMoveCount();
	const unsigned nout = noutpos + noutrhop + noutmove;
	AddOutCount(noutpos, noutrhop, noutmove);

	//-Graba ficheros con datos de particulas.
	//-Stores data files of particles.
	SavePartData_M1(npok, nout, idp, pos, vel, rhop, pore, press, mass, qf, nabvx, vonMises, gradVelSav, cellOSpr, gradvel, ndom, vdom, infoplus);

	//-Reinicia limites de dt.
	//-Reinitialises limits of dt.
	PartDtMin = DBL_MAX; PartDtMax = -DBL_MAX;

	//-Calculo de tiempo.
	//-Computation of time.
	if (Part > PartIni || Nstep) {
		TimerPart.Stop();
		double tpart = TimerPart.GetElapsedTimeD() / 1000;
		double tseg = tpart / (TimeStep - TimeStepM1);
		TimerSim.Stop();
		double tcalc = TimerSim.GetElapsedTimeD() / 1000;
		double tleft = (tcalc / (TimeStep - TimeStepIni)) * (TimeMax - TimeStep);
		Log->Printf("Part%s  %12.6f  %12d  %7d  %9.2f  %14s", suffixpartx.c_str(), TimeStep, (Nstep + 1), Nstep - PartNstep, tseg, fun::GetDateTimeAfter(int(tleft)).c_str());
	}
	else Log->Printf("Part%s        %u particles successfully stored", suffixpartx.c_str(), npok);


	//-Muestra info de particulas excluidas
	//-Shows info of the excluded particles
	if (nout) {
		PartOut += nout;
		Log->Printf("  Particles out: %u  (total: %u)", nout, PartOut);
	}

	if (SvDomainVtk)SaveDomainVtk(ndom, vdom);
	if (SaveDt)SaveDt->SaveData();
	if (GaugeSystem)GaugeSystem->SaveResults(Part);
}


///////////////////////////
// SaveData 11 (V32-Da): +float3 def, -float NabVx
///////////////////////////
void JSph::SaveData11_M(unsigned npok, const unsigned* idp, const tdouble3* pos, const tfloat3* vel, const float* rhop, const float* pore
	, const float* press, const float* mass, const tsymatrix3f* qf, const float* vonMises
	, const float* gradVelSav, unsigned* cellOSpr, tfloat3* gradvel, unsigned ndom, const tdouble3* vdom, const StInfoPartPlus* infoplus)
{
	string suffixpartx = fun::PrintStr("_%04d", Part);

	//-Contabiliza nuevas particulas excluidas.
	//-Counts new excluded particles.
	const unsigned noutpos = PartsOut->GetOutPosCount(), noutrhop = PartsOut->GetOutRhopCount(), noutmove = PartsOut->GetOutMoveCount();
	const unsigned nout = noutpos + noutrhop + noutmove;
	AddOutCount(noutpos, noutrhop, noutmove);

	//-Graba ficheros con datos de particulas.
	//-Stores data files of particles.
	SavePartData11_M(npok, nout, idp, pos, vel, rhop, pore, press, mass, qf, vonMises, gradVelSav, cellOSpr, gradvel, ndom, vdom, infoplus);

	//-Reinicia limites de dt.
	//-Reinitialises limits of dt.
	PartDtMin = DBL_MAX; PartDtMax = -DBL_MAX;

	//-Calculo de tiempo.
	//-Computation of time.
	if (Part > PartIni || Nstep) {
		TimerPart.Stop();
		double tpart = TimerPart.GetElapsedTimeD() / 1000;
		double tseg = tpart / (TimeStep - TimeStepM1);
		TimerSim.Stop();
		double tcalc = TimerSim.GetElapsedTimeD() / 1000;
		double tleft = (tcalc / (TimeStep - TimeStepIni)) * (TimeMax - TimeStep);
		Log->Printf("Part%s  %12.6f  %12d  %7d  %9.2f  %14s", suffixpartx.c_str(), TimeStep, (Nstep + 1), Nstep - PartNstep, tseg, fun::GetDateTimeAfter(int(tleft)).c_str());
	}
	else Log->Printf("Part%s        %u particles successfully stored", suffixpartx.c_str(), npok);


	//-Muestra info de particulas excluidas
	//-Shows info of the excluded particles
	if (nout) {
		PartOut += nout;
		Log->Printf("  Particles out: %u  (total: %u)", nout, PartOut);
	}

	if (SvDomainVtk)SaveDomainVtk(ndom, vdom);
	if (SaveDt)SaveDt->SaveData();
	if (GaugeSystem)GaugeSystem->SaveResults(Part);
}

//==============================================================================
/// Generates VTK file with domain of the particles.
/// Genera fichero VTK con el dominio de las particulas.
//==============================================================================
void JSph::SaveDomainVtk(unsigned ndom,const tdouble3 *vdom)const{
  if(vdom){
    string fname=fun::FileNameSec("Domain.vtk",Part);
    tfloat3 *vdomf3=new tfloat3[ndom*2];
    for(unsigned c=0;c<ndom*2;c++)vdomf3[c]=ToTFloat3(vdom[c]);
    JFormatFiles2::SaveVtkBoxes(DirDataOut+fname,ndom,vdomf3,H*0.5f);
    delete[] vdomf3;
  }
}

//==============================================================================
/// Saves initial domain of simulation in a VTK file (CasePosMin/Max,
/// MapRealPosMin/Max and Map_PosMin/Max).
///
/// Graba dominio inicial de simulacion en fichero VTK (CasePosMin/Max,
/// MapRealPosMin/Max and Map_PosMin/Max).
//==============================================================================
void JSph::SaveInitialDomainVtk()const{
  const unsigned nbox=(MapRealPosMin!=Map_PosMin || MapRealPosMax!=Map_PosMax? 3: 2);
  tfloat3 *vdomf3=new tfloat3[nbox*2];
  vdomf3[0]=ToTFloat3(CasePosMin);
  vdomf3[1]=ToTFloat3(CasePosMax);
  vdomf3[2]=ToTFloat3(MapRealPosMin);
  vdomf3[3]=ToTFloat3(MapRealPosMax);
  if(nbox==3){
    vdomf3[4]=ToTFloat3(Map_PosMin);
    vdomf3[5]=ToTFloat3(Map_PosMax);
  }
  const string file=DirOut+"CfgInit_Domain.vtk";
  Log->AddFileInfo(file,"Saves the limits of the case and the simulation domain limits.");
  JFormatFiles2::SaveVtkBoxes(file,nbox,vdomf3,0);
  delete[] vdomf3;
}

//==============================================================================
/// Returns size of VTK file with map cells.
/// Devuelve tamaño de fichero VTK con las celdas del mapa.
//==============================================================================
unsigned JSph::SaveMapCellsVtkSize()const{
  const tuint3 cells=OrderDecode(Map_Cells);
  unsigned nlin=cells.x+cells.z+2;//-Back lines.
  if(!Simulate2D){
    nlin+=cells.x+cells.y+2;//-Bottom lines.
    nlin+=cells.y+cells.z+2;//-Left lines.
  }
  const unsigned slin=sizeof(tfloat3)*2+sizeof(int)*4; //-Size per line is 40 bytes.
  return(nlin*slin);
}

//==============================================================================
/// Generates VTK file with map cells.
/// Genera fichero VTK con las celdas del mapa.
//==============================================================================
void JSph::SaveMapCellsVtk(float scell)const{
  const tuint3 cells=OrderDecode(Map_Cells);
  tdouble3 pmin=OrderDecode(MapRealPosMin);
  tdouble3 pmax=pmin+TDouble3(scell*cells.x,scell*cells.y,scell*cells.z);
  if(Simulate2D)pmin.y=pmax.y=Simulate2DPosY;
  //-Creates lines.
  std::vector<JFormatFiles2::StShapeData> shapes;
  //-Back lines.
  tdouble3 p0=TDouble3(pmin.x,pmax.y,pmin.z),p1=TDouble3(pmin.x,pmax.y,pmax.z);
  for(unsigned cx=0;cx<=cells.x;cx++)shapes.push_back(JFormatFiles2::DefineShape_Line(p0+TDouble3(scell*cx,0,0),p1+TDouble3(scell*cx,0,0),0,0));
  p1=TDouble3(pmax.x,pmax.y,pmin.z);
  for(unsigned cz=0;cz<=cells.z;cz++)shapes.push_back(JFormatFiles2::DefineShape_Line(p0+TDouble3(0,0,scell*cz),p1+TDouble3(0,0,scell*cz),0,0));
  if(!Simulate2D){
    //-Bottom lines.
    p0=TDouble3(pmin.x,pmin.y,pmin.z),p1=TDouble3(pmax.x,pmin.y,pmin.z);
    for(unsigned cy=0;cy<=cells.y;cy++)shapes.push_back(JFormatFiles2::DefineShape_Line(p0+TDouble3(0,scell*cy,0),p1+TDouble3(0,scell*cy,0),1,0));
    p1=TDouble3(pmin.x,pmax.y,pmin.z);
    for(unsigned cx=0;cx<=cells.x;cx++)shapes.push_back(JFormatFiles2::DefineShape_Line(p0+TDouble3(scell*cx,0,0),p1+TDouble3(scell*cx,0,0),1,0));
    //-Left lines.
    p0=TDouble3(pmin.x,pmin.y,pmin.z),p1=TDouble3(pmin.x,pmax.y,pmin.z);
    for(unsigned cz=0;cz<=cells.z;cz++)shapes.push_back(JFormatFiles2::DefineShape_Line(p0+TDouble3(0,0,scell*cz),p1+TDouble3(0,0,scell*cz),2,0));
    p1=TDouble3(pmin.x,pmin.y,pmax.z);
    for(unsigned cy=0;cy<=cells.y;cy++)shapes.push_back(JFormatFiles2::DefineShape_Line(p0+TDouble3(0,scell*cy,0),p1+TDouble3(0,scell*cy,0),2,0));
  }
  const string file=DirOut+"CfgInit_MapCells.vtk";
  Log->AddFileInfo(file,"Saves the cell division of the simulation domain.");
  JFormatFiles2::SaveVtkShapes(file,"axis","",shapes);
}

//==============================================================================
/// Adds basic information of resume to hinfo & dinfo.
/// Añade la informacion basica de resumen a hinfo y dinfo.
//==============================================================================
void JSph::GetResInfo(float tsim,float ttot,const std::string &headplus,const std::string &detplus,std::string &hinfo,std::string &dinfo){
  hinfo=hinfo+"#RunName;RunCode;DateTime;Np;TSimul;TSeg;TTotal;MemCpu;MemGpu;Steps;PartFiles;PartsOut;MaxParticles;MaxCells;Hw;StepAlgo;Kernel;Viscosity;ViscoValue;DeltaSPH;TMax;Nbound;Nfixed;H;RhopOut;PartsRhopOut;PartsVelOut;CellMode"+headplus;
  dinfo=dinfo+ RunName+ ";"+ RunCode+ ";"+ RunTimeDate+ ";"+ fun::UintStr(CaseNp);
  dinfo=dinfo+ ";"+ fun::FloatStr(tsim)+ ";"+ fun::FloatStr(tsim/float(TimeStep))+ ";"+ fun::FloatStr(ttot);
  dinfo=dinfo+ ";"+ fun::LongStr(MaxMemoryCpu)+ ";"+ fun::LongStr(MaxMemoryGpu);
  const unsigned nout=GetOutPosCount()+GetOutRhopCount()+GetOutMoveCount();
  dinfo=dinfo+ ";"+ fun::IntStr(Nstep)+ ";"+ fun::IntStr(Part)+ ";"+ fun::UintStr(nout);
  dinfo=dinfo+ ";"+ fun::UintStr(MaxParticles)+ ";"+ fun::UintStr(MaxCells);
  dinfo=dinfo+ ";"+ Hardware+ ";"+ GetStepName(TStep)+ ";"+ GetKernelName(TKernel)+ ";"+ GetViscoName(TVisco)+ ";"+ fun::FloatStr(Visco);
  dinfo=dinfo+ ";"+ fun::FloatStr(DeltaSph,"%G")+ ";"+ fun::FloatStr(float(TimeMax));
  dinfo=dinfo+ ";"+ fun::UintStr(CaseNbound)+ ";"+ fun::UintStr(CaseNfixed)+ ";"+ fun::FloatStr(H);
  std::string rhopcad;
  if(RhopOut)rhopcad=fun::PrintStr("(%G-%G)",RhopOutMin,RhopOutMax); else rhopcad="None";
  dinfo=dinfo+ ";"+ rhopcad+ ";"+ fun::UintStr(GetOutRhopCount())+ ";"+ fun::UintStr(GetOutMoveCount())+ ";"+ GetNameCellMode(CellMode)+ detplus;
}

//==============================================================================
/// Generates file Run.csv with resume of execution.
/// Genera fichero Run.csv con resumen de ejecucion.
//==============================================================================
void JSph::SaveRes(float tsim,float ttot,const std::string &headplus,const std::string &detplus){
  const char* met="SaveRes";
  const string fname=DirOut+"Run.csv";
  Log->AddFileInfo(fname,"One line CSV file with execution parameters and other simulation data.");
  ofstream pf;
  pf.open(fname.c_str());
  if(pf){
    string hinfo,dinfo;
    GetResInfo(tsim,ttot,headplus,detplus,hinfo,dinfo);
    pf << fun::StrCsvSep(CsvSepComa,hinfo) << endl << fun::StrCsvSep(CsvSepComa,dinfo) << endl;
    if(pf.fail())RunException(met,"Failed writing to file.",fname);
    pf.close();
  }
  else RunException(met,"File could not be opened.",fname);
}

//==============================================================================
/// Shows resume of execution.
/// Muestra resumen de ejecucion.
//==============================================================================
void JSph::ShowResume(bool stop,float tsim,float ttot,bool all,std::string infoplus){
  Log->Printf("\n[Simulation %s  %s]",(stop? "INTERRUPTED": "finished"),fun::GetDateTime().c_str());
  Log->Printf("Particles of simulation (initial): %u",CaseNp);
  if(NpDynamic)Log->Printf("Particles of simulation (total)..: %llu",TotalNp);
  if(all){
    Log->Printf("DTs adjusted to DtMin............: %d",DtModif);
    const unsigned nout=GetOutPosCount()+GetOutRhopCount()+GetOutMoveCount();
    Log->Printf("Excluded particles...............: %d",nout);
    if(GetOutRhopCount())Log->Printf("Excluded particles due to RhopOut: %u",GetOutRhopCount());
    if(GetOutMoveCount())Log->Printf("Excluded particles due to Velocity: %u",GetOutMoveCount());
  }
  Log->Printf("Total Runtime....................: %f sec.",ttot);
  Log->Printf("Simulation Runtime...............: %f sec.",tsim);
  if(all){
    float tseg=tsim/float(TimeStep);
    float nstepseg=float(Nstep)/tsim;
    Log->Printf("Time per second of simulation....: %f sec.",tseg);
    Log->Printf("Steps per second.................: %f",nstepseg);
    Log->Printf("Steps of simulation..............: %d",Nstep);
    Log->Printf("PART files.......................: %d",Part-PartIni);
    while(!infoplus.empty()){
      string lin=fun::StrSplit("#",infoplus);
      if(!lin.empty()){
        string tex=fun::StrSplit("=",lin);
        string val=fun::StrSplit("=",lin);
        while(tex.size()<33)tex=tex+".";
        Log->Print(tex+": "+val);
      }
    }
  }
  Log->Printf("Maximum number of particles......: %u",MaxParticles);
  Log->Printf("Maximum number of cells..........: %u",MaxCells);
  Log->Printf("CPU Memory.......................: %lld (%.2f MB)",MaxMemoryCpu,double(MaxMemoryCpu)/(1024*1024));
  if(MaxMemoryGpu)Log->Printf("GPU Memory.......................: %lld (%.2f MB)",MaxMemoryGpu,double(MaxMemoryGpu)/(1024*1024));
}

//========================================  ======================================
/// Returns text about PosDouble configuration.
/// Devuelve texto sobre la configuracion de PosDouble.
//==============================================================================
std::string JSph::GetPosDoubleName(bool psingle,bool svdouble){
  string tx;
  if(psingle && !svdouble)tx="0: Uses and stores in single precision";
  else if(!psingle && !svdouble)tx="1: Uses double and stores in single precision";
  else if(!psingle && svdouble)tx="2: Uses and stores in double precision";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns the name of the time algorithm in text format.
/// Devuelve el nombre del algoritmo en texto.
//==============================================================================
std::string JSph::GetStepName(TpStep tstep){
  string tx;
  if(tstep==STEP_Verlet)tx="Verlet";
  else if (tstep == STEP_Symplectic)tx = "Symplectic";
  else if (tstep == STEP_Euler)tx = "Euler";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns the name of the kernel in text format.
/// Devuelve el nombre del kernel en texto.
//==============================================================================
std::string JSph::GetKernelName(TpKernel tkernel){
  string tx;
  if(tkernel==KERNEL_Cubic)tx="Cubic";
  else if(tkernel==KERNEL_Wendland)tx="Wendland";
  else if(tkernel==KERNEL_Gaussian)tx="Gaussian";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns value of viscosity in text format.
/// Devuelve el nombre de la viscosidad en texto.
//==============================================================================
std::string JSph::GetViscoName(TpVisco tvisco){
  string tx;
  if(tvisco==VISCO_Artificial)tx="Artificial";
  else if(tvisco==VISCO_LaminarSPS)tx="Laminar+SPS";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns value of DeltaSPH in text format.
/// Devuelve el valor de DeltaSPH en texto.
//==============================================================================
std::string JSph::GetDeltaSphName(TpDeltaSph tdelta){
  string tx;
  if(tdelta==DELTA_None)tx="None";
  else if(tdelta==DELTA_Dynamic)tx="Dynamic";
  else if(tdelta==DELTA_DynamicExt)tx="DynamicExt";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns value of Shifting in text format.
/// Devuelve el valor de Shifting en texto.
//==============================================================================
std::string JSph::GetShiftingName(TpShifting tshift){
  string tx;
  if(tshift==SHIFT_None)tx="None";
  else if(tshift==SHIFT_NoBound)tx="NoBound";
  else if(tshift==SHIFT_NoFixed)tx="NoFixed";
  else if(tshift==SHIFT_Full)tx="Full";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns string with the name of timer and value.
/// Devuelve string con el nombre del temporizador y su valor.
//==============================================================================
std::string JSph::TimerToText(const std::string &name,float value){
  string ret=name;
  while(ret.length()<33)ret+=".";
  return(ret+": "+fun::FloatStr(value/1000)+" sec.");
}

//==============================================================================
/// Saves VTK file with particle data (degug).
/// Graba fichero VTK con datos de las particulas (degug).
//==============================================================================
void JSph::DgSaveVtkParticlesCpu(std::string filename,int numfile,unsigned pini,unsigned pfin
  ,const tdouble3 *pos,const typecode *code,const unsigned *idp,const tfloat4 *velrhop
  ,const tfloat3 *ace)const
{
  int mpirank=Log->GetMpiRank();
  if(mpirank>=0)filename=string("p")+fun::IntStr(mpirank)+"_"+filename;
  if(numfile>=0)filename=fun::FileNameSec(filename,numfile);
  filename=DirDataOut+filename;
  //-Allocates memory.
  const unsigned np=pfin-pini;
  tfloat3 *xpos=new tfloat3[np];
  tfloat3 *xvel=new tfloat3[np];
  tfloat3 *xace=(ace? new tfloat3[np]: NULL);
  float *xrhop=new float[np];
  byte *xtype=new byte[np];
  byte *xkind=new byte[np];
  for(unsigned p=0;p<np;p++){
    xpos[p]=ToTFloat3(pos[p+pini]);
    tfloat4 vr=velrhop[p+pini];
    xvel[p]=TFloat3(vr.x,vr.y,vr.z);
    if(xace)xace[p]=ace[p+pini];
    xrhop[p]=vr.w;
    typecode t=CODE_GetType(code[p+pini]);
    xtype[p]=(t==CODE_TYPE_FIXED? 0: (t==CODE_TYPE_MOVING? 1: (t==CODE_TYPE_FLOATING? 2: 3)));
    typecode k=CODE_GetSpecialValue(code[p+pini]);
    xkind[p]=(k==CODE_NORMAL? 0: (k==CODE_PERIODIC? 1: (k==CODE_OUTIGNORE? 2: 3)));
  }
  //-Generates VTK file.
  JFormatFiles2::StScalarData fields[10];
  unsigned nfields=0;
  if(idp){   fields[nfields]=JFormatFiles2::DefineField("Idp" ,JFormatFiles2::UInt32 ,1,idp+pini); nfields++; }
  if(xtype){ fields[nfields]=JFormatFiles2::DefineField("Type",JFormatFiles2::UChar8 ,1,xtype);    nfields++; }
  if(xkind){ fields[nfields]=JFormatFiles2::DefineField("Kind",JFormatFiles2::UChar8 ,1,xkind);    nfields++; }
  if(xvel){  fields[nfields]=JFormatFiles2::DefineField("Vel" ,JFormatFiles2::Float32,3,xvel);     nfields++; }
  if(xrhop){ fields[nfields]=JFormatFiles2::DefineField("Rhop",JFormatFiles2::Float32,1,xrhop);    nfields++; }
  if(xace){  fields[nfields]=JFormatFiles2::DefineField("Ace" ,JFormatFiles2::Float32,3,xace);     nfields++; }
  //string fname=DirOut+fun::FileNameSec("DgParts.vtk",numfile);
  JFormatFiles2::SaveVtk(filename,np,xpos,nfields,fields);
  //-Deallocates memory.
  delete[] xpos;
  delete[] xtype;
  delete[] xkind;
  delete[] xvel;
  delete[] xrhop;
  delete[] xace;
}

//==============================================================================
/// Saves VTK file with particle data (degug).
/// Graba fichero VTK con datos de las particulas (degug).
//==============================================================================
void JSph::DgSaveVtkParticlesCpu(std::string filename,int numfile,unsigned pini,unsigned pfin,const tfloat3 *pos,const byte *check,const unsigned *idp,const tfloat3 *vel,const float *rhop){
  int mpirank=Log->GetMpiRank();
  if(mpirank>=0)filename=string("p")+fun::IntStr(mpirank)+"_"+filename;
  if(numfile>=0)filename=fun::FileNameSec(filename,numfile);
  filename=DirDataOut+filename;
  //-Reserva memoria basica.
  const unsigned n=pfin-pini;
  unsigned *num=new unsigned[n];
  for(unsigned p=0;p<n;p++)num[p]=p;
  //-Generates VTK file.
  JFormatFiles2::StScalarData fields[10];
  unsigned nfields=0;
  if(idp){   fields[nfields]=JFormatFiles2::DefineField("Idp"  ,JFormatFiles2::UInt32 ,1,idp+pini);   nfields++; }
  if(vel){   fields[nfields]=JFormatFiles2::DefineField("Vel"  ,JFormatFiles2::Float32,3,vel+pini);   nfields++; }
  if(rhop){  fields[nfields]=JFormatFiles2::DefineField("Rhop" ,JFormatFiles2::Float32,1,rhop+pini);  nfields++; }
  if(check){ fields[nfields]=JFormatFiles2::DefineField("Check",JFormatFiles2::UChar8 ,1,check+pini); nfields++; }
  if(num){   fields[nfields]=JFormatFiles2::DefineField("Num"  ,JFormatFiles2::UInt32 ,1,num);        nfields++; }
  //-Generates VTK file.
  JFormatFiles2::SaveVtk(filename,n,pos+pini,nfields,fields);
  //-Deallocates memory.
  delete[] num;
}

//==============================================================================
/// Saves CSV file with particle data (degug).
/// Graba fichero CSV con datos de las particulas (degug).
//==============================================================================
void JSph::DgSaveCsvParticlesCpu(std::string filename,int numfile,unsigned pini,unsigned pfin,std::string head,const tfloat3 *pos,const unsigned *idp,const tfloat3 *vel,const float *rhop,const float *ar,const tfloat3 *ace,const tfloat3 *vcorr){
  const char met[]="DgSaveCsvParticlesCpu";
  int mpirank=Log->GetMpiRank();
  if(mpirank>=0)filename=string("p")+fun::IntStr(mpirank)+"_"+filename;
  if(numfile>=0)filename=fun::FileNameSec(filename,numfile);
  filename=DirOut+filename;
  //-Generates CSV file.
  ofstream pf;
  pf.open(filename.c_str());
  if(pf){
    if(!head.empty())pf << head << endl;
    pf << "Num";
    if(idp)pf << ";Idp";
    if(pos)pf << ";PosX;PosY;PosZ";
    if(vel)pf << ";VelX;VelY;VelZ";
    if(rhop)pf << ";Rhop";
    if(ar)pf << ";Ar";
    if(ace)pf << ";AceX;AceY;AceZ";
    if(vcorr)pf << ";VcorrX;VcorrY;VcorrZ";
    pf << endl;
    const char fmt1[]="%f"; //="%24.16f";
    const char fmt3[]="%f;%f;%f"; //="%24.16f;%24.16f;%24.16f";
    for(unsigned p=pini;p<pfin;p++){
      pf << fun::UintStr(p-pini);
      if(idp)pf << ";" << fun::UintStr(idp[p]);
      if(pos)pf << ";" << fun::Float3Str(pos[p],fmt3);
      if(vel)pf << ";" << fun::Float3Str(vel[p],fmt3);
      if(rhop)pf << ";" << fun::FloatStr(rhop[p],fmt1);
      if(ar)pf << ";" << fun::FloatStr(ar[p],fmt1);
      if(ace)pf << ";" << fun::Float3Str(ace[p],fmt3);
      if(vcorr)pf << ";" << fun::Float3Str(vcorr[p],fmt3);
      pf << endl;
    }
    if(pf.fail())RunException(met,"Failed writing to file.",filename);
    pf.close();
  }
  else RunException(met,"File could not be opened.",filename);
}
