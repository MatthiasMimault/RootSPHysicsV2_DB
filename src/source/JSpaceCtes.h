//HEAD_DSCODES
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

//:#############################################################################
//:# Cambios:
//:# =========
//:# - Se paso a usar double en lugar de float. (25-11-2013)
//:# - El valor de Eps pasa a ser opcional para mantener compatibilidad. (08-01-2015)
//:# - Se cambio Coefficient por CoefH pero manteniendo compatibilidad. (08-01-2015)
//:# - Se añadio SpeedSound para asignar valor de forma explicita. (08-01-2015)
//:# - Se añadieron comentarios al escribir el XML. (08-01-2015)
//:# - Se ampliaron los limites de CFLnumber de (0.1-0.5) a (0.001-1). (08-01-2015)
//:# - <speedsystem> y <speedsound> pasan a ser opcionales. (20-01-2015)
//:# - <eps> solo se pasa a <constants> cuando este definido en <constantsdef>. (20-01-2015)
//:# - Se muestran unidades de las constantes. (15-12-2015)
//:# - Nueva funcion estatica para calcular constantes. (19-01-2016)
//:# - Ahora se puede definir <coefh> o <hdp> pero no ambos a la vez. (29-01-2016)
//:# - Se cambio las unidades de la constante B a kg/(m*s^2). (08-06-2016)
//:# - Se cambio las unidades de la constante B a Pascal (Pa). (07-06-2017)
//:#############################################################################

/// \file JSpaceCtes.h \brief Declares the class \ref JSpaceCtes.

#ifndef _JSpaceCtes_
#define _JSpaceCtes_

#include <string>
#include <vector>
#include "JObject.h"
#include "TypesDef.h"

class JXml;
class TiXmlElement;

//##############################################################################
//# JSpaceCtes
//##############################################################################
/// \brief Manages the info of constants from the input XML file.

class JSpaceCtes : protected JObject 
{
public:

  /// Defines structure to calculate constants. 
  typedef struct StrConstants{
    bool data2d;
    tdouble3 gravity;
    double dp,coefh,coefhdp;
    double hswl,speedsystem,coefsound,speedsound;
    double gamma,rhop0;
    double cteh,cteb;
    double massbound;
    double massfluid;

    StrConstants(){ Clear(); }
    StrConstants(bool vdata2d,tdouble3 vgravity,double vdp,double vcoefh,double vcoefhdp,double vhswl
      ,double vspeedsystem,double vcoefsound,double vspeedsound,double vgamma,double vrhop0
      ,double vcteh,double vcteb,double vmassbound,double vmassfluid)
    {
      data2d=vdata2d; gravity=vgravity; dp=vdp; coefh=vcoefh; coefhdp=vcoefhdp; hswl=vhswl;
      speedsystem=vspeedsystem; coefsound=vcoefsound; speedsound=vspeedsound; gamma=vgamma; rhop0=vrhop0;
      cteh=vcteh; cteb=vcteb; massbound=vmassbound; massfluid=vmassfluid;
    }
    void Clear(){ 
      data2d=false; gravity=TDouble3(0);
      dp=hswl=speedsystem=coefsound=speedsound=coefh=coefhdp=gamma=rhop0=cteh=cteb=massbound=massfluid=0;
    }
  }StConstants;

private:
  int LatticeBound;       ///<Lattice to create boundary particles on its nodes.
  int LatticeFluid;       ///<Lattice to create fluid particles on its nodes.
  tdouble3 Gravity;       ///<Gravity acceleration.
  double CFLnumber;       ///<CFL number (0.001-1).
  bool HSwlAuto;          ///<Activates the automatic computation of H_Swl.
  double HSwl;            ///<Maximum height of the volume of fluid.
  bool SpeedSystemAuto;   ///<Activates the automatic computation of SpeedSystem.
  double SpeedSystem;     ///<Maximum system speed.
  double CoefSound;       ///<Coefficient to multiply speedsystem.
  bool SpeedSoundAuto;    ///<Activates the automatic computation of SpeedSound.
  double SpeedSound;      ///<Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem).

  double CoefH;           ///<Coefficient to calculate the smoothing length H (H=coefficient*sqrt(3*dp^2) in 3D).
  double CoefHdp;         ///<Relationship between h and dp. (it is optional).
  double Gamma;           ///<Polytropic constant. (1-7).
  double Rhop0;           ///<Density of reference.

  double Eps;             ///<Epsilon constant for XSPH variant.
  bool EpsDefined;        ///<Epsilon was defined in constantsdef.

  bool HAuto;             ///<Activates the automatic computation of H.
  bool BAuto;             ///<Activates the automatic computation of B.
  bool MassBoundAuto;     ///<Activates the automatic computation of MassBound.
  bool MassFluidAuto;     ///<Activates the automatic computation of MassFluid.
  double H;               ///<Smoothing length.
  double B;               ///<Constant that sets a limit for the maximum change in density.
  double MassBound;       ///<Mass of a boundary particle.
  double MassFluid;       ///<Mass of a fluid particle.

  // Matthias
  // Simulation choices
  int typeCase, typeCompression, typeGrowth, typeDivision, typeYoung;
  double Hmin, Hmax;

  // Bord mirroir
  double PlanMirror;
  // Extension domain
  double BordDomain;
  // Solid - anisotropic
  double Ef, Et, Gf, nuxy, nuyz;
  //double K;
  //double Mu;
  // Pore
  double PoreZero;
  // Mass
  double LambdaMass;
  // Cell division
  double SizeDivision_M, VelDivCoef_M;
  tdouble3 LocalDiv_M;
  float Spread_M;
  // Anisotropy
  tdouble3 AnisotropyK_M;
  tsymatrix3f AnisotropyG_M;

  //-Computed values:
  double Dp;              ///<Inter-particle distance.

  void ReadXmlElementAuto(JXml *sxml,TiXmlElement* node,bool optional,std::string name,double &value,bool &valueauto);
  void WriteXmlElementAuto(JXml *sxml,TiXmlElement* node,std::string name,double value,bool valueauto,std::string comment="",std::string unitscomment="")const;

  void WriteXmlElementComment(TiXmlElement* ele,std::string comment="",std::string unitscomment="")const;

  void ReadXmlDef(JXml *sxml,TiXmlElement* ele);
  void WriteXmlDef(JXml *sxml,TiXmlElement* ele)const;
  void ReadXmlRun(JXml *sxml, TiXmlElement* ele);
  void ReadXmlRun_T(JXml *sxml, TiXmlElement* node);
  void ReadAddXmlRun_M(JXml *sxml, TiXmlElement* ele);
  void WriteXmlRun(JXml *sxml,TiXmlElement* ele)const;


public:
  static StConstants CalcConstans(StConstants cte);
  JSpaceCtes();
  void Reset();
  void LoadDefault();
  void LoadXmlDef(JXml *sxml,const std::string &place);
  void SaveXmlDef(JXml *sxml,const std::string &place)const;
  void LoadXmlRun(JXml *sxml, const std::string &place);
  void LoadXmlRun_T(JXml *sxml, const std::string &place);
  void LoadAddXmlRun_M(JXml *sxml, const std::string &place);
  void SaveXmlRun(JXml *sxml,const std::string &place)const;

  int GetLatticeBound()const{ return(LatticeBound); }
  int GetLatticeFluid()const{ return(LatticeFluid); }
  tdouble3 GetGravity()const{ return(Gravity); }
  double GetCFLnumber()const { return(CFLnumber); }
  double GetPlanMirror()const { return(PlanMirror); }
  bool GetHSwlAuto()const{ return(HSwlAuto); }
  double GetHSwl()const{ return(HSwl); }
  bool GetSpeedSystemAuto()const{ return(SpeedSystemAuto); }
  double GetSpeedSystem()const{ return(SpeedSystem); }
  double GetCoefSound()const{ return(CoefSound); }
  bool GetSpeedSoundAuto()const{ return(SpeedSoundAuto); }
  double GetSpeedSound()const{ return(SpeedSound); }
  double GetCoefH()const{ return(CoefH); }
  double GetHmin()const{ return(Hmin); }
  double GetHmax()const{ return(Hmax); }
  double GetCoefHdp()const{ return(CoefHdp); }
  double GetCoefficient()const{ return(GetCoefH()); }
  double GetGamma()const{ return(Gamma); }
  double GetRhop0()const{ return(Rhop0); }
  double GetEps()const{ return(Eps); }
  
  // Simulation choices
  int GetCase()const { return typeCase; }
  int GetComp()const { return typeCompression; }
  int GetDiv()const { return typeDivision; }
  int GetGrow()const { return typeGrowth; }
  int GetYoung()const { return typeYoung; }

  // Extension Domain
  double GetBordDomain()const { return BordDomain; }

  // Solid - anisotropic
  //double GetYoung()const { return(K); }
  double GetYoungX()const { return(Ef); }
  double GetYoungY()const { return(Et); }
  double GetShear()const { return(Gf); }
  double GetPoissonXY()const { return nuxy; }
  double GetPoissonYZ()const { return nuyz; }
  //double GetShear()const { return(Mu); }

  // Pore Pressure
  double GetPoreZero()const { return(PoreZero); }
  // Mass assimilation
  double GetLambdaMass()const{ return(LambdaMass); }
  // Cell division
  double GetSizeDivision()const { return(SizeDivision_M); }
  tdouble3 GetLocalDivision()const { return LocalDiv_M; }
  double GetVelocityDivisionCoef()const { return VelDivCoef_M; }
  float GetSpreadDivision()const { return Spread_M; }
  // Anisotropy
  tdouble3 GetAnisotropyK()const { return AnisotropyK_M; }
  tsymatrix3f GetAnisotropyG()const { return AnisotropyG_M; }

  void SetLatticeBound(bool simple){ LatticeBound=(simple? 1: 2); }
  void SetLatticeFluid(bool simple){ LatticeFluid=(simple? 1: 2); }
  void SetGravity(const tdouble3& g){ Gravity=g; }
  void SetCFLnumber(double v){ 
    if(!v)RunException("SetCFLnumber","Value cannot be zero.");
    if(v>1)RunException("SetCFLnumber","Value cannot be greater than 1.");
    CFLnumber=v;
  }
  void SetHSwlAuto(bool on){ HSwlAuto=on; }
  void SetHSwl(double v){ HSwl=v; }
  void SetSpeedSystemAuto(bool on){ SpeedSystemAuto=on; }
  void SetSpeedSystem(double v){ SpeedSystem=v; }
  void SetCoefSound(double v){ CoefSound=v; }
  void SetSpeedSoundAuto(bool on){ SpeedSoundAuto=on; }
  void SetSpeedSound(double v){ SpeedSound=v; }
  void SetCoefH(double v){ CoefH=v; CoefHdp=0; }
  void SetCoefHdp(double v){ if(v){ CoefHdp=v; CoefH=0; } }
  void SetHmin(double v){ if(v){ Hmin=v; } }
  void SetHmax(double v){ if(v){ Hmax=v; } }
  void SetCoefficient(double v){ SetCoefH(v); }
  void SetGamma(double v){ Gamma=v; }
  void SetRhop0(double v){ Rhop0=v; }
  void SetEps(double v){ Eps=v; }
  // Matthias
  // Simulation choices
  void SetCase(int v) { typeCase = v; }
  void SetComp(int v) { typeCompression = v; }
  void SetDiv(int v) { typeDivision = v; }
  void SetGrow(int v) { typeGrowth = v; }
  void SetYoung(int v) { typeYoung = v; }

  // Plan mirroir
  void SetPlanMirror(double v) { PlanMirror = v; }
  // Extension Domain
  void SetBordDomain(double v) { BordDomain = v; }
  // Solid - anisotropic
  //void SetYoung(double v) { K = v; };
  //void SetShear(double v) { Mu = v; };
  void SetYoungX(double v) { Ef = v; };
  void SetYoungY(double v) { Et = v; };
  void SetShear(double v) { Gf = v; };
  void SetPoissonXY(double v) { nuxy = v; };
  void SetPoissonYZ(double v) { nuyz = v; };

  // Pore Pressure
  void SetPoreZero(double v) { PoreZero = v; };
  // Mass assimilation
  void SetLambdaMass(double v) { LambdaMass = v; };
  // Cell division
  void SetSizeDivision(double v) { SizeDivision_M = v; };
  void SetLocalDivision(tdouble3 v) { LocalDiv_M = v; }
  void SetVelocityDivisionCoef(double v) { VelDivCoef_M = v; }
  void SetSpreadDivision(float v){ Spread_M = v; }
  // Anisotropy
  void SetAnisotropyK(tdouble3 v) { AnisotropyK_M = v; }
  void SetAnisotropyG(tsymatrix3f v) { AnisotropyG_M = v; }

  bool GetHAuto()const{ return(HAuto); }
  bool GetBAuto()const{ return(BAuto); }
  bool GetMassBoundAuto()const{ return(MassBoundAuto); }
  bool GetMassFluidAuto()const{ return(MassFluidAuto); }
  double GetH()const{ return(H); }
  double GetB()const{ return(B); }
  double GetMassBound()const{ return(MassBound); }
  double GetMassFluid()const{ return(MassFluid); }

  void SetHAuto(bool on){ HAuto=on; }
  void SetBAuto(bool on){ BAuto=on; }
  void SetMassBoundAuto(bool on){ MassBoundAuto=on; }
  void SetMassFluidAuto(bool on){ MassFluidAuto=on; }
  void SetH(double v){ H=v; }
  void SetB(double v){ B=v; }
  void SetMassBound(double v){ MassBound=v; }
  void SetMassFluid(double v){ MassFluid=v; }

  double GetDp()const{ return(Dp); }
  void SetDp(double v){ Dp=v; }
};

#endif


