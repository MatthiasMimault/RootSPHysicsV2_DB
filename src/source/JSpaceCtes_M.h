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
  void ReadXmlRun_M(JXml *sxml,TiXmlElement* ele);
};

#endif


