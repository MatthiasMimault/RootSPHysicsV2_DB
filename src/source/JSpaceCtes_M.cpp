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

/// \file JSpaceCtes.cpp \brief Implements the class \ref JSpaceCtes.

#include "JSpaceCtes.h"
#include "JSpaceCtes_M.h"
#include "JXml.h"
#include <cmath>
#include <algorithm>

//##############################################################################
//# JSpaceCtes
//##############################################################################
//==============================================================================
/// Reads constants for execution of the case of xml node.
/// Version Matthias: Solid
//==============================================================================
void JSpaceCtes::ReadXmlRun_M(JXml *sxml,TiXmlElement* node){
  SetGravity(sxml->ReadElementDouble3(node,"gravity"));
  SetCFLnumber(sxml->ReadElementDouble(node,"cflnumber","value"));
  SetGamma(sxml->ReadElementDouble(node,"gamma","value"));
  SetRhop0(sxml->ReadElementDouble(node,"rhop0","value"));
  SetEps(sxml->ReadElementDouble(node,"eps","value",true,0));
  SetDp(sxml->ReadElementDouble(node,"dp","value"));
  SetH(sxml->ReadElementDouble(node,"h","value"));
  SetB(sxml->ReadElementDouble(node,"b","value"));
  SetMassBound(sxml->ReadElementDouble(node,"massbound","value"));
  SetMassFluid(sxml->ReadElementDouble(node,"massfluid","value"));
}
