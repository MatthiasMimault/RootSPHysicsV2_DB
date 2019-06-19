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

/// \file JCellDivGpu_ker.h \brief Declares functions and CUDA kernels to compute operations of the Neighbour List.

#ifndef _JCellDivGpu_ker_
#define _JCellDivGpu_ker_

#include "TypesDef.h"
#include "Types.h"
#include <cuda_runtime_api.h>

//:#define DG_LimitsCell //-En LimitsCell() comprueba que el resultado sea correcto.
//:#define DG_LimitsPos //-En LimitsPos() comprueba que el resultado sea correcto.
//:#define DG_GetRangeParticlesCells //-En GetParticlesCellRange() comprueba que el resultado sea correcto.

class JLog2;

#define DIVBSIZE 256

/// Implements a set of functions and CUDA kernels to compute operations of the Neighbour List.
namespace cudiv{

/// Axes (X,Y,Z) order for the reordering of particles in cells. | Orden de ejes en ordenacion de particulas en celdas.
typedef enum{ ORDER_XYZ=1,ORDER_YZX=2,ORDER_XZY=3 }TpCellOrder;  

/*:
//template <byte torder> __device__ float GetSortX(const float3 &v){ return(torder==2? v.y: v.x); }
//template <byte torder> __device__ float GetSortY(const float3 &v){ return(torder==1? v.y: v.z); }
//template <byte torder> __device__ float GetSortZ(const float3 &v){ return(torder==1? v.z: (torder==2? v.x: v.y)); }
//template <byte torder> __host__ __device__ float3 GetSort(const float3 &v){ return(torder==2? make_float3(v.y,v.z,v.x): (torder==3? make_float3(v.x,v.z,v.y): v)); }
//template <byte torder> __host__ __device__ float3 GetSort(const float3 &v){ 
//  if(torder==2){ float3 r; r.x=v.y; r.y=v.z; r.z=v.x; return(r); }
//  else if(torder==3){ float3 r; r.x=v.x; r.y=v.z; r.z=v.y; return(r); }
//  else return(v);
//}
:*/


inline float3 Float3(const tfloat3& v){ float3 p={v.x,v.y,v.z}; return(p); }
inline float3 Float3(float x,float y,float z){ float3 p={x,y,z}; return(p); }
inline tfloat3 ToTFloat3(const float3& v){ return(TFloat3(v.x,v.y,v.z)); }

void Sort(unsigned* keys,unsigned* values,unsigned size,bool stable);

dim3 GetGridSize(unsigned n,unsigned blocksize);
void ReduPosLimits(unsigned nblocks,float *aux,tfloat3 &pmin,tfloat3 &pmax,JLog2 *log);


inline unsigned LimitsPosSize(unsigned ndata){ ndata=(ndata>DIVBSIZE? ndata: DIVBSIZE); unsigned n=6,s=((ndata/DIVBSIZE)+1); return((s*n + ((s/DIVBSIZE)+1)*n) + DIVBSIZE); }


void LimitsCell(unsigned np,unsigned pini,unsigned cellcode,const unsigned *dcell,const typecode *code,unsigned *aux,tuint3 &celmin,tuint3 &celmax,JLog2 *log);
void CalcBeginEndCell(bool full,unsigned np,unsigned npb,unsigned sizebegcell,unsigned 
	,const unsigned *cellpart,int2 *begcell);

void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const unsigned *idp,const typecode *code,const unsigned *dcell,const double2 *posxy,const double *posz,const float4 *velrhop,unsigned *idp2,typecode *code2,unsigned *dcell2,double2 *posxy2,double *posz2,float4 *velrhop2);
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const float4 *a,float4 *a2);
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const float *a,const float *b,float *a2,float *b2);
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const double2 *a,const double *b,const float4 *c,double2 *a2,double *b2,float4 *c2);
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const tsymatrix3f *a,tsymatrix3f *a2);

//:inline unsigned GetRangeParticlesCellsSizeAux(unsigned celini,unsigned celfin){ unsigned n=2,s=(((celfin-celini)/DIVBSIZE)+1); return((s*n + ((s/DIVBSIZE)+1)*n) + DIVBSIZE); } 
//:void GetRangeParticlesCells(unsigned celini,unsigned celfin,const int2 *begcell,unsigned *aux,unsigned &pmin,unsigned &pmax,JLog2 *log);

//:inline unsigned GetParticlesCellsSizeAux(unsigned celini,unsigned celfin){ unsigned n=1,s=(((celfin-celini)/DIVBSIZE)+1); return((s*n + ((s/DIVBSIZE)+1)*n) + DIVBSIZE); }  
//:unsigned GetParticlesCells(unsigned celini,unsigned celfin,const int2 *begcell,unsigned *aux,JLog2 *log);

}

#endif


