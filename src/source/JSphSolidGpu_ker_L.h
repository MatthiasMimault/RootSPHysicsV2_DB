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

/// \file JSphSolidGpu_ker_l.h \brief Declares the class \ref JSphSolidGpu_ker_L

#ifndef _JSphSolidGpu_ker_
#define _JSphSolidGpu_ker_

#include "Types.h"
#include "JSphTimersGpu.h"
#include "JSph.h"
#include "JSphGpu_ker.h"
#include "JBlockSizeAuto.h"
#include <string>
#endif

namespace cuSol {
	void CteInteractionUp(const StCteInteraction *cte);
	void ComputeStepVerlet_L(bool floating, bool shift, unsigned np, unsigned npb
		, const float4 *velrhop1, const float4 *velrhop2
		, const float *ar, const float3 *ace, const float3 *shiftpos
		, double dt, double dt2, float rhopoutmin, float rhopoutmax
		, typecode *code, double2 *movxy, double *movz, float4 *velrhopnew
		, const tsymatrix3f *tau2, tsymatrix3f *JauTauDot_M, tsymatrix3f *taunew
		, const float *mass1, const float *mass2, float *massnew, float LambdaMass, float RhopZero);

	void ComputeStepVerlet_M(bool floating, bool shift, unsigned np, unsigned npb
		, const float4 *velrhop1, const float4 *velrhop2
		, const float *ar, const float3 *ace, const float3 *shiftpos
		, double dt, double dt2, float rhopoutmin, float rhopoutmax
		, typecode *code, double2 *movxy, double *movz, float4 *velrhopnew
		, const tsymatrix3f *tau2, tsymatrix3f *JauTauDot_M, tsymatrix3f *taunew
		, const float *mass1, const float *mass2, float *massnew);


	void Interaction_Forces_M(TpKernel TKernel, bool WithFloating, TpShifting TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM, TpCellMode cellmode
		, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
		, unsigned np, unsigned npb, unsigned npbok, tuint3 ncells
		, int2 *begincell, tuint3 cellmin, const unsigned *dcell
		, const double2 *posxy, const double *posz, const float4 *pospress
		, const float4 *velrhop, const unsigned *idp, const typecode *code
		, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f *omega
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, float* viscdt, float* ar, float3 *ace, float *delta
		, float3 *shiftpos, float *shiftdetect
		, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
		, const float *press, const float *pore, const float *mass, tsymatrix3f AnisotropyG, float Mu, const float *ftomassp);

	//==============================================================================
	void Interaction_Forces_M(TpKernel TKernel, bool WithFloating, TpShifting TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM
		, TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
		, unsigned np, unsigned npb, unsigned npbok
		, const tuint3 ncells, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
		, const double2 *posxy, double *posz, float4 *pospress
		, float4 *velrhop, unsigned *idp, typecode *code
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, float* viscdt, float* ar, float3 *ace, float *delta
		, float3 *shiftpos, float *shiftdetect
		, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
		, float *pore, float *mass, float *ftomassp);

	void Interaction_Forces_M(TpKernel TKernel, bool WithFloating, TpShifting TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM
		, TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
		, unsigned np, unsigned npb, unsigned npbok
		, const tuint3 ncells, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
		, const double2 *posxy, double *posz, float4 *pospress
		, float4 *velrhop, unsigned *idp, typecode *code
		, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f *omega
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, float* viscdt, float* ar, float3 *ace, float *delta
		, float3 *shiftpos, float *shiftdetect
		, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
		, float *press, float *pore, float *mass, tsymatrix3f AnisotropyG, float Mu, float *ftomassp);

	void Interaction_Forces_M(TpKernel TKernel, bool WithFloating, TpShifting TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM
		, TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
		, unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, int2 *begincell, tuint3 cellmin, const unsigned *dcell
		, const double2 *posxy, const double *posz, const float4 *pospress
		, const float4 *velrhop, const unsigned *idp, const typecode *code
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, float* viscdt, float* ar, float3 *ace, float *delta
		, float3 *shiftpos, float *shiftdetect
		, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
		, const float *pore, const float *mass, const float *ftomassp);

	void Interaction_Forces_M(TpKernel TKernel, bool WithFloating, TpShifting TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM
		, TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
		, unsigned np, unsigned npb, unsigned npbok
		, const tuint3 ncells, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
		, const double2 *posxy, double *posz, float4 *pospress
		, float4 *velrhop, unsigned *idp, typecode *code
		, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f *omega
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, float* viscdt, float* ar, float3 *ace, float *delta
		, float3 *shiftpos, float *shiftdetect
		, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
		, float3 *press, float *pore, float *mass, tsymatrix3f AnisotropyG, float Mu, float *ftomassp);

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesSolMass
	(unsigned n, unsigned pinit, int hdiv, int4 nc, unsigned cellfluid, float viscob, float viscof
		, const int2 *begincell, int3 cellzero, const unsigned *dcell, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const float *ftomassp, const float2 *tauff, float2 *gradvelff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, float *viscdt, float *ar, float3 *ace, float *delta
		, TpShifting tshifting, float3 *shiftpos, float *shiftdetect);

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesSolMass
	(unsigned n, unsigned pinit, int hdiv, int4 nc, unsigned cellfluid, float viscob, float viscof
		, const int2 *begincell, int3 cellzero, const unsigned *dcell, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const float *ftomassp, const float2 *tauff, float2 *gradvelff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, const float3 *press, const float *pore, const float *mass
		, float *viscdt, float *ar, float3 *ace, float *delta
		, TpShifting tshifting, float3 *shiftpos, float *shiftdetect);

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesSolid_M
	(unsigned n, unsigned pinit, int hdiv, int4 nc, unsigned cellfluid, float viscob, float viscof
		, const int2 *begincell, int3 cellzero, const unsigned *dcell
		, const float *ftomassp
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float *viscdt, float *ar, float3 *ace, float *delta
		, TpShifting tshifting, float3 *shiftpos, float *shiftdetect
		, const float *mass, const float *pore, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega);

	/*template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesSolidBoxPPM_M
	(bool boundp2, unsigned p1, const unsigned &pini, const unsigned &pfin, float visco
		, const float *ftomassp
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float massp2, float ftmassp1, bool ftp1
		, double3 posdp1, float3 posp1, float3 velp1, float3 pressp1, float rhopp1
		, float3 &acep1, float &arp1, float &visc, float &deltap1
		, TpShifting tshifting, float3 &shiftposp1, float &shiftdetectp1
		, float porep1, const float *pore, tsymatrix3f taup1, const tsymatrix3f* tau, tsymatrix3f &gradvelp1, tsymatrix3f &omegap1);*/


	__global__ void KerPressPoreC_L(
		unsigned n, const float4 *velrhop, const float RhopZero, float  *Pressg
		, tfloat3 Anisotropy, float CteB, float Gamma, float3 *Press3Dc
		, double2 *posxy, double *posz, tdouble3 LocDiv_M, float PoreZero, float Spread_M, float *Porec_M);

	__global__ void KerPressPore_M(
		unsigned n, const float4 *velrhop, const float RhopZero, float  *Pressg
		, float CteB, float Gamma
		, double2 *posxy, double *posz, float PoreZero, float *Poreg_M);

	void PressPoreC_L(unsigned np, const float4 *velrhop, const float RhopZero, float  *Pressg
		, tfloat3 Anisotropy, float CteB, float Gamma, float3 *Press3Dc
		, double2 *posxy, double *posz, tdouble3 LocDiv_M, float PoreZero, float Spread_M, float *Porec_M);

	void PressPore_M(unsigned np, const float4 *velrhop, const float RhopZero, float  *Pressg
		, float CteB, float Gamma
		, double2 *posxy, double *posz, float PoreZero, float *Porec_M);

	__global__ void KerComputeVelrhopBound(unsigned n, const float4* velrhopold, double armul, float4* velrhopnew, const float* Arg, float RhopZero);
	void ComputeVelrhopBound(unsigned n, const float4* velrhopold, double armul, float4* velrhopnew, const float* Arg, float RhopZero);
}