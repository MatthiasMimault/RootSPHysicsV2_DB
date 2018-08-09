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

/// \file JSphSolidGpu_ker_L.cu \brief Implements functions and CUDA kernels to compute operations for solids particles.
#include <stdio.h>
#include "JSphSolidGpu_ker_L.h"
#include "JSph.h"
#include "JCellDivGpu_ker.h"
#include "Types.h"
#include <float.h>
#include <cmath>
#include "JLog2.h"
#include "JSphGpu_ker.h"
#include "JBlockSizeAuto.h"
#include "JGauge_ker.h"
#include <math_constants.h>
//:#include "JDgKerPrint.h"
//:#include "JDgKerPrint_ker.h"
#include <cstdio>
#include <string>

#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
__constant__ StCteInteraction CTE;

namespace cuSol {

	//==============================================================================
	/// Stores constants for the GPU interaction.
	/// Graba constantes para la interaccion a la GPU.
	//==============================================================================
	void CteInteractionUp(const StCteInteraction *cte) {
		cudaMemcpyToSymbol(CTE, cte, sizeof(StCteInteraction));
	}


#include "FunctionsMath_ker.cu"

	//==============================================================================
	/// Checks error and ends execution.
	/// Comprueba error y finaliza ejecucion.
	//==============================================================================
#define CheckErrorCuda(text)  __CheckErrorCuda(text,__FILE__,__LINE__)
	void __CheckErrorCuda(const char *text, const char *file, const int line) {
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			char cad[2048];
			sprintf(cad, "%s (CUDA error: %s -> %s:%i).\n", text, cudaGetErrorString(err), file, line);
			throw std::string(cad);
		}
	}

	//==============================================================================
	/// Returns size of gridsize according to parameters.
	/// Devuelve tamaño de gridsize segun parametros.
	//==============================================================================
	dim3 GetGridSize(unsigned n, unsigned blocksize) {
		dim3 sgrid;//=dim3(1,2,3);
		unsigned nb = unsigned(n + blocksize - 1) / blocksize; //-Total number of blocks to execute.
		sgrid.x = (nb <= 65535 ? nb : unsigned(sqrt(float(nb))));
		sgrid.y = (nb <= 65535 ? 1 : unsigned((nb + sgrid.x - 1) / sgrid.x));
		sgrid.z = 1;
		return(sgrid);
	}

	//==============================================================================
	/// Reduction using maximum of float values in shared memory for a warp.
	/// Reduccion mediante maximo de valores float en memoria shared para un warp.
	//==============================================================================
	template <unsigned blockSize> __device__ void KerReduMaxFloatWarp(volatile float* sdat, unsigned tid) {
		if (blockSize >= 64)sdat[tid] = max(sdat[tid], sdat[tid + 32]);
		if (blockSize >= 32)sdat[tid] = max(sdat[tid], sdat[tid + 16]);
		if (blockSize >= 16)sdat[tid] = max(sdat[tid], sdat[tid + 8]);
		if (blockSize >= 8)sdat[tid] = max(sdat[tid], sdat[tid + 4]);
		if (blockSize >= 4)sdat[tid] = max(sdat[tid], sdat[tid + 2]);
		if (blockSize >= 2)sdat[tid] = max(sdat[tid], sdat[tid + 1]);
	}

	//==============================================================================
	/// Accumulates the maximum of n values of array dat[], storing the result in 
	/// the beginning of res[].(Many positions of res[] are used as blocks, 
	/// storing the final result in res[0]).
	///
	/// Acumula el maximo de n valores del vector dat[], guardando el resultado al 
	/// principio de res[] (Se usan tantas posiciones del res[] como bloques, 
	/// quedando el resultado final en res[0]).
	//==============================================================================
	template <unsigned blockSize> __global__ void KerReduMaxFloat(unsigned n, unsigned ini, const float *dat, float *res) {
		extern __shared__ float sdat[];
		unsigned tid = threadIdx.x;
		unsigned c = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
		sdat[tid] = (c<n ? dat[c + ini] : -FLT_MAX);
		__syncthreads();
		if (blockSize >= 512) { if (tid<256)sdat[tid] = max(sdat[tid], sdat[tid + 256]);  __syncthreads(); }
		if (blockSize >= 256) { if (tid<128)sdat[tid] = max(sdat[tid], sdat[tid + 128]);  __syncthreads(); }
		if (blockSize >= 128) { if (tid<64) sdat[tid] = max(sdat[tid], sdat[tid + 64]);   __syncthreads(); }
		if (tid<32)KerReduMaxFloatWarp<blockSize>(sdat, tid);
		if (tid == 0)res[blockIdx.y*gridDim.x + blockIdx.x] = sdat[0];
	}

	//==============================================================================
	/// Returns the maximum of an array, using resu[] as auxiliar array.
	/// Size of resu[] must be >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
	///
	/// Devuelve el maximo de un vector, usando resu[] como vector auxiliar. El tamaño
	/// de resu[] debe ser >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
	//==============================================================================
	float ReduMaxFloat(unsigned ndata, unsigned inidata, float* data, float* resu) {
		float resf;
		if (1) {

			unsigned n = ndata, ini = inidata;
			unsigned smemSize = SPHBSIZE * sizeof(float);
			dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
			unsigned n_blocks = sgrid.x*sgrid.y;
			float *dat = data;
			float *resu1 = resu, *resu2 = resu + n_blocks;
			float *res = resu1;
			while (n > 1) {
				KerReduMaxFloat<SPHBSIZE> << <sgrid, SPHBSIZE, smemSize >> > (n, ini, dat, res);
				n = n_blocks; ini = 0;
				sgrid = cuSol::GetGridSize(n, SPHBSIZE);
				n_blocks = sgrid.x*sgrid.y;
				if (n > 1) {
					dat = res; res = (dat == resu1 ? resu2 : resu1);
				}
			}

			if (ndata > 1) {
				cudaMemcpy(&resf, res, sizeof(float), cudaMemcpyDeviceToHost);
			}
			else {
				cudaMemcpy(&resf, data, sizeof(float), cudaMemcpyDeviceToHost);

			}

		}

		//else{//-Using Thrust library is slower than ReduMasFloat() with ndata < 5M.
		//  thrust::device_ptr<float> dev_ptr(data);
		//  resf=thrust::reduce(dev_ptr,dev_ptr+ndata,-FLT_MAX,thrust::maximum<float>());
		//}

		return(resf);
	}

	//==============================================================================
	/// Accumulates the sum of n values of array dat[], storing the result in 
	/// the beginning of res[].(Many positions of res[] are used as blocks, 
	/// storing the final result in res[0]).
	///
	/// Acumula la suma de n valores del vector dat[].w, guardando el resultado al 
	/// principio de res[] (Se usan tantas posiciones del res[] como bloques, 
	/// quedando el resultado final en res[0]).
	//==============================================================================
	template <unsigned blockSize> __global__ void KerReduMaxFloat_w(unsigned n, unsigned ini, const float4 *dat, float *res) {
		extern __shared__ float sdat[];
		unsigned tid = threadIdx.x;
		unsigned c = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
		sdat[tid] = (c<n ? dat[c + ini].w : -FLT_MAX);
		__syncthreads();
		if (blockSize >= 512) { if (tid<256)sdat[tid] = max(sdat[tid], sdat[tid + 256]);  __syncthreads(); }
		if (blockSize >= 256) { if (tid<128)sdat[tid] = max(sdat[tid], sdat[tid + 128]);  __syncthreads(); }
		if (blockSize >= 128) { if (tid<64) sdat[tid] = max(sdat[tid], sdat[tid + 64]);   __syncthreads(); }
		if (tid<32)KerReduMaxFloatWarp<blockSize>(sdat, tid);
		if (tid == 0)res[blockIdx.y*gridDim.x + blockIdx.x] = sdat[0];
	}

	//==============================================================================
	/// Returns the maximum of an array, using resu[] as auxiliar array.
	/// Size of resu[] must be >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE).
	///
	/// Devuelve el maximo de la componente w de un vector float4, usando resu[] como 
	/// vector auxiliar. El tamaño de resu[] debe ser >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE).
	//==============================================================================
	float ReduMaxFloat_w(unsigned ndata, unsigned inidata, float4* data, float* resu) {
		unsigned n = ndata, ini = inidata;
		unsigned smemSize = SPHBSIZE * sizeof(float);
		dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
		unsigned n_blocks = sgrid.x*sgrid.y;
		float *dat = NULL;
		float *resu1 = resu, *resu2 = resu + n_blocks;
		float *res = resu1;
		while (n>1) {
			if (!dat)KerReduMaxFloat_w<SPHBSIZE> << <sgrid, SPHBSIZE, smemSize >> >(n, ini, data, res);
			else KerReduMaxFloat<SPHBSIZE> << <sgrid, SPHBSIZE, smemSize >> >(n, ini, dat, res);
			n = n_blocks; ini = 0;
			sgrid = cuSol::GetGridSize(n, SPHBSIZE);
			n_blocks = sgrid.x*sgrid.y;
			if (n>1) {
				dat = res; res = (dat == resu1 ? resu2 : resu1);
			}
		}
		float resf;
		if (ndata>1)cudaMemcpy(&resf, res, sizeof(float), cudaMemcpyDeviceToHost);
		else {
			float4 resf4;
			cudaMemcpy(&resf4, data, sizeof(float4), cudaMemcpyDeviceToHost);
			resf = resf4.w;
		}
		return(resf);
	}


	//------------------------------------------------------------------------------
	/// Initialises array with the indicated value.
	/// Inicializa array con el valor indicado.
	//------------------------------------------------------------------------------
	__global__ void KerInitArray(unsigned n, float3 *v, float3 value)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p<n)v[p] = value;
	}

	//==============================================================================
	/// Initialises array with the indicated value.
	/// Inicializa array con el valor indicado.
	//==============================================================================
	void InitArray(unsigned n, float3 *v, tfloat3 value) {
		if (n) {
			dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
			KerInitArray << <sgrid, SPHBSIZE >> > (n, v, cudiv::Float3(value));
		}
	}

	//------------------------------------------------------------------------------
	/// Sets v[].y to zero.
	/// Pone v[].y a cero.
	//------------------------------------------------------------------------------
	__global__ void KerResety(unsigned n, unsigned ini, float3 *v)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p<n)v[p + ini].y = 0;
	}

	//==============================================================================
	/// Sets v[].y to zero.
	/// Pone v[].y a cero.
	//==============================================================================
	void Resety(unsigned n, unsigned ini, float3 *v) {
		if (n) {
			dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
			KerResety << <sgrid, SPHBSIZE >> > (n, ini, v);
		}
	}

	//------------------------------------------------------------------------------
	/// Calculates module^2 of ace.
	//------------------------------------------------------------------------------
	__global__ void KerComputeAceMod(unsigned n, const float3 *ace, float *acemod)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p<n) {
			const float3 r = ace[p];
			acemod[p] = r.x*r.x + r.y*r.y + r.z*r.z;
		}
	}

	//==============================================================================
	/// Calculates module^2 of ace.
	//==============================================================================
	void ComputeAceMod(unsigned n, const float3 *ace, float *acemod) {
		if (n) {
			dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
			KerComputeAceMod << <sgrid, SPHBSIZE >> > (n, ace, acemod);
		}
	}

	//------------------------------------------------------------------------------
	/// Calculates module^2 of ace, comprobando que la particula sea normal.
	//------------------------------------------------------------------------------
	__global__ void KerComputeAceMod(unsigned n, const typecode *code, const float3 *ace, float *acemod)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p<n) {
			const float3 r = (CODE_IsNormal(code[p]) ? ace[p] : make_float3(0, 0, 0));
			acemod[p] = r.x*r.x + r.y*r.y + r.z*r.z;
		}
	}

	//==============================================================================
	/// Calculates module^2 of ace, comprobando que la particula sea normal.
	//==============================================================================
	void ComputeAceMod(unsigned n, const typecode *code, const float3 *ace, float *acemod) {
		if (n) {
			dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
			KerComputeAceMod << <sgrid, SPHBSIZE >> > (n, code, ace, acemod);
		}
	}


	//##############################################################################
	//# Other kernels...
	//# Otros kernels...
	//##############################################################################
	//------------------------------------------------------------------------------
	/// Calculates module^2 of vel.
	//------------------------------------------------------------------------------
	__global__ void KerComputeVelMod(unsigned n, const float4 *vel, float *velmod)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p<n) {
			const float4 r = vel[p];
			velmod[p] = r.x*r.x + r.y*r.y + r.z*r.z;
		}
	}

	//==============================================================================
	/// Calculates module^2 of vel.
	//==============================================================================
	void ComputeVelMod(unsigned n, const float4 *vel, float *velmod) {
		if (n) {
			dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
			KerComputeVelMod << <sgrid, SPHBSIZE >> > (n, vel, velmod);
		}
	}


	//##############################################################################
	//# Kernels for preparing force computation with Pos-Single.
	//# Kernels para preparar calculo de fuerzas con Pos-Single.
	//##############################################################################
	//------------------------------------------------------------------------------
	/// Prepare variables for Pos-Single interaction.
	/// Prepara variables para interaccion Pos-Single.
	//------------------------------------------------------------------------------
	__global__ void KerPreInteractionSingle(unsigned n, const double2 *posxy, const double *posz
		, const float4 *velrhop, float4 *pospress, float cteb, float gamma)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Particle number.
		if (p<n) {
			//Computes press in single or double precision,although the latter does not have any significant positive effect,
			//and like PosDouble if it is previously calculated and read the interaction can incur losses of
			//performance of 6% or 15% (GTX480 or k20c) so it is best calculated as always simple.
			//
			//Calcular press en simple o doble precision no parece tener ningun efecto positivo significativo,
			//y como para PosDouble si se calcula antes y se lee en la interaccion supondria una perdida de 
			//rendimiento del 6% o 15% (gtx480 o k20c) mejor se calcula en simple siempre.
			const float rrhop = velrhop[p].w;
			float press = cteb * (powf(rrhop*CTE.ovrhopzero, gamma) - 1.0f);
			double2 rpos = posxy[p];
			pospress[p] = make_float4(float(rpos.x), float(rpos.y), float(posz[p]), press);
		}
	}

	//==============================================================================
	/// Prepare variables for Pos-Single interaction.
	/// Prepara variables para interaccion Pos-Single.
	//==============================================================================
	void PreInteractionSingle(unsigned np, const double2 *posxy, const double *posz
		, const float4 *velrhop, float4 *pospress, float cteb, float ctegamma)
	{
		if (np) {
			dim3 sgrid = cuSol::GetGridSize(np, SPHBSIZE);
			KerPreInteractionSingle << <sgrid, SPHBSIZE >> > (np, posxy, posz, velrhop, pospress, cteb, ctegamma);
		}
	}


	//##############################################################################
	//# Auxiliary kernels for the interaction.
	//# Kernels auxiliares para interaccion.
	//##############################################################################
	//------------------------------------------------------------------------------
	/// Returns position, vel, rhop and press of a particle.
	/// Devuelve posicion, vel, rhop y press de particula.
	//------------------------------------------------------------------------------
	template<bool psingle> __device__ void KerGetParticleData(unsigned p1
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop
		, float3 &velp1, float &rhopp1, double3 &posdp1, float3 &posp1, float &pressp1)
	{
		float4 r = velrhop[p1];
		velp1 = make_float3(r.x, r.y, r.z);
		rhopp1 = r.w;
		if (psingle) {
			float4 pxy = pospress[p1];
			posp1 = make_float3(pxy.x, pxy.y, pxy.z);
			pressp1 = pxy.w;
		}
		else {
			double2 pxy = posxy[p1];
			posdp1 = make_double3(pxy.x, pxy.y, posz[p1]);
			pressp1 = (CTE.cteb*(powf(rhopp1*CTE.ovrhopzero, CTE.gamma) - 1.0f));
		}
	}

	//------------------------------------------------------------------------------
	/// Returns postion and vel of a particle.
	/// Devuelve posicion y vel de particula.
	//------------------------------------------------------------------------------
	template<bool psingle> __device__ void KerGetParticleData(unsigned p1
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop
		, float3 &velp1, double3 &posdp1, float3 &posp1)
	{
		float4 r = velrhop[p1];
		velp1 = make_float3(r.x, r.y, r.z);
		if (psingle) {
			float4 pxy = pospress[p1];
			posp1 = make_float3(pxy.x, pxy.y, pxy.z);
		}
		else {
			double2 pxy = posxy[p1];
			posdp1 = make_double3(pxy.x, pxy.y, posz[p1]);
		}
	}

	//------------------------------------------------------------------------------
	/// Returns particle postion.
	/// Devuelve posicion de particula.
	//------------------------------------------------------------------------------
	template<bool psingle> __device__ void KerGetParticleData(unsigned p1
		, const double2 *posxy, const double *posz, const float4 *pospress
		, double3 &posdp1, float3 &posp1)
	{
		if (psingle) {
			float4 pxy = pospress[p1];
			posp1 = make_float3(pxy.x, pxy.y, pxy.z);
		}
		else {
			double2 pxy = posxy[p1];
			posdp1 = make_double3(pxy.x, pxy.y, posz[p1]);
		}
	}

	//------------------------------------------------------------------------------
	/// Returns position, vel, rhop,press of a particle. + tau & Pore - Lucas
	//------------------------------------------------------------------------------
	template<bool psingle> __device__ void KerGetParticleData(unsigned p1
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop
		, float3 &velp1, float &rhopp1, double3 &posdp1, float3 &posp1, float &pressp1, const tsymatrix3f* tau, const float *pore,tsymatrix3f &taup1,float &porep1 )
	{
		float4 r = velrhop[p1];
		velp1 = make_float3(r.x, r.y, r.z);
		rhopp1 = r.w;
		const tsymatrix3f taup = tau[p1];
		taup1 = taup;
		const float por = pore[p1];
		porep1 = por;
		if (psingle) {
			float4 pxy = pospress[p1];
			posp1 = make_float3(pxy.x, pxy.y, pxy.z);
			pressp1 = pxy.w;
		}
		else {
			double2 pxy = posxy[p1];
			posdp1 = make_double3(pxy.x, pxy.y, posz[p1]);
			pressp1 = (CTE.cteb*(powf(rhopp1*CTE.ovrhopzero, CTE.gamma) - 1.0f));
		}
	}

	//------------------------------------------------------------------------------
	/// Returns drx, dry and drz between the particles.
	/// Devuelve drx, dry y drz entre dos particulas.
	//------------------------------------------------------------------------------
	template<bool psingle> __device__ void KerGetParticlesDr(int p2
		, const double2 *posxy, const double *posz, const float4 *pospress
		, const double3 &posdp1, const float3 &posp1
		, float &drx, float &dry, float &drz, float &pressp2)
	{
		if (psingle) {
			float4 posp2 = pospress[p2];
			drx = posp1.x - posp2.x;
			dry = posp1.y - posp2.y;
			drz = posp1.z - posp2.z;
			pressp2 = posp2.w;
		}
		else {
			double2 posp2 = posxy[p2];
			drx = float(posdp1.x - posp2.x);
			dry = float(posdp1.y - posp2.y);
			drz = float(posdp1.z - posz[p2]);
			pressp2 = 0;
		}
	}

	//------------------------------------------------------------------------------
	/// Returns drx, dry and drz between the particles.
	/// Devuelve drx, dry y drz entre dos particulas.
	//------------------------------------------------------------------------------
	template<bool psingle> __device__ void KerGetParticlesDr(int p2
		, const double2 *posxy, const double *posz, const float4 *pospress
		, const double3 &posdp1, const float3 &posp1
		, float &drx, float &dry, float &drz)
	{
		if (psingle) {
			float4 posp2 = pospress[p2];
			drx = posp1.x - posp2.x;
			dry = posp1.y - posp2.y;
			drz = posp1.z - posp2.z;
		}
		else {
			double2 posp2 = posxy[p2];
			drx = float(posdp1.x - posp2.x);
			dry = float(posdp1.y - posp2.y);
			drz = float(posdp1.z - posz[p2]);
		}
	}

	//------------------------------------------------------------------------------
	/// Returns cell limits for the interaction.
	/// Devuelve limites de celdas para interaccion.
	//------------------------------------------------------------------------------
	__device__ void KerGetInteractionCells(unsigned rcell
		, int hdiv, const int4 &nc, const int3 &cellzero
		, int &cxini, int &cxfin, int &yini, int &yfin, int &zini, int &zfin)
	{
		//-Obtains interaction limits.
		const int cx = PC__Cellx(CTE.cellcode, rcell) - cellzero.x;
		const int cy = PC__Celly(CTE.cellcode, rcell) - cellzero.y;
		const int cz = PC__Cellz(CTE.cellcode, rcell) - cellzero.z;
		//-Code for hdiv 1 or 2 but not zero.
		//-Codigo para hdiv 1 o 2 pero no cero.
		cxini = cx - min(cx, hdiv);
		cxfin = cx + min(nc.x - cx - 1, hdiv) + 1;
		yini = cy - min(cy, hdiv);
		yfin = cy + min(nc.y - cy - 1, hdiv) + 1;
		zini = cz - min(cz, hdiv);
		zfin = cz + min(nc.z - cz - 1, hdiv) + 1;
	}

	//------------------------------------------------------------------------------
	/// Returns cell limits for the interaction.
	/// Devuelve limites de celdas para interaccion.
	//------------------------------------------------------------------------------
	__device__ void KerGetInteractionCells(double px, double py, double pz
		, int hdiv, const int4 &nc, const int3 &cellzero
		, int &cxini, int &cxfin, int &yini, int &yfin, int &zini, int &zfin)
	{
		//-Obtains interaction limits.
		const int cx = int((px - CTE.domposminx) / CTE.scell) - cellzero.x;
		const int cy = int((py - CTE.domposminy) / CTE.scell) - cellzero.y;
		const int cz = int((pz - CTE.domposminz) / CTE.scell) - cellzero.z;
		//-Code for hdiv 1 or 2 but not zero.
		//-Codigo para hdiv 1 o 2 pero no cero.
		cxini = cx - min(cx, hdiv);
		cxfin = cx + min(nc.x - cx - 1, hdiv) + 1;
		yini = cy - min(cy, hdiv);
		yfin = cy + min(nc.y - cy - 1, hdiv) + 1;
		zini = cz - min(cz, hdiv);
		zfin = cz + min(nc.z - cz - 1, hdiv) + 1;
	}

	//------------------------------------------------------------------------------
	/// Returns Wendland kernel values: frx, fry and frz.
	/// Devuelve valores del kernel Wendland: frx, fry y frz.
	//------------------------------------------------------------------------------
	__device__ void KerGetKernelWendland(float rr2, float drx, float dry, float drz
		, float &frx, float &fry, float &frz)
	{
		const float rad = sqrt(rr2);
		const float qq = rad / CTE.h;
		//-Wendland kernel.
		const float wqq1 = 1.f - 0.5f*qq;
		const float fac = CTE.bwen*qq*wqq1*wqq1*wqq1 / rad;
		frx = fac * drx; fry = fac * dry; frz = fac * drz;
	}

	//------------------------------------------------------------------------------
	/// Returns Gaussian kernel values: frx, fry and frz.
	/// Devuelve valores del kernel Gaussian: frx, fry y frz.
	//------------------------------------------------------------------------------
	__device__ void KerGetKernelGaussian(float rr2, float drx, float dry, float drz
		, float &frx, float &fry, float &frz)
	{
		const float rad = sqrt(rr2);
		const float qq = rad / CTE.h;
		//-Gaussian kernel.
		const float qqexp = -4.0f*qq*qq;
		//const float wab=CTE.agau*expf(qqexp);
		const float fac = CTE.bgau*qq*expf(qqexp) / rad;
		frx = fac * drx; fry = fac * dry; frz = fac * drz;
	}

	//------------------------------------------------------------------------------
	/// Return values of kernel Cubic without tensil correction, gradients: frx, fry and frz.
	/// Devuelve valores de kernel Cubic sin correccion tensil, gradients: frx, fry y frz.
	//------------------------------------------------------------------------------
	__device__ void KerGetKernelCubic(float rr2, float drx, float dry, float drz
		, float &frx, float &fry, float &frz)
	{
		const float rad = sqrt(rr2);
		const float qq = rad / CTE.h;
		//-Cubic Spline kernel.
		float fac;
		if (rad>CTE.h) {
			float wqq1 = 2.0f - qq;
			float wqq2 = wqq1 * wqq1;
			fac = CTE.cubic_c2*wqq2 / rad;
		}
		else {
			float wqq2 = qq * qq;
			fac = (CTE.cubic_c1*qq + CTE.cubic_d1*wqq2) / rad;
		}
		//-Gradients.
		frx = fac * drx; fry = fac * dry; frz = fac * drz;
	}

	//------------------------------------------------------------------------------
	/// Return tensil correction for kernel Cubic.
	/// Devuelve correccion tensil para kernel Cubic.
	//------------------------------------------------------------------------------
	__device__ float KerGetKernelCubicTensil(float rr2
		, float rhopp1, float pressp1, float rhopp2, float pressp2)
	{
		const float rad = sqrt(rr2);
		const float qq = rad / CTE.h;
		//-Cubic Spline kernel.
		float wab;
		if (rad>CTE.h) {
			float wqq1 = 2.0f - qq;
			float wqq2 = wqq1 * wqq1;
			wab = CTE.cubic_a24*(wqq2*wqq1);
		}
		else {
			float wqq2 = qq * qq;
			float wqq3 = wqq2 * qq;
			wab = CTE.cubic_a2*(1.0f - 1.5f*wqq2 + 0.75f*wqq3);
		}
		//-Tensile correction.
		float fab = wab * CTE.cubic_odwdeltap;
		fab *= fab; fab *= fab; //fab=fab^4
		const float tensilp1 = (pressp1 / (rhopp1*rhopp1))*(pressp1>0 ? 0.01f : -0.2f);
		const float tensilp2 = (pressp2 / (rhopp2*rhopp2))*(pressp2>0 ? 0.01f : -0.2f);
		return(fab*(tensilp1 + tensilp2));
	}
	
	//##############################################################################
	//# Kernels for calculating forces (Pos-Double).
	//# Kernels para calculo de fuerzas (Pos-Double).
	//##############################################################################
	//------------------------------------------------------------------------------
	/// Interaction of a particle with a set of particles. Bound-Fluid/Float
	/// Realiza la interaccion de una particula con un conjunto de ellas. Bound-Fluid/Float
	//------------------------------------------------------------------------------
	template<bool psingle, TpKernel tker, TpFtMode ftmode> __device__ void KerInteractionForcesBoundBox
	(unsigned p1, const unsigned &pini, const unsigned &pfin
		, const float *ftomassp
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned* idp
		, float massf, double3 posdp1, float3 posp1, float3 velp1, float &arp1, float &visc)
	{
		for (int p2 = pini; p2 < pfin; p2++) {
			float drx, dry, drz;
			KerGetParticlesDr<psingle>(p2, posxy, posz, pospress, posdp1, posp1, drx, dry, drz);
			float rr2 = drx * drx + dry * dry + drz * drz;
			if (rr2 <= CTE.fourh2 && rr2 >= ALMOSTZERO) {
				//-Cubic Spline, Wendland or Gaussian kernel.
				float frx, fry, frz;
				if (tker == KERNEL_Wendland)KerGetKernelWendland(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Gaussian)KerGetKernelGaussian(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Cubic)KerGetKernelCubic(rr2, drx, dry, drz, frx, fry, frz);

				const float4 velrhop2 = velrhop[p2];
				//-Obtains particle mass p2 if there are floating bodies.
				//-Obtiene masa de particula p2 en caso de existir floatings.
				float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massf si es fluid.
				bool compute = true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				if (USE_FLOATING) {
					const typecode cod = code[p2];
					bool ftp2 = CODE_IsFloating(cod);
					ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massf);
					compute = !(USE_DEM && ftp2); //-Deactivated when DEM is used and is bound-float. | Se desactiva cuando se usa DEM y es bound-float.
				}

				if (compute) {
					//-Density derivative.
					const float dvx = velp1.x - velrhop2.x, dvy = velp1.y - velrhop2.y, dvz = velp1.z - velrhop2.z;
					arp1 += (USE_FLOATING ? ftmassp2 : massf)*(dvx*frx + dvy * fry + dvz * frz);

					{//===== Viscosity ===== 
						const float dot = drx * dvx + dry * dvy + drz * dvz;
						const float dot_rr2 = dot / (rr2 + CTE.eta2);
						visc = max(dot_rr2, visc);
					}
				}
			}
		}
	}

	//------------------------------------------------------------------------------
	/// Particle interaction. Bound-Fluid/Float
	/// Realiza interaccion entre particulas. Bound-Fluid/Float
	//------------------------------------------------------------------------------
	template<bool psingle, TpKernel tker, TpFtMode ftmode> __global__ void KerInteractionForcesBound
	(unsigned n, int hdiv, int4 nc, const int2 *begincell, int3 cellzero, const unsigned *dcell
		, const float *ftomassp
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float *viscdt, float *ar)
	{
		unsigned p1 = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p1 < n) {
			float visc = 0, arp1 = 0;

			//-Loads particle p1 data.
			double3 posdp1;
			float3 posp1, velp1;
			KerGetParticleData<psingle>(p1, posxy, posz, pospress, velrhop, velp1, posdp1, posp1);

			//-Obtains interaction limits.
			int cxini, cxfin, yini, yfin, zini, zfin;
			KerGetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);

			//-Boundary-Fluid interaction.
			for (int z = zini; z < zfin; z++) {
				int zmod = (nc.w)*z + (nc.w*nc.z + 1);//-Adds Nct + 1 which is the first cell fluid. | Le suma Nct+1 que es la primera celda de fluido.
				for (int y = yini; y < yfin; y++) {
					int ymod = zmod + nc.x*y;
					unsigned pini, pfin = 0;
					for (int x = cxini; x < cxfin; x++) {
						int2 cbeg = begincell[x + ymod];
						if (cbeg.y) {
							if (!pfin)pini = cbeg.x;
							pfin = cbeg.y;
						}
					}
					if (pfin)KerInteractionForcesBoundBox<psingle, tker, ftmode>(p1, pini, pfin, ftomassp, posxy, posz, pospress, velrhop, code, idp, CTE.massf, posdp1, posp1, velp1, arp1, visc);
				}
			}
			//-Stores results.
			if (arp1 || visc) {
				ar[p1] += arp1;
				if (visc > viscdt[p1])viscdt[p1] = visc;
			}
		}
	}

	//------------------------------------------------------------------------------
	/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
	/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
	//------------------------------------------------------------------------------
	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __device__ void KerInteractionForcesFluidBox
	(bool boundp2, unsigned p1, const unsigned &pini, const unsigned &pfin, float visco
		, const float *ftomassp, const float2 *tauff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float massp2, float ftmassp1, bool ftp1
		, double3 posdp1, float3 posp1, float3 velp1, float pressp1, float rhopp1
		, const float2 &taup1_xx_xy, const float2 &taup1_xz_yy, const float2 &taup1_yz_zz
		, float2 &grap1_xx_xy, float2 &grap1_xz_yy, float2 &grap1_yz_zz
		, float3 &acep1, float &arp1, float &visc, float &deltap1
		, TpShifting tshifting, float3 &shiftposp1, float &shiftdetectp1)
	{
		for (int p2 = pini; p2 < pfin; p2++) {
			printf("Acep123");
			float drx, dry, drz, pressp2;
			KerGetParticlesDr<psingle>(p2, posxy, posz, pospress, posdp1, posp1, drx, dry, drz, pressp2);
			float rr2 = drx * drx + dry * dry + drz * drz;
			if (rr2 <= CTE.fourh2 && rr2 >= ALMOSTZERO) {
				//-Cubic Spline, Wendland or Gaussian kernel.
				float frx, fry, frz;
				if (tker == KERNEL_Wendland)KerGetKernelWendland(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Gaussian)KerGetKernelGaussian(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Cubic)KerGetKernelCubic(rr2, drx, dry, drz, frx, fry, frz);

				//-Obtains mass of particle p2 if any floating bodies exist.
				//-Obtiene masa de particula p2 en caso de existir floatings.
				bool ftp2;         //-Indicates if it is floating. | Indica si es floating.
				float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
				bool compute = true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				if (USE_FLOATING) {
					const typecode cod = code[p2];
					ftp2 = CODE_IsFloating(cod);
					ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
					if (ftp2 && ftmassp2 <= (massp2*1.2f) && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#else
					if (ftp2 && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#endif
					if (ftp2 && shift && tshifting == SHIFT_NoBound)shiftposp1.x = FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
					compute = !(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				}

				const float4 velrhop2 = velrhop[p2];

				//===== Aceleration ===== 
				if (compute) {
					if (!psingle)pressp2 = (CTE.cteb*(powf(velrhop2.w*CTE.ovrhopzero, CTE.gamma) - 1.0f));
					const float prs = (pressp1 + pressp2) / (rhopp1*velrhop2.w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop2.w, pressp2) : 0);
					const float p_vpm = -prs * (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2);
					acep1.x += p_vpm * frx; acep1.y += p_vpm * fry; acep1.z += p_vpm * frz;
				}

				//-Density derivative.
				const float dvx = velp1.x - velrhop2.x, dvy = velp1.y - velrhop2.y, dvz = velp1.z - velrhop2.z;
				if (compute)arp1 += (USE_FLOATING ? ftmassp2 : massp2)*(dvx*frx + dvy * fry + dvz * frz);

				const float cbar = CTE.cs0;
				//-Density derivative (DeltaSPH Molteni).
				if ((tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt)) {
					const float rhop1over2 = rhopp1 / velrhop2.w;
					const float visc_densi = CTE.delta2h*cbar*(rhop1over2 - 1.f) / (rr2 + CTE.eta2);
					const float dot3 = (drx*frx + dry * fry + drz * frz);
					const float delta = visc_densi * dot3*(USE_FLOATING ? ftmassp2 : massp2);
					if (USE_FLOATING)deltap1 = (boundp2 || deltap1 == FLT_MAX ? FLT_MAX : deltap1 + delta); //-Con floating bodies entre el fluido. //-For floating bodies within the fluid
					else deltap1 = (boundp2 ? FLT_MAX : deltap1 + delta);
				}

				//-Shifting correction.
				if (shift && shiftposp1.x != FLT_MAX) {
					const float massrhop = (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
					const bool noshift = (boundp2 && (tshifting == SHIFT_NoBound || (tshifting == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
					shiftposp1.x = (noshift ? FLT_MAX : shiftposp1.x + massrhop * frx); //-Removes shifting for the boundaries. | Con boundary anula shifting.
					shiftposp1.y += massrhop * fry;
					shiftposp1.z += massrhop * frz;
					shiftdetectp1 -= massrhop * (drx*frx + dry * fry + drz * frz);
				}

				//===== Viscosity ===== 
				if (compute) {
					const float dot = drx * dvx + dry * dvy + drz * dvz;
					const float dot_rr2 = dot / (rr2 + CTE.eta2);
					visc = max(dot_rr2, visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
					if (!lamsps) {//-Artificial viscosity.
						if (dot < 0) {
							const float amubar = CTE.h*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
							const float robar = (rhopp1 + velrhop2.w)*0.5f;
							const float pi_visc = (-visco * cbar*amubar / robar)*(USE_FLOATING ? ftmassp2 * ftmassp1 : massp2);
							acep1.x -= pi_visc * frx; acep1.y -= pi_visc * fry; acep1.z -= pi_visc * frz;
						}
					}
					else {//-Laminar+SPS viscosity.
						{//-Laminar contribution.
							const float robar2 = (rhopp1 + velrhop2.w);
							const float temp = 4.f*visco / ((rr2 + CTE.eta2)*robar2);  //-Simplication of temp=2.0f*visco/((rr2+CTE.eta2)*robar); robar=(rhopp1+velrhop2.w)*0.5f;
							const float vtemp = (USE_FLOATING ? ftmassp2 : massp2)*temp*(drx*frx + dry * fry + drz * frz);
							acep1.x += vtemp * dvx; acep1.y += vtemp * dvy; acep1.z += vtemp * dvz;
						}
						//-SPS turbulence model.
						float2 taup2_xx_xy = taup1_xx_xy; //-taup1 is always zero when p1 is not fluid. | taup1 siempre es cero cuando p1 no es fluid.
						float2 taup2_xz_yy = taup1_xz_yy;
						float2 taup2_yz_zz = taup1_yz_zz;
						if (!boundp2 && (USE_NOFLOATING || !ftp2)) {//-When p2 is fluid.
							float2 taup2 = tauff[p2 * 3];     taup2_xx_xy.x += taup2.x; taup2_xx_xy.y += taup2.y;
							taup2 = tauff[p2 * 3 + 1];   taup2_xz_yy.x += taup2.x; taup2_xz_yy.y += taup2.y;
							taup2 = tauff[p2 * 3 + 2];   taup2_yz_zz.x += taup2.x; taup2_yz_zz.y += taup2.y;
						}
						acep1.x += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xx_xy.x*frx + taup2_xx_xy.y*fry + taup2_xz_yy.x*frz);
						acep1.y += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xx_xy.y*frx + taup2_xz_yy.y*fry + taup2_yz_zz.x*frz);
						acep1.z += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xz_yy.x*frx + taup2_yz_zz.x*fry + taup2_yz_zz.y*frz);
						//-Velocity gradients.
						if (USE_NOFLOATING || !ftp1) {//-When p1 is fluid.
							const float volp2 = -(USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
							float dv = dvx * volp2; grap1_xx_xy.x += dv * frx; grap1_xx_xy.y += dv * fry; grap1_xz_yy.x += dv * frz;
							dv = dvy * volp2; grap1_xx_xy.y += dv * frx; grap1_xz_yy.y += dv * fry; grap1_yz_zz.x += dv * frz;
							dv = dvz * volp2; grap1_xz_yy.x += dv * frx; grap1_yz_zz.x += dv * fry; grap1_yz_zz.y += dv * frz;
							// to compute tau terms we assume that gradvel.xy=gradvel.dudy+gradvel.dvdx, gradvel.xz=gradvel.dudz+gradvel.dwdx, gradvel.yz=gradvel.dvdz+gradvel.dwdy
							// so only 6 elements are needed instead of 3x3.
						}
					}
				}
			}
		}
	}

	//------------------------------------------------------------------------------
	/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound) with Pore pressure
	//------------------------------------------------------------------------------
	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __device__ void KerInteractionForcesFluidBoxPP
	(bool boundp2, unsigned p1, const unsigned &pini, const unsigned &pfin, float visco
		, const float *ftomassp, const float2 *tauff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float massp2, float ftmassp1, bool ftp1
		, double3 posdp1, float3 posp1, float3 velp1, float pressp1, float rhopp1, float porep1
		, const float2 &taup1_xx_xy, const float2 &taup1_xz_yy, const float2 &taup1_yz_zz
		, float2 &grap1_xx_xy, float2 &grap1_xz_yy, float2 &grap1_yz_zz
		, float3 &acep1, float &arp1, float &visc, float &deltap1
		, TpShifting tshifting, float3 &shiftposp1, float &shiftdetectp1, tsymatrix3f taup1, const tsymatrix3f* tau, const float *press, const float *pore, tsymatrix3f gradvelp1, tsymatrix3f omegap1)
	{
		for (int p2 = pini; p2 < pfin; p2++) {
			printf("Acep12");
			float drx, dry, drz, pressp2;
			KerGetParticlesDr<psingle>(p2, posxy, posz, pospress, posdp1, posp1, drx, dry, drz, pressp2);
			float rr2 = drx * drx + dry * dry + drz * drz;
			if (rr2 <= CTE.fourh2 && rr2 >= ALMOSTZERO) {
				//-Cubic Spline, Wendland or Gaussian kernel.
				float frx, fry, frz;
				if (tker == KERNEL_Wendland)KerGetKernelWendland(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Gaussian)KerGetKernelGaussian(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Cubic)KerGetKernelCubic(rr2, drx, dry, drz, frx, fry, frz);

				//-Obtains mass of particle p2 if any floating bodies exist.
				//-Obtiene masa de particula p2 en caso de existir floatings.
				bool ftp2;         //-Indicates if it is floating. | Indica si es floating.
				float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
				bool compute = true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				if (USE_FLOATING) {
					const typecode cod = code[p2];
					ftp2 = CODE_IsFloating(cod);
					ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
					if (ftp2 && ftmassp2 <= (massp2*1.2f) && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#else
					if (ftp2 && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#endif
					if (ftp2 && shift && tshifting == SHIFT_NoBound)shiftposp1.x = FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
					compute = !(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				}

				const float4 velrhop2 = velrhop[p2];

				//===== Aceleration ===== 
				/*if (compute) {
				if (!psingle)pressp2 = (CTE.cteb*(powf(velrhop2.w*CTE.ovrhopzero, CTE.gamma) - 1.0f));
				const float prs = (pressp1 + pressp2) / (rhopp1*velrhop2.w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop2.w, pressp2) : 0);
				const float p_vpm = -prs * (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2);
				acep1.x += p_vpm * frx; acep1.y += p_vpm * fry; acep1.z += p_vpm * frz;
				}*/
				if (compute) {
					const tsymatrix3f prs = {
						(pressp1 + porep1 - taup1.xx + press[p2] + pore[p2] - tau[p2].xx) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop[p2].w, press[p2]) : 0),
						-(taup1.xy + tau[p2].xy) / (rhopp1*velrhop[p2].w),
						-(taup1.xz + tau[p2].xz) / (rhopp1*velrhop[p2].w),
						(pressp1 + porep1 - taup1.yy + press[p2] + pore[p2] - tau[p2].yy) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop[p2].w, press[p2]) : 0),
						-(taup1.yz + tau[p2].yz) / (rhopp1*velrhop[p2].w),
						(pressp1 + porep1 - taup1.zz + press[p2] + pore[p2] - tau[p2].zz) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop[p2].w, press[p2]) : 0)
					};
					const tsymatrix3f p_vpm3 = {
						-prs.xx*massp2*ftmassp1, -prs.xy*massp2*ftmassp1, -prs.xz*massp2*ftmassp1,
						-prs.yy*massp2*ftmassp1, -prs.yz*massp2*ftmassp1, -prs.zz*massp2*ftmassp1
					};

					acep1.x += p_vpm3.xx*frx + p_vpm3.xy*fry + p_vpm3.xz*frz;
					acep1.y += p_vpm3.xy*frx + p_vpm3.yy*fry + p_vpm3.yz*frz;
					acep1.z += p_vpm3.xz*frx + p_vpm3.yz*fry + p_vpm3.zz*frz;
				}

				//-Density derivative.
				const float dvx = velp1.x - velrhop2.x, dvy = velp1.y - velrhop2.y, dvz = velp1.z - velrhop2.z;
				if (compute)arp1 += (USE_FLOATING ? ftmassp2 : massp2)*(dvx*frx + dvy * fry + dvz * frz);

				const float cbar = CTE.cs0;
				//-Density derivative (DeltaSPH Molteni).
				if ((tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt)) {
					const float rhop1over2 = rhopp1 / velrhop2.w;
					const float visc_densi = CTE.delta2h*cbar*(rhop1over2 - 1.f) / (rr2 + CTE.eta2);
					const float dot3 = (drx*frx + dry * fry + drz * frz);
					const float delta = visc_densi * dot3*(USE_FLOATING ? ftmassp2 : massp2);
					if (USE_FLOATING)deltap1 = (boundp2 || deltap1 == FLT_MAX ? FLT_MAX : deltap1 + delta); //-Con floating bodies entre el fluido. //-For floating bodies within the fluid
					else deltap1 = (boundp2 ? FLT_MAX : deltap1 + delta);
				}

				//-Shifting correction.
				if (shift && shiftposp1.x != FLT_MAX) {
					const float massrhop = (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
					const bool noshift = (boundp2 && (tshifting == SHIFT_NoBound || (tshifting == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
					shiftposp1.x = (noshift ? FLT_MAX : shiftposp1.x + massrhop * frx); //-Removes shifting for the boundaries. | Con boundary anula shifting.
					shiftposp1.y += massrhop * fry;
					shiftposp1.z += massrhop * frz;
					shiftdetectp1 -= massrhop * (drx*frx + dry * fry + drz * frz);
				}

				//===== Viscosity ===== 
				if (compute) {
					const float dot = drx * dvx + dry * dvy + drz * dvz;
					const float dot_rr2 = dot / (rr2 + CTE.eta2);
					visc = max(dot_rr2, visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
					if (!lamsps) {//-Artificial viscosity.
						if (dot < 0) {
							const float amubar = CTE.h*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
							const float robar = (rhopp1 + velrhop2.w)*0.5f;
							const float pi_visc = (-visco * cbar*amubar / robar)*(USE_FLOATING ? ftmassp2 * ftmassp1 : massp2);
							acep1.x -= pi_visc * frx; acep1.y -= pi_visc * fry; acep1.z -= pi_visc * frz;
						}
					}
					/*else {//-Laminar+SPS viscosity.
					{//-Laminar contribution.
					const float robar2 = (rhopp1 + velrhop2.w);
					const float temp = 4.f*visco / ((rr2 + CTE.eta2)*robar2);  //-Simplication of temp=2.0f*visco/((rr2+CTE.eta2)*robar); robar=(rhopp1+velrhop2.w)*0.5f;
					const float vtemp = (USE_FLOATING ? ftmassp2 : massp2)*temp*(drx*frx + dry * fry + drz * frz);
					acep1.x += vtemp * dvx; acep1.y += vtemp * dvy; acep1.z += vtemp * dvz;
					}
					//-SPS turbulence model.
					float2 taup2_xx_xy = taup1_xx_xy; //-taup1 is always zero when p1 is not fluid. | taup1 siempre es cero cuando p1 no es fluid.
					float2 taup2_xz_yy = taup1_xz_yy;
					float2 taup2_yz_zz = taup1_yz_zz;
					if (!boundp2 && (USE_NOFLOATING || !ftp2)) {//-When p2 is fluid.
					float2 taup2 = tauff[p2 * 3];     taup2_xx_xy.x += taup2.x; taup2_xx_xy.y += taup2.y;
					taup2 = tauff[p2 * 3 + 1];   taup2_xz_yy.x += taup2.x; taup2_xz_yy.y += taup2.y;
					taup2 = tauff[p2 * 3 + 2];   taup2_yz_zz.x += taup2.x; taup2_yz_zz.y += taup2.y;
					}
					acep1.x += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xx_xy.x*frx + taup2_xx_xy.y*fry + taup2_xz_yy.x*frz);
					acep1.y += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xx_xy.y*frx + taup2_xz_yy.y*fry + taup2_yz_zz.x*frz);
					acep1.z += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xz_yy.x*frx + taup2_yz_zz.x*fry + taup2_yz_zz.y*frz);
					//-Velocity gradients.
					if (USE_NOFLOATING || !ftp1) {//-When p1 is fluid.
					const float volp2 = -(USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
					float dv = dvx * volp2; grap1_xx_xy.x += dv * frx; grap1_xx_xy.y += dv * fry; grap1_xz_yy.x += dv * frz;
					dv = dvy * volp2; grap1_xx_xy.y += dv * frx; grap1_xz_yy.y += dv * fry; grap1_yz_zz.x += dv * frz;
					dv = dvz * volp2; grap1_xz_yy.x += dv * frx; grap1_yz_zz.x += dv * fry; grap1_yz_zz.y += dv * frz;
					// to compute tau terms we assume that gradvel.xy=gradvel.dudy+gradvel.dvdx, gradvel.xz=gradvel.dudz+gradvel.dwdx, gradvel.yz=gradvel.dvdz+gradvel.dwdy
					// so only 6 elements are needed instead of 3x3.
					}
					}*/
				}
				//===== Velocity gradients ===== 
				if (compute) {
					if (!ftp1) {//-When p1 is a fluid particle / Cuando p1 es fluido. 
						const float volp2 = -massp2 / velrhop[p2].w;
						float dv = dvx * volp2;
						gradvelp1.xx += dv * frx; gradvelp1.xy += 0.5f*dv*fry; gradvelp1.xz += 0.5f*dv*frz;
						omegap1.xy += 0.5f*dv*fry; omegap1.xz += 0.5f*dv*frz;

						dv = dvy * volp2;
						gradvelp1.xy += 0.5f*dv*frx; gradvelp1.yy += dv * fry; gradvelp1.yz += 0.5f*dv*frz;
						omegap1.xy -= 0.5f*dv*frx; omegap1.yz += 0.5f*dv*frz;

						dv = dvz * volp2;
						gradvelp1.xz += 0.5f*dv*frx; gradvelp1.yz += 0.5f*dv*fry; gradvelp1.zz += dv * frz;
						omegap1.xz -= 0.5f*dv*frx; omegap1.yz -= 0.5f*dv*fry;
					}
				}
			}
		}
	}
	//------------------------------------------------------------------------------
	/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound) with Pore pressure 3D 
	//------------------------------------------------------------------------------
	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __device__ void KerInteractionForcesFluidBoxPPM
	(bool boundp2, unsigned p1, const unsigned &pini, const unsigned &pfin, float visco
		, const float *ftomassp, const float2 *tauff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float massp2, float ftmassp1, bool ftp1
		, double3 posdp1, float3 posp1, float3 velp1, float3 pressp1, float rhopp1, float porep1
		, float3 &acep1, float &arp1, float &visc, float &deltap1
		, TpShifting tshifting, float3 &shiftposp1, float &shiftdetectp1, tsymatrix3f taup1, const tsymatrix3f* tau, const float3 *press, const float *pore, const float *mass, tsymatrix3f &gradvelp1, tsymatrix3f &omegap1)
	{
		for (int p2 = pini; p2 < pfin; p2++) {
			float drx, dry, drz, pressp2;
			KerGetParticlesDr<psingle>(p2, posxy, posz, pospress, posdp1, posp1, drx, dry, drz, pressp2);
			float rr2 = drx * drx + dry * dry + drz * drz;
			if (rr2 <= CTE.fourh2 && rr2 >= ALMOSTZERO) {
				//Cubic Spline, Wendland or Gaussian kernel.
				float frx, fry, frz;
				if (tker == KERNEL_Wendland)KerGetKernelWendland(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Gaussian)KerGetKernelGaussian(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Cubic)KerGetKernelCubic(rr2, drx, dry, drz, frx, fry, frz);

				//-Obtains mass of particle p2 if any floating bodies exist.
				//-Obtiene masa de particula p2 en caso de existir floatings.
				bool ftp2;         //-Indicates if it is floating. | Indica si es floating.
				float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
				bool compute = true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				if (USE_FLOATING) {
					const typecode cod = code[p2];
					ftp2 = CODE_IsFloating(cod);
					ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
					if (ftp2 && ftmassp2 <= (massp2*1.2f) && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#else
					if (ftp2 && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#endif
					if (ftp2 && shift && tshifting == SHIFT_NoBound)shiftposp1.x = FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
					compute = !(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				}

				const float4 velrhop2 = velrhop[p2];

				//===== Aceleration ===== 
				/*if (compute) {
					if (!psingle)pressp2 = (CTE.cteb*(powf(velrhop2.w*CTE.ovrhopzero, CTE.gamma) - 1.0f));
					const float prs = (pressp1 + pressp2) / (rhopp1*velrhop2.w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop2.w, pressp2) : 0);
					const float p_vpm = -prs * (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2);
					acep1.x += p_vpm * frx; acep1.y += p_vpm * fry; acep1.z += p_vpm * frz;
				}*/
				if (compute) {
					const tsymatrix3f prs = {
						(pressp1.x + porep1 - taup1.xx + press[p2].x + pore[p2] - tau[p2].xx) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1.x, velrhop[p2].w, press[p2].x) : 0),
						-(taup1.xy + tau[p2].xy) / (rhopp1*velrhop[p2].w),
						-(taup1.xz + tau[p2].xz) / (rhopp1*velrhop[p2].w),
						(pressp1.y + porep1 - taup1.yy + press[p2].y + pore[p2] - tau[p2].yy) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1.y, velrhop[p2].w, press[p2].y) : 0),
						-(taup1.yz + tau[p2].yz) / (rhopp1*velrhop[p2].w),
						(pressp1.z + porep1 - taup1.zz + press[p2].z + pore[p2] - tau[p2].zz) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1.z, velrhop[p2].w, press[p2].z) : 0)
					};
					const tsymatrix3f p_vpm3 = {
					-prs.xx*massp2*ftmassp1, -prs.xy*massp2*ftmassp1, -prs.xz*massp2*ftmassp1,
						-prs.yy*massp2*ftmassp1, -prs.yz*massp2*ftmassp1, -prs.zz*massp2*ftmassp1
					};
					acep1.x += p_vpm3.xx*frx + p_vpm3.xy*fry + p_vpm3.xz*frz;
					acep1.y += p_vpm3.xy*frx + p_vpm3.yy*fry + p_vpm3.yz*frz;
					acep1.z += p_vpm3.xz*frx + p_vpm3.yz*fry + p_vpm3.zz*frz;
				}

				//-Density derivative.
				const float dvx = velp1.x - velrhop2.x, dvy = velp1.y - velrhop2.y, dvz = velp1.z - velrhop2.z;
				if (compute)arp1 += (USE_FLOATING ? ftmassp2 : massp2)*(dvx*frx + dvy * fry + dvz * frz);

				const float cbar = CTE.cs0;
				//-Density derivative (DeltaSPH Molteni).
				if ((tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt)) {
					const float rhop1over2 = rhopp1 / velrhop2.w;
					const float visc_densi = CTE.delta2h*cbar*(rhop1over2 - 1.f) / (rr2 + CTE.eta2);
					const float dot3 = (drx*frx + dry * fry + drz * frz);
					const float delta = visc_densi * dot3*(USE_FLOATING ? ftmassp2 : massp2);
					if (USE_FLOATING)deltap1 = (boundp2 || deltap1 == FLT_MAX ? FLT_MAX : deltap1 + delta); //-Con floating bodies entre el fluido. //-For floating bodies within the fluid
					else deltap1 = (boundp2 ? FLT_MAX : deltap1 + delta);
				}

				//-Shifting correction.
				if (shift && shiftposp1.x != FLT_MAX) {
					const float massrhop = (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
					const bool noshift = (boundp2 && (tshifting == SHIFT_NoBound || (tshifting == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
					shiftposp1.x = (noshift ? FLT_MAX : shiftposp1.x + massrhop * frx); //-Removes shifting for the boundaries. | Con boundary anula shifting.
					shiftposp1.y += massrhop * fry;
					shiftposp1.z += massrhop * frz;
					shiftdetectp1 -= massrhop * (drx*frx + dry * fry + drz * frz);
				}

				//===== Viscosity ===== 
				if (compute) {
					const float dot = drx * dvx + dry * dvy + drz * dvz;
					const float dot_rr2 = dot / (rr2 + CTE.eta2);
					visc = max(dot_rr2, visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
					if (!lamsps) {//-Artificial viscosity.
						if (dot < 0) {
							const float amubar = CTE.h*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
							const float robar = (rhopp1 + velrhop2.w)*0.5f;
							const float pi_visc = (-visco * cbar*amubar / robar)*(USE_FLOATING ? ftmassp2 * ftmassp1 : massp2);
							acep1.x -= pi_visc * frx; acep1.y -= pi_visc * fry; acep1.z -= pi_visc * frz;
						}
					}
					/*else {//-Laminar+SPS viscosity.
						{//-Laminar contribution.
							const float robar2 = (rhopp1 + velrhop2.w);
							const float temp = 4.f*visco / ((rr2 + CTE.eta2)*robar2);  //-Simplication of temp=2.0f*visco/((rr2+CTE.eta2)*robar); robar=(rhopp1+velrhop2.w)*0.5f;
							const float vtemp = (USE_FLOATING ? ftmassp2 : massp2)*temp*(drx*frx + dry * fry + drz * frz);
							acep1.x += vtemp * dvx; acep1.y += vtemp * dvy; acep1.z += vtemp * dvz;
						}
						//-SPS turbulence model.
						float2 taup2_xx_xy = taup1_xx_xy; //-taup1 is always zero when p1 is not fluid. | taup1 siempre es cero cuando p1 no es fluid.
						float2 taup2_xz_yy = taup1_xz_yy;
						float2 taup2_yz_zz = taup1_yz_zz;
						if (!boundp2 && (USE_NOFLOATING || !ftp2)) {//-When p2 is fluid.
							float2 taup2 = tauff[p2 * 3];     taup2_xx_xy.x += taup2.x; taup2_xx_xy.y += taup2.y;
							taup2 = tauff[p2 * 3 + 1];   taup2_xz_yy.x += taup2.x; taup2_xz_yy.y += taup2.y;
							taup2 = tauff[p2 * 3 + 2];   taup2_yz_zz.x += taup2.x; taup2_yz_zz.y += taup2.y;
						}
						acep1.x += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xx_xy.x*frx + taup2_xx_xy.y*fry + taup2_xz_yy.x*frz);
						acep1.y += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xx_xy.y*frx + taup2_xz_yy.y*fry + taup2_yz_zz.x*frz);
						acep1.z += (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2)*(taup2_xz_yy.x*frx + taup2_yz_zz.x*fry + taup2_yz_zz.y*frz);
						//-Velocity gradients.
						if (USE_NOFLOATING || !ftp1) {//-When p1 is fluid.
							const float volp2 = -(USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
							float dv = dvx * volp2; grap1_xx_xy.x += dv * frx; grap1_xx_xy.y += dv * fry; grap1_xz_yy.x += dv * frz;
							dv = dvy * volp2; grap1_xx_xy.y += dv * frx; grap1_xz_yy.y += dv * fry; grap1_yz_zz.x += dv * frz;
							dv = dvz * volp2; grap1_xz_yy.x += dv * frx; grap1_yz_zz.x += dv * fry; grap1_yz_zz.y += dv * frz;
							// to compute tau terms we assume that gradvel.xy=gradvel.dudy+gradvel.dvdx, gradvel.xz=gradvel.dudz+gradvel.dwdx, gradvel.yz=gradvel.dvdz+gradvel.dwdy
							// so only 6 elements are needed instead of 3x3.
						}
					}*/
				}
				//===== Velocity gradients ===== 
				if (compute) {
					if (!ftp1) {//-When p1 is a fluid particle / Cuando p1 es fluido. 
						const float volp2 = -massp2 / velrhop[p2].w;
						float dv = dvx * volp2;
						gradvelp1.xx += dv * frx; gradvelp1.xy += 0.5f*dv*fry; gradvelp1.xz += 0.5f*dv*frz;
						omegap1.xy += 0.5f*dv*fry; omegap1.xz += 0.5f*dv*frz;

						dv = dvy * volp2;
						gradvelp1.xy += 0.5f*dv*frx; gradvelp1.yy += dv * fry; gradvelp1.yz += 0.5f*dv*frz;
						omegap1.xy -= 0.5f*dv*frx; omegap1.yz += 0.5f*dv*frz;

						dv = dvz * volp2;
						gradvelp1.xz += 0.5f*dv*frx; gradvelp1.yz += 0.5f*dv*fry; gradvelp1.zz += dv * frz;
						omegap1.xz -= 0.5f*dv*frx; omegap1.yz -= 0.5f*dv*fry;

						
					}
				}
			}
		}
	}

	//------------------------------------------------------------------------------
	/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound) with Pore pressure 1D WIP
	//------------------------------------------------------------------------------
	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __device__ void KerInteractionForcesFluidBoxPPM
	(bool boundp2, unsigned p1, const unsigned &pini, const unsigned &pfin, float visco
		, const float *ftomassp, const float2 *tauff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float massp2, float ftmassp1, bool ftp1
		, double3 posdp1, float3 posp1, float3 velp1, float pressp1, float rhopp1, float porep1
		, float3 &acep1, float &arp1, float &visc, float &deltap1
		, TpShifting tshifting, float3 &shiftposp1, float &shiftdetectp1, tsymatrix3f taup1, const tsymatrix3f* tau, const float *press, const float *pore, const float *mass, tsymatrix3f &gradvelp1, tsymatrix3f &omegap1)
	{
		for (int p2 = pini; p2 < pfin; p2++) {
			float drx, dry, drz, pressp2;
			KerGetParticlesDr<psingle>(p2, posxy, posz, pospress, posdp1, posp1, drx, dry, drz, pressp2);
			float rr2 = drx * drx + dry * dry + drz * drz;
			if (rr2 <= CTE.fourh2 && rr2 >= ALMOSTZERO) {
				//Cubic Spline, Wendland or Gaussian kernel.
				float frx, fry, frz;
				if (tker == KERNEL_Wendland)KerGetKernelWendland(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Gaussian)KerGetKernelGaussian(rr2, drx, dry, drz, frx, fry, frz);
				else if (tker == KERNEL_Cubic)KerGetKernelCubic(rr2, drx, dry, drz, frx, fry, frz);

				//-Obtains mass of particle p2 if any floating bodies exist.
				//-Obtiene masa de particula p2 en caso de existir floatings.
				//printf("Mass enregistree = %f \n", mass[p2]);				//massp2 = mass[p2];
				bool ftp2;         //-Indicates if it is floating. | Indica si es floating.
				float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
				bool compute = true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				if (USE_FLOATING) {
					const typecode cod = code[p2];
					ftp2 = CODE_IsFloating(cod);
					ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
					if (ftp2 && ftmassp2 <= (massp2*1.2f) && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#else
					if (ftp2 && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
#endif
					if (ftp2 && shift && tshifting == SHIFT_NoBound)shiftposp1.x = FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
					//compute = !(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
				}
				const float4 velrhop2 = velrhop[p2];

				//===== Aceleration ===== 
				/*if (compute) {
				if (!psingle)pressp2 = (CTE.cteb*(powf(velrhop2.w*CTE.ovrhopzero, CTE.gamma) - 1.0f));
				const float prs = (pressp1 + pressp2) / (rhopp1*velrhop2.w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop2.w, pressp2) : 0);
				const float p_vpm = -prs * (USE_FLOATING ? ftmassp2 * ftmassp1 : massp2);
				acep1.x += p_vpm * frx; acep1.y += p_vpm * fry; acep1.z += p_vpm * frz;
				}*/
				if (compute) {
					const tsymatrix3f prs = {
						(pressp1 + porep1 - taup1.xx + press[p2] + pore[p2] - tau[p2].xx) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop[p2].w, press[p2]) : 0),
						-(taup1.xy + tau[p2].xy) / (rhopp1*velrhop[p2].w),
						-(taup1.xz + tau[p2].xz) / (rhopp1*velrhop[p2].w),
						(pressp1 + porep1 - taup1.yy + press[p2] + pore[p2] - tau[p2].yy) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop[p2].w, press[p2]) : 0),
						-(taup1.yz + tau[p2].yz) / (rhopp1*velrhop[p2].w),
						(pressp1+ porep1 - taup1.zz + press[p2] + pore[p2] - tau[p2].zz) / (rhopp1*velrhop[p2].w) + (tker == KERNEL_Cubic ? KerGetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop[p2].w, press[p2]) : 0)
					};
					const tsymatrix3f p_vpm3 = {
						-prs.xx*massp2*ftmassp1, -prs.xy*massp2*ftmassp1, -prs.xz*massp2*ftmassp1,
						-prs.yy*massp2*ftmassp1, -prs.yz*massp2*ftmassp1, -prs.zz*massp2*ftmassp1
					};
					acep1.x += p_vpm3.xx*frx + p_vpm3.xy*fry + p_vpm3.xz*frz;
					acep1.y += p_vpm3.xy*frx + p_vpm3.yy*fry + p_vpm3.yz*frz;
					acep1.z += p_vpm3.xz*frx + p_vpm3.yz*fry + p_vpm3.zz*frz;
				}

				//-Density derivative.
				const float dvx = velp1.x - velrhop2.x, dvy = velp1.y - velrhop2.y, dvz = velp1.z - velrhop2.z;
				if (compute)arp1 += massp2*(dvx*frx + dvy * fry + dvz * frz);


				const float cbar = CTE.cs0;
				//-Density derivative (DeltaSPH Molteni).
				if ((tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt)) {
					const float rhop1over2 = rhopp1 / velrhop2.w;
					const float visc_densi = CTE.delta2h*cbar*(rhop1over2 - 1.f) / (rr2 + CTE.eta2);
					const float dot3 = (drx*frx + dry * fry + drz * frz);
					const float delta = visc_densi * dot3*massp2;
					if (USE_FLOATING)deltap1 = (boundp2 || deltap1 == FLT_MAX ? FLT_MAX : deltap1 + delta); //-Con floating bodies entre el fluido. //-For floating bodies within the fluid
					else deltap1 = (boundp2 ? FLT_MAX : deltap1 + delta);
				}

				//-Shifting correction.
				if (shift && shiftposp1.x != FLT_MAX) {
					const float massrhop = massp2 / velrhop2.w;
					const bool noshift = (boundp2 && (tshifting == SHIFT_NoBound || (tshifting == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
					shiftposp1.x = (noshift ? FLT_MAX : shiftposp1.x + massrhop * frx); //-Removes shifting for the boundaries. | Con boundary anula shifting.
					shiftposp1.y += massrhop * fry;
					shiftposp1.z += massrhop * frz;
					shiftdetectp1 -= massrhop * (drx*frx + dry * fry + drz * frz);
				}

				//===== Viscosity ===== 
				if (compute) {
					const float dot = drx * dvx + dry * dvy + drz * dvz;
					const float dot_rr2 = dot / (rr2 + CTE.eta2);
					visc = max(dot_rr2, visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
					if (!lamsps) {//-Artificial viscosity.
						if (dot < 0) {
							const float amubar = CTE.h*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
							const float robar = (rhopp1 + velrhop2.w)*0.5f;
							const float pi_visc = (-visco * cbar*amubar / robar)*massp2*ftmassp1;
							acep1.x -= pi_visc * frx; acep1.y -= pi_visc * fry; acep1.z -= pi_visc * frz;
						}
					}
			
				}
				//===== Velocity gradients ===== 
				if (compute) {
					if (!ftp1) {//-When p1 is a fluid particle / Cuando p1 es fluido. 
						const float volp2 = -massp2 / velrhop[p2].w;
						float dv = dvx * volp2;
						gradvelp1.xx += dv * frx; gradvelp1.xy += 0.5f*dv*fry; gradvelp1.xz += 0.5f*dv*frz;
						omegap1.xy += 0.5f*dv*fry; omegap1.xz += 0.5f*dv*frz;

						dv = dvy * volp2;
						gradvelp1.xy += 0.5f*dv*frx; gradvelp1.yy += dv * fry; gradvelp1.yz += 0.5f*dv*frz;
						omegap1.xy -= 0.5f*dv*frx; omegap1.yz += 0.5f*dv*frz;

						dv = dvz * volp2;
						gradvelp1.xz += 0.5f*dv*frx; gradvelp1.yz += 0.5f*dv*fry; gradvelp1.zz += dv * frz;
						omegap1.xz -= 0.5f*dv*frx; omegap1.yz -= 0.5f*dv*fry;
					}
				}
			}
		}
	}
	//------------------------------------------------------------------------------
	/// Interaction between particles. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
	/// Includes artificial/laminar viscosity and normal/DEM floating bodies.
	///
	/// Realiza interaccion entre particulas. Fluid/Float-Fluid/Float or Fluid/Float-Bound
	/// Incluye visco artificial/laminar y floatings normales/dem.
	//------------------------------------------------------------------------------
	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesFluid
	(unsigned n, unsigned pinit, int hdiv, int4 nc, unsigned cellfluid, float viscob, float viscof
		, const int2 *begincell, int3 cellzero, const unsigned *dcell
		, const float *ftomassp, const float2 *tauff, float2 *gradvelff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, float *viscdt, float *ar, float3 *ace, float *delta
		, TpShifting tshifting, float3 *shiftpos, float *shiftdetect)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p < n) {
			unsigned p1 = p + pinit;      //-Number of particle.
			float visc = 0, arp1 = 0, deltap1 = 0;
			float3 acep1 = make_float3(0, 0, 0);

			//-Variables for Shifting.
			float3 shiftposp1;
			float shiftdetectp1;
			if (shift) {
				shiftposp1 = make_float3(0, 0, 0);
				shiftdetectp1 = 0;
			}

			//-Obtains data of particle p1 in case there are floating bodies.
			//-Obtiene datos de particula p1 en caso de existir floatings.
			bool ftp1;       //-Indicates if it is floating. | Indica si es floating.
			float ftmassp1;  //-Contains floating particle mass or 1.0f if it is fluid. | Contiene masa de particula floating o 1.0f si es fluid.
			if (USE_FLOATING) {
				const typecode cod = code[p1];
				ftp1 = CODE_IsFloating(cod);
				ftmassp1 = (ftp1 ? ftomassp[CODE_GetTypeValue(cod)] : 1.f);
				if (ftp1 && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
				if (ftp1 && shift)shiftposp1.x = FLT_MAX; //-Shifting is not calculated for floating bodies. | Para floatings no se calcula shifting.
			}

			//-Obtains basic data of particle p1.
			double3 posdp1;
			float3 posp1, velp1;
			float rhopp1, pressp1;
			KerGetParticleData<psingle>(p1, posxy, posz, pospress, velrhop, velp1, rhopp1, posdp1, posp1, pressp1);

			//-Variables for Laminar+SPS.
			float2 taup1_xx_xy, taup1_xz_yy, taup1_yz_zz;
			if (lamsps) {
				taup1_xx_xy = tauff[p1 * 3];
				taup1_xz_yy = tauff[p1 * 3 + 1];
				taup1_yz_zz = tauff[p1 * 3 + 2];
			}
			//-Variables for Laminar+SPS (computation).
			float2 grap1_xx_xy, grap1_xz_yy, grap1_yz_zz;
			if (lamsps) {
				grap1_xx_xy = make_float2(0, 0);
				grap1_xz_yy = make_float2(0, 0);
				grap1_yz_zz = make_float2(0, 0);
			}

			//-Obtains interaction limits.
			int cxini, cxfin, yini, yfin, zini, zfin;
			KerGetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);

			//-Interaction with fluids.
			for (int z = zini; z < zfin; z++) {
				int zmod = (nc.w)*z + cellfluid; //-The sum showing where fluid cells start. | Le suma donde empiezan las celdas de fluido.
				for (int y = yini; y < yfin; y++) {
					int ymod = zmod + nc.x*y;
					unsigned pini, pfin = 0;
					for (int x = cxini; x < cxfin; x++) {
						int2 cbeg = begincell[x + ymod];
						if (cbeg.y) {
							if (!pfin)pini = cbeg.x;
							pfin = cbeg.y;
						}
					}
					if (pfin)KerInteractionForcesFluidBox<psingle, tker, ftmode, lamsps, tdelta, shift>(false, p1, pini, pfin, viscof, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massf, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1, taup1_xx_xy, taup1_xz_yy, taup1_yz_zz, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1);
				}
			}
			//-Interaction with boundaries.
			for (int z = zini; z < zfin; z++) {
				int zmod = (nc.w)*z;
				for (int y = yini; y < yfin; y++) {
					int ymod = zmod + nc.x*y;
					unsigned pini, pfin = 0;
					for (int x = cxini; x < cxfin; x++) {
						int2 cbeg = begincell[x + ymod];
						if (cbeg.y) {
							if (!pfin)pini = cbeg.x;
							pfin = cbeg.y;
						}
					}
					if (pfin)KerInteractionForcesFluidBox<psingle, tker, ftmode, lamsps, tdelta, shift>(true, p1, pini, pfin, viscob, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massb, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1, taup1_xx_xy, taup1_xz_yy, taup1_yz_zz, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1);
				}
			}

			//-Stores results.
			if (shift || arp1 || acep1.x || acep1.y || acep1.z || visc) {
				if (tdelta == DELTA_Dynamic && deltap1 != FLT_MAX)arp1 += deltap1;
				if (tdelta == DELTA_DynamicExt) {
					float rdelta = delta[p1];
					delta[p1] = (rdelta == FLT_MAX || deltap1 == FLT_MAX ? FLT_MAX : rdelta + deltap1);
				}
				ar[p1] += arp1;
				float3 r = ace[p1]; r.x += acep1.x; r.y += acep1.y; r.z += acep1.z; ace[p1] = r;
				if (visc > viscdt[p1])viscdt[p1] = visc;
				if (lamsps) {
					gradvelff[p1 * 3] = grap1_xx_xy;
					gradvelff[p1 * 3 + 1] = grap1_xz_yy;
					gradvelff[p1 * 3 + 2] = grap1_yz_zz;
				}
				if (shift) {
					shiftpos[p1] = shiftposp1;
					if (shiftdetect)shiftdetect[p1] = shiftdetectp1;
				}
			}
		}
	}
	//==============================================================================
	/// Perform interaction between particles: Solid - Lucas 
	//==============================================================================
	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesSolid
	(unsigned n, unsigned pinit, int hdiv, int4 nc, unsigned cellfluid, float viscob, float viscof
		, const int2 *begincell, int3 cellzero, const unsigned *dcell, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const float *ftomassp, const float2 *tauff, float2 *gradvelff
		, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore
		, float *viscdt, float *ar, float3 *ace, float *delta
		, TpShifting tshifting, float3 *shiftpos, float *shiftdetect)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p < n) {
			unsigned p1 = p + pinit;      //-Number of particle.
			float visc = 0, arp1 = 0, deltap1 = 0;
			float3 acep1 = make_float3(0, 0, 0);
			tsymatrix3f gradvelp1 = { 0, 0, 0, 0, 0, 0 };
			tsymatrix3f omegap1 = { 0, 0, 0, 0, 0, 0 };
			float3 shiftposp1 = make_float3(0, 0, 0);
			float shiftdetectp1 = 0;

			//-Obtains data of particle p1 in case there are floating bodies.
			//-Obtiene datos de particula p1 en caso de existir floatings.
			bool ftp1;       //-Indicates if it is floating. | Indica si es floating.
			float ftmassp1;  //-Contains floating particle mass or 1.0f if it is fluid. | Contiene masa de particula floating o 1.0f si es fluid.
			if (USE_FLOATING) {
				const typecode cod = code[p1];
				ftp1 = CODE_IsFloating(cod);
				ftmassp1 = (ftp1 ? ftomassp[CODE_GetTypeValue(cod)] : 1.f);
				if (ftp1 && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
				if (ftp1 && shift)shiftposp1.x = FLT_MAX; //-Shifting is not calculated for floating bodies. | Para floatings no se calcula shifting.
			}
			//-Obtains basic data of particle p1.
			double3 posdp1;
			float3 posp1, velp1;
			float rhopp1, pressp1;
			KerGetParticleData<psingle>(p1, posxy, posz, pospress, velrhop, velp1, rhopp1, posdp1, posp1, pressp1);

			// Matthias
			const tsymatrix3f taup1 = tau[p1];
			const float porep1 = pore[p1];

			//-Variables for Laminar+SPS.
			float2 taup1_xx_xy, taup1_xz_yy, taup1_yz_zz;
			if (lamsps) {
				taup1_xx_xy = tauff[p1 * 3];
				taup1_xz_yy = tauff[p1 * 3 + 1];
				taup1_yz_zz = tauff[p1 * 3 + 2];
			}
			//-Variables for Laminar+SPS (computation).
			float2 grap1_xx_xy, grap1_xz_yy, grap1_yz_zz;
			if (lamsps) {
				grap1_xx_xy = make_float2(0, 0);
				grap1_xz_yy = make_float2(0, 0);
				grap1_yz_zz = make_float2(0, 0);
			}


			//-Obtains interaction limits.
			int cxini, cxfin, yini, yfin, zini, zfin;
			KerGetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);

			//-Interaction with fluids.
			for (int z = zini; z < zfin; z++) {
				int zmod = (nc.w)*z + cellfluid; //-The sum showing where fluid cells start. 
				for (int y = yini; y < yfin; y++) {
					int ymod = zmod + nc.x*y;
					unsigned pini, pfin = 0;
					for (int x = cxini; x < cxfin; x++) {
						int2 cbeg = begincell[x + ymod];
						if (cbeg.y) {
							if (!pfin)pini = cbeg.x;
							pfin = cbeg.y;
						}
					}
					if (pfin)KerInteractionForcesFluidBoxPP<psingle, tker, ftmode, lamsps, tdelta, shift>(false, p1, pini, pfin, viscof, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massf, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1,porep1, taup1_xx_xy, taup1_xz_yy, taup1_yz_zz, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1, taup1, tau, press, pore, gradvelp1, omegap1);
				}
			}
			//-Interaction with boundaries.
			for (int z = zini; z < zfin; z++) {
				int zmod = (nc.w)*z;
				for (int y = yini; y < yfin; y++) {
					int ymod = zmod + nc.x*y;
					unsigned pini, pfin = 0;
					for (int x = cxini; x < cxfin; x++) {
						int2 cbeg = begincell[x + ymod];
						if (cbeg.y) {
							if (!pfin)pini = cbeg.x;
							pfin = cbeg.y;
						}
					}
					if (pfin)KerInteractionForcesFluidBoxPP<psingle, tker, ftmode, lamsps, tdelta, shift>(true, p1, pini, pfin, viscob, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massb, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1,porep1, taup1_xx_xy, taup1_xz_yy, taup1_yz_zz, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1, taup1, tau, press, pore, gradvelp1, omegap1);
				}
			}
			//-Stores results
			if (shift || arp1 || acep1.x || acep1.y || acep1.z || visc) {
				if (tdelta == DELTA_Dynamic && deltap1 != FLT_MAX)arp1 += deltap1;
				if (tdelta == DELTA_DynamicExt) {
					float rdelta = delta[p1];
					delta[p1] = (rdelta == FLT_MAX || deltap1 == FLT_MAX ? FLT_MAX : rdelta + deltap1);
				}
				ar[p1] += arp1;
				float3 r = ace[p1]; r.x += acep1.x; r.y += acep1.y; r.z += acep1.z; ace[p1] = r;
				if (visc > viscdt[p1])viscdt[p1] = visc;

				if (lamsps) {
					gradvelff[p1 * 3] = grap1_xx_xy;
					gradvelff[p1 * 3 + 1] = grap1_xz_yy;
					gradvelff[p1 * 3 + 2] = grap1_yz_zz;
				}
				if (shift) {
					shiftpos[p1] = shiftposp1;
					if (shiftdetect)shiftdetect[p1] = shiftdetectp1;
				}

				// Gradvel and rotation tensor
				gradvel[p1].xx += gradvelp1.xx;
				gradvel[p1].xy += gradvelp1.xy;
				gradvel[p1].xz += gradvelp1.xz;
				gradvel[p1].yy += gradvelp1.yy;
				gradvel[p1].yz += gradvelp1.yz;
				gradvel[p1].zz += gradvelp1.zz;

				omega[p1].xx += omegap1.xx;
				omega[p1].xy += omegap1.xy;
				omega[p1].xz += omegap1.xz;
				omega[p1].yy += omegap1.yy;
				omega[p1].yz += omegap1.yz;
				omega[p1].zz += omegap1.zz;
			}
		}
	}

	//##############################################################################
	//# Kernels for ComputeStep (vel & rhop) --Lucas--
	//# Kernels para ComputeStep (vel & rhop).
	//##############################################################################

	//------------------------------------------------------------------------------
	/// Computes new values for Pos, Check, Vel and Ros (using Verlet).
	/// The value of Vel always set to be reset.
	///
	/// Calcula nuevos valores de  Pos, Check, Vel y Rhop (usando Verlet).
	/// El valor de Vel para bound siempre se pone a cero.
	//------------------------------------------------------------------------------
	template<bool floating, bool shift> __global__ void KerComputeStepVerlet_L
	(unsigned n, unsigned npb, float rhopoutmin, float rhopoutmax
		, const float4 *velrhop1, const float4 *velrhop2
		, const float *ar, const float3 *ace, const float3 *shiftpos
		, double dt, double dt205, double dt2
		, double2 *movxy, double *movz, typecode *code, float4 *velrhopnew
		, const tsymatrix3f *tau2, tsymatrix3f *JauTauDot_M, tsymatrix3f *taunew
		, const float *mass1,const float *mass2, float *massnew
		,float LambdaMass, float RhopZero)
	{
		unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
		if (p<n) {
			if (p<npb) { //-Particles: Fixed & Moving.
				float rrhop = float(double(velrhop2[p].w) + dt2 * ar[p]);
				rrhop = (rrhop<CTE.rhopzero ? CTE.rhopzero : rrhop); //-To prevent absorption of fluid particles by boundaries. | Evita q las boundary absorvan a las fluidas.
				velrhopnew[p] = make_float4(0, 0, 0, rrhop);
			}
			else { //-Particles: Floating & Fluid.
				   //-Updates density.
				const float volu = float(double(mass1[p]) / double(velrhop1[p].w));
				const float amass = float(LambdaMass * (RhopZero / velrhop1[p].w - 1.0f));
						
				float4 rvelrhop2 = velrhop2[p];
				rvelrhop2.w = float(double(rvelrhop2.w) + dt2 * ar[p]+ amass);
				//rvelrhop2.w = float(double(rvelrhop2.w));
				//printf("\n new = %f / %f /%f /%f /%f", rvelrhop2.w, velrhop1[p].x, velrhop1[p].y, velrhop1[p].z ,velrhop1[p].w);
				float4 rvel1 = velrhop1[p];
				if (!floating || CODE_IsFluid(code[p])) { //-Particles: Fluid.
														  //-Checks rhop limits.
					if (rvelrhop2.w<rhopoutmin || rvelrhop2.w>rhopoutmax) { //-Only brands as excluded normal particles (not periodic). | Solo marca como excluidas las normales (no periodicas).
						const typecode rcode = code[p];
						if (CODE_IsNormal(rcode))code[p] = CODE_SetOutRhop(rcode);
					}
					//-Computes and stores position displacement.
					const float3 race = ace[p];
					double dx = double(rvel1.x)*dt + double(race.x)*dt205;
					double dy = double(rvel1.y)*dt + double(race.y)*dt205;
					double dz = double(rvel1.z)*dt + double(race.z)*dt205;
					if (shift) {
						const float3 rshiftpos = shiftpos[p];
						dx += double(rshiftpos.x);
						dy += double(rshiftpos.y);
						dz += double(rshiftpos.z);
					}
					movxy[p] = make_double2(dx, dy);
					movz[p] = dz;
					//printf("DANS COMPUTE %f,%f,%f \n", ace[p].x, ace[p].y, ace[p].z);
					//printf("rvell ,race = %f,%f,%f,%f,%f,%f",rvel1.x, rvel1.y, rvel1.z, race.x, race.y, race.z);
					//-Updates velocity.
					rvelrhop2.x = float(double(rvelrhop2.x) + double(race.x)*dt2);
					rvelrhop2.y = float(double(rvelrhop2.y) + double(race.y)*dt2);
					rvelrhop2.z = float(double(rvelrhop2.z) + double(race.z)*dt2);
					velrhopnew[p] = rvelrhop2;

					// Update Shear stress
					taunew[p].xx = float(double(tau2[p].xx) + double(JauTauDot_M[p].xx)*dt2);
					taunew[p].xy = float(double(tau2[p].xy) + double(JauTauDot_M[p].xy)*dt2);
					taunew[p].xz = float(double(tau2[p].xz) + double(JauTauDot_M[p].xz)*dt2);
					taunew[p].yy = float(double(tau2[p].yy) + double(JauTauDot_M[p].yy)*dt2);
					taunew[p].yz = float(double(tau2[p].yz) + double(JauTauDot_M[p].yz)*dt2);
					taunew[p].zz = float(double(tau2[p].zz) + double(JauTauDot_M[p].zz)*dt2);
					//printf("Ici = %f/%f/%f/%f/%f/%f", tau2[p].xx, tau2[p].xy, tau2[p].xz, tau2[p].yy, tau2[p].yz, tau2[p].zz);

					//Update mass
					 massnew[p] = float(double(mass2[p]) + double(amass / volu)*dt2);


				}
				else { //-Particles: Floating.
					rvel1.w = (rvelrhop2.w<CTE.rhopzero ? CTE.rhopzero : rvelrhop2.w); //-To prevent absorption of fluid particles by boundaries. | Evita q las floating absorvan a las fluidas.
					velrhopnew[p] = rvel1;
				}
			}
		}
	}

	void ComputeStepVerlet_L(bool floating, bool shift, unsigned np, unsigned npb, const float4 * velrhop1, const float4 * velrhop2, const float * ar, const float3 * ace, const float3 * shiftpos, double dt, double dt2, float rhopoutmin, float rhopoutmax, typecode * code, double2 * movxy, double * movz, float4 * velrhopnew, const tsymatrix3f * tau2, tsymatrix3f * JauTauDot_M, tsymatrix3f * taunew, const float * mass1, const float * mass2, float * massnew,float LambdaMass, float RhopZero)
	{
		double dt205 = (0.5*dt*dt);
		if (np) {
			dim3 sgrid = cuSol::GetGridSize(np, SPHBSIZE);
			if (shift) {
				const bool shift = true;
				if (floating)cuSol::KerComputeStepVerlet_L<true, shift> << <sgrid, SPHBSIZE >> > (np, npb, rhopoutmin, rhopoutmax, velrhop1, velrhop2, ar, ace, shiftpos, dt, dt205, dt2, movxy, movz, code, velrhopnew, tau2, JauTauDot_M, taunew,mass1, mass2, massnew,LambdaMass,RhopZero);
				else         cuSol::KerComputeStepVerlet_L<false, shift> << <sgrid, SPHBSIZE >> > (np, npb, rhopoutmin, rhopoutmax, velrhop1, velrhop2, ar, ace, shiftpos, dt, dt205, dt2, movxy, movz, code, velrhopnew, tau2, JauTauDot_M, taunew,mass1, mass2, massnew, LambdaMass, RhopZero);
			}
			else {
				const bool shift = false;
				if (floating)cuSol::KerComputeStepVerlet_L<true, shift> << <sgrid, SPHBSIZE >> > (np, npb, rhopoutmin, rhopoutmax, velrhop1, velrhop2, ar, ace, shiftpos, dt, dt205, dt2, movxy, movz, code, velrhopnew, tau2, JauTauDot_M, taunew,mass1, mass2, massnew, LambdaMass, RhopZero);
				else         cuSol::KerComputeStepVerlet_L<false, shift> << <sgrid, SPHBSIZE >> > (np, npb, rhopoutmin, rhopoutmax, velrhop1, velrhop2, ar, ace, shiftpos, dt, dt205, dt2, movxy, movz, code, velrhopnew, tau2, JauTauDot_M, taunew,mass1, mass2, massnew, LambdaMass, RhopZero);
			}
		}
	}


//==============================================================================
/// Perform interaction between particles with Solid, pore and mass variation - Lucas
//==============================================================================
template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesSolMass
(unsigned n, unsigned pinit, int hdiv, int4 nc, unsigned cellfluid, float viscob, float viscof
	, const int2 *begincell, int3 cellzero, const unsigned *dcell, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
	, const float *ftomassp, const float2 *tauff, float2 *gradvelff
	, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
	, const float3 *press, const float *pore, const float *mass
	, float *viscdt, float *ar, float3 *ace, float *delta
	, TpShifting tshifting, float3 *shiftpos, float *shiftdetect)
{
	unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p < n) {
		unsigned p1 = p + pinit;      //-Number of particle.
		float visc = 0, arp1 = 0, deltap1 = 0;
		float3 acep1 = make_float3(0, 0, 0);
		tsymatrix3f gradvelp1 = { 0, 0, 0, 0, 0, 0 };
		tsymatrix3f omegap1 = { 0, 0, 0, 0, 0, 0 };
		float3 shiftposp1 = make_float3(0, 0, 0);
		float shiftdetectp1 = 0;


		//-Obtains data of particle p1 in case there are floating bodies.
		//-Obtiene datos de particula p1 en caso de existir floatings.
		bool ftp1;       //-Indicates if it is floating. | Indica si es floating.
		float ftmassp1;  //-Contains floating particle mass or 1.0f if it is fluid. | Contiene masa de particula floating o 1.0f si es fluid.
		if (USE_FLOATING) {
			const typecode cod = code[p1];
			ftp1 = CODE_IsFloating(cod);
			ftmassp1 = (ftp1 ? ftomassp[CODE_GetTypeValue(cod)] : 1.f);
			if (ftp1 && (tdelta == DELTA_Dynamic || tdelta == DELTA_DynamicExt))deltap1 = FLT_MAX;
			if (ftp1 && shift)shiftposp1.x = FLT_MAX; //-Shifting is not calculated for floating bodies. | Para floatings no se calcula shifting.
		}

		//-Obtains basic data of particle p1.
		double3 posdp1;
		float3 posp1, velp1;
		float rhopp1, pressptemp;
		const float3 pressp1 = press[p1];
		const tsymatrix3f taup1 = tau[p1];
		const float porep1 = pore[p1];

		cuSol::KerGetParticleData<psingle>(p1, posxy, posz, pospress, velrhop, velp1, rhopp1, posdp1, posp1, pressptemp);

		//-Obtains interaction limits.
		int cxini, cxfin, yini, yfin, zini, zfin;
		cuSol::KerGetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);

		//-Interaction with fluids.
		for (int z = zini; z < zfin; z++) {
			int zmod = (nc.w)*z + cellfluid; //-The sum showing where fluid cells start. 
			for (int y = yini; y < yfin; y++) {
				int ymod = zmod + nc.x*y;
				unsigned pini, pfin = 0;
				for (int x = cxini; x < cxfin; x++) {
					int2 cbeg = begincell[x + ymod];
					if (cbeg.y) {
						if (!pfin)pini = cbeg.x;
						pfin = cbeg.y;
					}
				}
				if (pfin)cuSol::KerInteractionForcesFluidBoxPPM<psingle, tker, ftmode, lamsps, tdelta, shift>(false, p1, pini, pfin, viscof, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massf, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1, porep1, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1, taup1, tau, press, pore, mass, gradvelp1, omegap1);
			}
		}
		//-Interaction with boundaries.
		for (int z = zini; z < zfin; z++) {
			int zmod = (nc.w)*z;
			for (int y = yini; y < yfin; y++) {
				int ymod = zmod + nc.x*y;
				unsigned pini, pfin = 0;
				for (int x = cxini; x < cxfin; x++) {
					int2 cbeg = begincell[x + ymod];
					if (cbeg.y) {
						if (!pfin)pini = cbeg.x;
						pfin = cbeg.y;
					}
				}
				if (pfin)cuSol::KerInteractionForcesFluidBoxPPM<psingle, tker, ftmode, lamsps, tdelta, shift>(true, p1, pini, pfin, viscob, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massb, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1, porep1, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1, taup1, tau, press, pore, mass, gradvelp1, omegap1);
			}
		}
		__syncthreads();

		if (shift || arp1 || acep1.x || acep1.y || acep1.z || visc) {
			if (tdelta == DELTA_Dynamic && deltap1 != FLT_MAX)arp1 += deltap1;
			if (tdelta == DELTA_DynamicExt) {
				float rdelta = delta[p1];
				delta[p1] = (rdelta == FLT_MAX || deltap1 == FLT_MAX ? FLT_MAX : rdelta + deltap1);
			}
			ar[p1] += arp1;
			float3 r = ace[p1]; r.x += acep1.x; r.y += acep1.y; r.z += acep1.z; ace[p1] = r;
			if (visc > viscdt[p1])viscdt[p1] = visc;

		}
		if (shift) {
			shiftpos[p1] = shiftposp1;
			if (shiftdetect)shiftdetect[p1] = shiftdetectp1;
		}
		gradvel[p1].xx += gradvelp1.xx;
		gradvel[p1].xy += gradvelp1.xy;
		gradvel[p1].xz += gradvelp1.xz;
		gradvel[p1].yy += gradvelp1.yy;
		gradvel[p1].yz += gradvelp1.yz;
		gradvel[p1].zz += gradvelp1.zz;

		omega[p1].xx += omegap1.xx;
		omega[p1].xy += omegap1.xy;
		omega[p1].xz += omegap1.xz;
		omega[p1].yy += omegap1.yy;
		omega[p1].yz += omegap1.yz;
		omega[p1].zz += omegap1.zz;
		}
	}


//==============================================================================
/// Perform interaction between particles with Solid, pore and mass variation - Lucas Press Simple
//==============================================================================
template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> __global__ void KerInteractionForcesSolMass
(unsigned n, unsigned pinit, int hdiv, int4 nc, unsigned cellfluid, float viscob, float viscof
	, const int2 *begincell, int3 cellzero, const unsigned *dcell, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
	, const float *ftomassp, const float2 *tauff, float2 *gradvelff
	, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
	, const float *press, const float *pore, const float *mass
	, float *viscdt, float *ar, float3 *ace, float *delta
	, TpShifting tshifting, float3 *shiftpos, float *shiftdetect)
{
	unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p < n) {
		unsigned p1 = p + pinit;      //-Number of particle.
		//printf(" ace = %f /%f /%f", ace[p1].x, ace[p1].y, ace[p1].z);
		float visc = 0, arp1 = 0, deltap1 = 0;
		float3 acep1 = make_float3(0, 0, 0);
		tsymatrix3f gradvelp1 = { 0, 0, 0, 0, 0, 0 };
		tsymatrix3f omegap1 = { 0, 0, 0, 0, 0, 0 };
		float3 shiftposp1 = make_float3(0, 0, 0);
		float shiftdetectp1 = 0;


		//-Obtains data of particle p1 in case there are floating bodies.
		//-Obtiene datos de particula p1 en caso de existir floatings.
		bool ftp1=false;       //-Indicates if it is floating. | Indica si es floating.
		float ftmassp1= 1.f;  //-Contains floating particle mass or 1.0f if it is fluid. | Contiene masa de particula floating o 1.0f si es fluid.

		//-Obtains basic data of particle p1.
		double3 posdp1;
		float3 posp1, velp1;
		float rhopp1, pressptemp;
		const float pressp1 = press[p1];
		const tsymatrix3f taup1 = tau[p1];
		const float porep1 = pore[p1];

		cuSol::KerGetParticleData<psingle>(p1, posxy, posz, pospress, velrhop, velp1, rhopp1, posdp1, posp1, pressptemp);

		//-Obtains interaction limits.
		int cxini, cxfin, yini, yfin, zini, zfin;
		cuSol::KerGetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);


		//-Interaction with fluids.
		for (int z = zini; z < zfin; z++) {
			int zmod = (nc.w)*z + cellfluid; //-The sum showing where fluid cells start. 
			for (int y = yini; y < yfin; y++) {
				int ymod = zmod + nc.x*y;
				unsigned pini, pfin = 0;
				for (int x = cxini; x < cxfin; x++) {
					int2 cbeg = begincell[x + ymod];
					if (cbeg.y) {
						if (!pfin)pini = cbeg.x;
						pfin = cbeg.y;
					}
				}
				if (pfin)cuSol::KerInteractionForcesFluidBoxPPM<psingle, tker, ftmode, lamsps, tdelta, shift>(false, p1, pini, pfin, viscof, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massf, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1, porep1, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1, taup1, tau, press, pore,mass,gradvelp1, omegap1);
			}
		}
		//-Interaction with boundaries.
		for (int z = zini; z < zfin; z++) {
			int zmod = (nc.w)*z;
			for (int y = yini; y < yfin; y++) {
				int ymod = zmod + nc.x*y;
				unsigned pini, pfin = 0;
				for (int x = cxini; x < cxfin; x++) {
					int2 cbeg = begincell[x + ymod];
					if (cbeg.y) {
						if (!pfin)pini = cbeg.x;
						pfin = cbeg.y;
					}
				}
				if (pfin)cuSol::KerInteractionForcesFluidBoxPPM<psingle, tker, ftmode, lamsps, tdelta, shift>(true, p1, pini, pfin, viscob, ftomassp, tauff, posxy, posz, pospress, velrhop, code, idp, CTE.massb, ftmassp1, ftp1, posdp1, posp1, velp1, pressp1, rhopp1, porep1, acep1, arp1, visc, deltap1, tshifting, shiftposp1, shiftdetectp1, taup1, tau, press, pore,mass,gradvelp1, omegap1);
			}
		}
		__syncthreads();

		if (shift || arp1 || acep1.x || acep1.y || acep1.z || visc) {
			if (tdelta == DELTA_Dynamic && deltap1 != FLT_MAX)arp1 += deltap1;
			if (tdelta == DELTA_DynamicExt) {
				float rdelta = delta[p1];
				delta[p1] = (rdelta == FLT_MAX || deltap1 == FLT_MAX ? FLT_MAX : rdelta + deltap1);
			}
			ar[p1] += arp1;
			float3 r = ace[p1]; r.x += acep1.x; r.y += acep1.y; r.z += acep1.z; ace[p1] = r;
			if (visc > viscdt[p1])viscdt[p1] = visc;

			}
			if (shift) {
				shiftpos[p1] = shiftposp1;
				if (shiftdetect)shiftdetect[p1] = shiftdetectp1;
			}
			gradvel[p1].xx += gradvelp1.xx;
			gradvel[p1].xy += gradvelp1.xy;
			gradvel[p1].xz += gradvelp1.xz;
			gradvel[p1].yy += gradvelp1.yy;
			gradvel[p1].yz += gradvelp1.yz;
			gradvel[p1].zz += gradvelp1.zz;

			omega[p1].xx += omegap1.xx;
			omega[p1].xy += omegap1.xy;
			omega[p1].xz += omegap1.xz;
			omega[p1].yy += omegap1.yy;
			omega[p1].yz += omegap1.yz;
			omega[p1].zz += omegap1.zz;

		}
	}


//##############################################################################
//# Kernels for DEM interaction.
//# Kernels para interaccion DEM.
//##############################################################################
//------------------------------------------------------------------------------
/// DEM interaction of a particle with a set of particles. (Float-Float/Bound)
/// Realiza la interaccion DEM de una particula con un conjunto de ellas. (Float-Float/Bound)
//------------------------------------------------------------------------------
template<bool psingle> __device__ void KerInteractionForcesDemBox
(bool boundp2, const unsigned &pini, const unsigned &pfin
	, const float4 *demdata, float dtforce
	, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
	, double3 posdp1, float3 posp1, float3 velp1, typecode tavp1, float masstotp1, float taup1, float kfricp1, float restitup1
	, float3 &acep1, float &demdtp1)
{
	for (int p2 = pini; p2<pfin; p2++) {
		const typecode codep2 = code[p2];
		if (CODE_IsNotFluid(codep2) && tavp1 != CODE_GetTypeAndValue(codep2)) {
			float drx, dry, drz;
			cuSol::KerGetParticlesDr<psingle>(p2, posxy, posz, pospress, posdp1, posp1, drx, dry, drz);
			const float rr2 = drx * drx + dry * dry + drz * drz;
			const float rad = sqrt(rr2);

			//-Computes maximum value of demdt.
			float4 demdatap2 = demdata[CODE_GetTypeAndValue(codep2)];
			const float nu_mass = (boundp2 ? masstotp1 / 2 : masstotp1 * demdatap2.x / (masstotp1 + demdatap2.x)); //-With boundary takes the actual mass of floating 1. | Con boundary toma la propia masa del floating 1.
			const float kn = 4 / (3 * (taup1 + demdatap2.y))*sqrt(CTE.dp / 4); //-Generalized rigidity - Lemieux 2008.
			const float dvx = velp1.x - velrhop[p2].x, dvy = velp1.y - velrhop[p2].y, dvz = velp1.z - velrhop[p2].z; //vji
			const float nx = drx / rad, ny = dry / rad, nz = drz / rad; //-normal_ji             
			const float vn = dvx * nx + dvy * ny + dvz * nz; //-vji.nji    
			const float demvisc = 0.2f / (3.21f*(pow(nu_mass / kn, 0.4f)*pow(fabs(vn), -0.2f)) / 40.f);
			if (demdtp1<demvisc)demdtp1 = demvisc;

			const float over_lap = 1.0f*CTE.dp - rad; //-(ri+rj)-|dij|
			if (over_lap>0.0f) { //-Contact.
								 //-Normal.
				const float eij = (restitup1 + demdatap2.w) / 2;
				const float gn = -(2.0f*log(eij)*sqrt(nu_mass*kn)) / (sqrt(float(PI) + log(eij)*log(eij))); //-Generalized damping - Cummins 2010.
																											//const float gn=0.08f*sqrt(nu_mass*sqrt(CTE.dp/2)/((taup1+demdatap2.y)/2)); //-generalized damping - Lemieux 2008.
				float rep = kn * pow(over_lap, 1.5f);
				float fn = rep - gn * pow(over_lap, 0.25f)*vn;
				acep1.x += (fn*nx); acep1.y += (fn*ny); acep1.z += (fn*nz); //-Force is applied in the normal between the particles.
																			//-Tangencial.
				float dvxt = dvx - vn * nx, dvyt = dvy - vn * ny, dvzt = dvz - vn * nz; //Vji_t
				float vt = sqrt(dvxt*dvxt + dvyt * dvyt + dvzt * dvzt);
				float tx = (vt != 0 ? dvxt / vt : 0), ty = (vt != 0 ? dvyt / vt : 0), tz = (vt != 0 ? dvzt / vt : 0); //-Tang vel unit vector.
				float ft_elast = 2 * (kn*dtforce - gn)*vt / 7; //-Elastic frictional string -->  ft_elast=2*(kn*fdispl-gn*vt)/7; fdispl=dtforce*vt;
				const float kfric_ij = (kfricp1 + demdatap2.z) / 2;
				float ft = kfric_ij * fn*tanh(8 * vt);  //-Coulomb.
				ft = (ft<ft_elast ? ft : ft_elast);   //-Not above yield criteria, visco-elastic model.
				acep1.x += (ft*tx); acep1.y += (ft*ty); acep1.z += (ft*tz);
			}
		}
	}
}

//------------------------------------------------------------------------------
/// Interaction between particles. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes artificial/laminar viscosity and normal/DEM floating bodies.
///
/// Realiza interaccion entre particulas. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
//------------------------------------------------------------------------------
template<bool psingle> __global__ void KerInteractionForcesDem
(unsigned nfloat, int hdiv, int4 nc, unsigned cellfluid
	, const int2 *begincell, int3 cellzero, const unsigned *dcell
	, const unsigned *ftridp, const float4 *demdata, float dtforce
	, const double2 *posxy, const double *posz, const float4 *pospress, const float4 *velrhop, const typecode *code, const unsigned *idp
	, float *viscdt, float3 *ace)
{
	unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p<nfloat) {
		const unsigned p1 = ftridp[p]; //-Number of particle.
		if (p1 != UINT_MAX) {
			float demdtp1 = 0;
			float3 acep1 = make_float3(0, 0, 0);

			//-Obtains basic data of particle p1.
			double3 posdp1;
			float3 posp1, velp1;
			cuSol::KerGetParticleData<psingle>(p1, posxy, posz, pospress, velrhop, velp1, posdp1, posp1);
			const typecode tavp1 = CODE_GetTypeAndValue(code[p1]);
			float4 rdata = demdata[tavp1];
			const float masstotp1 = rdata.x;
			const float taup1 = rdata.y;
			const float kfricp1 = rdata.z;
			const float restitup1 = rdata.w;

			//-Obtains interaction limits.
			int cxini, cxfin, yini, yfin, zini, zfin;
			cuSol:: KerGetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);

			//-Interaction with boundaries.
			for (int z = zini; z<zfin; z++) {
				int zmod = (nc.w)*z;
				for (int y = yini; y<yfin; y++) {
					int ymod = zmod + nc.x*y;
					unsigned pini, pfin = 0;
					for (int x = cxini; x<cxfin; x++) {
						int2 cbeg = begincell[x + ymod];
						if (cbeg.y) {
							if (!pfin)pini = cbeg.x;
							pfin = cbeg.y;
						}
					}
					if (pfin)KerInteractionForcesDemBox<psingle>(true, pini, pfin, demdata, dtforce, posxy, posz, pospress, velrhop, code, idp, posdp1, posp1, velp1, tavp1, masstotp1, taup1, kfricp1, restitup1, acep1, demdtp1);
				}
			}

			//-Interaction with fluids.
			for (int z = zini; z<zfin; z++) {
				int zmod = (nc.w)*z + cellfluid; //-The sum showing where fluid cells start. | Le suma donde empiezan las celdas de fluido.
				for (int y = yini; y<yfin; y++) {
					int ymod = zmod + nc.x*y;
					unsigned pini, pfin = 0;
					for (int x = cxini; x<cxfin; x++) {
						int2 cbeg = begincell[x + ymod];
						if (cbeg.y) {
							if (!pfin)pini = cbeg.x;
							pfin = cbeg.y;
						}
					}
					if (pfin)KerInteractionForcesDemBox<psingle>(false, pini, pfin, demdata, dtforce, posxy, posz, pospress, velrhop, code, idp, posdp1, posp1, velp1, tavp1, masstotp1, taup1, kfricp1, restitup1, acep1, demdtp1);
				}
			}
			//-Stores results.
			if (acep1.x || acep1.y || acep1.z || demdtp1) {
				float3 r = ace[p1]; r.x += acep1.x; r.y += acep1.y; r.z += acep1.z; ace[p1] = r;
				if (viscdt[p1]<demdtp1)viscdt[p1] = demdtp1;
			}
		}
	}
}


//------------------------------------------------------------------------------
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.
//------------------------------------------------------------------------------
__global__ void KerComputeSpsTau(unsigned n, unsigned pini, float smag, float blin
	, const float4 *velrhop, const float2 *gradvelff, float2 *tauff)
{
	unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	if (p < n) {
		const unsigned p1 = p + pini;
		float2 rr = gradvelff[p1 * 3];   const float grad_xx = rr.x, grad_xy = rr.y;
		rr = gradvelff[p1 * 3 + 1]; const float grad_xz = rr.x, grad_yy = rr.y;
		rr = gradvelff[p1 * 3 + 2]; const float grad_yz = rr.x, grad_zz = rr.y;
		const float pow1 = grad_xx * grad_xx + grad_yy * grad_yy + grad_zz * grad_zz;
		const float prr = grad_xy * grad_xy + grad_xz * grad_xz + grad_yz * grad_yz + pow1 + pow1;
		const float visc_sps = smag * sqrt(prr);
		const float div_u = grad_xx + grad_yy + grad_zz;
		const float sps_k = (2.0f / 3.0f)*visc_sps*div_u;
		const float sps_blin = blin * prr;
		const float sumsps = -(sps_k + sps_blin);
		const float twovisc_sps = (visc_sps + visc_sps);
		float one_rho2 = 1.0f / velrhop[p1].w;
		//-Computes new values of tau[].
		const float tau_xx = one_rho2 * (twovisc_sps*grad_xx + sumsps);
		const float tau_xy = one_rho2 * (visc_sps   *grad_xy);
		tauff[p1 * 3] = make_float2(tau_xx, tau_xy);
		const float tau_xz = one_rho2 * (visc_sps   *grad_xz);
		const float tau_yy = one_rho2 * (twovisc_sps*grad_yy + sumsps);
		tauff[p1 * 3 + 1] = make_float2(tau_xz, tau_yy);
		const float tau_yz = one_rho2 * (visc_sps   *grad_yz);
		const float tau_zz = one_rho2 * (twovisc_sps*grad_zz + sumsps);
		tauff[p1 * 3 + 2] = make_float2(tau_yz, tau_zz);
	}
}

//==============================================================================
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.
//==============================================================================
void ComputeSpsTau(unsigned np, unsigned npb, float smag, float blin
	, const float4 *velrhop, const tsymatrix3f *gradvelg, tsymatrix3f *tau)
{
	const unsigned npf = np - npb;
	if (npf) {
		dim3 sgridf = cuSol::GetGridSize(npf, DIVBSIZE);
		KerComputeSpsTau << < sgridf, DIVBSIZE >> > (npf, npb, smag, blin, velrhop, (const float2*)gradvelg, (float2*)tau);
	}
}
//==============================================================================
/// Computes stress tensor rate for solid - Matthias 
//==============================================================================


__global__ void KerComputeJauTauDot(unsigned n, unsigned pini,tsymatrix3f *taudot, tsymatrix3f *JauTauc2_M, tsymatrix3f *JauGradvelc2_M, tsymatrix3f *JauOmega_M, tsymatrix3f AnisotropyG_M, float Mu)
{
	unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	if (p < n) {
		const tsymatrix3f tau = JauTauc2_M[p];
		const tsymatrix3f gradvel = JauGradvelc2_M[p];
		const tsymatrix3f omega = JauOmega_M[p];

		const tsymatrix3f E = {
			2.0f / 3.0f * gradvel.xx - 1.0f / 3.0f * gradvel.yy - 1.0f / 3.0f * gradvel.zz,
			gradvel.xy,
			gradvel.xz,
			2.0f / 3.0f * gradvel.yy - 1.0f / 3.0f * gradvel.xx - 1.0f / 3.0f * gradvel.zz,
			gradvel.yz,
			2.0f / 3.0f * gradvel.zz - 1.0f / 3.0f * gradvel.xx - 1.0f / 3.0f * gradvel.yy };

		taudot[p].xx = 2.0f*AnisotropyG_M.xx*Mu*E.xx + 2.0f*tau.xy*omega.xy + 2.0f*tau.xz*omega.xz;
		taudot[p].xy = 2.0f*AnisotropyG_M.xy*Mu*E.xy + (tau.yy - tau.xx)*omega.xy + tau.xz*omega.yz + tau.yz*omega.xz;
		taudot[p].xz = 2.0f*AnisotropyG_M.xz*Mu*E.xz + (tau.zz - tau.xx)*omega.xz - tau.xy*omega.yz + tau.yz*omega.xy;
		taudot[p].yy = 2.0f*AnisotropyG_M.yy*Mu*E.yy - 2.0f*tau.xy*omega.xy + 2.0f*tau.yz*omega.yz;
		taudot[p].yz = 2.0f*AnisotropyG_M.yz*Mu*E.yz + (tau.zz - tau.yy)*omega.yz - tau.xz*omega.xy - tau.xy*omega.xz;
		taudot[p].zz = 2.0f*AnisotropyG_M.zz*Mu*E.zz - 2.0f*tau.xz*omega.xz - 2.0f*tau.yz*omega.yz;

	}
}



void ComputeJauTauDot(unsigned np, unsigned npb, tsymatrix3f *taudot, tsymatrix3f *JauTau, tsymatrix3f *JauGradvel, tsymatrix3f *JauOmega, tsymatrix3f AnisotropyG_M, float Mu)
{
	const unsigned npf = np - npb;
	if (npf) {
		dim3 sgrid = cuSol::GetGridSize(npf, DIVBSIZE);
		KerComputeJauTauDot << <sgrid, DIVBSIZE >> > (npf, npb,taudot, JauTau, JauGradvel, JauOmega, AnisotropyG_M, Mu);
	}
}

//==============================================================================
/// Interaction for the force computation.
/// Interaccion para el calculo de fuerzas.
//==============================================================================
template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
(TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
	, unsigned np, unsigned npb, unsigned npbok, tuint3 ncells
	, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
	, const double2 *posxy, const double *posz, const float4 *pospress
	, const float4 *velrhop, const typecode *code, const unsigned *idp
	, const float *ftomassp, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f * omega
	, tsymatrix3f *JauTauc2_M, tsymatrix3f *JauGradvelc2_M, tsymatrix3f *taudot
	, tsymatrix3f *JauOmega_M, tsymatrix3f AnisotropyG_M, float Mu
	, float *viscdt, float* ar, float3 *ace, float *delta
	, TpShifting tshifting, float3 *shiftpos, float *shiftdetect
	, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto) {

	printf("1");
		//-Executes particle interactions.
		unsigned npf = np - npb;
		const int hdiv = (cellmode == CELLMODE_H ? 2 : 1);
		const int4 nc = make_int4(int(ncells.x), int(ncells.y), int(ncells.z), int(ncells.x*ncells.y));
		const unsigned cellfluid = nc.w*nc.z + 1;
		const int3 cellzero = make_int3(cellmin.x, cellmin.y, cellmin.z);
		//-Interaction Fluid-Fluid & Fluid-Bound.
		if (npf) {
			dim3 sgridf = cuSol::GetGridSize(npf, bsfluid);
			//cuSol::KerInteractionForcesFluid<psingle, tker, ftmode, lamsps, tdelta, shift> << <sgridf, bsfluid >> > (npf, npb, hdiv, nc, cellfluid, viscob, viscof, begincell, cellzero, dcell, ftomassp, (const float2*)tau, (float2*)gradvel, posxy, posz, pospress, velrhop, code, idp, viscdt, ar, ace, delta, tshifting, shiftpos, shiftdetect);
		}

		
		//-Interaction Boundary-Fluid.
		if (npbok) {
			dim3 sgridb = cuSol::GetGridSize(npbok, bsbound);
			//printf("bsbound:%u\n",bsbound);
		//	cuSol::KerInteractionForcesBound<psingle, tker, ftmode> << <sgridb, bsbound >> > (npbok, hdiv, nc, begincell, cellzero, dcell, ftomassp, posxy, posz, pospress, velrhop, code, idp, viscdt, ar);
		}
#ifndef DISABLE_BSMODES
	}
#endif

//==============================================================================
/// Surcharche solide de IntForceT - Matthias
//==============================================================================
template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
(TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
	, unsigned np, unsigned npb, unsigned npbok, tuint3 ncells
	, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
	, const double2 *posxy, const double *posz, const float4 *pospress
	, const float4 *velrhop, const typecode *code, const unsigned *idp
	, const float *ftomassp, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f * omega
	, tsymatrix3f *JauTau, tsymatrix3f *JauGradvel, tsymatrix3f *taudot
	, tsymatrix3f *JauOmega, tsymatrix3f AnisotropyG_M, float Mu
	, float *viscdt, float* ar, float3 *ace, float *delta
	, TpShifting tshifting, float3 *shiftpos, float *shiftdetect
	, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
	, const float *press, const float *pore, const float *mass) {
	const unsigned npf = np - npb;
	const int4 nc = make_int4(ncells.x, ncells.y, ncells.z, ncells.x*ncells.y);
	const unsigned cellfluid = nc.w*nc.z + 1;
	const int3 cellzero = make_int3(int(cellmin.x), int(cellmin.y), int(cellmin.z));
	const int hdiv = (cellmode == CELLMODE_H ? 2 : 1);

	if (npf) {
		dim3 sgridf = cuSol::GetGridSize(npf, bsfluid);
		cuSol::KerInteractionForcesSolMass <psingle, tker, ftmode, lamsps, tdelta, shift> << <sgridf, bsfluid >> > (npf, npb, hdiv, nc, cellfluid, viscob, viscof, begincell, cellzero, dcell, JauTau, JauGradvel, JauOmega, ftomassp, (const float2*)tau, (float2*)gradvel, posxy, posz, pospress, velrhop, code, idp, press, pore, mass, viscdt, ar, ace, delta, tshifting, shiftpos, shiftdetect);
	}
		// Compute Sdot
	ComputeJauTauDot(npf, npb, taudot, JauTau, JauGradvel, JauOmega, AnisotropyG_M, Mu);

		//-Interaction Boundary-Fluid.
	if (npbok) {
		dim3 sgridb = cuSol::GetGridSize(npbok, bsbound);
		//cuSol::KerInteractionForcesBound<psingle, tker, ftmode> << <sgridb, bsbound >> > (npbok, hdiv, nc, begincell, cellzero, dcell, ftomassp, posxy, posz, pospress, velrhop, code, idp, viscdt, ar);
	}
}

//==============================================================================
/// Surcharche solide de IntForceT - Matthias
//==============================================================================
template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
(TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
	, unsigned np, unsigned npb, unsigned npbok, tuint3 ncells
	, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
	, const double2 *posxy, const double *posz, const float4 *pospress
	, const float4 *velrhop, const typecode *code, const unsigned *idp
	, const float *ftomassp, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f * omega
	, tsymatrix3f *JauTauc2_M, tsymatrix3f *JauGradvelc2_M, tsymatrix3f *taudot
	, tsymatrix3f *JauOmega_M, tsymatrix3f AnisotropyG_M, float Mu
	, float *viscdt, float* ar, float3 *ace, float *delta
	, TpShifting tshifting, float3 *shiftpos, float *shiftdetect
	, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
	, const float *press, const float *pore) {
	printf("3");
	const unsigned npf = np - npb;
//	const int4 nc = make_int4(ncells.x, ncells.y, ncells.z, ncells.x*ncells.y);
	//const unsigned cellfluid = nc.w*nc.z + 1;
	//const int3 cellzero = make_int3(cellmin.x, cellmin.y, cellmin.z);
	//const int hdiv = (cellmode == CELLMODE_H ? 2 : 1);

	if (npf) {
		/*//-Interaction Fluid-Fluid.
		InteractionForcesSolid<psingle, tker, ftmode, lamsps, tdelta, shift>(npf, npb, nc, hdiv, cellfluid, Visco, begincell, cellzero, dcell, jautau, jaugradvel, jauomega, pos, pspos, velrhop, code, idp, press, pore, viscdt, ar, ace, delta, tshifting, shiftpos, shiftdetect);
		//-Interaction Fluid-Bound.
		InteractionForcesSolid<psingle, tker, ftmode, lamsps, tdelta, shift>(npf, npb, nc, hdiv, 0, Visco*ViscoBoundFactor, begincell, cellzero, dcell, jautau, jaugradvel, jauomega, pos, pspos, velrhop, code, idp, press, pore, viscdt, ar, ace, delta, tshifting, shiftpos, shiftdetect);
		*/
		dim3 sgridf = cuSol::GetGridSize(npf, bsfluid);
		//-Interaction Fluid-Fluid.
		//KerInteractionForcesSolMass <psingle, tker, ftmode, lamsps, tdelta, shift> << <sgridf, bsfluid >> > (npf, npb, nc, hdiv, cellfluid, Visco, begincell, cellzero, dcell, jautau, jaugradvel, jauomega, pos, pspos, velrhop, code, idp, press, pore, mass, viscdt, ar, ace, delta, tshifting, shiftpos, shiftdetect);
		//-Interaction Fluid-Bound.
		//KerInteractionForcesSolMass <psingle, tker, ftmode, lamsps, tdelta, shift> << <sgridf, bsfluid >> > (npf, npb, nc, hdiv, 0, Visco*ViscoBoundFactor, begincell, cellzero, dcell, jautau, jaugradvel, jauomega, pos, pspos, velrhop, code, idp, press, pore, mass, viscdt, ar, ace, delta, tshifting, shiftpos, shiftdetect);
	}

		//-Computes tau for Laminar+SPS.
		//if (lamsps)ComputeSpsTau(npf, npb, velrhop, spsgradvel, spstau);

		// Compute Sdot
		//ComputeJauTauDot(npf, npb, gradvel, tau, taudot, omega, velrhop, JauTauc2_M, JauGradvelc2_M, JauOmega_M, AnisotropyG_M, Mu);
	
		//-Interaction Boundary-Fluid.
	if (npbok) {
		dim3 sgridb = cuSol::GetGridSize(npbok, bsbound);
		//printf("bsbound:%u\n",bsbound);
		//cuSol::KerInteractionForcesBound<psingle, tker, ftmode> << <sgridb, bsbound >> > (npbok, hdiv, nc, begincell, cellzero, dcell, ftomassp, posxy, posz, pospress, velrhop, code, idp, viscdt, ar);
	}
}

//==============================================================================
/// Surcharche solide de IntForceT - Matthias
//==============================================================================
template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
(TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
	, unsigned np, unsigned npb, unsigned npbok, tuint3 ncells
	, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
	, const double2 *posxy, const double *posz, const float4 *pospress
	, const float4 *velrhop, const typecode *code, const unsigned *idp
	, const float *ftomassp, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f * omega
	, tsymatrix3f *JauTau, tsymatrix3f *JauGradvel, tsymatrix3f *taudot
	, tsymatrix3f *JauOmega, tsymatrix3f AnisotropyG_M, float Mu
	, float *viscdt, float* ar, float3 *ace, float *delta
	, TpShifting tshifting, float3 *shiftpos, float *shiftdetect
	, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
	, const float3 *press, const float *pore, const float *mass) {
	const unsigned npf = np - npb;
	const int4 nc = make_int4(ncells.x, ncells.y, ncells.z, ncells.x*ncells.y);
	const unsigned cellfluid = nc.w*nc.z + 1;
	const int3 cellzero = make_int3(cellmin.x, cellmin.y, cellmin.z);
	const int hdiv = (cellmode == CELLMODE_H ? 2 : 1);

	if (npf) {
		dim3 sgridf = cuSol::GetGridSize(npf, bsfluid);
		//-Interaction Fluid-Fluid.
		cuSol::KerInteractionForcesSolMass <psingle, tker, ftmode, lamsps, tdelta, shift> << <sgridf, bsfluid >> > (npf, npb, hdiv, nc, cellfluid, 0, viscof, begincell, cellzero,dcell, JauTau, JauGradvel, JauOmega, ftomassp, (const float2*)tau, (float2*)gradvel,posxy, posz, pospress, velrhop, code, idp, press, pore, mass, viscdt, ar, ace, delta, tshifting, shiftpos,shiftdetect);
		//-Interaction Fluid-Bound.
		cuSol::KerInteractionForcesSolMass <psingle, tker, ftmode, lamsps, tdelta, shift> << <sgridf, bsfluid >> > (npf, npb, hdiv, nc, 0, viscob, 0, begincell, cellzero, dcell, JauTau, JauGradvel, JauOmega, ftomassp, (const float2*)tau, (float2*)gradvel, posxy, posz, pospress, velrhop, code, idp, press, pore, mass, viscdt, ar, ace, delta, tshifting, shiftpos, shiftdetect);
	}
		//-Computes tau for Laminar+SPS.
		//if (lamsps)ComputeSpsTau(npf, npb, velrhop, spsgradvel, spstau);

		// Compute Sdot
	    ComputeJauTauDot(npf, npb,taudot,JauTau, JauGradvel, JauOmega ,AnisotropyG_M, Mu);
	
		//-Interaction Boundary-Fluid.
	if (npbok) {
		dim3 sgridb = cuSol::GetGridSize(npbok, bsbound);
		cuSol::KerInteractionForcesBound<psingle, tker, ftmode> << <sgridb, bsbound >> > (npbok, hdiv, nc, begincell, cellzero, dcell, ftomassp, posxy, posz, pospress, velrhop, code, idp, viscdt, ar);
	}
}


//==============================================================================
/// Selection of template parameters for Interaction_ForcesX.
/// Seleccion de parametros template para Interaction_ForcesX.
//==============================================================================
void Interaction_Forces_M(TpKernel TKernel, bool WithFloating, TpShifting TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM
	, TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
	, unsigned np, unsigned npb, unsigned npbok
	, const tuint3 ncells, const int2 *begincell, tuint3 cellmin, const unsigned *dcell
	, const double2 *posxy,  double *posz,  float4 *pospress
	,  float4 *velrhop,  unsigned *idp,  typecode *code
	, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f *omega
	, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
	, float* viscdt, float* ar, float3 *ace, float *delta
	, float3 *shiftpos, float *shiftdetect
	, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
	,  float3 *press,  float *pore,  float *mass, tsymatrix3f AnisotropyG, float Mu,  float *ftomassp)
{
	const bool psingle = false;
	if (TKernel == KERNEL_Wendland) {
		const TpKernel tker = KERNEL_Wendland;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt) Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Gaussian) {
		const TpKernel tker = KERNEL_Gaussian;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp, ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Cubic) {
		const TpKernel tker = KERNEL_Cubic;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
	}
}

//==============================================================================
/// Selection of template parameters for Interaction_ForcesX.
/// Seleccion de parametros template para Interaction_ForcesX.
//==============================================================================
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
	, float *press, float *pore, float *mass, tsymatrix3f AnisotropyG, float Mu, float *ftomassp)
{
	const bool psingle = false;
	if (TKernel == KERNEL_Wendland) {
		const TpKernel tker = KERNEL_Wendland;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Gaussian) {
		const TpKernel tker = KERNEL_Gaussian;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Cubic) {
		const TpKernel tker = KERNEL_Cubic;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore, mass);
				}
			}
		}
	}
}

//==============================================================================
/// Selection of template parameters for Interaction_ForcesX.
/// Seleccion de parametros template para Interaction_ForcesX.
//==============================================================================
void Interaction_Forces_M(TpKernel TKernel, bool WithFloating, TpShifting TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM
	, TpCellMode cellmode, float viscob, float viscof, unsigned bsbound, unsigned bsfluid
	, unsigned np, unsigned npb, unsigned npbok
	, tuint3 ncells, int2 *begincell, tuint3 cellmin, const unsigned *dcell
	, const double2 *posxy, const double *posz, const float4 *pospress
	, const float4 *velrhop, const unsigned *idp, const typecode *code
	, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f *omega
	, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
	, float* viscdt, float* ar, float3 *ace, float *delta
	, float3 *shiftpos, float *shiftdetect
	, bool simulate2d, StKerInfo *kerinfo, JBlockSizeAuto *bsauto
	, const float *press, const float *pore, tsymatrix3f AnisotropyG, float Mu, const float *ftomassp)
{
	const bool psingle = false;
	if (TKernel == KERNEL_Wendland) {
		const TpKernel tker = KERNEL_Wendland;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Gaussian) {
		const TpKernel tker = KERNEL_Gaussian;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Cubic) {
		const TpKernel tker = KERNEL_Cubic;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(cellmode, viscob, viscof, bsbound, bsfluid, np, npb, npbok, ncells, begincell, cellmin, dcell, posxy, posz, pospress, velrhop, code, idp,ftomassp, tau, gradvel, omega, jautau, jaugradvel, jautaudot, jauomega, AnisotropyG, Mu, viscdt, ar, ace, delta, TShifting, shiftpos, shiftdetect, simulate2d, kerinfo, bsauto, press, pore);
				}
			}
		}
	}
}




/*
//==============================================================================
/// Selection of template parameters for Interaction_ForcesX.
/// Seleccion de parametros template para Interaction_ForcesX.
//==============================================================================
void Interaction_Forces(TpKernel TKernel, bool WithFloating, bool TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM, unsigned np, unsigned npb, unsigned npbok
	, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
	, const tdouble3 *pos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
	, const float *press
	, float &viscdt, float* ar, tfloat3 *ace, float *delta
	, tsymatrix3f *spstau, tsymatrix3f *spsgradvel
	, tfloat3 *shiftpos, float *shiftdetect)
{
	float3 *pspos = NULL;
	const bool psingle = false;
	if (TKernel == KERNEL_Wendland) {
		const TpKernel tker = KERNEL_Wendland;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Gaussian) {
		const TpKernel tker = KERNEL_Gaussian;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Cubic) {
		const TpKernel tker = KERNEL_Cubic;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
	}
}

//==============================================================================
/// Selection of template parameters for Interaction_ForcesX.
/// Seleccion de parametros template para Interaction_ForcesX.
//==============================================================================
void InteractionSimple_Forces(TpKernel TKernel, bool WithFloating, bool TShifting, TpVisco TVisco, TpDeltaSph TDeltaSph, bool UseDEM, unsigned np, unsigned npb, unsigned npbok
	, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
	, const tfloat3 *pspos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
	, const float *press
	, float &viscdt, float* ar, tfloat3 *ace, float *delta
	, tsymatrix3f *spstau, tsymatrix3f *spsgradvel
	, tfloat3 *shiftpos, float *shiftdetect)
{
	double3 *pos = NULL;
	const bool psingle = true;
	if (TKernel == KERNEL_Wendland) {
		const TpKernel tker = KERNEL_Wendland;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Gaussian) {
		const TpKernel tker = KERNEL_Gaussian;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
	}
	else if (TKernel == KERNEL_Cubic) {
		const TpKernel tker = KERNEL_Cubic;
		if (!WithFloating) {
			const TpFtMode ftmode = FTMODE_None;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else if (!UseDEM) {
			const TpFtMode ftmode = FTMODE_Sph;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
		else {
			const TpFtMode ftmode = FTMODE_Dem;
			if (TShifting) {
				const bool tshift = true;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
			else {
				const bool tshift = false;
				if (TVisco == VISCO_LaminarSPS) {
					const bool lamsps = true;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
				else {
					const bool lamsps = false;
					if (TDeltaSph == DELTA_None)      Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_None, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_Dynamic)   Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_Dynamic, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
					if (TDeltaSph == DELTA_DynamicExt)Interaction_ForcesT<psingle, tker, ftmode, lamsps, DELTA_DynamicExt, tshift>(np, npb, npbok, ncells, begincell, cellmin, dcell, pos, pspos, velrhop, code, idp, press, viscdt, ar, ace, delta, spstau, spsgradvel, TShifting, shiftpos, shiftdetect);
				}
			}
		}
	}
}
*/


//------------------------------------------------------------------------------
/// Calculations for Press + porec //Lucas
//------------------------------------------------------------------------------
__global__ void KerPressPoreC_L(
	unsigned n,const float4 *velrhop,const float RhopZero,float  *Pressg
	,tfloat3 Anisotropy,float CteB , float Gamma,float3 *Press3Dc
	,double2 *posxy,double *posz,tdouble3 LocDiv_M,float PoreZero,float Spread_M, float *Porec_M)
{
	unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p<n) {
		
		const float rhop = velrhop[p].w, rhop_r0 = rhop / RhopZero;
		 Pressg[p] = CteB * (pow(rhop_r0, Gamma) - 1.0f);
		 Press3Dc[p].x = Anisotropy.x * (CteB * (pow(rhop_r0, Gamma) - 1.0f));
		 Press3Dc[p].y = Anisotropy.y * (CteB * (pow(rhop_r0, Gamma) - 1.0f));
		 Press3Dc[p].z = Anisotropy.z * (CteB * (pow(rhop_r0, Gamma) - 1.0f));

		 float distance2x = posxy[p].x - LocDiv_M.x;
		 float distance2y = posxy[p].y - LocDiv_M.y;
		 float distance2z = posz[p] - LocDiv_M.z;
		 Porec_M[p] = PoreZero;
		//Porec_M[p] = PoreZero / (1 + exp(-(TimeStep-2))) * exp(-(pow(distance2x,2) + pow(distance2.y, 2) + pow(distance2.z, 2)) / Spread_M);
	//Porec_M[p] = PoreZero / sqrt(2 * Spread_M*PI) * exp(-(pow(distance2x, 2) + pow(distance2y, 2) + pow(distance2z, 2)) / Spread_M);
	  //  Porec_M[p] = PoreZero  / sqrt(2*Spread_M*PI) * exp(-(pow(distance2x,2)) / Spread_M);
	}
}

//==============================================================================
/// Calculations for Press + porec //Lucas
//==============================================================================
void PressPoreC_L(unsigned np, const float4 *velrhop, const float RhopZero, float  *Pressg
, tfloat3 Anisotropy, float CteB, float Gamma, float3 *Press3Dc
, double2 *posxy, double *posz, tdouble3 LocDiv_M, float PoreZero, float Spread_M,float *Porec_M)
{
	if (np) {
		dim3 sgrid = cuSol::GetGridSize(np, SPHBSIZE);
		KerPressPoreC_L << <sgrid, SPHBSIZE >> > (np, velrhop, RhopZero, Pressg, Anisotropy, CteB, Gamma, Press3Dc, posxy, posz, LocDiv_M,PoreZero, Spread_M,Porec_M);

	}
}

void ComputeVelrhopBound(unsigned n, const float4* velrhopold, double armul, float4* velrhopnew, const float* Arg, float RhopZero)
{
	if (n) {
		dim3 sgrid = cuSol::GetGridSize(n, SPHBSIZE);
		KerComputeVelrhopBound << <sgrid, SPHBSIZE >> > (n, velrhopold, armul, velrhopnew, Arg, RhopZero);
	}

}
__global__ void KerComputeVelrhopBound(unsigned n,const float4* velrhopold, double armul, float4* velrhopnew,const float* Arg,float RhopZero)
{
unsigned p = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if (p < n) {
	const float rhopnew = float(double(velrhopold[p].w) + armul * Arg[p]);
	velrhopnew[p].x = 0;
	velrhopnew[p].y = 0;
	velrhopnew[p].z = 0;
	velrhopnew[p].w = (rhopnew < RhopZero ? RhopZero : rhopnew);
}
}

}
