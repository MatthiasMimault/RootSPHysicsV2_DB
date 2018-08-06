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

/// \file JSphSolidGpu.cpp \brief Implements the class \ref JSphSolidGpu.

#include "JSphSolidGpu_L.h"
#include "JSphGpu_ker.h"
#include "JCellDivGpu.h"
#include "JPartFloatBi4.h"
#include "Functions.h"
#include "JSphMotion.h"
#include "JArraysGpu.h"
#include "JSphDtFixed.h"
#include "JWaveGen.h"
#include "JDamping.h"
#include "JXml.h"
#include "JSaveDt.h"
#include "JTimeOut.h"
#include "JSphAccInput.h"
#include "JGaugeSystem.h"
#include "TypesDef.h"
#include "JSphGpu.h"
#include "FunctionsCuda.h"

#include <climits>

using namespace std;

/*

//==============================================================================
/// Constructor.
//==============================================================================

JSphSolidGpu::JSphSolidGpu(bool withmpi) :JSph(true, withmpi) {
	ClassName = "JSphSolidGpu";
	CellDiv = NULL;
	ArraysGpu = new JArraysGpu;
	InitVars();
	TmgCreation(Timers, false);
}

//==============================================================================
/// Destructor.
//==============================================================================

JSphSolidGpu::~JSphSolidGpu() {
	DestructorActive = true;
	FreeCpuMemoryParticles();
	FreeCpuMemoryFixed();
	delete ArraysGpu;
	TmgDestruction(Timers);
}
//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSphSolidGpu::InitVars() {
	RunMode = "";

	Np = Npb = NpbOk = 0;
	NpbPer = NpfPer = 0;
	WithFloating = false;

	Idpc = NULL; Codec = NULL; Dcellc = NULL; Posc = NULL; Velrhopg = NULL;
	VelrhopM1c = NULL;                //-Verlet
	PosPrec = NULL; VelrhopPrec = NULL; //-Symplectic
	PsPosg = NULL;                    //-Interaccion Pos-Single.
	SpsTaug = NULL; SpsGradvelg = NULL; //-Laminar+SPS. 
										// Matthias
										//JauTauc_M = NULL; JauGradvelc_M = NULL; // Jaumann Solid
	JauTauc2_M = NULL; JauGradvelc2_M = NULL; // Jaumann Solid
	JauTauM1c2_M = NULL;
	MassM1c_M = NULL;
	JauTauDot_M = NULL; JauOmega_M = NULL; // Jaumann Solid

	Arg = NULL; Aceg = NULL; Deltag = NULL;
	ShiftPosg = NULL; ShiftDetectg = NULL; //-Shifting.
	Pressg = NULL;
	Press3Dg = NULL;
	// Matthias
	Porec_M = NULL;
	Massc_M = NULL;
	Divisionc_M = NULL;
	//Amassc_M = NULL;
	//Voluc_M = NULL;

	RidpMove = NULL;
	FtRidp = NULL;
	FtoForces = NULL;
	FtoForcesRes = NULL;
	FreeCpuMemoryParticles();
	FreeCpuMemoryFixed();
}
//==============================================================================
/// Deallocate fixed memory on CPU for moving and floating bodies.
/// Libera memoria fija en cpu para moving y floating.
//==============================================================================
void JSphSolidGpu::FreeCpuMemoryFixed() {
	MemGpuFixed = 0;
	delete[] RidpMove;     RidpMove = NULL;
	delete[] FtRidp;       FtRidp = NULL;
	delete[] FtoForces;    FtoForces = NULL;
	delete[] FtoForcesRes; FtoForcesRes = NULL;
}

//==============================================================================
/// Allocates memory for arrays with fixed size (motion and floating bodies).
//==============================================================================
void JSphSolidGpu::AllocCpuMemoryFixed() {
	MemGpuFixed = 0;
	try {
		//-Allocates memory for moving objects.
		if (CaseNmoving) {
			RidpMove = new unsigned[CaseNmoving];  MemGpuFixed += (sizeof(unsigned)*CaseNmoving);
		}
		//-Allocates memory for floating bodies.
		if (CaseNfloat) {
			FtRidp = new unsigned[CaseNfloat];     MemGpuFixed += (sizeof(unsigned)*CaseNfloat);
			FtoForces = new StFtoForces[FtCount];     MemGpuFixed += (sizeof(StFtoForces)*FtCount);
			FtoForcesRes = new StFtoForcesRes[FtCount];  MemGpuFixed += (sizeof(StFtoForcesRes)*FtCount);
		}
	}
	catch (const std::bad_alloc) {
		RunException("AllocMemoryFixed", "Could not allocate the requested memory.");
	}
}

//==============================================================================
/// Deallocate memory in CPU for particles.
/// Libera memoria en cpu para particulas.
//==============================================================================
void JSphSolidGpu::FreeCpuMemoryParticles() {
	GpuParticlesSize = 0;
	MemGpuParticles = 0;
	ArraysGpu->Reset();
}

//==============================================================================
/// Allocte memory on CPU for the particles. 
/// Reserva memoria en Cpu para las particulas. 
//==============================================================================

void JSphSolidGpu::AllocCpuMemoryParticles(unsigned np, float over) {
	const char* met = "AllocCpuMemoryParticles";
	FreeCpuMemoryParticles();
	//-Calculate number of partices with reserved memory | Calcula numero de particulas para las que se reserva memoria.
	const unsigned np2 = (over>0 ? unsigned(over*np) : np);
	GpuParticlesSize = np2 + PARTICLES_OVERMEMORY_MIN;
	//-Define number or arrays to use. | Establece numero de arrays a usar.
	ArraysGpu->SetArraySize(GpuParticlesSize);
#ifdef CODE_SIZE4
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_4B, 2);  //-code,code2
#else
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_2B, 2);  //-code,code2
#endif
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_4B, 5);  //-idp,ar,viscdt,dcell,prrhop
	if (TDeltaSph == DELTA_DynamicExt) {
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_4B, 1);
	}  //-delta
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_12B, 1); //-ace
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_16B, 1); //-velrhop
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_24B, 2); //-pos
	if (Psingle) {
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_12B, 1); //-pspos
	}
	if (TStep == STEP_Verlet) {
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_16B, 1); //-velrhopm1
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_24B, 1); //-JauTauM12
	}
	else if (TStep == STEP_Symplectic) {
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_24B, 1); //-pospre
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_16B, 1); //-velrhoppre
	}
	if (TVisco == VISCO_LaminarSPS) {
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_24B, 1); //-SpsTau,SpsGradvel
	}
	if (TShifting != SHIFT_None) {
		ArraysGpu->AddArrayCount(JArraysGpu::SIZE_12B, 1); //-shiftpos
	}

	// Matthias
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_1B, 1);  //division
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_4B, 1); // Pore
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_4B, 1); // Mass
													  //ArraysGpu->AddArrayCount(JArraysGpu::SIZE_36B, 1); //-JauGradvel, JauTau
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_24B, 3); //-JauGradvel, JauTau2, Omega and Taudot
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_4B, 4); // SaveFields
	ArraysGpu->AddArrayCount(JArraysGpu::SIZE_12B, 1); // Press3D

													   //-Shows the allocated memory.
	MemGpuParticles = ArraysGpu->GetAllocMemoryGpu();
	PrintSizeNp(GpuParticlesSize, MemGpuParticles);
}
/*
//==============================================================================
/// Resizes space in CPU memory for particles.
//==============================================================================
void JSphSolidGpu::ResizeCpuMemoryParticles(unsigned npnew) {
	npnew = npnew + PARTICLES_OVERMEMORY_MIN;
	//-Saves current data from CPU.
	unsigned    *idp = SaveArrayCpu(Np, Idpc);
	typecode    *code = SaveArrayCpu(Np, Codec);
	unsigned    *dcell = SaveArrayCpu(Np, Dcellc);
	tdouble3    *pos = SaveArrayCpu(Np, Posc);
	tfloat4     *velrhop = SaveArrayCpu(Np, Velrhopg);
	tfloat4     *velrhopm1 = SaveArrayCpu(Np, VelrhopM1c);
	tdouble3    *pospre = SaveArrayCpu(Np, PosPrec);
	tfloat4     *velrhoppre = SaveArrayCpu(Np, VelrhopPrec);
	tsymatrix3f *spstau = SaveArrayCpu(Np, SpsTauc);
	// Matthias
	bool		  *division = SaveArrayCpu(Np, Divisionc_M);
	float		  *pore = SaveArrayCpu(Np, Porec_M);
	float		  *mass = SaveArrayCpu(Np, Massc_M);
	float		  *massm1 = SaveArrayCpu(Np, MassM1c_M);
	//float		  *volu = SaveArrayCpu(Np, Voluc_M);	
	//tmatrix3f *jautau = SaveArrayCpu(Np, JauTauc_M);
	tsymatrix3f *jautau2 = SaveArrayCpu(Np, JauTauc2_M);
	tsymatrix3f *jautaum12 = SaveArrayCpu(Np, JauTauM1c2_M);

	//-Frees pointers.
	ArraysGpu->Free(Idpc);
	ArraysGpu->Free(Codec);
	ArraysGpu->Free(Dcellc);
	ArraysGpu->Free(Posc);
	ArraysGpu->Free(Velrhopc);
	ArraysGpu->Free(VelrhopM1c);
	ArraysGpu->Free(PosPrec);
	ArraysGpu->Free(VelrhopPrec);
	ArraysGpu->Free(SpsTauc);
	// Matthias
	ArraysGpu->Free(Divisionc_M);
	ArraysGpu->Free(Porec_M);
	ArraysGpu->Free(Massc_M);
	ArraysGpu->Free(MassM1c_M);
	//ArraysGpu->Free(Voluc_M);
	//ArraysGpu->Free(JauTauc_M);
	ArraysGpu->Free(JauTauc2_M);
	ArraysGpu->Free(JauTauM1c2_M);

	//-Resizes CPU memory allocation.
	const double mbparticle = (double(MemGpuParticles) / (1024 * 1024)) / GpuParticlesSize; //-MB por particula.
	Log->Printf("**JSphSolidGpu: Requesting cpu memory for %u particles: %.1f MB.", npnew, mbparticle*npnew);
	ArraysGpu->SetArraySize(npnew);

	//-Reserve pointers.
	Idpc = ArraysGpu->ReserveUint();
	Codec = ArraysGpu->ReserveTypeCode();
	Dcellc = ArraysGpu->ReserveUint();
	Posc = ArraysGpu->ReserveDouble3();
	Velrhopc = ArraysGpu->ReserveFloat4();
	if (velrhopm1) VelrhopM1c = ArraysGpu->ReserveFloat4();
	if (pospre)    PosPrec = ArraysGpu->ReserveDouble3();
	if (velrhoppre)VelrhopPrec = ArraysGpu->ReserveFloat4();
	if (spstau)    SpsTauc = ArraysGpu->ReserveSymatrix3f();
	// Matthias
	Divisionc_M = ArraysGpu->ReserveBool();
	Porec_M = ArraysGpu->ReserveFloat();
	Massc_M = ArraysGpu->ReserveFloat();
	MassM1c_M = ArraysGpu->ReserveFloat();
	//Voluc_M = ArraysGpu->ReserveFloat();
	//JauTauc_M = ArraysGpu->ReserveMatrix3f_M();
	JauTauc2_M = ArraysGpu->ReserveSymatrix3f();
	if (velrhopm1) JauTauM1c2_M = ArraysGpu->ReserveSymatrix3f();

	//-Restore data in CPU memory.
	RestoreArrayCpu(Np, idp, Idpc);
	RestoreArrayCpu(Np, code, Codec);
	RestoreArrayCpu(Np, dcell, Dcellc);
	RestoreArrayCpu(Np, pos, Posc);
	RestoreArrayCpu(Np, velrhop, Velrhopc);
	RestoreArrayCpu(Np, velrhopm1, VelrhopM1c);
	RestoreArrayCpu(Np, pospre, PosPrec);
	RestoreArrayCpu(Np, velrhoppre, VelrhopPrec);
	RestoreArrayCpu(Np, spstau, SpsTauc);
	// Matthias
	RestoreArrayCpu(Np, division, Divisionc_M);
	RestoreArrayCpu(Np, pore, Porec_M);
	RestoreArrayCpu(Np, mass, Massc_M);
	RestoreArrayCpu(Np, massm1, MassM1c_M);
	//RestoreArrayCpu(Np, volu, Porec_M);
	//RestoreArrayCpu(Np, jautau, JauTauc_M);
	RestoreArrayCpu(Np, jautau2, JauTauc2_M);
	RestoreArrayCpu(Np, jautaum12, JauTauM1c2_M);
	//-Updates values.
	GpuParticlesSize = npnew;
	MemGpuParticles = ArraysGpu->GetAllocMemoryGpu();
}
*/
//==============================================================================
/// Saves a CPU array in CPU memory. 
//==============================================================================
/*
template<class T> T* JSphSolidGpu::TSaveArrayCpu(unsigned np, const T *datasrc)const {
	T *data = NULL;
	if (datasrc) {
		try {
			data = new T[np];
		}
		catch (const std::bad_alloc) {
			RunException("TSaveArrayCpu", "Could not allocate the requested memory.");
		}
		memcpy(data, datasrc, sizeof(T)*np);
	}
	return(data);
}

//==============================================================================
/// Restores an array (generic) from CPU memory. 
//==============================================================================
template<class T> void JSphSolidGpu::TRestoreArrayCpu(unsigned np, T *data, T *datanew)const {
	if (data&&datanew)memcpy(datanew, data, sizeof(T)*np);
	delete[] data;
}

//==============================================================================
/// Arrays for basic particle data. 
/// Arrays para datos basicos de las particulas. 
//==============================================================================
void JSphSolidGpu::ReserveBasicArraysCpu() {
	Idpc = ArraysGpu->ReserveUint();
	Codec = ArraysGpu->ReserveTypeCode();
	Dcellc = ArraysGpu->ReserveUint();
	Posc = ArraysGpu->ReserveDouble3();
	Velrhopg = ArraysGpu->ReserveFloat4();
	if (TStep == STEP_Verlet) {
		VelrhopM1c = ArraysGpu->ReserveFloat4();
		MassM1c_M = ArraysGpu->ReserveFloat();
		JauTauM1c2_M = ArraysGpu->ReserveSymatrix3f();
	}
	if (TVisco == VISCO_LaminarSPS)SpsTaug = ArraysGpu->ReserveSymatrix3f();

	// Matthias
	Divisionc_M = ArraysGpu->ReserveBool();
	Porec_M = ArraysGpu->ReserveFloat();
	Massc_M = ArraysGpu->ReserveFloat();
	//JauTauc_M = ArraysGpu->ReserveMatrix3f_M();
	JauTauc2_M = ArraysGpu->ReserveSymatrix3f();
}

//==============================================================================
/// Return memory reserved on CPU.
/// Devuelve la memoria reservada en cpu.
//==============================================================================
llong JSphSolidGpu::GetAllocMemoryCpu()const {
	llong s = JSph::GetAllocMemoryCpu();
	//-Reserved in AllocCpuMemoryParticles().
	s += MemGpuParticles;
	//-Reserved in AllocCpuMemoryFixed().
	s += MemGpuFixed;
	//-Reserved in other objects.
	return(s);
}

//==============================================================================
/// Visualize the reserved memory.
/// Visualiza la memoria reservada.
//==============================================================================
void JSphSolidGpu::PrintAllocMemory(llong mgpu)const {
	Log->Printf("Allocated memory in GPU: %lld (%.2f MB)", mgpu, double(mgpu) / (1024 * 1024));
}

//==============================================================================
/// Collect data from a range of particles and return the number of particles that 
/// will be less than n and eliminate the periodic ones
/// - cellorderdecode: Reorder components of position (pos) and velocity (vel) according to CellOrder.
/// - onlynormal: Only keep the normal ones and eliminate the periodic particles.
///
/// Recupera datos de un rango de particulas y devuelve el numero de particulas que
/// sera menor que n si se eliminaron las periodicas.
/// - cellorderdecode: Reordena componentes de pos y vel segun CellOrder.
/// - onlynormal: Solo se queda con las normales, elimina las particulas periodicas.
//==============================================================================
unsigned JSphSolidGpu::GetParticlesData(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
	, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, typecode *code)
{
	const char met[] = "GetParticlesData";
	unsigned num = n;
	//-Copy selected values.
	if (code)cudaMemcpy(code, Codec + pini, sizeof(typecode)*n, cudaMemcpyHostToDevice);
	if (idp)cudaMemcpy(idp, Idpc + pini, sizeof(unsigned)*n, cudaMemcpyHostToDevice);
	if (pos)cudaMemcpy(pos, Posc + pini, sizeof(tdouble3)*n, cudaMemcpyHostToDevice);
	if (vel && rhop) {
		for (unsigned p = 0; p<n; p++) {
			float4 vr = Velrhopg[p + pini];
			vel[p] = TFloat3(vr.x, vr.y, vr.z);
			rhop[p] = vr.w;
		}
	}
	else {
		if (vel) for (unsigned p = 0; p<n; p++) { float4 vr = Velrhopg[p + pini]; vel[p] = TFloat3(vr.x, vr.y, vr.z); }
		if (rhop)for (unsigned p = 0; p<n; p++)rhop[p] = Velrhopg[p + pini].w;
	}
	//-Eliminate non-normal particles (periodic & others). | Elimina particulas no normales (periodicas y otras).
	if (onlynormal) {
		if (!idp || !pos || !vel || !rhop)RunException(met, "Pointers without data.");
		typecode *code2 = code;
		if (!code2) {
			code2 = ArraysGpu->ReserveTypeCode();
			memcpy(code2, Codec + pini, sizeof(typecode)*n);
		}
		unsigned ndel = 0;
		for (unsigned p = 0; p<n; p++) {
			bool normal = CODE_IsNormal(code2[p]);
			if (ndel && normal) {
				const unsigned pdel = p - ndel;
				idp[pdel] = idp[p];
				pos[pdel] = pos[p];
				vel[pdel] = vel[p];
				rhop[pdel] = rhop[p];
				code2[pdel] = code2[p];
			}
			if (!normal)ndel++;
		}
		num -= ndel;
		if (!code)ArraysGpu->Free(code2);
	}
	//-Reorder components in their original order. | Reordena componentes en su orden original.
	if (cellorderdecode)DecodeCellOrder(n, pos, vel);
	return(num);
}

//==============================================================================
/// Collect data from a range of particles and return the number of particles that 
/// will be less than n and eliminate the periodic ones
/// - cellorderdecode: Reorder components of position (pos) and velocity (vel) according to CellOrder.
/// - onlynormal: Only keep the normal ones and eliminate the periodic particles.
///
/// Recupera datos de un rango de particulas y devuelve el numero de particulas que
/// sera menor que n si se eliminaron las periodicas.
/// - cellorderdecode: Reordena componentes de pos y vel segun CellOrder.
/// - onlynormal: Solo se queda con las normales, elimina las particulas periodicas.
//==============================================================================

unsigned JSphSolidGpu::GetParticlesData_M(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
	, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, float *pore, tfloat3 *press, float* mass, tsymatrix3f *tau, typecode *code)
{
	const char met[] = "GetParticlesData";
	unsigned num = n;
	//-Copy selected values.
	if (code)memcpy(code, Codec + pini, sizeof(typecode)*n);
	if (idp)memcpy(idp, Idpc + pini, sizeof(unsigned)*n);
	if (pos)memcpy(pos, Posc + pini, sizeof(tdouble3)*n);
	if (vel && rhop) {
		for (unsigned p = 0; p<n; p++) {
			float4 vr = Velrhopg[p + pini];
			vel[p] = TFloat3(vr.x, vr.y, vr.z);
			rhop[p] = vr.w;
		}
	}
	else {
		if (vel) for (unsigned p = 0; p<n; p++) { float4 vr = Velrhopg[p + pini]; vel[p] = TFloat3(vr.x, vr.y, vr.z); }
		if (rhop)for (unsigned p = 0; p<n; p++)rhop[p] = Velrhopg[p + pini].w;
	}

	// Matthias
	if (pore)memcpy(pore, Porec_M + pini, sizeof(float)*n);
	if (press) {
		for (unsigned p = 0; p<n; p++) {
			tfloat3 pre = AnisotropyK_M * TFloat3(CteB * (pow(Velrhopg[p + pini].w / RhopZero, Gamma) - 1.0f));
			press[p] = TFloat3(pre.x, pre.y, pre.z);
		}
	}
	//if (press)memcpy(press, Press3Dc + pini, sizeof(tfloat3)*n); // Not used, but Pressure seems to be recorded anyway, as well for tau
	if (mass)memcpy(mass, Massc_M + pini, sizeof(float)*n);
	if (tau)memcpy(tau, JauTauc2_M + pini, sizeof(tsymatrix3f)*n);

	//-Eliminate non-normal particles (periodic & others). | Elimina particulas no normales (periodicas y otras).
	if (onlynormal) {
		if (!idp || !pos || !vel || !rhop)RunException(met, "Pointers without data.");
		typecode *code2 = code;
		if (!code2) {
			code2 = ArraysGpu->ReserveTypeCode();
			memcpy(code2, Codec + pini, sizeof(typecode)*n);
		}
		unsigned ndel = 0;
		for (unsigned p = 0; p<n; p++) {
			bool normal = CODE_IsNormal(code2[p]);
			if (ndel && normal) {
				const unsigned pdel = p - ndel;
				idp[pdel] = idp[p];
				pos[pdel] = pos[p];
				vel[pdel] = vel[p];
				rhop[pdel] = rhop[p];
				code2[pdel] = code2[p];
			}
			if (!normal)ndel++;
		}
		num -= ndel;
		if (!code)ArraysGpu->Free(code2);
	}
	//-Reorder components in their original order. | Reordena componentes en su orden original.
	if (cellorderdecode)DecodeCellOrder(n, pos, vel);
	return(num);
}

//==============================================================================
/// Initialisation of arrays and variables for execution.
/// Inicializa vectores y variables para la ejecucion.
//==============================================================================
void JSphSolidGpu::InitRun() {
	const char met[] = "InitRun";
	WithFloating = (CaseNfloat>0);
	if (TStep == STEP_Verlet) {
		cudaMemcpy(VelrhopM1c, Velrhopg, sizeof(tfloat4)*Np, cudaMemcpyDeviceToDevice);
		cudaMemset(JauTauM1c2_M, 0, sizeof(tsymatrix3f)*Np);
		VerletStep = 0;
	}
	else if (TStep == STEP_Symplectic)DtPre = DtIni;
	if (TVisco == VISCO_LaminarSPS)cudaMemset(SpsTaug, 0, sizeof(tsymatrix3f)*Np);

	// Matthias
	//memset(JauTauc_M, 0, sizeof(tmatrix3f)*Np);
	cudaMemset(JauTauc2_M, 0, sizeof(tsymatrix3f)*Np);
	cudaMemset(Divisionc_M, 0, sizeof(bool)*Np);
	for (unsigned p = 0; p < Np; p++) {
		Massc_M[p] = MassFluid;
		MassM1c_M[p] = MassFluid;
	}


	if (UseDEM)DemDtForce = DtIni; //(DEM)
	if (CaseNfloat)InitFloating();

	//-Adjust paramaters to start.
	PartIni = PartBeginFirst;
	TimeStepIni = (!PartIni ? 0 : PartBeginTimeStep);
	//-Adjust motion for the instant of the loaded PART.
	if (CaseNmoving) {
		MotionTimeMod = (!PartIni ? PartBeginTimeStep : 0);
		Motion->ProcesTime(JSphMotion::MOMT_Simple, 0, TimeStepIni + MotionTimeMod);
	}

	//-Uses Inlet information from PART read.
	if (PartBeginTimeStep && PartBeginTotalNp) {
		TotalNp = PartBeginTotalNp;
		IdMax = unsigned(TotalNp - 1);
	}

	//-Shows Initialize configuration.
	if (InitializeInfo.size()) {
		Log->Print("Initialization configuration:");
		Log->Print(InitializeInfo);
		Log->Print(" ");
	}

	//-Process Special configurations in XML.
	JXml xml; xml.LoadFile(FileXml);

	//-Configuration of GaugeSystem.
	GaugeSystem->Config(Simulate2D, Simulate2DPosY, TimeMax, TimePart, Dp, DomPosMin, DomPosMax, Scell, Hdiv, H, MassFluid);
	if (xml.GetNode("case.execution.special.gauges", false))GaugeSystem->LoadXml(&xml, "case.execution.special.gauges");

	//-Prepares WaveGen configuration.
	if (WaveGen) {
		Log->Print("Wave paddles configuration:");
		WaveGen->Init(GaugeSystem, MkInfo, TimeMax, Gravity);
		WaveGen->VisuConfig("", " ");
	}

	//-Prepares Damping configuration.
	if (Damping) {
		Damping->Config(CellOrder);
		Damping->VisuConfig("Damping configuration:", " ");
	}

	//-Prepares AccInput configuration.
	if (AccInput) {
		Log->Print("AccInput configuration:");
		AccInput->Init(TimeMax);
		AccInput->VisuConfig("", " ");
	}

	//-Configuration of SaveDt.
	if (xml.GetNode("case.execution.special.savedt", false)) {
		SaveDt = new JSaveDt(Log);
		SaveDt->Config(&xml, "case.execution.special.savedt", TimeMax, TimePart);
		SaveDt->VisuConfig("SaveDt configuration:", " ");
	}

	//-Shows configuration of JGaugeSystem.
	if (GaugeSystem->GetCount())GaugeSystem->VisuConfig("GaugeSystem configuration:", " ");

	//-Shows configuration of JTimeOut.
	if (TimeOut->UseSpecialConfig())TimeOut->VisuConfig(Log, "TimeOut configuration:", " ");

	Part = PartIni; Nstep = 0; PartNstep = 0; PartOut = 0;
	TimeStep = TimeStepIni; TimeStepM1 = TimeStep;
	if (DtFixed)DtIni = DtFixed->GetDt(TimeStep, DtIni);
	TimePartNext = TimeOut->GetNextTime(TimeStep);
}

//==============================================================================
/// Prepares variables for interaction "INTER_Forces" or "INTER_ForcesCorr".
/// Prepara variables para interaccion "INTER_Forces" o "INTER_ForcesCorr".
//==============================================================================
void JSphSolidGpu::PreInteraction_Forces(TpInter tinter) {
	TmgStart(Timers, TMG_CfPreForces);
	//-Allocates memory.
	ViscDtg = ArraysGpu->ReserveFloat();
	Arg = ArraysGpu->ReserveFloat();
	Aceg = ArraysGpu->ReserveFloat3();
	if (TDeltaSph == DELTA_DynamicExt)Deltag = ArraysGpu->ReserveFloat();
	if (TShifting != SHIFT_None) {
		ShiftPosg = ArraysGpu->ReserveFloat3();
		if (ShiftTFS)ShiftDetectg = ArraysGpu->ReserveFloat();
	}
	Pressg = ArraysGpu->ReserveFloat();
	Press3Dg = ArraysGpu->ReserveFloat3();
	if (TVisco == VISCO_LaminarSPS)SpsGradvelg = ArraysGpu->ReserveSymatrix3f();

	// Matthias
	JauGradvelc2_M = ArraysGpu->ReserveSymatrix3f();
	JauTauDot_M = ArraysGpu->ReserveSymatrix3f();
	JauOmega_M = ArraysGpu->ReserveSymatrix3f();

	//-Prepares data for interation Pos-Single.
	if (Psingle) {
		PsPospressg = ArraysGpu->ReserveFloat4();
		cusph::PreInteractionSingle(Np, Posxyg, Poszg, Velrhopg, PsPospressg, CteB, Gamma);
	}
	//-Initialises arrays.
	PreInteractionVars_Forces(tinter, Np, Npb);

	//-Computes VelMax: Includes the particles from floating bodies and does not affect the periodic conditions.
	//-Calcula VelMax: Se incluyen las particulas floatings y no afecta el uso de condiciones periodicas.
	const unsigned pini = (DtAllParticles ? 0 : Npb);
	cusph::ComputeVelMod(Np - pini, Velrhopg + pini, ViscDtg);
	cudaDeviceSynchronize();
	float velmax = cusph::ReduMaxFloat(Np - pini, 0, ViscDtg, CellDiv->GetAuxMem(cusph::ReduMaxFloatSize(Np - pini)));
	VelMax = sqrt(velmax);
	cudaMemset(ViscDtg, 0, sizeof(float)*Np);           //ViscDtg[]=0
	ViscDtMax = 0;
	//JSphGpu::CheckCudaError("PreInteraction_Forces", "Failed calculating VelMax.");
	TmgStop(Timers, TMG_CfPreForces);
}

//==============================================================================
/// Free memory assigned to ArraysCpu.
/// Libera memoria asignada de ArraysCpu.
//==============================================================================
void JSphSolidGpu::PosInteraction_Forces() {
	//-Free memory assigned in PreInteraction_Forces(). | Libera memoria asignada en PreInteraction_Forces().
	ArraysGpu->Free(Arg);          Arg = NULL;
	ArraysGpu->Free(Aceg);         Aceg = NULL;
	ArraysGpu->Free(Deltag);       Deltag = NULL;
	ArraysGpu->Free(ShiftPosg);    ShiftPosg = NULL;
	ArraysGpu->Free(ShiftDetectg); ShiftDetectg = NULL;
	ArraysGpu->Free(Pressg);       Pressg = NULL;
	ArraysGpu->Free(Press3Dg);       Press3Dg = NULL;
	ArraysGpu->Free(PsPosg);       PsPosg = NULL;
	ArraysGpu->Free(SpsGradvelg);  SpsGradvelg = NULL;
	// Matthias
	//ArraysCpu->Free(Porec_M);       Porec_M = NULL; // Pending suppression of this line - Matthias
	//ArraysCpu->Free(JauGradvelc_M);  JauGradvelc_M = NULL;
	ArraysGpu->Free(JauGradvelc2_M);  JauGradvelc2_M = NULL;
	ArraysGpu->Free(JauTauDot_M);  JauTauDot_M = NULL;
	ArraysGpu->Free(JauOmega_M);  JauOmega_M = NULL;

}

//==============================================================================
/// Returns values of kernel Wendland, gradients: frx, fry and frz.
/// Devuelve valores de kernel Wendland, gradients: frx, fry y frz.
//==============================================================================
void JSphSolidGpu::GetKernelWendland(float rr2, float drx, float dry, float drz
	, float &frx, float &fry, float &frz)const
{
	const float rad = sqrt(rr2);
	const float qq = rad / H;
	//-Wendland kernel.
	const float wqq1 = 1.f - 0.5f*qq;
	const float fac = Bwen * qq*wqq1*wqq1*wqq1 / rad;
	frx = fac * drx; fry = fac * dry; frz = fac * drz;
}

//==============================================================================
/// Returns values of kernel Gaussian, gradients: frx, fry and frz.
/// Devuelve valores de kernel Gaussian, gradients: frx, fry y frz.
//==============================================================================
void JSphSolidGpu::GetKernelGaussian(float rr2, float drx, float dry, float drz
	, float &frx, float &fry, float &frz)const
{
	const float rad = sqrt(rr2);
	const float qq = rad / H;
	//-Gaussian kernel.
	const float qqexp = -4.0f*qq*qq;
	//const float wab=Agau*expf(qqexp);
	const float fac = Bgau * qq*expf(qqexp) / rad;
	frx = fac * drx; fry = fac * dry; frz = fac * drz;
}

//==============================================================================
/// Return values of kernel Cubic without tensil correction, gradients: frx, fry and frz.
/// Devuelve valores de kernel Cubic sin correccion tensil, gradients: frx, fry y frz.
//==============================================================================
void JSphSolidGpu::GetKernelCubic(float rr2, float drx, float dry, float drz
	, float &frx, float &fry, float &frz)const
{
	const float rad = sqrt(rr2);
	const float qq = rad / H;
	//-Cubic Spline kernel.
	float fac;
	if (rad>H) {
		float wqq1 = 2.0f - qq;
		float wqq2 = wqq1 * wqq1;
		fac = CubicCte.c2*wqq2 / rad;
	}
	else {
		float wqq2 = qq * qq;
		fac = (CubicCte.c1*qq + CubicCte.d1*wqq2) / rad;
	}
	//-Gradients.
	frx = fac * drx; fry = fac * dry; frz = fac * drz;
}

//==============================================================================
/// Return tensil correction for kernel Cubic.
/// Devuelve correccion tensil para kernel Cubic.
//==============================================================================
float JSphSolidGpu::GetKernelCubicTensil(float rr2, float rhopp1, float pressp1, float rhopp2, float pressp2)const {
	const float rad = sqrt(rr2);
	const float qq = rad / H;
	//-Cubic Spline kernel.
	float wab;
	if (rad>H) {
		float wqq1 = 2.0f - qq;
		float wqq2 = wqq1 * wqq1;
		wab = CubicCte.a24*(wqq2*wqq1);
	}
	else {
		float wqq2 = qq * qq;
		float wqq3 = wqq2 * qq;
		wab = CubicCte.a2*(1.0f - 1.5f*wqq2 + 0.75f*wqq3);
	}
	//-Tensile correction.
	float fab = wab * CubicCte.od_wdeltap;
	fab *= fab; fab *= fab; //fab=fab^4
	const float tensilp1 = (pressp1 / (rhopp1*rhopp1))*(pressp1>0 ? 0.01f : -0.2f);
	const float tensilp2 = (pressp2 / (rhopp2*rhopp2))*(pressp2>0 ? 0.01f : -0.2f);
	return(fab*(tensilp1 + tensilp2));
}

//==============================================================================
/// Return cell limits for interaction starting from cell coordinates.
/// Devuelve limites de celdas para interaccion a partir de coordenadas de celda.
//==============================================================================
void JSphSolidGpu::GetInteractionCells(unsigned rcell
	, int hdiv, const tint4 &nc, const tint3 &cellzero
	, int &cxini, int &cxfin, int &yini, int &yfin, int &zini, int &zfin)const
{
	//-Get interaction limits. | Obtiene limites de interaccion.
	const int cx = PC__Cellx(DomCellCode, rcell) - cellzero.x;
	const int cy = PC__Celly(DomCellCode, rcell) - cellzero.y;
	const int cz = PC__Cellz(DomCellCode, rcell) - cellzero.z;
	//-Code for hdiv 1 or 2 but not zero. | Codigo para hdiv 1 o 2 pero no cero.
	cxini = cx - min(cx, hdiv);
	cxfin = cx + min(nc.x - cx - 1, hdiv) + 1;
	yini = cy - min(cy, hdiv);
	yfin = cy + min(nc.y - cy - 1, hdiv) + 1;
	zini = cz - min(cz, hdiv);
	zfin = cz + min(nc.z - cz - 1, hdiv) + 1;
}


//==============================================================================
/// Interaction for force computation.
/// Interaccion para el calculo de fuerzas.
//==============================================================================
void JSphSolidGpu::Interaction_Forces(TpInter tinter) {

	const char met[] = "Interaction_Forces";
	PreInteraction_Forces(tinter);
	TmgStart(Timers, TMG_CfForces);

	const bool lamsps = (TVisco == VISCO_LaminarSPS);
	unsigned bsfluid = BlockSizes.forcesfluid;
	unsigned bsbound = BlockSizes.forcesbound;
	//debut de la modif
	if (BsAuto && !(Nstep%BsAuto->GetStepsInterval())) { //-Every certain number of steps. | Cada cierto numero de pasos.
		cusph::Interaction_Forces(Psingle, TKernel, WithFloating, UseDEM, lamsps, TDeltaSph, CellMode, Visco*ViscoBoundFactor, Visco, bsbound, bsfluid, Np, Npb, NpbOk, CellDivSingle->GetNcells(), CellDivSingle->GetBeginCell(), CellDivSingle->GetCellDomainMin(), Dcellg, Posxyg, Poszg, PsPospressg, Velrhopg, Codeg, Idpg, FtoMasspg, SpsTaug, SpsGradvelg, ViscDtg, Arg, Aceg, Deltag, TShifting, ShiftPosg, ShiftDetectg, Simulate2D, NULL, BsAuto);
		PreInteractionVars_Forces(tinter, Np, Npb);
		BsAuto->ProcessTimes(TimeStep, Nstep);
		bsfluid = BlockSizes.forcesfluid = BsAuto->GetKernel(0)->GetOptimumBs();
		bsbound = BlockSizes.forcesbound = BsAuto->GetKernel(1)->GetOptimumBs();
	}


	//-Interaction Fluid-Fluid/Bound & Bound-Fluid.
	cusph::Interaction_Forces(Psingle, TKernel, WithFloating, UseDEM, lamsps, TDeltaSph, CellMode, Visco*ViscoBoundFactor, Visco, bsbound, bsfluid, Np, Npb, NpbOk, CellDivSingle->GetNcells(), CellDivSingle->GetBeginCell(), CellDivSingle->GetCellDomainMin(), Dcellg, Posxyg, Poszg, PsPospressg, Velrhopg, Codeg, Idpg, FtoMasspg, SpsTaug, SpsGradvelg, ViscDtg, Arg, Aceg, Deltag, TShifting, ShiftPosg, ShiftDetectg, Simulate2D, NULL, NULL);

	//-Interaction DEM Floating-Bound & Floating-Floating. //(DEM)
	if (UseDEM)cusph::Interaction_ForcesDem(Psingle, CellMode, BlockSizes.forcesdem, CaseNfloat, CellDivSingle->GetNcells(), CellDivSingle->GetBeginCell(), CellDivSingle->GetCellDomainMin(), Dcellg, FtRidpg, DemDatag, float(DemDtForce), Posxyg, Poszg, PsPospressg, Velrhopg, Codeg, Idpg, ViscDtg, Aceg, NULL);

	// fin de la modif


	//-For 2D simulations always overrides the 2nd component (Y axis).
	//-Para simulaciones 2D anula siempre la 2º componente.
	if (Simulate2D)cusph::Resety(Np - Npb, Npb, Aceg);

	//-Computes Tau for Laminar+SPS.
	if (lamsps)cusph::ComputeSpsTau(Np, Npb, SpsSmag, SpsBlin, Velrhopg, SpsGradvelg, SpsTaug);

	if (Deltag)cusph::AddDelta(Np - Npb, Deltag + Npb, Arg + Npb);//-Adds the Delta-SPH correction for the density. | Añade correccion de Delta-SPH a Arg[]. 
	CheckCudaError(met, "Failed while executing kernels of interaction.");

	//-Calculates maximum value of ViscDt. 
	if (Np)ViscDtMax = cusph::ReduMaxFloat(Np, 0, ViscDtg, CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(Np)));

	TmgStop(Timers, TMG_CfForces);

	//-Calculates maximum value of Ace using ViscDtg like auxiliar memory.
	AceMax = ComputeAceMax(ViscDtg);

	CheckCudaError(met, "Failed in reduction of viscdt.");
}
*/