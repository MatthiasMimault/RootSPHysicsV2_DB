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

/// \file JSphSolidCpu.h \brief Declares the class \ref JSphSolidCpu.

#ifndef _JSphSolidCpu_
#define _JSphSolidCpu_

#include "Types.h"
#include "JSphTimersCpu.h"
#include "JPartsLoad4.h"
#include "JSph.h"
#include <string>

class JPartsOut;
class JArraysCpu;
class JCellDivCpu;

//##############################################################################
//# JSphSolidCpu
//##############################################################################
/// \brief Defines the attributes and functions to be used only in CPU simulations.

class JSphSolidCpu : public JSph
{
private:
	JCellDivCpu * CellDiv;

protected:
	int OmpThreads;        ///<Max number of OpenMP threads in execution on CPU host (minimum 1). | Numero maximo de hilos OpenMP en ejecucion por host en CPU (minimo 1).
	std::string RunMode;   ///<Overall mode of execution (symmetry, openmp, load balancing). |  Almacena modo de ejecucion (simetria,openmp,balanceo,...).

						   //-Number of particles in domain | Numero de particulas del dominio.
	unsigned Np;        ///<Total number of particles (including periodic duplicates). | Numero total de particulas (incluidas las duplicadas periodicas).
	unsigned Npb;       ///<Total number of boundary particles (including periodic boundaries). | Numero de particulas contorno (incluidas las contorno periodicas).
	unsigned NpbOk;     ///<Total number of boundary particles near fluid (including periodic duplicates). | Numero de particulas contorno cerca del fluido (incluidas las contorno periodicas).

	unsigned NpfPer;    ///<Number of periodic floating-fluid particles. | Numero de particulas fluidas-floating periodicas.
	unsigned NpbPer;    ///<Number of periodic boundary particles. | Numero de particulas contorno periodicas.
	unsigned NpfPerM1;  ///<Number of periodic floating-fluid particles (previous values). | Numero de particulas fluidas-floating periodicas (valores anteriores).
	unsigned NpbPerM1;  ///<Number of periodic boundary particles (previous values). | Numero de particulas contorno periodicas (valores anteriores).

	bool WithFloating;
	bool BoundChanged;  ///<Indicates if selected boundary has changed since last call of divide. | Indica si el contorno seleccionado a cambiado desde el ultimo divide.

	unsigned CpuParticlesSize;  ///<Number of particles with reserved memory on the CPU. | Numero de particulas para las cuales se reservo memoria en cpu.
	llong MemCpuParticles;      ///<Memory reserved for particles' vectors. | Mermoria reservada para vectores de datos de particulas.
	llong MemCpuFixed;          ///<Memory reserved in AllocMemoryFixed. | Mermoria reservada en AllocMemoryFixed.

								//-Particle Position according to id. | Posicion de particula segun id.
	unsigned *RidpMove; ///<Only for moving boundary particles [CaseNmoving] and when CaseNmoving!=0 | Solo para boundary moving particles [CaseNmoving] y cuando CaseNmoving!=0 

						//-List of particle arrays on CPU. | Lista de arrays en CPU para particulas.
	JArraysCpu* ArraysCpu;

	//-Execution Variables for particles (size=ParticlesSize). | Variables con datos de las particulas para ejecucion (size=ParticlesSize).
	unsigned *Idpc;    ///<Identifier of particle | Identificador de particula.
	typecode *Codec;   ///<Indicator of group of particles & other special markers. | Indica el grupo de las particulas y otras marcas especiales.
	unsigned *Dcellc;  ///<Cells inside DomCells coded with DomCellCode. | Celda dentro de DomCells codificada con DomCellCode.
	tdouble3 *Posc;
	tfloat4 *Velrhopc;

	//-Variables for compute step: VERLET. | Vars. para compute step: VERLET.
	tfloat4 *VelrhopM1c;  ///<Verlet: in order to keep previous values. | Verlet: para guardar valores anteriores.
	int VerletStep;
	float *MassM1c_M;

	//-Variables for compute step: SYMPLECTIC. | Vars. para compute step: SYMPLECTIC.
	tdouble3 *PosPrec;    ///<Sympletic: in order to keep previous values. | Sympletic: para guardar valores en predictor.
	tfloat4 *VelrhopPrec;
	double DtPre;

	// Additional variables for #Symplectic_M
	float *MassPrec_M;
	tsymatrix3f *TauPrec_M;
	tsymatrix3f *QuadFormPrec_M;


	//-Variables for floating bodies.
	unsigned *FtRidp;             ///<Identifier to access to the particles of the floating object [CaseNfloat].
	StFtoForces *FtoForces;       ///<Stores forces of floatings [FtCount].
	StFtoForcesRes *FtoForcesRes; ///<Stores data to update floatings [FtCount].

								  //-Variables for computation of forces | Vars. para computo de fuerzas.
	tfloat3 *PsPosc;       ///<Position and prrhop for Pos-Single interaction | Posicion y prrhop para interaccion Pos-Single.

	tfloat3 *Acec;         ///<Sum of interaction forces | Acumula fuerzas de interaccion
	float *Arc;
	float *Deltac;         ///<Adjusted sum with Delta-SPH with DELTA_DynamicExt | Acumula ajuste de Delta-SPH con DELTA_DynamicExt

	tfloat3 *ShiftPosc;    ///<Particle displacement using Shifting.
	float *ShiftDetectc;   ///<Used to detect free surface with Shifting.

	double VelMax;        ///<Maximum value of Vel[] sqrt(vel.x^2 + vel.y^2 + vel.z^2) computed in PreInteraction_Forces().
	double AceMax;        ///<Maximum value of Ace[] sqrt(ace.x^2 + ace.y^2 + ace.z^2) computed in Interaction_Forces().
	float ViscDtMax;      ///<Max value of ViscDt calculated in Interaction_Forces() / Valor maximo de ViscDt calculado en Interaction_Forces().

						  //-Variables for computing forces [INTER_Forces,INTER_ForcesCorr] | Vars. derivadas para computo de fuerzas [INTER_Forces,INTER_ForcesCorr]
	float *Pressc;       ///< Press[]=B*((Rhop/Rhop0)^gamma-1)
	//tfloat3 *Press3Dc_M;       ///< Press[]=B*((Rhop/Rhop0)^gamma-1)

	// Matthias - Pore pressure
	bool *Divisionc_M;
	float *Porec_M; 
	float *Massc_M; // Mass, Delta mass
	float *NabVx_M;
	//float TimeGoing;
	// Augustin
	float* VonMises3D;
	float* GradVelSave;
	unsigned* CellOffSpring;
	tfloat3* StrainDotSave;

	// Matthias - Root geometry data
	float maxPosX;

	// Direct density estimation - #temp
	float* DirectRhop_M;

						 //-Variables for Laminar+SPS viscosity.  
	tsymatrix3f *SpsTauc;       ///<SPS sub-particle stress tensor.
	tsymatrix3f *SpsGradvelc;   ///<Velocity gradients.

								// Matthias - Solid
	//tmatrix3f *JauTauc_M;
	tsymatrix3f *Tauc_M;
	tsymatrix3f *TauM1c_M;
	//tmatrix3f *JauGradvelc_M;
	tsymatrix3f *StrainDotc_M;
	tsymatrix3f *TauDotc_M;
	tsymatrix3f *Spinc_M;

	tsymatrix3f *QuadFormc_M;
	tsymatrix3f *QuadFormM1c_M;

	// NSPH
	tmatrix3f   *L_M;
	float* Lo_M;

	TimersCpu Timers;


	void InitVars();

	void FreeCpuMemoryFixed();
	void AllocCpuMemoryFixed();
	void FreeCpuMemoryParticles();
	void AllocCpuMemoryParticles(unsigned np, float over);

	void ResizeCpuMemoryParticles(unsigned np);
	void ReserveBasicArraysCpu();

	bool CheckCpuParticlesSize(unsigned requirednp) { return(requirednp + PARTICLES_OVERMEMORY_MIN <= CpuParticlesSize); }

	template<class T> T* TSaveArrayCpu(unsigned np, const T *datasrc)const;
	word*        SaveArrayCpu(unsigned np, const word        *datasrc)const { return(TSaveArrayCpu<word>(np, datasrc)); }
	unsigned*    SaveArrayCpu(unsigned np, const unsigned    *datasrc)const { return(TSaveArrayCpu<unsigned>(np, datasrc)); }
	int*         SaveArrayCpu(unsigned np, const int         *datasrc)const { return(TSaveArrayCpu<int>(np, datasrc)); }
	float*       SaveArrayCpu(unsigned np, const float       *datasrc)const { return(TSaveArrayCpu<float>(np, datasrc)); }
	tfloat4*     SaveArrayCpu(unsigned np, const tfloat4     *datasrc)const { return(TSaveArrayCpu<tfloat4>(np, datasrc)); }
	double*      SaveArrayCpu(unsigned np, const double      *datasrc)const { return(TSaveArrayCpu<double>(np, datasrc)); }
	tdouble3*    SaveArrayCpu(unsigned np, const tdouble3    *datasrc)const { return(TSaveArrayCpu<tdouble3>(np, datasrc)); }
	tsymatrix3f* SaveArrayCpu(unsigned np, const tsymatrix3f *datasrc)const { return(TSaveArrayCpu<tsymatrix3f>(np, datasrc)); }
	// Matthias
	tmatrix3f*   SaveArrayCpu(unsigned np, const tmatrix3f *datasrc)const { return(TSaveArrayCpu<tmatrix3f>(np, datasrc)); }
	tfloat3*   SaveArrayCpu(unsigned np, const tfloat3 *datasrc)const { return(TSaveArrayCpu<tfloat3>(np, datasrc)); }
	bool*		 SaveArrayCpu(unsigned np, const bool      *datasrc)const { return(TSaveArrayCpu<bool>(np, datasrc)); }

	template<class T> void TRestoreArrayCpu(unsigned np, T *data, T *datanew)const;
	void RestoreArrayCpu(unsigned np, word        *data, word        *datanew)const { TRestoreArrayCpu<word>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, unsigned    *data, unsigned    *datanew)const { TRestoreArrayCpu<unsigned>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, int         *data, int         *datanew)const { TRestoreArrayCpu<int>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, float       *data, float       *datanew)const { TRestoreArrayCpu<float>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, tfloat4     *data, tfloat4     *datanew)const { TRestoreArrayCpu<tfloat4>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, double      *data, double      *datanew)const { TRestoreArrayCpu<double>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, tdouble3    *data, tdouble3    *datanew)const { TRestoreArrayCpu<tdouble3>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, tsymatrix3f *data, tsymatrix3f *datanew)const { TRestoreArrayCpu<tsymatrix3f>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, tmatrix3f *data, tmatrix3f *datanew)const { TRestoreArrayCpu<tmatrix3f>(np, data, datanew); }
	// Matthias
	void RestoreArrayCpu(unsigned np, bool *data, bool*datanew)const { TRestoreArrayCpu<bool>(np, data, datanew); }
	void RestoreArrayCpu(unsigned np, tfloat3* data, tfloat3* datanew)const { TRestoreArrayCpu<tfloat3>(np, data, datanew); }

	llong GetAllocMemoryCpu()const;
	void PrintAllocMemory(llong mcpu)const;

	unsigned GetParticlesData(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
		, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, typecode *code);
	unsigned GetParticlesData_M(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
		, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, float *pore, tfloat3 *press, float* mass, tsymatrix3f *tau, typecode *code);
	//unsigned GetParticlesData_M(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal, unsigned* idp, tdouble3* pos, tfloat3* vel, float* rhop, float* pore, tfloat3* press, float* mass, tsymatrix3f* tau, float* vonMises, typecode* code);
	unsigned GetParticlesData_M(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
		, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, float *pore, tfloat3 *press, float* mass, tsymatrix3f *gradvel, tsymatrix3f *tau, typecode *code);
	unsigned GetParticlesData_M(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
		, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, float *pore, tfloat3 *press, float* mass, tsymatrix3f *gradvel, tsymatrix3f *tau, tsymatrix3f *qf, typecode *code);
	unsigned GetParticlesData_M(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
		, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, float *pore, float *press, float* mass, tsymatrix3f *qf, typecode *code);
	unsigned GetParticlesData_M(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
		, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, float *pore, float *press, float* mass, tsymatrix3f *qf, float *nabvx, typecode *code);
	unsigned GetParticlesData_A(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
		, unsigned* idp, tdouble3* pos, tfloat3* vel, float* rhop, float* pore, float* press, float* mass, tsymatrix3f* qf
		, float* nabvx, float* vonMises, float* grVelSav, unsigned* cellOSpr, typecode* code);
	unsigned GetParticlesData_M1(unsigned n, unsigned pini, bool cellorderdecode, bool onlynormal
			, unsigned *idp, tdouble3 *pos, tfloat3 *vel, float *rhop, float *pore, float *press, float* mass, tsymatrix3f *qf
			, float *nabvx, float* vonMises, float* grVelSav, unsigned* cellOSpr, tfloat3* gradvel, typecode *code);

	void ConfigOmp(const JCfgRun *cfg);

	void ConfigRunMode(const JCfgRun *cfg, std::string preinfo = "");
	void ConfigCellDiv(JCellDivCpu* celldiv) { CellDiv = celldiv; }
	void InitFloating(); 
	void InitRun();

	// Matthias
	void InitRun_Mixed_M();
	void InitRun_Uni_M();
	void InitRun_T(JPartsLoad4 *pl); 

	void AddAccInput();

	float CalcVelMaxSeq(unsigned np, const tfloat4* velrhop)const;
	float CalcVelMaxOmp(unsigned np, const tfloat4* velrhop)const;

	void PreInteractionVars_Forces(TpInter tinter, unsigned np, unsigned npb);
	void PreInteraction_Forces(TpInter tinter);
	void PosInteraction_Forces();

	inline void GetKernelWendland(float rr2, float drx, float dry, float drz, float &frx, float &fry, float &frz)const;
	inline void GetKernelGaussian(float rr2, float drx, float dry, float drz, float &frx, float &fry, float &frz)const;
	inline void GetKernelCubic(float rr2, float drx, float dry, float drz, float &frx, float &fry, float &frz)const;
	inline float GetKernelCubicTensil(float rr2, float rhopp1, float pressp1, float rhopp2, float pressp2)const;
	inline void GetKernelDirectWend_M(float rr2, float& f)const;

	inline void GetInteractionCells(unsigned rcell
		, int hdiv, const tint4 &nc, const tint3 &cellzero
		, int &cxini, int &cxfin, int &yini, int &yfin, int &zini, int &zfin)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode> void InteractionForcesBound
	(unsigned n, unsigned pini, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhopp, const typecode *code, const unsigned *id
		, float &viscdt, float *ar)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode> void InteractionForcesBound11_M
	(unsigned n, unsigned pini, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhopp, const typecode *code, const unsigned *id
		, float &viscdt, float *ar, tsymatrix3f* gradvel, tsymatrix3f* omega, tmatrix3f* L)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode> void InteractionForcesBound12_M
	(unsigned n, unsigned pini, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned* beginendcell, tint3 cellzero, const unsigned* dcell
		, const tdouble3* pos, const tfloat3* pspos, const tfloat4* velrhopp, const typecode* code, const unsigned* id
		, const float* mass
		, float& viscdt, float* ar, tsymatrix3f* gradvel, tsymatrix3f* omega, tmatrix3f* L)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesFluid
	(unsigned n, unsigned pini, tint4 nc, int hdiv, unsigned cellfluid, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesSolid
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesSolMass_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const; 

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesSolMass_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const tfloat3 *press, const float *pore, const float *mass
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesNSPH_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker> void ComputeNsphCorrection
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop
		, const float *mass, tmatrix3f *L)const;

	template<bool psingle, TpKernel tker> void ComputeNsphCorrection11
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop
		, const float *mass, tmatrix3f *L)const;

	template<bool psingle, TpKernel tker> void ComputeNsphCorrection12
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop
		, const float *mass, tmatrix3f *L)const;

	template<bool psingle, TpKernel tker> void ComputeNsphCorrectionX
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop
		, const float *mass, tmatrix3f *L)const;

	template<bool psingle, TpKernel tker> void ComputeNsphCorrection13
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned* beginendcell, tint3 cellzero, const unsigned* dcell
		, const tdouble3* pos, const tfloat3* pspos, const tfloat4* velrhop
		, const float* mass, tmatrix3f* L)const;

	template<bool psingle, TpKernel tker> void ComputeNsphCorrection14
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial
		, const unsigned* beginendcell, tint3 cellzero, const unsigned* dcell
		, const tdouble3* pos, const tfloat3* pspos, const tfloat4* velrhop
		, const float* mass, tmatrix3f* L)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesNSPH11_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesNSPH12_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForcesNSPH13_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForces_VelCst_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForces_CstSig_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float *ar, tfloat3 *ace, float *delta
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void InteractionForces_V11b_M
	(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco
		, const unsigned* beginendcell, tint3 cellzero, const unsigned* dcell
		, const tsymatrix3f* tau, tsymatrix3f* gradvel, tsymatrix3f* omega
		, const tdouble3* pos, const tfloat3* pspos, const tfloat4* velrhop, const typecode* code, const unsigned* idp
		, const float* press, const float* pore, const float* mass
		, tmatrix3f* L
		, float& viscdt, float* ar, tfloat3* ace, float* delta
		, TpShifting tshifting, tfloat3* shiftpos, float* shiftdetect)const;

	template<bool psingle> void InteractionForcesDEM
	(unsigned nfloat, tint4 nc, int hdiv, unsigned cellfluid
		, const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
		, const unsigned *ftridp, const StDemData* demobjs
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, float &viscdt, tfloat3 *ace)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
	(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *spstau, tsymatrix3f *spsgradvel
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
	(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
	(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
	(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const tfloat3 *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
	(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega, tsymatrix3f *qf
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	template<bool psingle, TpKernel tker, TpFtMode ftmode, bool lamsps, TpDeltaSph tdelta, bool shift> void Interaction_ForcesT
	(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat3 *pspos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tmatrix3f *L
		, TpShifting tshifting, tfloat3 *shiftpos, float *shiftdetect)const;

	void Interaction_Forces(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *spstau, tsymatrix3f *spsgradvel
		, tfloat3 *shiftpos, float *shiftdetect)const;

	void InteractionSimple_Forces(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tfloat3 *pspos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *spstau, tsymatrix3f *spsgradvel
		, tfloat3 *shiftpos, float *shiftdetect)const;

	void Interaction_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;
	
	void Interaction_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;

	void InteractionSimple_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tfloat3 *pspos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;
	
	void InteractionSimple_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tfloat3 *pspos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;
	
	// Press3D
	void InteractionSimple_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tfloat3 *pspos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const tfloat3 *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;

	void Interaction_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const tfloat3 *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;

	// Press1D + QuadForm
	void InteractionSimple_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tfloat3 *pspos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega, tsymatrix3f *qf
		, tfloat3 *shiftpos, float *shiftdetect)const;

	void Interaction_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore, const float *mass
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega, tsymatrix3f *qf
		, tfloat3 *shiftpos, float *shiftdetect)const;

	// P1SM - NSPH
	void InteractionSimple_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tfloat3 *pspos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;

	void Interaction_Forces_M(unsigned np, unsigned npb, unsigned npbok
		, tuint3 ncells, const unsigned *begincell, tuint3 cellmin, const unsigned *dcell
		, const tdouble3 *pos, const tfloat4 *velrhop, const unsigned *idp, const typecode *code
		, const float *press, const float *pore, const float *mass
		, tmatrix3f *L
		, float &viscdt, float* ar, tfloat3 *ace, float *delta
		, tsymatrix3f *jautau, tsymatrix3f *jaugradvel, tsymatrix3f *jautaudot, tsymatrix3f *jauomega
		, tfloat3 *shiftpos, float *shiftdetect)const;


	void ComputeSpsTau(unsigned n, unsigned pini, const tfloat4 *velrhop, const tsymatrix3f *gradvel, tsymatrix3f *tau)const;
	void ComputeJauTauDot_M(unsigned n, unsigned pini, const tsymatrix3f *gradvel, tsymatrix3f *tau, tsymatrix3f *taudot, tsymatrix3f *omega)const;
	void ComputeTauDot_Gradual_M(unsigned n, unsigned pini, tsymatrix3f *taudot)const;
	//void ComputeJauTauDotImplicit_M(unsigned n, unsigned pini, const double dt, const tsymatrix3f *gradvel, tsymatrix3f *tau, tsymatrix3f *taudot, tsymatrix3f *omega)const;

	template<bool shift> void ComputeVerletVarsFluid(const tfloat4 *velrhop1, const tfloat4 *velrhop2, double dt, double dt2, tdouble3 *pos, unsigned *cell, typecode *code, tfloat4 *velrhopnew)const;
	// Matthias
	template<bool shift> void ComputeVerletVarsSolid_M(const tfloat4 *velrhop1, const tfloat4 *velrhop2, const tsymatrix3f *tau1, const tsymatrix3f *tau2, double dt, double dt2
		, tdouble3 *pos, unsigned *dcell, typecode *code, tfloat4 *velrhopnew, tsymatrix3f *taunew)const;
	template<bool shift> void ComputeVerletVarsSolMass_M(const tfloat4 *velrhop1, const tfloat4 *velrhop2
		, const tsymatrix3f *tau1, const tsymatrix3f *tau2, const float *mass1, const float *mass2
		, double dt, double dt2, tdouble3 *pos, unsigned *dcell, typecode *code, tfloat4 *velrhopnew, tsymatrix3f *taunew, float *massnew)const;
	template<bool shift> void ComputeVerletVarsQuad_M(const tfloat4 *velrhop1, const tfloat4 *velrhop2
		, const tsymatrix3f *tau1, const tsymatrix3f *tau2, const tsymatrix3f *qf1, const tsymatrix3f *qf2, const float *mass1, const float *mass2
		, double dt, double dt2, tdouble3 *pos, unsigned *dcell, typecode *code, tfloat4 *velrhopnew, tsymatrix3f *taunew, tsymatrix3f *qfnew, float *massnew)const;

	void ComputeVelrhopBound(const tfloat4* velrhopold, double armul, tfloat4* velrhopnew)const;

	// Matthias
	template<bool shift> void ComputeEulerVarsFluid_M(tfloat4 *velrhop, double dt, tdouble3 *pos, unsigned *dcell, word *code)const;
	template<bool shift> void ComputeEulerVarsSolid_M(tfloat4 *velrhop, double dt, tdouble3 *pos, tsymatrix3f *tau, unsigned *dcell, word *code)const;
	template<bool shift> void ComputeEulerVarsSolidImplicit_M(tfloat4 *velrhop, double dt, tdouble3 *pos, tsymatrix3f *tau, tsymatrix3f *gradvel, tsymatrix3f *omega, unsigned *dcell, word *code)const;
	
	void ComputeVerlet(double dt);
	template<bool shift> void ComputeSymplecticPreT(double dt);
	void ComputeSymplecticPre(double dt);
	template<bool shift> void ComputeSymplecticCorrT(double dt);
	void ComputeSymplecticCorr(double dt);
	double DtVariable(bool final);
	// Matthias
	void ComputeEuler_M(double dt);

	template<bool shift> void ComputeSymplecticPreT_M(double dt);
	template<bool shift> void ComputeSymplecticPreT_BlockBdy_M(double dt);
	template<bool shift> void ComputeSymplecticPreT_CompressBdy_M(double dt);

	void ComputeSymplecticPre_M(double dt);
	template<bool shift> void ComputeSymplecticCorrT_M(double dt);
	template<bool shift> void ComputeSymplecticCorrT_BlockBdy_M(double dt);
	template<bool shift> void ComputeSymplecticCorrT_CompressBdy_M(double dt);
	void ComputeSymplecticCorr_M(double dt);

	template<bool shift> void ComputeSymplecticPreVcT_M(double dt);
	void ComputeSymplecticPre_VelCst_M(double dt);
	template<bool shift> void ComputeSymplecticCorrVcT_M(double dt);
	void ComputeSymplecticCorr_VelCst_M(double dt);

	void GrowthCell_M(double dt);
	float GrowthRateSpace(float pos);
	float GrowthRateSpaceNormalised(float pos);
	float GrowthRateGaussian(float pos);
	double GrowthRateSpaceNormalised(double pos);
	double GrowthRate2(double pos, double tip);
	float MaxValueParticles(float* field); 
	tfloat3 MaxPosition();

	template<bool shift> void ComputeSymplecticPreT_SigCst_M(double dt);
	void ComputeSymplecticPre_SigCst_M(double dt);
	template<bool shift> void ComputeSymplecticCorrT_SigCst_M(double dt);
	void ComputeSymplecticCorr_SigCst_M(double dt);
	// End Matthias

	// T19
	template<bool shift> void ComputeSymplecticPreT_T19(double dt);
	void ComputeSymplecticPre_T19(double dt);
	template<bool shift> void ComputeSymplecticCorrT_T19(double dt);
	void ComputeSymplecticCorr_T19(double dt);

	void RunShifting(double dt);

	void CalcRidp(bool periactive, unsigned np, unsigned pini, unsigned idini, unsigned idfin, const typecode *code, const unsigned *idp, unsigned *ridp)const;
	void MoveLinBound(unsigned np, unsigned ini, const tdouble3 &mvpos, const tfloat3 &mvvel, const unsigned *ridp, tdouble3 *pos, unsigned *dcell, tfloat4 *velrhop, typecode *code)const;
	void MoveMatBound(unsigned np, unsigned ini, tmatrix4d m, double dt, const unsigned *ridpmv, tdouble3 *pos, unsigned *dcell, tfloat4 *velrhop, typecode *code)const;
	void RunMotion(double stepdt);
	void RunDamping(double dt, unsigned np, unsigned npb, const tdouble3 *pos, const typecode *code, tfloat4 *velrhop)const;

	void ShowTimers(bool onlyfile = false); 
	void GetTimersInfo(std::string &hinfo, std::string &dinfo)const; 
	unsigned TimerGetCount()const { return(TmcGetCount()); }
	bool TimerIsActive(unsigned ct)const { return(TmcIsActive(Timers, (CsTypeTimerCPU)ct)); }
	float TimerGetValue(unsigned ct)const { return(TmcGetValue(Timers, (CsTypeTimerCPU)ct)); }
	const double* TimerGetPtrValue(unsigned ct)const { return(TmcGetPtrValue(Timers, (CsTypeTimerCPU)ct)); }
	std::string TimerGetName(unsigned ct)const { return(TmcGetName((CsTypeTimerCPU)ct)); }
	std::string TimerToText(unsigned ct)const { return(JSph::TimerToText(TimerGetName(ct), TimerGetValue(ct))); }

public:
	JSphSolidCpu(bool withmpi);
	~JSphSolidCpu();

	void UpdatePos(tdouble3 pos0, double dx, double dy, double dz, bool outrhop, unsigned p, tdouble3 *pos, unsigned *cell, typecode *code)const;
};

#endif


