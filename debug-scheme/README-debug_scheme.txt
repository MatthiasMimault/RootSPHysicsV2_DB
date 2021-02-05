# Debug log of debug_scheme
- Renovate numerical scheme "single step-two stages"
- Control switch with bool typeDev (therefore an "if")
- Update XML file from cluster and move typeDev to Numerics
- Update Vtk script with csv/vtk/img
- Change working folder to /debug-scheme/

# Settings
Turgor only, with one control based on previous version of code (typeDev false)
AI setup with length sample 9, Time 0.4, p100
>> The control will require a growth component: it is the critical reason of 
the refactoring

# Tests
A. Control
Build OK
| Ctrl	| T0.2 p100 A2P3S1	| 2min	| Stable, soft but not meaningful
| Ctrl	| T0.4 p200 #1		| 3min	| Stable, visible deformation
|-------|-----------------------|-------|
| CtL10	| G1 T0.4 p200 #1	| 3min	| Stable but small
| CtLe3	| G1 T0.4 p200 #1	| 3min	| Too hard
| CtLe2	| G1 T0.4 p200 #1	| 3min	| Deformed and a bit unstable
	
B. Version 1
Scheme implemented, with new Growth interface method and modification to const
of a lot of dependencies (because read/compute only)
Debug: all the values were not systematically updated. Some forgotten were left
to zero and caused NaN apparition (Mass)
Run seems smooth
| DvLe2	| G1 T0.4 p200 #1	| 4min	| Impact visible but very negligible
|-------|-----------------------|-------|
The code runs and output and fairly faint variation of the control. It means
that the numerical scheme impact amounts little in the computation of the 
solution
>> Test on cluster
>> Publish as Version 37d (because code is still in validation)
