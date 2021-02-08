# Debug log of theta_transition
X change folder to /debug-theta_transition/
X update XML to 6 (pm,pp,smsp,cm,cp) parameters
X update XMl to laptop/debug/ 
- update XML to cluster/root_experiments_2021
X change code to enable theta transition
> validate code implementation
- build on cluster

# Settings
Turgor only, with one control based on previous version of code (typeDev false)
AI setup with length sample 9, Time 0.4
p varies from 1 to 0.3
>> Design tests to reflect different dynamics

# Tests
A. Control - debug_c
Parameters pairs from Python study (08/02)
| p1.0 P4 S4 C0.5	| T0.4 	| 5min	| Velocity variation over half body
| p0.3 P2 S2 C1.0	| T0.4 	| 	| Velocity variation only at the tip
|-----------------------|-------|-------|

B. Dev version - debug_d
| p1.00 Mix_transition	| T0.4 	| 	| Slow computation: What is the load ?
| 			| 	| 	| % Check coefficients value: Fixed, 
| 			| 	| 	| % wrong set call (_m instead of _p)
| p0.85 Mix_transition	| T0.4 	| 	| Close to p1.00 and stable
| p0.65 Mix_transition	| T0.4 	| 	|
| p0.45 Mix_transition	| T0.4 	| 	|
| p0.30 Mix_transition	| T0.4 	| 	| 
|-----------------------|-------|-------| 
Validated !