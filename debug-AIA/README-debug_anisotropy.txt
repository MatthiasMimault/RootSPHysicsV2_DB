# Updates
- switch for anisotropy balance
- shift tip of Dp/2
- update XML file, node "Anisotropy" with new variables (p, s, c)

# Settings
Turgor only, with three cases of anisotropy. Aim to detect any defect in the 
anisotropy balance formulation
Length sample 9, Time 0.4, p100

# Tests
Control
A1-Isotropy 	
  p50  T2: too slow (15 minutes)
  p100 T02: 7 steps per round, 2 minutes, visible change >> T++
  p100 T04: 40 steps, 3 minutes, 
A2-Anisoropy

Balance	
1. P0.5 S1: exact same as isotropic
2. P0.5 S0.5: sample length was wrong, P++, and cut down time, T--
An odd change in dynamics around 5.5: 
Pystats process all the csvs again and again >> Debug it
3. P4 S0.5: dynamics re switch with isotropy on wrong side
4. P4 S1: dynamics close to isotropy but inconclusive
5. P4 S2: large take-off of the root sides, no visibility of a transition
6. P2 S2: A swap between the assignation of baseline and spread has been found

C-Balance fixed
6. P2 S2: Swapped dynamics iso-aniso clearly visible
7. P3 S7: Failure, data not regenerated and same as C6

D-Sigmoid fixed
7. P3 S7: Correction of the sigmoid in the balance distribution
== Data validating the fix
== Commit changes and close debug session -- 22/01