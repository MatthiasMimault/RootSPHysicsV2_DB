# Debug log of theta_transition
- Investigated the source of wrong Qf in csv
- Fix it

# Settings
Turgor only, with one control based on previous version of code (typeDev false)
AI setup with length sample 9, Time 0.4
p 1.0

# Tests
A. Investigation - debug_d
Parameters pairs from RG4
| p1.0 P4 S4 C0.5	| T0.4 	| 5min	| 
|-----------------------|-------|-------|

== Fixed: Qf update of boundary particles