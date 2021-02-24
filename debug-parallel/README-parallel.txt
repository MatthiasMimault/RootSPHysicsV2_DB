# Debug of parallel performance

# Settings
Turgor only. Control do not divide and Test divide a lot
AI setup with length sample 9, Time 0.4
p 1.0

RunDivision is not parallelised

# Control
Run loop times
1. Compute step: 700 ms
2. RunDivision: 1 ms
3. Save: 280 ms (sporadically)

# Test 1: all division
1. Compute step before: 700 ms
2. Run Division: 400 ms
3. Compute step after: 2400 ms
4. Save: 840 ms


# Test 2: all division - Single core
1. Compute step before: 1200 ms
2. Run Division: 400 ms (MarkedDivision takes 400 ms)
3. Compute step after: 3800 ms
4. Save: 840 ms (GetParticleData takes 800 ms)

Division and Save processes are single core
Can they be made parallelised ?
Division is easier and more urgent

== Actually the problem lies in PreInteractions_Value, where the computation of
K is too complex to be parallised. It contains a switch and value calls that 
are cumbersome.
Similar functions are performing reading and writing more efficiently (IntForce
ComputeStep) but they are different
IntForce is given the arrays of value
ComputeStep does not have switches
== Refactor CalcK to do not have such problem: No
== Just fixing the value of MaxPos X out of the OpenMP loop solved the problem
== Save and Division remain big cpu wasters
>> Fix GetParticleData
>> Fix MarkedDivision
