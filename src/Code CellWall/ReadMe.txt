To make it work, you must build it thanks to the Makefile and GCC, then, in the bash script, write :
- the number of threads for the parallelisation
- then : ./ContDens [path of the CSVs]/[name *] [coefficient for the variable smoothing length] [**]

* : if the aim is to compute one image, write the whole file name (without '.csv'), otherwise write the namecase (cut before the '_')
** : optional, it is the number of the last image in the simulation