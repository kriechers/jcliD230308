# Readme -  jcliD230308 

This github repository contains all the code used to generate the
results presented in ** Glacial abrupt climate change as a
multi-scale phenomenon resulting from monostable excitable
dynamics ** by Keno Riechers, Georg Gottwald and Niklas Boers
accepted for publication in Journal of Climate under the Number
jcliD230308.

The code was written using **python 3.8.10**. The file
**requirements.txt** contains a list of packages which were
installed for that python version at the time when the code was
executed to produce the outputs for the publication. Not all
packages indicates in the **requirements.txt** are necessary for
executing the code.

Executing the file **master.py** successively produces the
results presented in the publication, except for figures 2 and 3.
Notice that a random number generator is used for some
simulations. The results shown in the paper were generated
without setting the seed of the generator. Hence, an exact
reproduction of the results based on random numbers is not
possible.

Running **master.py** automatically creates a repository
**output**, where the results (including figures) will be stored.

The directory **proxy_data** contains data that was published
as supplementary material of other research articles. The
corresponding research articles are indicated in the individual
files as well as in the main text of our manuscript. 


Two create figures 2 and 3 please run **fig02.py** and
**fig03.py**. 

