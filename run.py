from numpy import savetxt
from new import *

solution = relabel(7, 6, [1, 2, 3, 4, 7, 10, 19])
savetxt('solution.csv', solution, delimiter=',')