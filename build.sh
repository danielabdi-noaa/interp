#!/bin/sh
mpicxx -o interp interp.cpp -O3 -march=native -funroll-loops -fopenmp -DNDEBUG -I /usr/include/eigen3/ -leccodes -Wl,-rpath=/usr/local/lib
