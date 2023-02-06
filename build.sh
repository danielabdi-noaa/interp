#!/bin/sh
mpicxx -o interp interp.cpp -O3 -march=native -funroll-loops -DNDEBUG -I /usr/include/eigen3/ -leccodes
