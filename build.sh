#!/bin/sh
mpicxx -O3 -march=native -funroll-loops -DNDEBUG -I /usr/include/eigen3/ interp_fields.cpp -o interp
