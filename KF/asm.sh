#!/bin/bash
echo ""
echo "***************************************"
echo "Used for LY's GPS project"
echo "assemble matrix.c/.h kalman.c/.h gps.c/.h & gps_test.c"
echo "into kalman_* file(* could be modified)"
echo "please input file name: "
read filename
gcc -std=c11 matrix.c -c -o matrix.o
gcc -std=c11 kalman.c -c -o kalman.o
gcc -std=c11 gps.c    -c -o gps.o
gcc -std=c11 gps_test.c matrix.o kalman.o gps.o -lm -o run_kalman_$filename
echo "please screen and run, have fun"
echo "****************************************"
echo ""

