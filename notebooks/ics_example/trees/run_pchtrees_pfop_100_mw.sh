#!/bin/bash

NTREES=100
PARAMS=pchtrees_pfop_100_mw.toml

# These are the upper and lower values of z=0 dark matter halo mass at the root
# of the tree. With the other options below, the code randomly samples in log10
# mass between these values. The values should be in h^-1 Msolar, i.e., if you
# want a mass of exactly 1x10^12 Msol and h=0.73, you should pass
# 0.73*(1x10^12).

MLO=2.308463e11
MHI=2.308463e12

# You can remove the --mmax option to generate trees with mass exactly equal to
# the value of --mphalo.

./pchtrees --ntrees ${NTREES} --mphalo ${MLO} --mmax ${MHI} --params ${PARAMS} --no-output-trees --process-first-order-progenitors --loguniform
