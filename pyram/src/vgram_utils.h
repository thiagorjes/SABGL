#ifndef VGRAM_UTILS_H_
#define VGRAM_UTILS_H_

#include <stdio.h>

double minOf3(double x, double y, double z);
double minOfN(double *values, int size);
int argminOf3(double x, double y, double z);
int argminOfN(double *values, int size);

int index2D(int row_i, int col_j, int rows, int cols);
int index3D(int neuron, int row_i, int col_j, int rows, int cols);

void copy_reverse(int *data, size_t size);

void SaveLocalCostMatrix(int *cost, int number_of_neurons, int height, int width);
void SaveMeanCostMatrix(double *cost, int height, int width);
void SaveAccumCostMatrix(double *cost, int height, int width);

#endif /* VGRAM_UTILS_H_ */
