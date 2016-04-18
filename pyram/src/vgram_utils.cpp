#include "vgram_utils.h"
#include "vgram_base.h"
#include <stdlib.h>
#include <sys/time.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>
#include <float.h> //DBL_MAX


double
minOf3(double x, double y, double z)
{
	return std::min(std::min(x, y), z);
}


int
argminOf3(double x, double y, double z)
{
	return (x < y ? (x < z ? 0 : 2) : y < z ? 1 : 2);
}


double
minOfN(double *values, int size)
{
	double minval = DBL_MAX;
	for(int c=0; c < size; c++)
		if (values[c] <= minval)
			minval = values[c];
	return minval;
}


int
argminOfN(double *values, int size)
{
	int index = 0;
	double minval = DBL_MAX;
	for(int c=0; c < size; c++)
		if (values[c] <= minval)
		{
			minval = values[c];
			index = c;
		}
	return index;
}


int
index3D(int neuron, int row_i, int col_j, int rows, int cols)
{
	return neuron * rows * cols + row_i * cols + col_j;
}


int
index2D(int row_i, int col_j, int rows, int cols)
{
	return row_i * cols + col_j;
}


void
copy_reverse(int *data, size_t size)
{
	for (size_t i=0; i < size/2; i++)
	{
		int tmp = data[i];
		data[i] = data[size - 1 - i];
		data[size - 1 - i] = tmp;
	}
}


void
SaveImage(const char *name, int *data, int height, int width, bool normalized = false)
{
	long double timestamp;
	struct timeval tv;
	std::stringstream filename;
//	usleep(100000);
	gettimeofday(&tv, NULL);
	timestamp = 1000000 * tv.tv_sec + tv.tv_usec;
	filename << name << std::setprecision(17) << timestamp << ".pgm";
	std::ofstream fout (filename.str().c_str());

	// find max value
	int max_value = -1;
	int min_value = 1000;
	for(int i=0; i < height; i++)
	{
		for(int j=0; j < width; j++)
		{
			int value = data[index2D(i, j, height, width)];
			if (value > max_value)
				max_value = value;
			if (value < min_value)
				min_value = value;
		}
	}

	// write the header
	fout << "P2 " << width << " " << height << " ";
	if (normalized)
		fout << 255 << std::endl;
	else
		fout << max_value << std::endl;

	// write the data
	for(int i=height-1; i >= 0; i--)
	{
		for(int j=0; j < width; j++)
		{
			int value = data[index2D(i, j, height, width)];
			if (normalized)
			{
				int normalized_value = (int)( (float)(value - min_value) * 255.0 / (float)(max_value - min_value));
				fout << normalized_value << " ";
			}
			else
			{
				fout << value << " ";
			}
		}
		fout << std::endl;
	}
	// close the stream
	fout.close();
}


void
SaveLocalCostMatrix(int *cost, int number_of_neurons, int height, int width)
{
	int *data = (int *) AllocMemory((size_t) height * width * sizeof(int));
	for(int i=0; i < height; i++)
	{
		for(int j=0; j < width; j++)
		{
			long int sum = 0;
			for (int neuron = 0; neuron < number_of_neurons; neuron++)
			{
				sum += cost[index3D(neuron, i, j, height, width)];
			}
			data[index2D(i, j, height, width)] = sum / number_of_neurons;
		}
	}
	SaveImage("neuron_cost_matrix_mean", data, height, width);
	SaveImage("neuron_cost_matrix_zero", &cost[height*width], height, width);
	SaveImage("neuron_cost_matrix_one", &cost[1*height*width], height, width);
	SaveImage("neuron_cost_matrix_two", &cost[2*height*width], height, width);
	SaveImage("neuron_cost_matrix_centre", &cost[(number_of_neurons/2)*height*width], height, width);
	free(data);
}


void
SaveMeanCostMatrix(double *cost, int height, int width)
{
	int *data = (int *) AllocMemory((size_t) height * width * sizeof(int));
	for(int i=0; i < height; i++)
	{
		for(int j=0; j < width; j++)
		{
			data[index2D(i, j, height, width)] = (int)cost[index2D(i, j, height, width)];
		}
	}
	SaveImage("mean_cost_matrix", data, height, width);
	free(data);
}


void
SaveAccumCostMatrix(double *cost, int height, int width)
{
	int *data = (int *) AllocMemory((size_t) height * width * sizeof(int));
	for(int i=0; i < height; i++)
	{
		for(int j=0; j < width; j++)
		{
			data[index2D(i, j, height, width)] = (int)cost[index2D(i, j, height, width)];
		}
	}
	SaveImage("accumulated_cost_matrix", data, height, width, true);
	free(data);
}


