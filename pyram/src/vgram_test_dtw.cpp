#include "vgram.h"
#include "vgram_base.h"
#include "vgram_error.h"
#include "vgram_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm> //min/max
#include <float.h> //DBL_MAX
#include <cmath> //sqrt
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

//#include <sys/time.h>

void
BuildBitPatternForQuery(VG_RAM_WNN *vg_ram_wnn, DATA_SET *testing_set, int *query)
{
	int sample, neuron;
	int nearest_pattern;
	int *network_input;

	#pragma omp parallel default(none) \
			private(sample,neuron,network_input) \
			shared(vg_ram_wnn,testing_set,query)
	for (sample = 0; sample < testing_set->num_samples; sample++)
	{
		network_input = &(testing_set->sample[sample * vg_ram_wnn->input_size]);
		for (neuron = 0; neuron < vg_ram_wnn->number_of_neurons; neuron++)
		{
			BuildBitPattern((unsigned int*)
					GetNeuronMemoryByNeuronAndSample(query, testing_set->num_samples,
							vg_ram_wnn->memory_bit_group_size,
							neuron, sample),
							&(vg_ram_wnn->synapses[neuron * vg_ram_wnn->number_of_synapses_per_neuron]),
							network_input, vg_ram_wnn->number_of_synapses_per_neuron);
		}
	}
}


void
ComputeLocalCostForNeurons(VG_RAM_WNN *vg_ram_wnn, int *query, int query_size, int *neuron_cost)
{
	int *data = vg_ram_wnn->memories;
	int data_size = vg_ram_wnn->memory_size;
	int number_of_neurons = vg_ram_wnn->number_of_neurons;
	int memory_bit_group_size = vg_ram_wnn->memory_bit_group_size;

	#pragma omp parallel default(none) \
	shared(data,query,data_size,query_size,memory_bit_group_size,number_of_neurons,neuron_cost)
	{
		for (int neuron = 0; neuron < number_of_neurons; neuron++)
		{
			//percorrer memoria do neuronio
			int *seqData = GetNeuronMemoryByNeuron(data, data_size, memory_bit_group_size, neuron);
			//percorrer amostras de testes e extrair features para memoria temporaria
			int *seqQuery = GetNeuronMemoryByNeuron(query, query_size, memory_bit_group_size, neuron);
			//calcular matriz de custo local para cada neuronio

			#pragma omp for
			for (int i = 0; i < query_size; i++)
			{
				for (int j = 0; j < data_size; j++)
				{
					int distance = HammingDistance(
							(unsigned int *)GetNeuronMemoryBySample(seqData, memory_bit_group_size, j),
							(unsigned int *)GetNeuronMemoryBySample(seqQuery, memory_bit_group_size, i),
							memory_bit_group_size);
					neuron_cost[index3D(neuron, i, j, query_size, data_size)] = distance;
				}
			}
		}
	}
}


void
ComputeMeanCostForNeurons(int *neuron_cost, int query_size, int data_size, int number_of_neurons, double *mean_cost)
{
	double sum;
	for(int i=0; i < query_size; i++)
	{
		for(int j=0; j < data_size; j++)
		{
			sum = 0.0;
			#pragma omp parallel for reduction(+:sum)
			for (int neuron = 0; neuron < number_of_neurons; neuron++)
				sum += (double)neuron_cost[index3D(neuron, i, j, query_size, data_size)];
			mean_cost[index2D(i, j, query_size, data_size)] = sum / (double)number_of_neurons;
		}
	}
}


void
ComputeAccumulatedCost(double *mean_cost, int query_size, int data_size, int *step_list, int step_list_size, double *accumulated_cost)
{
	int step_max_i = -1;
	int step_max_j = -1;

	for(int k=0; k < step_list_size; k++)
	{
		step_max_i = std::max(step_max_i, step_list[k*2]);
		step_max_j = std::max(step_max_j, step_list[k*2+1]);
	}

	// zero out the accumulated_cost matrix
	for(int i=0; i < query_size; i++)
		for(int j=0; j < data_size; j++)
			accumulated_cost[index2D(i, j, query_size, data_size)] = 0.0;

	// compute the accumulated cost for the first element (0,0) of all classifiers
	accumulated_cost[index2D(0, 0, query_size, data_size)] = mean_cost[index2D(0, 0, query_size, data_size)];

	// compute the accumulated cost along the first (or last) column of all classifiers (note the sum of mean cost across rows)
	for(int i=1; i < query_size; i++)
		accumulated_cost[index2D(i, 0, query_size, data_size)] = accumulated_cost[index2D(i-1, 0, query_size, data_size)] + mean_cost[index2D(i, 0, query_size, data_size)];

	// compute the accumulated cost along the first (or last) row of all classifiers
	for(int j=1; j < data_size; j++)
		accumulated_cost[index2D(0, j, query_size, data_size)] = mean_cost[index2D(0, j, query_size, data_size)];

	// compute the accumulated cost using dynamic programming
	for(int i = step_max_i; i < query_size; i++)
	{
		for(int j = step_max_j; j < data_size; j++)
		{
			double cost_list[step_list_size];
			for(int k = 0; k < step_list_size; k++)
				cost_list[k] = DBL_MAX;
			for(int k = 0; k < step_list_size; k++)
			{
				int step_i = i-step_list[2*k];
				int step_j = j-step_list[2*k+1];
				if ( (step_i >= 0) and (step_j >= 0) )
					cost_list[k] = accumulated_cost[index2D(step_i, step_j, query_size, data_size)];
			}
			accumulated_cost[index2D(i, j, query_size, data_size)] = minOfN(cost_list, step_list_size) + mean_cost[index2D(i, j, query_size, data_size)];
		}
	}
}


void
ComputeDTWHammingDistance(VG_RAM_WNN *vg_ram_wnn, DATA_SET *testing_set, double *accumulated_cost, int *step_list, int step_list_size)
{
	int data_size = vg_ram_wnn->memory_size;
	int query_size = testing_set->num_samples;
	int number_of_neurons = vg_ram_wnn->number_of_neurons;
	int memory_bit_group_size = vg_ram_wnn->memory_bit_group_size;

	int *query = (int *) AllocMemory((size_t) number_of_neurons * query_size * (memory_bit_group_size + 1) * sizeof(int));
	int *neuron_cost = (int *) AllocMemory((size_t) number_of_neurons * data_size * query_size * sizeof(int));
	double *mean_cost = (double *) AllocMemory((size_t) data_size * query_size * sizeof(double));

	BuildBitPatternForQuery(vg_ram_wnn, testing_set, query);

	ComputeLocalCostForNeurons(vg_ram_wnn, query, query_size, neuron_cost);
//	SaveLocalCostMatrix(neuron_cost, number_of_neurons, query_size, data_size);

	ComputeMeanCostForNeurons(neuron_cost, query_size, data_size, number_of_neurons, mean_cost);
//	SaveMeanCostMatrix(mean_cost, query_size, data_size);

	ComputeAccumulatedCost(mean_cost, query_size, data_size, step_list, step_list_size, accumulated_cost);
//	SaveAccumCostMatrix(accumulated_cost, query_size, data_size);

	free(query);
	free(mean_cost);
	free(neuron_cost);
}


int
FindOptimalWarpingPath(double *cost_matrix, int *query_path, int *data_path, int max_path_size, int query_size, int data_size, int i, int j, int *step_list, int step_list_size)
{
	int path_size = 1;
	int step_max_i = -1;
	int step_max_j = -1;

	for(int k=0; k < step_list_size; k++)
	{
		step_max_i = std::max(step_max_i, step_list[k*2]);
		step_max_j = std::max(step_max_j, step_list[k*2+1]);
	}

	query_path[0] = i;
	data_path[0] = j;

	while ((i-step_max_i > 0) || (j-step_max_j > 0))
	{
		double cost_list[step_list_size];
		for(int k = 0; k < step_list_size; k++)
			cost_list[k] = DBL_MAX;
		for(int k = 0; k < step_list_size; k++)
		{
			int step_i = i-step_list[2*k];
			int step_j = j-step_list[2*k+1];
			if ( (step_i >= 0) and (step_j >= 0) )
				cost_list[k] = cost_matrix[index2D(step_i, step_j, query_size, data_size)];
		}
		int k = argminOfN(cost_list, step_list_size);
		i = i - step_list[2*k];
		j = j - step_list[2*k+1];

		if ( (i >= 0) && (j >= 0) && (path_size < max_path_size) )
		{
			query_path[path_size] = i;
			data_path[path_size] = j;
			path_size++;
		}
	}

	copy_reverse(query_path, path_size);
	copy_reverse(data_path, path_size);

	return path_size;
}


int FindUpperBound(double *cost, int size)
{
	int argmin = -1;
	double valmin = DBL_MAX;
	for (int i = 0; i < size; i++)
	{
		if (cost[i] < valmin)
		{
			valmin = cost[i];
			argmin = i;
		}
	}
	return argmin;
}


int FindLowerBound(int *query_path, int path_size)
{
	int k = -1;
	while ( (k+1 < path_size) && (query_path[k+1] == query_path[0]) ) //finds the last query path repeating element
	{
		k++;
	}
	return k;
}


int
FindSubsequenceWarpingPath(double *cost, int *query_path, int *data_path, int max_path_size, int query_size, int data_size, int *step_list, int step_list_size)
{
	int a_star, b_star, path_size, full_path_size;
	b_star = FindUpperBound(&(cost[index2D(query_size-1, 0, query_size, data_size)]), data_size);

	full_path_size = FindOptimalWarpingPath(cost, query_path, data_path, max_path_size, query_size, data_size, query_size-1, b_star, step_list, step_list_size);

	a_star = FindLowerBound(query_path, full_path_size);

	path_size = 0;
	for(int k=0; k < full_path_size; k++)
	{
		if ( (data_path[k] >= a_star) and (data_path[k] <= b_star) )
		{
			query_path[path_size] = query_path[k];
			data_path[path_size] = data_path[k];
			path_size++;
		}
	}
	return path_size;
}


void
TestSequence(VG_RAM_WNN *vg_ram_wnn, DATA_SET *testing_set, int *step_list, int step_list_size)
{
	double *cost;
	int *query_path;
	int *data_path;
	int path_size;
	int data_size = vg_ram_wnn->memory_size;
	int query_size = vg_ram_wnn->test_size;
	int max_path_size = std::max(data_size, query_size);

	//struct timeval start, end;
	//gettimeofday(&start, NULL);

	if ((cost = (double *) AllocMemory((size_t) data_size * query_size * sizeof(double))) == NULL)
		Error("Could not allocate memory in TestSequence().","","");
	if ((query_path = (int *) AllocMemory((size_t) max_path_size * sizeof(int))) == NULL)
		Error("Could not allocate memory in TestSequence().","","");
	if ((data_path = (int *) AllocMemory((size_t) max_path_size * sizeof(int))) == NULL)
		Error("Could not allocate memory in TestSequence().","","");

	ComputeDTWHammingDistance(vg_ram_wnn, testing_set, cost, step_list, step_list_size);

	path_size = FindSubsequenceWarpingPath(cost, query_path, data_path, max_path_size, query_size, data_size, step_list, step_list_size);

	for(int k=path_size-1; k >= 0; k--)
	{
		//FIXME: query_path[k] overwrites potential duplicated indices
		int data_label = GetNeuronMemoryBySample(vg_ram_wnn->memories, vg_ram_wnn->memory_bit_group_size, data_path[k])[vg_ram_wnn->memory_bit_group_size];
		vg_ram_wnn->network_output[query_path[k]] = data_label;
	}

	free(query_path);
	free(data_path);
	free(cost);

	//gettimeofday(&end, NULL);
	//printf("Elapsed Time: %.2lfs\n", ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1e6);
}
