#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include "vgram.h"
#include "vgram_base.cuh"


__device__ int
cudaHammingDistance(unsigned int *bit_pattern1, unsigned int *bit_pattern2,  int memory_bit_group_size)
{
	int i;
	unsigned int bit_difference;
	int hamming_distance;

	hamming_distance = 0;
	for (i = 0; i < memory_bit_group_size; i++)
	{
		bit_difference = bit_pattern1[i] ^ bit_pattern2[i];
		hamming_distance += __popc(bit_difference);
	}
	return hamming_distance;
}


__device__ void
cudaFindNearestPattern(unsigned int *bit_pattern,
		int *neuron_memory,
		int *neuron_output,
		int memory_size,
		int memory_bit_group_size)
{
	int hd_register;
	__shared__ int short_hd;

	int best_hd = 1<<30;
	int best_sample;
	short_hd = 1<<30;

	__syncthreads();

	for (int learned_pattern = threadIdx.x; learned_pattern < memory_size; learned_pattern += blockDim.x)
	{
		hd_register = cudaHammingDistance((unsigned int *)
				cudaGetNeuronMemoryBySample(neuron_memory, memory_bit_group_size, learned_pattern),
				bit_pattern, memory_bit_group_size);

		if(hd_register < best_hd)
		{
			best_hd = hd_register;
			best_sample = learned_pattern;
		}
	}

	atomicMin(&short_hd, best_hd);

	__syncthreads();

	if(short_hd == best_hd)
	{
		*neuron_output = cudaGetNeuronMemoryBySample(neuron_memory, memory_bit_group_size, best_sample)[memory_bit_group_size];
	}

}


__global__ void
cudaNeuronTest(VG_RAM_WNN vg_ram_wnn, DATA_SET testing_set, int sample)
{
	__shared__
	unsigned int bit_pattern[BLOCK_DIM];

	for (int neuron = blockIdx.x; neuron < vg_ram_wnn.number_of_neurons; neuron += gridDim.x)
	{
		cudaBuildBitPattern(bit_pattern,
				&(vg_ram_wnn.d_synapses[neuron * vg_ram_wnn.number_of_synapses_per_neuron]),
				&(testing_set.d_sample[sample * vg_ram_wnn.input_size]),
				vg_ram_wnn.number_of_synapses_per_neuron);

		cudaFindNearestPattern(bit_pattern,
				cudaGetNeuronMemoryByNeuron(vg_ram_wnn.d_memories,
						vg_ram_wnn.memory_size,
						vg_ram_wnn.memory_bit_group_size,
						neuron),
						&(vg_ram_wnn.d_network_output[sample * vg_ram_wnn.number_of_neurons + neuron]),
						vg_ram_wnn.memory_size,
						vg_ram_wnn.memory_bit_group_size);
	}
}


void
Test(VG_RAM_WNN *vg_ram_wnn, DATA_SET *testing_set)
{
	int *d_sample_class;
	int *d_sample;
	int *d_network_output;

	#ifdef TRAIN_ON_CPU
	int *d_synapses;
	int *d_memories;

	cudaCheckError(cudaMalloc((int **) &(d_synapses),
			vg_ram_wnn->number_of_synapses_per_neuron * vg_ram_wnn->number_of_neurons * sizeof(int)));

	cudaMemcpy(d_synapses, vg_ram_wnn->synapses,
			vg_ram_wnn->number_of_neurons * vg_ram_wnn->number_of_synapses_per_neuron * sizeof(int),
			cudaMemcpyHostToDevice);

	vg_ram_wnn->d_synapses = d_synapses;

	cudaCheckError(cudaMalloc((int**) &(d_memories),
			vg_ram_wnn->number_of_neurons * vg_ram_wnn->memory_size * (vg_ram_wnn->memory_bit_group_size + 1) * sizeof(int)));

	cudaMemcpy(d_memories, vg_ram_wnn->memories,
			vg_ram_wnn->number_of_neurons * vg_ram_wnn->memory_size * (vg_ram_wnn->memory_bit_group_size + 1) * sizeof(int),
			cudaMemcpyHostToDevice);

	vg_ram_wnn->d_memories = d_memories;
	#endif

	cudaCheckError(cudaMalloc((int **) &(d_sample_class),
			testing_set->num_samples * sizeof(int)));
	cudaCheckError(cudaMalloc((int **) &(d_sample),
			testing_set->num_inputs * testing_set->num_samples * sizeof(int)));

	cudaMemcpy(d_sample_class, testing_set->sample_class,
			testing_set->num_samples * sizeof(int),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample, testing_set->sample,
			testing_set->num_samples * testing_set->num_inputs * sizeof(int),
			cudaMemcpyHostToDevice);

	testing_set->d_sample = d_sample;
	testing_set->d_sample_class = d_sample_class;

	cudaCheckError(cudaMalloc((int **) &(d_network_output),
			vg_ram_wnn->test_size * vg_ram_wnn->number_of_neurons * sizeof(int)));

	vg_ram_wnn->d_network_output = d_network_output;

	for (int sample = 0; sample < testing_set->num_samples; sample++)
	{
		cudaNeuronTest<<<1024,BLOCK_DIM>>>(*vg_ram_wnn,*testing_set,sample);
	}

	cudaMemcpy(vg_ram_wnn->network_output, d_network_output,
			vg_ram_wnn->test_size * vg_ram_wnn->number_of_neurons * sizeof(int),
			cudaMemcpyDeviceToHost);

	cudaFree(d_sample);
	cudaFree(d_sample_class);
	cudaFree(d_network_output);
	cudaFree(vg_ram_wnn->d_synapses);
	cudaFree(vg_ram_wnn->d_memories);
}

