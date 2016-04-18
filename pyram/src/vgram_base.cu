#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include "vgram.h"
#include "vgram_base.cuh"

#define BLOCK_DIM (PATTERN_UNIT_SIZE*8)

__device__ void
cudaSetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample, int value)
{
	mem[(neuron * memory_size + sample) * (memory_bit_group_size+1) + memory_bit_group_size] = value;
}


__device__ int
cudaGetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample)
{
	return mem[(neuron * memory_size + sample) * (memory_bit_group_size+1) + memory_bit_group_size];
}


__device__ int *
cudaGetNeuronMemoryByNeuronAndSample(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample)
{
	return &mem[(neuron * memory_size + sample) * (memory_bit_group_size+1)];
}


__device__ int *
cudaGetNeuronMemoryByNeuron(int *mem, int memory_size, int memory_bit_group_size, int neuron)
{
	return &mem[neuron * memory_size * (memory_bit_group_size+1)];
}


__device__ int *
cudaGetNeuronMemoryBySample(int *mem, int memory_bit_group_size, int sample)
{
	return &mem[sample * (memory_bit_group_size+1)];
}


__device__ void
cudaBuildBitPattern(unsigned int *bit_pattern, int *synapses, int *network_input, int number_of_synapses_per_neuron)
{
	int synapse;
	int current_bit_pattern_group;
	int tid = threadIdx.x%PATTERN_UNIT_SIZE;
	__shared__ unsigned int bp[BLOCK_DIM];
	unsigned int aux;

	if(threadIdx.x<BLOCK_DIM) bp[threadIdx.x]=0;

	__syncthreads();

	for (synapse = threadIdx.x; synapse < number_of_synapses_per_neuron -1; synapse += blockDim.x)
	{
		// Cada grupo de padrão de bits tem tamanho igual a PATTERN_UNIT_SIZE.
		current_bit_pattern_group = synapse / PATTERN_UNIT_SIZE;

		// Minchington: Uma sinapse compara com a próxima
		aux = (network_input[synapses[synapse]] > network_input[synapses[synapse+1]]) ? 1 : 0;

		aux  = aux << tid;

		atomicOr(&bp[current_bit_pattern_group],aux);

	}
	__syncthreads();

	if(threadIdx.x == blockDim.x)
	{
		current_bit_pattern_group = synapse / PATTERN_UNIT_SIZE;
		aux  = aux << tid;
		aux = (network_input[synapses[synapse]] > network_input[synapses[0]]) ? 1 : 0;
		atomicOr(&bp[current_bit_pattern_group],aux);
	}

	__syncthreads();

	if(threadIdx.x < number_of_synapses_per_neuron / PATTERN_UNIT_SIZE)
	{
		bit_pattern[threadIdx.x] =  bp[threadIdx.x];
	}
}

void cudaAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Error assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
