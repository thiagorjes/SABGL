#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include "vgram.h"

#define BLOCK_DIM (PATTERN_UNIT_SIZE*8)

__device__ void
cudaSetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample, int value);

__device__ int
cudaGetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample);

__device__ int *
cudaGetNeuronMemoryByNeuronAndSample(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample);

__device__ int *
cudaGetNeuronMemoryByNeuron(int *mem, int memory_size, int memory_bit_group_size, int neuron);

__device__ int *
cudaGetNeuronMemoryBySample(int *mem, int memory_bit_group_size, int sample);

__device__ void
cudaBuildBitPattern(unsigned int *bit_pattern, int *synapses, int *network_input, int number_of_synapses_per_neuron);

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }

void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true);
