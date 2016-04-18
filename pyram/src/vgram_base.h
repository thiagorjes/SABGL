#ifndef VGRAM_BASE_H_
#define VGRAM_BASE_H_

#include <stdio.h>

void *AllocMemory (size_t size);

void SetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample, int value);

int GetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample);

int *GetNeuronMemoryByNeuronAndSample(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample);

int *GetNeuronMemoryByNeuron(int *mem, int memory_size, int memory_bit_group_size, int neuron);

int *GetNeuronMemoryBySample(int *mem, int memory_bit_group_size, int sample);

void BuildBitPattern(unsigned int *bit_pattern, int *synapses, int *network_input, int number_of_synapses_per_neuron);

int HammingDistance(unsigned int *bit_pattern1, unsigned int *bit_pattern2, int memory_bit_group_size);

#endif /* VGRAM_BASE_H_ */
