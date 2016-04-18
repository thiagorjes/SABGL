#include "vgram.h"
#include "vgram_base.h"
#include "vgram_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *
AllocMemory (size_t size)
{
	void *pointer;

	if ((pointer = calloc (1, size)) == (void *) NULL)
		Error("cannot alloc more memory.", "", "");

	return (pointer);
}


void
SetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample, int value)
{
	mem[(neuron * memory_size + sample) * (memory_bit_group_size+1) + memory_bit_group_size] = value;
}


int
GetNeuronMemory(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample)
{
	return mem[(neuron * memory_size + sample) * (memory_bit_group_size+1) + memory_bit_group_size];
}


int *
GetNeuronMemoryByNeuronAndSample(int *mem, int memory_size, int memory_bit_group_size, int neuron, int sample)
{
	return &mem[(neuron * memory_size + sample) * (memory_bit_group_size+1)];
}


int *
GetNeuronMemoryByNeuron(int *mem, int memory_size, int memory_bit_group_size, int neuron)
{
	return &mem[neuron * memory_size * (memory_bit_group_size+1)];
}


int *
GetNeuronMemoryBySample(int *mem, int memory_bit_group_size, int sample)
{
	return &mem[sample * (memory_bit_group_size+1)];
}


void
BuildBitPattern(unsigned int *bit_pattern, int *synapses, int *network_input, int number_of_synapses_per_neuron)
{
	int shift;
	int synapse;
	int current_bit_pattern_group;
	int bit_value;

	current_bit_pattern_group = -1;
	for (synapse = 0; synapse < number_of_synapses_per_neuron; synapse++)
	{
		shift=synapse % PATTERN_UNIT_SIZE;
		current_bit_pattern_group=synapse / PATTERN_UNIT_SIZE;
		if (!shift)
			bit_pattern[current_bit_pattern_group] = 0; // Se começou um novo grupo, zero o mesmo.

		if (synapse == (number_of_synapses_per_neuron - 1))
			// Minchington: Última sinapse compara com a primeira
			bit_value = (network_input[synapses[synapse]] > network_input[synapses[0]]) ? 1 : 0;

		else
		{	// Minchington: Uma sinapse compara com a próxima
			bit_value = (network_input[synapses[synapse]] > network_input[synapses[synapse+1]]) ? 1 :0;
		}
		// Novos bits são inseridos na parte alta do padrão de bits.
		bit_pattern[current_bit_pattern_group] |= bit_value << shift;
	}
	// Novos bits são inseridos na parte alta do padrão de bits. Assim, ao fim da inserção, alinha os bits junto a parte baixa.

}


int
HammingDistance(unsigned int *bit_pattern1, unsigned int *bit_pattern2, int memory_bit_group_size)
{
	int i;
	unsigned int bit_difference;
	int hamming_distance;

	hamming_distance = 0;
	for (i = 0; i < memory_bit_group_size; i++)
	{
		bit_difference = bit_pattern1[i] ^ bit_pattern2[i];
		hamming_distance += __builtin_popcount(bit_difference);
	}
	return hamming_distance;
}


