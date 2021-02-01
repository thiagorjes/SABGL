#include "vgram.h"
#include "vgram_base.h"
#include "vgram_error.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif


int
FindNearestPattern(unsigned int *bit_pattern, int *neuron_memory, int *candidate_patterns, int memory_size, int number_of_synapses_per_neuron, int memory_bit_group_size)
{
	int learned_pattern, nearest_pattern;
	int cur_ham_dist, new_ham_dist;
	int num_candidates = 1;

	cur_ham_dist = number_of_synapses_per_neuron;
	for (learned_pattern = 0; learned_pattern < memory_size; learned_pattern++)
	{
		new_ham_dist = HammingDistance((unsigned int *)
				GetNeuronMemoryBySample(neuron_memory, memory_bit_group_size, learned_pattern),
				bit_pattern, memory_bit_group_size);
		if (new_ham_dist < cur_ham_dist)
		{
			candidate_patterns[0] = learned_pattern;
			num_candidates = 1;
			cur_ham_dist = new_ham_dist;
		}
		else if (new_ham_dist == cur_ham_dist)
		{
			candidate_patterns[num_candidates] = learned_pattern;
			num_candidates++;
		}
	}
	nearest_pattern = candidate_patterns[rand() % num_candidates];
	return (nearest_pattern);
}


void
TestAll(VG_RAM_WNN *vg_ram_wnn, DATA_SET *testing_set)
{
	int sample, neuron;
	int nearest_pattern;
#if defined(_OPENMP)
	int num_threads=omp_get_max_threads();
#else
	int num_threads=1;
#endif
	int *network_input;
	unsigned int *bit_pattern[num_threads];
	int *candidate_patterns[num_threads];
	int tid;

	for(tid = 0; tid < num_threads; tid++)
	{
		if ((bit_pattern[tid] = (unsigned int *) AllocMemory((size_t)vg_ram_wnn->memory_bit_group_size * PATTERN_UNIT_SIZE/8)) == NULL)
			Error("Could not allocate bit_pattern in Test().", "", "");

		if ((candidate_patterns[tid] = (int *) AllocMemory((size_t)vg_ram_wnn->memory_size * sizeof(int))) == NULL)
			Error("Could not allocate candidate_patterns in Test().", "", "");
	}

	#pragma omp parallel default(none) \
			private(tid,sample,neuron,nearest_pattern,network_input) \
			shared(vg_ram_wnn,testing_set,candidate_patterns,bit_pattern)
	{
#if defined(_OPENMP)
		tid = omp_get_thread_num();
#else
		tid = 0;
#endif
		//Distribui as entries (samples de teste) entre as threads que ficarão responsáveis por cada subconjunto do dataset de testes.
		#pragma omp for
		for (sample = 0; sample < testing_set->num_samples; sample++)
		{
			network_input = &(testing_set->sample[sample * vg_ram_wnn->input_size]);
			for (neuron = 0; neuron < vg_ram_wnn->number_of_neurons; neuron++)
			{
				BuildBitPattern(bit_pattern[tid], 
				&(vg_ram_wnn->synapses[neuron * vg_ram_wnn->number_of_synapses_per_neuron]),
				network_input, vg_ram_wnn->number_of_synapses_per_neuron);
				
				nearest_pattern = FindNearestPattern(bit_pattern[tid], 
				GetNeuronMemoryByNeuron(vg_ram_wnn->memories,
							vg_ram_wnn->memory_size,
							vg_ram_wnn->memory_bit_group_size,
							neuron),
				candidate_patterns[tid],
				vg_ram_wnn->memory_size,
				vg_ram_wnn->number_of_synapses_per_neuron, vg_ram_wnn->memory_bit_group_size);
				
				// O último inteiro do vetor de memória contém a saída memorizada junto com o padrão de bits
				vg_ram_wnn->network_output[sample * vg_ram_wnn->number_of_neurons + neuron] =
					GetNeuronMemoryBySample(vg_ram_wnn->memories, vg_ram_wnn->memory_bit_group_size, nearest_pattern)[vg_ram_wnn->memory_bit_group_size];
			}
		}

	}
}


void
Test(VG_RAM_WNN *vg_ram_wnn, int *testing_set, int sample_class_i, int idx)
{
	int sample, neuron;
	int nearest_pattern;
	int *network_input;
	unsigned int *bit_pattern[1];
	int *candidate_patterns[1];

	sample = idx;
	
	network_input = testing_set;


	if ((bit_pattern[0] = (unsigned int *) AllocMemory((size_t)vg_ram_wnn->memory_bit_group_size * PATTERN_UNIT_SIZE/8)) == NULL)
		Error("Could not allocate bit_pattern in Test().", "", "");

	if ((candidate_patterns[0] = (int *) AllocMemory((size_t)vg_ram_wnn->memory_size * sizeof(int))) == NULL)
		Error("Could not allocate candidate_patterns in Test().", "", "");
	
	
	for (neuron = 0; neuron < vg_ram_wnn->number_of_neurons; neuron++)
	{
		BuildBitPattern(bit_pattern[0], 
		&(vg_ram_wnn->synapses[neuron * vg_ram_wnn->number_of_synapses_per_neuron]),
		network_input, vg_ram_wnn->number_of_synapses_per_neuron);
		
		nearest_pattern = FindNearestPattern(bit_pattern[0], 
		GetNeuronMemoryByNeuron(vg_ram_wnn->memories,
					vg_ram_wnn->memory_size,
					vg_ram_wnn->memory_bit_group_size,
					neuron),
		candidate_patterns[0],
		vg_ram_wnn->memory_size,
		vg_ram_wnn->number_of_synapses_per_neuron, vg_ram_wnn->memory_bit_group_size);
		
		// O último inteiro do vetor de memória contém a saída memorizada junto com o padrão de bits
		vg_ram_wnn->network_output[sample * vg_ram_wnn->number_of_neurons + neuron] =
			GetNeuronMemoryBySample(vg_ram_wnn->memories, vg_ram_wnn->memory_bit_group_size, nearest_pattern)[vg_ram_wnn->memory_bit_group_size];
	}
}

