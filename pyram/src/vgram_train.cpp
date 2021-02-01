#include "vgram.h"
#include "vgram_base.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

void
TrainAll(VG_RAM_WNN *vg_ram_wnn, DATA_SET *training_set)
{
	int sample, neuron;
	int sample_class;
	int *network_input;

	// #pragma omp parallel for default(none) \
	// 		private(sample,neuron,network_input,sample_class) \
	// 		shared(training_set,vg_ram_wnn)
	for (sample = 0; sample < training_set->num_samples; sample++)
	{
		//printf("for 1\n");
		network_input = &training_set->sample[sample * vg_ram_wnn->input_size];
		//printf("network input\n");
		sample_class = training_set->sample_class[sample];
		//printf("sample class %d\n",sample_class);
		for (neuron = 0; neuron < vg_ram_wnn->number_of_neurons; neuron++)
		{
			//printf("for 2\n");
			// Escreve diretamente na memória do neurônio o padrão de bits gerado a partir da sample
			BuildBitPattern((unsigned int*)
					GetNeuronMemoryByNeuronAndSample(
							vg_ram_wnn->memories,
							vg_ram_wnn->memory_size,
							vg_ram_wnn->memory_bit_group_size,
							neuron, sample),
					&(vg_ram_wnn->synapses[neuron * vg_ram_wnn->number_of_synapses_per_neuron]),
					network_input, vg_ram_wnn->number_of_synapses_per_neuron);
			// Escreve diretamente na memória do neurônio a classe associada ao padrão de bits
			SetNeuronMemory(vg_ram_wnn->memories,
					vg_ram_wnn->memory_size,
					vg_ram_wnn->memory_bit_group_size,
					neuron, sample, sample_class);
			//printf("saindo do for 2");
		}
	}
}

void
Train(VG_RAM_WNN *vg_ram_wnn, int *training_set, int sample_class_i, int idx)
{
	int sample, neuron;
	int sample_class;
	int *network_input;

	sample = idx;
	
	network_input = training_set;
	
	sample_class = sample_class_i;

	for (neuron = 0; neuron < vg_ram_wnn->number_of_neurons; neuron++)
	{
		// Escreve diretamente na memória do neurônio o padrão de bits gerado a partir da sample
		BuildBitPattern((unsigned int*)
				GetNeuronMemoryByNeuronAndSample(
						vg_ram_wnn->memories,
						vg_ram_wnn->memory_size,
						vg_ram_wnn->memory_bit_group_size,
						neuron, sample),
				&(vg_ram_wnn->synapses[neuron * vg_ram_wnn->number_of_synapses_per_neuron]),
				network_input, vg_ram_wnn->number_of_synapses_per_neuron);
		// Escreve diretamente na memória do neurônio a classe associada ao padrão de bits
		SetNeuronMemory(vg_ram_wnn->memories,
				vg_ram_wnn->memory_size,
				vg_ram_wnn->memory_bit_group_size,
				neuron, sample, sample_class);
	}
}

