#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include "vgram.h"
#include "vgram_base.cuh"


__global__ void
cudaNeuronTrain(VG_RAM_WNN vg_ram_wnn, DATA_SET training_set, int sample)
{
	int sample_class;

	sample_class = training_set.d_sample_class[sample];
	for (int neuron = blockIdx.x; neuron < vg_ram_wnn.number_of_neurons; neuron += gridDim.x)
	{
		// Escreve diretamente na memória do neurônio o padrão de bits gerado a partir da sample
		cudaBuildBitPattern((unsigned int *)
				cudaGetNeuronMemoryByNeuronAndSample(
						vg_ram_wnn.d_memories,
						vg_ram_wnn.memory_size,
						vg_ram_wnn.memory_bit_group_size,
						neuron, sample),
						&(vg_ram_wnn.d_synapses[neuron * vg_ram_wnn.number_of_synapses_per_neuron]),
						&(training_set.d_sample[sample * vg_ram_wnn.input_size]),
						vg_ram_wnn.number_of_synapses_per_neuron);

		// Escreve diretamente na memória do neurônio a classe associada ao padrão de bits
		cudaSetNeuronMemory(vg_ram_wnn.d_memories,
				vg_ram_wnn.memory_size,
				vg_ram_wnn.memory_bit_group_size,
				neuron, sample, sample_class);
	}

}


void
Train(VG_RAM_WNN *vg_ram_wnn, DATA_SET *training_set)
{
	int *d_sample;
	int *d_sample_class;
	int *d_synapses;
	int *d_memories;

	cudaCheckError(cudaMalloc((int **) &(d_sample_class),
			training_set->num_samples * sizeof(int)));
	cudaCheckError(cudaMalloc((int **) &(d_sample),
			training_set->num_inputs * training_set->num_samples * sizeof(int)));

	cudaMemcpy(d_sample_class, training_set->sample_class,
			training_set->num_samples * sizeof(int),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample, training_set->sample,
			training_set->num_samples * training_set->num_inputs * sizeof(int),
			cudaMemcpyHostToDevice);

	training_set->d_sample = d_sample;
	training_set->d_sample_class = d_sample_class;

	cudaCheckError(cudaMalloc((int **) &(d_synapses),
			vg_ram_wnn->number_of_synapses_per_neuron * vg_ram_wnn->number_of_neurons * sizeof(int)));

	cudaMemcpy(d_synapses, vg_ram_wnn->synapses,
			vg_ram_wnn->number_of_neurons * vg_ram_wnn->number_of_synapses_per_neuron * sizeof(int),
			cudaMemcpyHostToDevice);

	vg_ram_wnn->d_synapses = d_synapses;

	cudaCheckError(cudaMalloc((int**) &(d_memories),
			vg_ram_wnn->number_of_neurons * vg_ram_wnn->memory_size * (vg_ram_wnn->memory_bit_group_size + 1) * sizeof(int)));

	vg_ram_wnn->d_memories = d_memories;

	for (int sample = 0; sample < training_set->num_samples; sample ++)
	{
		cudaNeuronTrain<<<1024,BLOCK_DIM>>>(*vg_ram_wnn,*training_set,sample);
	}
	#ifdef TEST_ON_CPU
	cudaMemcpy(vg_ram_wnn->memories, d_memories,
			vg_ram_wnn->number_of_neurons * vg_ram_wnn->memory_size * (vg_ram_wnn->memory_bit_group_size + 1) * sizeof(int),
			cudaMemcpyDeviceToHost);
	#endif
	cudaFree(d_sample);
	cudaFree(d_sample_class);
}


