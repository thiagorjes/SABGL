#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vgram.h"


float
EvaluateNetworkOutput(int *network_output_class, int *network_output, int number_of_neurons)
{
	int neuron_i, neuron_j;
	int class_aux, most_voted_class = 0;
	int aux, max = 0;
	float confidence;

	for (neuron_i = 0; neuron_i < number_of_neurons; neuron_i++)
	{
		class_aux = network_output[neuron_i];
		aux = 1;
		for (neuron_j = neuron_i + 1; neuron_j < number_of_neurons; neuron_j++)
		{
			if (network_output[neuron_j] == class_aux)
				aux++;
		}
		if (aux > max)
		{
			max = aux;
			most_voted_class = class_aux;
		}
	}
	confidence = ((float) max / (float) number_of_neurons);

	*network_output_class = most_voted_class;
	return confidence;
}


void
EvaluateNetworkOutputs(const VG_RAM_WNN *vg_ram_wnn, const DATA_SET *testing_set)
{
	int sample;
	int correct;
	int sample_class;
	int network_output_class;
	float largest_function_f_value;

	correct = 0;
	for (sample = 0; sample < testing_set->num_samples; sample ++)
	{
		sample_class = testing_set->sample_class[sample];

		largest_function_f_value = EvaluateNetworkOutput(
				&network_output_class,
				&(vg_ram_wnn->network_output[sample * vg_ram_wnn->number_of_neurons]),
				vg_ram_wnn->number_of_neurons);

		if (network_output_class == sample_class)
		{
			printf("Sample = %d, NetworkOutputClass = %d, ExpectedOutput = %d, FunctionFValue %0.2f\n", sample, network_output_class, sample_class, largest_function_f_value);
			correct++;
		}
		else
			printf("Sample = %d, NetworkOutputClass = %d, ExpectedOutput = %d, FunctionFValue %0.2f *\n", sample, network_output_class, sample_class, largest_function_f_value);
	}
	printf("Percentage correct = %0.2f%%\n", 100.0 * (float) correct / (float) testing_set->num_samples);
}

