#include "vgram.h"
#include "vgram_base.h"
#include "vgram_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define 	NUM_GREYLEVELS	(256 * 256 * 256)

#define 	RED(pixel)		((pixel & 0x000000ffL) >> 0)
#define 	GREEN(pixel)	((pixel & 0x0000ff00L) >> 8)
#define 	BLUE(pixel)		((pixel & 0x00ff0000L) >> 16)
#define 	PIXEL(r, g, b)  (((r & 0x000000ffL) << 0) | ((g & 0x000000ffL) << 8) | ((b & 0x000000ffL) << 16))

void
InitializeNetwork(VG_RAM_WNN *vg_ram_wnn)
{
	srand(5);
	vg_ram_wnn->synapses = NULL;
	vg_ram_wnn->memories = NULL;
	vg_ram_wnn->network_output = NULL;
	vg_ram_wnn->number_of_neurons = vg_ram_wnn->network_layer_width * vg_ram_wnn->network_layer_height;
}


void
AllocateNetworkSynapses(VG_RAM_WNN *vg_ram_wnn)
{
	if ((vg_ram_wnn->synapses = (int *) AllocMemory((size_t)vg_ram_wnn->number_of_neurons * vg_ram_wnn->number_of_synapses_per_neuron * sizeof(int))) == NULL)
		Error("Could not allocate memory in AllocateNetworkSynapses() for synapses.", "", "");
}


void
DeallocateNetworkSynapses(VG_RAM_WNN *vg_ram_wnn)
{
	if (vg_ram_wnn->synapses)
	{
		free(vg_ram_wnn->synapses);
		vg_ram_wnn->synapses = NULL;
	}
}


void
AllocateNetworkMemories(VG_RAM_WNN *vg_ram_wnn)
{
	if ((vg_ram_wnn->number_of_synapses_per_neuron % PATTERN_UNIT_SIZE) == 0)
		vg_ram_wnn->memory_bit_group_size = vg_ram_wnn->number_of_synapses_per_neuron / PATTERN_UNIT_SIZE;
	else
		vg_ram_wnn->memory_bit_group_size = 1 + vg_ram_wnn->number_of_synapses_per_neuron / PATTERN_UNIT_SIZE;
	if ((vg_ram_wnn->memories = (int *) AllocMemory((size_t)vg_ram_wnn->number_of_neurons * vg_ram_wnn->memory_size * (vg_ram_wnn->memory_bit_group_size + 1) * sizeof(int))) == NULL)
		Error("Could not allocate memory in AllocateNetworkMemories() for memories.","","");
}


void
DeallocateNetworkMemories(VG_RAM_WNN *vg_ram_wnn)
{
	if (vg_ram_wnn->memories)
	{
		free(vg_ram_wnn->memories);
		vg_ram_wnn->memories = NULL;
	}
}


void
AllocateNetworkOutput(VG_RAM_WNN *vg_ram_wnn, int size, int value)
{
	if ((vg_ram_wnn->network_output = (int *) AllocMemory((size_t)size * sizeof(int))) == NULL)
		Error("Could not allocate memory in AllocateNetworkOutput() for network_output.","","");
	if (value != 0)
		for(int i=0; i < size; i++)
			vg_ram_wnn->network_output[i] = value;
}


void
DeallocateNetworkOutput(VG_RAM_WNN *vg_ram_wnn)
{
	if (vg_ram_wnn->network_output)
	{
		free(vg_ram_wnn->network_output);
		vg_ram_wnn->network_output = NULL;
	}
}


void InitializeDataSet(DATA_SET *data_set)
{
	data_set->sample_class = NULL;
	data_set->sample = NULL;
}


void
AllocateDataSet(DATA_SET *data_set)
{
	if ((data_set->sample_class = (int *) AllocMemory((size_t)data_set->num_samples * sizeof(int))) == NULL)
		Error("Could not allocate memory in AllocateNetworkSynapses() for sample_class.","","");
	if ((data_set->sample = (int *) AllocMemory((size_t)data_set->num_samples * data_set->num_inputs * sizeof(int))) == NULL)
		Error("Could not allocate memory in AllocateNetworkSynapses() for sample.","","");
}


void
DeallocateDataSet(DATA_SET *data_set)
{
	if (data_set->sample)
	{
		free(data_set->sample);
		data_set->sample = NULL;
	}
	if (data_set->sample_class)
	{
		free(data_set->sample_class);
		data_set->sample_class = NULL;
	}
}


void
CopyDataSet(DATA_SET *data_set, int *input_data, int *input_class)
{
	memcpy(data_set->sample, input_data, data_set->num_inputs * data_set->num_samples * sizeof(int));
	if (input_class)
		memcpy(data_set->sample_class, input_class, data_set->num_samples * sizeof(int));
}


