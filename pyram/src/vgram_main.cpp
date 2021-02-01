/**
 * Controle de versão: 1.1
 *
 * Esta versão foi modificada a partir do original fornecido pelo Alberto.
 * I - PATTERN_UNIT_SIZE estava especificado em bytes e nao em bits, agora este
 * valor está especificado em bits... mas eh dividido por 8 nos mallocs, pois estes ainda trabalham em bytes.
 * II - A quase totalidade dos int's foi alterada para unsigned int, uma vez
 * que inteiros com sinal sofrem o Shift aritimetico (http://www.cs.uaf.edu/~cs301/notes/Chapter5/node3.html)
 * quando deslocados para a direita em vez do Shift logico, que era o esperado.
 * III - A taxa de acerto da RN para o Linux permanece a mesma, porem eh alterada para o Windows.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vgram.h"
#include "vgram_files.h"
#include "vgram_output.h"

VG_RAM_WNN vg_ram_wnn;
DATA_SET training_set;
DATA_SET testing_set;

int
main (int argc, char *argv[])
{
	if (argc != 5)
	{
		printf("Wrong number of arguments.\nUsage: ./vgram network-configuration.csv synaptic-interconnection-pattern.csv training-set.csv testing-set.csv\n");
		exit(0);
	}

	LoadNetworkConfiguration(argv[1], &vg_ram_wnn);

	LoadTrainingSet(argv[3], &training_set, &vg_ram_wnn);

	AllocateNetworkSynapses(&vg_ram_wnn);

	AllocateNetworkMemories(&vg_ram_wnn);

	LoadSynapticInterconnectionPattern(argv[2], &vg_ram_wnn);

	TrainAll(&vg_ram_wnn, &training_set);

	//Save(&host_vg_ram_wnn, &host_training_set, "memory_seq.log");

	printf("Fim do treinamento.\n");

	LoadTestingSet(argv[4], &testing_set, &vg_ram_wnn);

	AllocateNetworkOutput(&vg_ram_wnn, vg_ram_wnn.test_size * vg_ram_wnn.number_of_neurons, 0);

	TestAll(&vg_ram_wnn, &testing_set);

	printf("Fim do teste.\n");

	EvaluateNetworkOutputs(&vg_ram_wnn, &testing_set);

	DeallocateNetworkSynapses(&vg_ram_wnn);

	DeallocateNetworkMemories(&vg_ram_wnn);

	DeallocateNetworkOutput(&vg_ram_wnn);

	DeallocateDataSet(&training_set);

	DeallocateDataSet(&testing_set);

	printf("Program finished OK!\n");

	return 0;
}
