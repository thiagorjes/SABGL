#ifndef VGRAM_H_
#define VGRAM_H_

#define PATTERN_UNIT_SIZE (8*sizeof(int))

// Tipos //
typedef struct
{
	int network_layer_width;
	int network_layer_height;
	int number_of_neurons;				// |O|
	int number_of_synapses_per_neuron;	// |X|
	int input_size;						// |N|
	int memory_bit_group_size;			// tamanho de cada linha da matriz de memória dos neurônios
	int memory_size;					// quantidade de exemplos de treino (quantidade de linhas da matriz de memória dos neurônios)
	int test_size;						// quantidade de exemplos de teste

	// network_output é um vetor que armazena a saída corrente da rede neural
	int *network_output;

	// synapses é um vetor de vetores de sinapses: um vetor de sinapses para cada neurônio da rede neural.
	// O vetor de sinapses de cada neurônio é lido uma vez e não é mais alterado
	int *synapses;

	// memories é um vetor de matrizes de memórias: uma matriz de memórias para cada neurônio da rede neural.
	// Cada linha na matriz de memórias de um neurônio contęm um vetor de inteiros. Com excessão do último elemento,
	// todos os elementos de cada um destes vetores compôem, em conjunto, um padrão de bits aprendido durante a fase de
	// treino para uma dada entrada da rede neural. O último elemento de cada vetor é a saída vista junto
	// com esta dada entrada.
	int *memories;
	/**
	 * A memória do neurônio está organizada da seguinte forma considerando um cubo:
	 * com profundidade z, representando as camadas de neurônios
	 * com largura x, representando as amostras treinadas
	 * com altura y, representando o agrupamento de bits (memória propriamente dita do neurônio) e a classe da amostra associada.
	 *
	 * Um exemplo de representação linear da memória para 4 amostras de treinamento e agrupamento de bits igual a 2 + 1 classe de amostra:
	 * N0-S0-P0P1P2 N0-S1-P0P1P2 N0-S2-P0P1P2 N0-S3-P0P1P2
	 * N1-S0-P0P1P2 ...
	 */

	// CUDA device memory pointers
	int *d_network_output;
	int *d_synapses;
	int *d_memories;
}VG_RAM_WNN;


typedef struct
{
	int num_samples;
	int num_inputs;
	int *sample_class;
	int *sample;
	// CUDA device memory pointers
	int *d_sample_class;
	int *d_sample;
}DATA_SET;

extern "C" {

void InitializeNetwork(VG_RAM_WNN *vg_ram_wnn);

void AllocateNetworkSynapses(VG_RAM_WNN *vg_ram_wnn);

void AllocateNetworkMemories(VG_RAM_WNN *vg_ram_wnn);

void AllocateNetworkOutput(VG_RAM_WNN *vg_ram_wnn, int size, int value);

void DeallocateNetworkSynapses(VG_RAM_WNN *vg_ram_wnn);

void DeallocateNetworkMemories(VG_RAM_WNN *vg_ram_wnn);

void DeallocateNetworkOutput(VG_RAM_WNN *vg_ram_wnn);

void InitializeDataSet(DATA_SET *data_set);

void AllocateDataSet(DATA_SET *data_set);

void DeallocateDataSet(DATA_SET *data_set);

void CopyDataSet(DATA_SET *data_set, int *input_data, int *input_class);

void CreateSynapsesRandom(VG_RAM_WNN *vg_ram_wnn, int nun_inputs_per_neuron, int synapse_offset,
		int input_offset, int input_width, int input_height, int output_width, int output_height);

void CreateSynapsesLogPolar(VG_RAM_WNN *vg_ram_wnn, int nun_inputs_per_neuron, int synapse_offset,
		int input_offset, int input_width, int input_height, int output_width, int output_height,
		double gaussian_radius, double log_factor, bool same_interconnection_pattern);

void CreateSynapsesGaussian(VG_RAM_WNN *vg_ram_wnn, int nun_inputs_per_neuron, int synapse_offset,
		int input_offset, int input_width, int input_height, int output_width, int output_height,
		double gaussian_radius, bool same_interconnection_pattern);

void Train(VG_RAM_WNN *vg_ram_wnn, int *training_set, int sample_class_i, int idx);

void TrainAll(VG_RAM_WNN *vg_ram_wnn, DATA_SET *training_set);

void Test(VG_RAM_WNN *vg_ram_wnn, int *testing_set, int sample_class_i, int idx);

void TestAll(VG_RAM_WNN *vg_ram_wnn, DATA_SET *testing_set);

void TestSequence(VG_RAM_WNN *vg_ram_wnn, DATA_SET *testing_set, int *step_list, int step_list_size);

}

#endif /* VGRAM_H_ */
