#include <math.h>
#include "vgram_base.h"
#include "vgram_error.h"
#include "vgram_files.h"

FILE *
OpenCsvFile(const char *file_name, const char *csv_head)
{
	FILE *csv_file;
	char csv_head_readed[MAX_CSV_HEAD_SIZE];

	if ((csv_file = fopen(file_name, "r")) == NULL)
		return NULL;

	if (fgets(csv_head_readed, MAX_CSV_HEAD_SIZE, csv_file) == NULL)
		Error("Could not read the head of csv file", file_name, "in OpenCsvFile().");

	if (strcmp(csv_head_readed, csv_head) != 0)
	{
		printf("Head of csv file %s does not match. Head expected: %s, head readed: %s\n", file_name, csv_head, csv_head_readed);
		return NULL;
	}
	return csv_file;
}


void *
ReadCsvFileLine(FILE *csv_file, const char *line_format)
{
	int i;
	int integer;
	float floating_point;
	void *csv_line;
	char csv_str_line[MAX_CSV_LINE_SIZE];
	char *aux;

	if ((csv_line = (void *) AllocMemory((size_t)strlen(line_format) * sizeof(int))) == NULL)	// Assumimos aqui que int e float ocupam o mesmo número de bytes de memória
		Error("Could not allocate memory in ReadCsvFileLine().", "", "");

	if (fgets(csv_str_line, MAX_CSV_LINE_SIZE-1, csv_file) == NULL)	// fim do arquivo
		return NULL;

	aux = csv_str_line;
	for (i = 0; i < (int) strlen(line_format); i++)
	{
		if (!aux)
			Error("Unexpected end of line in ReadCsvFileLine()", "", "");
		switch(line_format[i])
		{
			case 'i':
				integer = atoi(aux);
				((int *) csv_line)[i] = integer;
				break;
			case 'f':
				floating_point = (float) atof(aux);
				((float *) csv_line)[i] = floating_point;
				break;
			default:
				Error("Unknown character in csv file line_format descriptor: %s", line_format, "");
		}
		aux = strchr(aux, ';');
		if (aux)
			aux++;
	}
	return csv_line;
}


void
LoadNetworkConfiguration(char *file_name, VG_RAM_WNN *host_vg_ram_wnn)
{
	FILE *csv_file;
	void *csv_file_line;

	if ((csv_file = OpenCsvFile(file_name, "O;X;N\n")) == NULL)
		Error("Could not open file", file_name, "in LoadNetworkConfiguration().");

	if ((csv_file_line = ReadCsvFileLine(csv_file, "iii")) == NULL)
		Error("Could not read a csv file line from", file_name, "in LoadNetworkConfiguration().");

	host_vg_ram_wnn->number_of_neurons = ((int *) csv_file_line)[0];
	host_vg_ram_wnn->number_of_synapses_per_neuron = ((int *) csv_file_line)[1];
	host_vg_ram_wnn->input_size = ((int *) csv_file_line)[2];

	free(csv_file_line);
}


char *
BuildSynapticInterconectionPatternCsvFileHeader(int num_synapses)
{
	int i;
	char *synaptic_interconection_pattern_csv_file_header;
	char *aux;

	aux = synaptic_interconection_pattern_csv_file_header = (char *) AllocMemory((size_t)num_synapses * 10 * sizeof(char)); // permite 10000+ synapses

	aux += sprintf(aux, "O");
	for (i = 0; i < num_synapses; i++)
		aux += sprintf(aux, ";x%d", i);
	sprintf(aux, "\n");

	return synaptic_interconection_pattern_csv_file_header;
}


char *
BuildSynapticInterconectionPatternCsvLineFormat(int num_synapses)
{
	int i;
	char *synaptic_interconection_pattern_csv_line_format;

	synaptic_interconection_pattern_csv_line_format = (char *) AllocMemory((size_t)(num_synapses + 2) * sizeof(char));

	synaptic_interconection_pattern_csv_line_format[0] = 'i'; // formato do número do neurônio, O
	for (i = 1; i < num_synapses + 1; i++)
		synaptic_interconection_pattern_csv_line_format[i] = 'i';	// formato do número da entrada ni onde a synapse se conecta

	synaptic_interconection_pattern_csv_line_format[i] = '\0';

	return synaptic_interconection_pattern_csv_line_format;
}


void
LoadSynapticInterconnectionPattern(const char *file_name, VG_RAM_WNN *host_vg_ram_wnn)
{
	int neuron, synapse;
	FILE *csv_file;
	void *csv_file_line;
	char *synaptic_interconection_pattern_csv_file_header;
	char *synaptic_interconection_pattern_csv_line_format;

	synaptic_interconection_pattern_csv_file_header = BuildSynapticInterconectionPatternCsvFileHeader(host_vg_ram_wnn->number_of_synapses_per_neuron);
	if ((csv_file = OpenCsvFile(file_name, synaptic_interconection_pattern_csv_file_header)) == NULL)
		Error("Could not open file", file_name, "in LoadSynapticInterconnectionPattern().");
	free(synaptic_interconection_pattern_csv_file_header);

	synaptic_interconection_pattern_csv_line_format = BuildSynapticInterconectionPatternCsvLineFormat(host_vg_ram_wnn->number_of_synapses_per_neuron);
	for (neuron = 0; neuron < host_vg_ram_wnn->number_of_neurons; neuron++)
	{
		if ((csv_file_line = ReadCsvFileLine(csv_file, synaptic_interconection_pattern_csv_line_format)) == NULL)
			Error("Could not read a csv file line from", file_name, "in LoadSynapticInterconnectionPattern().");

		for (synapse = 0; synapse < host_vg_ram_wnn->number_of_synapses_per_neuron; synapse++)
			host_vg_ram_wnn->synapses[neuron * host_vg_ram_wnn->number_of_synapses_per_neuron + synapse] = ((int *)csv_file_line)[synapse+1]; // primeiro elemento da linha do csv é o número do neurônio

		free(csv_file_line);
	}
	free(synaptic_interconection_pattern_csv_line_format);
}

char *
BuildDataSetCsvFileHeader(int input_size)
{
	int i;
	char *data_set_csv_file_header;
	char *aux;

	aux = data_set_csv_file_header = (char *) AllocMemory((size_t)input_size * 10 * sizeof(char)); // permite 10000+ inputs

	for (i = 0; i < input_size; i++)
		aux += sprintf(aux, "v%d;", i);
	sprintf(aux, "ci\n");

	return data_set_csv_file_header;
}


char *
BuildDataSetCsvLineFormat(int input_size)
{
	int i;
	char format;
	char *data_set_csv_line_format;
	format = 'i';

	data_set_csv_line_format = (char *) AllocMemory((size_t)(input_size + 2) * sizeof(char));

	for (i = 0; i < input_size; i++)
		data_set_csv_line_format[i] = format; // formato do número que representa cada termo

	data_set_csv_line_format[i] = 'i';		// formato do número que representa a classe ci
	data_set_csv_line_format[i+1] = '\0';

	return data_set_csv_line_format;
}


int
CountNumberOfSamplesInDataSet(char *file_name,
		char *data_set_csv_file_header,
		char *data_set_csv_line_format,
		int num_inputs)
{
	FILE *csv_file;
	int num_samples;

	if ((csv_file = OpenCsvFile(file_name, data_set_csv_file_header)) == NULL)
		Error("Could not open file", file_name, "in CountNumberOfSamplesInDataSet().");

	num_samples = 0;
	fseek(csv_file, (long)num_samples, SEEK_SET);
	while ((ReadCsvFileLine(csv_file, data_set_csv_line_format)) != NULL)
		num_samples++;

	fclose(csv_file);
	return num_samples-1;//discards header
}


void
LoadDataSet(char *file_name, DATA_SET *data_set)
{
	int sample, term;
	FILE *csv_file;
	void *csv_file_line;
	char *data_set_csv_file_header;
	char *data_set_csv_line_format;

	data_set_csv_file_header = BuildDataSetCsvFileHeader(data_set->num_inputs);
	data_set_csv_line_format = BuildDataSetCsvLineFormat(data_set->num_inputs);

	data_set->num_samples = CountNumberOfSamplesInDataSet(file_name, data_set_csv_file_header, data_set_csv_line_format, data_set->num_inputs);
	AllocateDataSet(data_set);

	if ((csv_file = OpenCsvFile(file_name, data_set_csv_file_header)) == NULL)
		Error("Could not open file", file_name, "in LoadDataSet().");
	free(data_set_csv_file_header);

	for (sample = 0; sample < data_set->num_samples; sample++)
	{
		csv_file_line = ReadCsvFileLine(csv_file, data_set_csv_line_format);
		for (term = 0; term < data_set->num_inputs; term++)
			data_set->sample[sample * data_set->num_inputs + term] = ((int *)csv_file_line)[term];

		data_set->sample_class[sample] = ((int *)csv_file_line)[term];
		free(csv_file_line);
	}
	free(data_set_csv_line_format);
	fclose(csv_file);
}

void
LoadTrainingSet(char *file_name, DATA_SET *training_set, VG_RAM_WNN *vg_ram_wnn)
{
	training_set->num_inputs = vg_ram_wnn->input_size;
	LoadDataSet(file_name, training_set);
	vg_ram_wnn->memory_size = training_set->num_samples;
}

void
LoadTestingSet(char *file_name, DATA_SET *testing_set, VG_RAM_WNN *vg_ram_wnn)
{
	testing_set->num_inputs = vg_ram_wnn->input_size;
	LoadDataSet(file_name, testing_set);
	vg_ram_wnn->test_size = testing_set->num_samples;
}


void
PrintPattern(FILE *fp, int memory_bit_group_size, int *bit_pattern)
{
	int bit_group, j;
	int bit_group_index;
	char bits[PATTERN_UNIT_SIZE+1];

	for (bit_group_index = memory_bit_group_size; bit_group_index--;)
	{
		bit_group = bit_pattern[bit_group_index];
		for (j = 0; j < PATTERN_UNIT_SIZE; bit_group>>=1, j++)
			bits[PATTERN_UNIT_SIZE-j-1] = bit_group & 0x1 ? '1' : '0';
		bits[PATTERN_UNIT_SIZE] = 0;
		fprintf(fp, "%s", bits);
	}
	// Imprime a classe associada ao padrão de bits
	fprintf(fp, "%d\n", bit_pattern[memory_bit_group_size]);
}

void
Save(VG_RAM_WNN *host_vg_ram_wnn, DATA_SET *host_training_set, const char *filename)
{
	FILE *fp;
	int sample, neuron;

	// Reference: http://www.cprogramming.com/tutorial/cfileio.html
	fp=fopen(filename, "w");

	for (sample = 0; sample < host_training_set->num_samples; sample++)
	{
		for (neuron = 0; neuron < host_vg_ram_wnn->number_of_neurons; neuron++)
		{
			// Imprime o padrão de bits gerado a partir da sample
			PrintPattern(fp, host_vg_ram_wnn->memory_bit_group_size,
					&(host_vg_ram_wnn->memories[(neuron * host_training_set->num_samples + sample) * (host_vg_ram_wnn->memory_bit_group_size+1)])
					);
		}
	}

	fclose(fp);
}

