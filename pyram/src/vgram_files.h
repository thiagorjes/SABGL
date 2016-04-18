#ifndef VGRAM_FILES_H_
#define VGRAM_FILES_H_

#include "vgram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CSV_HEAD_SIZE 100000
#define MAX_CSV_LINE_SIZE 100000

#ifdef __cplusplus
extern "C" {
#endif

FILE *
OpenCsvFile(const char *file_name, const char *csv_head);

void *
ReadCsvFileLine(FILE *csv_file, const char *line_format);

void
LoadNetworkConfiguration(char *file_name, VG_RAM_WNN *host_vg_ram_wnn);

char *
BuildSynapticInterconectionPatternCsvFileHeader(int num_synapses);

char *
BuildSynapticInterconectionPatternCsvLineFormat(int num_synapses);

void
LoadSynapticInterconnectionPattern(const char *file_name, VG_RAM_WNN *host_vg_ram_wnn);

char *
BuildDataSetCsvFileHeader(int network_input_size);

char *
BuildDataSetCsvLineFormat(int network_input_size);

int
CountNumberOfSamplesInDataSet(char *file_name,
		char *data_set_csv_file_header,
		char *data_set_csv_line_format,
		int num_inputs);

void
LoadDataSet(char *file_name, DATA_SET *data_set);

void
LoadTrainingSet(char *file_name, DATA_SET *training_set, VG_RAM_WNN *vg_ram_wnn);

void
LoadTestingSet(char *file_name, DATA_SET *testing_set, VG_RAM_WNN *vg_ram_wnn);

void
Save(VG_RAM_WNN *host_vg_ram_wnn, DATA_SET *host_training_set, const char *filename);

#ifdef __cplusplus
}
#endif

#endif /* VGRAM_FILES_H_ */
