#include "vgram.h"
#include "vgram_error.h"
#include <math.h>
#include <stdlib.h>

#define LRAND48_MAX ((unsigned int) -1 >> 1)

double
distance_from_image_center (double wi, double hi, double w, double h, double u, double log_factor)
{
	double exp_val, x, y;

	x = (u / (w / 2.0)) * log_factor;
	exp_val = (wi / 2.0) * (exp(log(log_factor) * (x - log_factor) / log_factor) - (1.0 / log_factor)) * (log_factor / (log_factor - 1.0));

	return (exp_val);
}


void
map_v1_to_image(int *xi, int *yi, double wi, double hi, double u, double v, double w, double h, double x_center, double y_center, double log_factor)
{
	double d, theta;

	if ( ((int)w % 2) == 0 ) //
		Error("This function is only defined for odd values of w.", "", "in map_v1_to_image().");

	if (u < ((w - 1.0) / 2.0))
	{
		d = distance_from_image_center(wi, hi, w, h, (w - 1.0) / 2.0 - u, log_factor);
		theta = M_PI * ((h * (3.0 / 2.0) - v) /  h) - M_PI / (2.0 * h);
	}
	else
	{
		d = distance_from_image_center(wi, hi, w, h, u - (w - 1.0) / 2.0, log_factor);
		theta = M_PI * ((h * (3.0 / 2.0) + v) /  h) + M_PI / (2.0 * h);
	}

	*xi = (int) (d * cos(theta)) + x_center;
	*yi = (int) (d * sin(theta)) + y_center;
}


double
gaussrand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if (phase == 0)
	{
		do
		{
			double U1 = (double) rand() / (double) LRAND48_MAX;
			double U2 = (double) rand() / (double) LRAND48_MAX;

			V1 = 2.0 * U1 - 1.0;
			V2 = 2.0 * U2 - 1.0;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1.0 || S == 0.0);

		X = V1 * sqrt(-2.0 * log(S) / S);
	}
	else
	{
		X = V2 * sqrt(-2.0 * log(S) / S);
	}

	phase = 1 - phase;

	return (X);
}


void
CreateSynapsesLogPolar(VG_RAM_WNN *vg_ram_wnn, int nun_inputs_per_neuron, int synapse_offset,
		int input_offset, int input_width, int input_height, int output_width, int output_height,
		double gaussian_radius, double log_factor, bool same_interconnection_pattern)
{
	int gx, gy;
	int xi, yi, x_center, y_center;
	double grx, gry;

	double first_interconnection_pattern[nun_inputs_per_neuron*2];

	int synapse_memory_offset = synapse_offset * output_height * output_width;

	x_center = input_width/2 ;
	y_center = input_height/2 ;

	for (int y = 0; y < output_height; y++)
	{
		for (int x = 0; x < output_width; x++)
		{
			//Here, we perform the log-polar mapping for determining the neuron position in the map
			map_v1_to_image (&xi, &yi, input_width, input_height, x, y, output_width, output_height, x_center, y_center, log_factor);

			for (int i = 0; i < nun_inputs_per_neuron; i++)
			{
				grx = gaussrand () * gaussian_radius + 0.5;
				gry = gaussrand () * gaussian_radius + 0.5;

				if (y == 0 && x == 0 && same_interconnection_pattern)
				{
					first_interconnection_pattern[i*2] = grx;
					first_interconnection_pattern[i*2+1] = gry;
				}
				else if (same_interconnection_pattern)
				{
					grx = first_interconnection_pattern[i*2];
					gry = first_interconnection_pattern[i*2+1];
				}

				gx = (int)((double) xi + grx);
				gy = (int)((double) yi + gry);

				//Boundary checks must be kept in order to avoid invalid synapse positioning
				if ((gx < 0) || (gy < 0) || (gx >= input_width) || (gy >= input_height))
					vg_ram_wnn->synapses[synapse_memory_offset + y + output_height * (x + output_width * i)] = input_offset + 0; //FIXME should be a neuron_with_output_zero or sample until valid
				else
					vg_ram_wnn->synapses[synapse_memory_offset + y + output_height * (x + output_width * i)] = input_offset + gx + gy * input_width;
			}
		}
	}
}


void
CreateSynapsesGaussian(VG_RAM_WNN *vg_ram_wnn, int nun_inputs_per_neuron, int synapse_offset,
		int input_offset, int input_width, int input_height, int output_width, int output_height,
		double gaussian_radius, bool same_interconnection_pattern)
{
	int tries;
	int nx_src, ny_src, nx_dst, ny_dst, gx, gy;
	double grx, gry;
	float x_factor, y_factor;

	double first_interconnection_pattern[nun_inputs_per_neuron*2];

	int synapse_memory_offset = synapse_offset * output_height * output_width;

	if (same_interconnection_pattern)
	{
		nx_src = input_width - (int) (2.0 * 4.0 * gaussian_radius + 0.5);
		ny_src = input_height - (int) (2.0 * 4.0 * gaussian_radius + 0.5);
	}
	else
	{
		nx_src = input_width;
		ny_src = input_height;
	}
	nx_dst = output_width;
	ny_dst = output_height;

	x_factor = (float) nx_src / (float) nx_dst;
	y_factor = (float) ny_src / (float) ny_dst;

	for (int y = 0; y < output_height; y++)
	{
		for (int x = 0; x < output_width; x++)
		{
			for (int i = 0; i < nun_inputs_per_neuron; i++)
			{
				if (same_interconnection_pattern)
				{
					if ((y != 0) || (x != 0))
					{
						grx = first_interconnection_pattern[i*2];
						gry = first_interconnection_pattern[i*2+1];
					}
					else
					{
						if (i == 0) // This is necessary for minchinton_center_surround neurons and should not affect other types of neuron
						{
							grx = 0.0;
							gry = 0.0;
						}
						else
						{
							grx = gaussrand () * gaussian_radius + 0.5;
							gry = gaussrand () * gaussian_radius + 0.5;
						}
						first_interconnection_pattern[i*2] = grx;
						first_interconnection_pattern[i*2+1] = gry;
					}

					if (x_factor > 1.0)
						gx = (int)((double) x * x_factor + 4.0 * gaussian_radius + x_factor / 2.0 + grx);
					else
						gx = (int)((double) x * x_factor + 4.0 * gaussian_radius + grx);

					if (y_factor > 1.0)
						gy = (int)((double) y * y_factor + 4.0 * gaussian_radius + y_factor / 2.0 + gry);
					else
						gy = (int)((double) y * y_factor + 4.0 * gaussian_radius + gry);
				}
				else	// different interconnection pattern (synapses can randomize into the image)
				{
					tries = 0;
					do
					{
						if (i == 0) // This is necessary for minchinton_center_surround neurons and should not affect other types of neuron
						{
							grx = 0.0;
							gry = 0.0;
						}
						else
						{
							grx = gaussrand () * gaussian_radius + 0.5;
							gry = gaussrand () * gaussian_radius + 0.5;
						}

						if (x_factor > 1.0)
							gx = (int)((double) x * x_factor + x_factor / 2.0 + grx);
						else
							gx = (int)((double) x * x_factor + grx);

						if (y_factor > 1.0)
							gy = (int)((double) y * y_factor + y_factor / 2.0 + gry);
						else
							gy = (int)((double) y * y_factor + gry);

						tries++;
					}
					while (((gx < 0) || (gy < 0) || (gx >= input_width) || (gy >= input_height)) && (tries < 100));

					if (tries == 100)
					{
						Error("Could not connect neuron layers.", "Check if gaussian_radius isn't too small for the current number of synapses", ".");
					}
				}

				if ((gx >= 0) && (gy >= 0) && (gx < input_width) && (gy < input_height))
					vg_ram_wnn->synapses[synapse_memory_offset + y + output_height * (x + output_width * i)] = input_offset + gx + gy * input_width;
			}
		}
	}
}


void
CreateSynapsesRandom(VG_RAM_WNN *vg_ram_wnn, int nun_inputs_per_neuron, int synapse_offset, int input_offset, int input_width, int input_height, int output_width, int output_height)
{
	int synapse_memory_offset = synapse_offset * output_height * output_width;
	for (int y = 0; y < output_height; y++)
	{
		for (int x = 0; x < output_width; x++)
		{
			for (int i = 0; i < nun_inputs_per_neuron; i++)
			{

				int input_x = rand () % input_width;
				int input_y = rand () % input_height;

				vg_ram_wnn->synapses[synapse_memory_offset + y + output_height * (x + output_width * i)] = input_offset + input_x + input_y * input_width;

			}
		}
	}
}

