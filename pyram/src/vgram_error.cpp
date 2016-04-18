#include "vgram_error.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
Error(const char *message1, const char *message2, const char *message3)
{
	printf("Error: %s %s %s\n", message1, message2, message3);
	exit(1);
}
