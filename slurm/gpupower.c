# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/resource.h>
#include <limits.h>

#include <errno.h>
extern int errno;

#include <slurm/spank.h>

#define SPANK_GPUPOWER_VERSION "0.0.1"

static int gpupower_default = 150;
static int gpupower_min = 100;
static int gpupower_max = 250;
static int gpupower_set = 150;

#define xinfo slurm_info
#define xerror slurm_error
#define xdebug slurm_debug

/*
 * All spank plugins must define this macro for the SLURM plugin loader.
 */
SPANK_PLUGIN(gpupower, 1);

static int _str2int(const char *str, int *p2int)
{
	long int l;
	char *p;

	l = strtol(str, &p, 10);

	if (l == LONG_MIN || l == LONG_MAX)
		return (-1);

	*p2int = (int)l;

	return (0);
}

static int _gpupower_option_process(int val, const char *optarg, int remote)
{
	if (optarg == NULL)
	{
		return 0;
	}

	if (strncmp("no", optarg, 2) == 0)
	{
		gpupower_default = gpupower_max;
		xdebug("gpupower: disabled on user request");
	}
	else if (strncmp("yes", optarg, 3) == 0)
	{
		xdebug("gpupower: enabled on user request");
	}
	else
	{
		int watts_requested = 0;
		if (_str2int(optarg, &watts_requested) < 0)
		{
			xerror("gpupower: ignoring invalid watts: %s", optarg);
			return 1;
		}
		else if (watts_requested > gpupower_max)
		{
			xerror("gpupower: requested %s, max %d", optarg, gpupower_max);
			watts_requested = gpupower_max;
		}
		else if (watts_requested < gpupower_min)
		{
			xerror("gpupower: requested %s, min %d", optarg, gpupower_min);
			watts_requested = gpupower_min;
		}
		gpupower_set = watts_requested;
	}

	return (0);
}

struct spank_option spank_options_reg[] =
	{
		{"gpupower", "[yes|no|WATTS]", "Change GPU power policy for job", 2, 0,
		 (spank_opt_cb_f) _gpupower_option_process},
		SPANK_OPTIONS_TABLE_END};

int slurm_spank_init(spank_t sp, int ac, char **av)
{
	int min = gpupower_min, max = gpupower_max;
	char buf[64] = {0};

    if (spank_context() != S_CTX_ALLOCATOR) 
	{
        if (spank_option_register(sp, spank_options_reg) != ESPANK_SUCCESS)
		{
	    	slurm_error("spank_option_register error");
			return(-1);
		}
	}

	/* do something in remote mode only */
	if (!spank_remote(sp))
		return 0;

    


	if (max > 0 || min > 0)
	{
		gpupower_max = max;
		gpupower_min = min;
		xdebug("gpupower: configuration is min=%d max=%d (version %s)",
			   gpupower_min, gpupower_max,
			   SPANK_GPUPOWER_VERSION);
	}

	snprintf(buf, sizeof(buf), "%d", gpupower_set);

	if (spank_setenv (sp, "SLURM_GPUPOWER", buf, 1) != ESPANK_SUCCESS) {
			slurm_error ("gpupower.so: spank_setenv (SLURM_GPUPOWER, \"%s\"): %m", buf);
			return (-1);
		}	

	return 0;
}
