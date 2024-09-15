/**
 * @file saxpy.cu
 * @brief This file contains a basic CUDA programming example
 *	  Reference: https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c
 *
 * @author Sundar Subramaniyan
 *
 * @date 9/7/2024
 */

#include <stdio.h>
#include <getopt.h>
#include <time.h>

/**
 * @brief Default size of the vectors
 */
#define TEST_VECTOR_SIZE (1 << 30)

/**
 * @brief Nanoseconds per second
 */
#define NS_PER_SECOND 1000000000UL

/**
 * @brief Summary output buffer length (Per task summary)
 */
#define SUMMARY_BUF_LEN 128

/**
 * @brief Tab spaces
 */
#define TABSPACES 8

/**
 * @brief Number of Tabs to add to the summary
 *	  before the time in seconds
 */
#define SUMMARY_NUM_TABS 7

/**
 * @brief Profile Task type
 *
 * @param PROF_TASK_SETUP		Setup time
 * @param PROF_TASK_HOST_ALLOC		Host memory allocation time
 * @param PROF_TASK_HOST_FREE		Host memory free time
 * @param PROF_TASK_HOST_MEMCPY		Host memcpy time
 * @param PROF_TASK_HOST_INIT		Host Vector initialisation time
 * @param PROF_TASK_HOST_VERIFY		Host verification time
 * @param PROF_TASK_HOST_COMPUTE	Host compute time
 * @param PROF_TASK_DEVICE_ALLOC	Device memory allocation time
 * @param PROF_TASK_DEVICE_FREE		Device memory free time
 * @param PROF_TASK_DEVICE_MEMCPY	Device memcpy time
 * @param PROF_TASK_DEVICE_COMPUTE	Device compute time
 * @param PROF_TASK_MAX_TYPES		Maximum number of Task types
 */
enum profile_task_type {
	PROF_TASK_SETUP,
	PROF_TASK_HOST_ALLOC,
	PROF_TASK_HOST_FREE,
	PROF_TASK_HOST_MEMCPY,
	PROF_TASK_HOST_INIT,
	PROF_TASK_HOST_VERIFY,
	PROF_TASK_HOST_COMPUTE,
	PROF_TASK_DEVICE_ALLOC,
	PROF_TASK_DEVICE_FREE,
	PROF_TASK_DEVICE_MEMCPY,
	PROF_TASK_DEVICE_COMPUTE,
	PROF_TASK_MAX_TYPES,
};

/**
 * @brief Profile context
 *
 */
struct profile_ctx {
	const char *title;		/**< Title of the task being profiled */
	enum profile_task_type type;	/**< Task type being profiled */
	struct timespec start;		/**< Profiling start time */
	struct timespec finish;		/**< Profiling finish time */
	struct timespec delta;		/**< Profiling delta time */
};

/**
 * @brief Instance of the profile context group
 *
 * Each index in the array points to a specific
 * task being profiled.
 */
static struct profile_ctx prof_ctx_grp[PROF_TASK_MAX_TYPES] = {
	{
		.title = "Host Setup",
		.type = PROF_TASK_SETUP,
	},
	{
		.title = "Host Alloc",
		.type = PROF_TASK_HOST_ALLOC,
	},
	{
		.title = "Host Free",
		.type = PROF_TASK_HOST_FREE,
	},
	{
		.title = "Host Memcpy",
		.type = PROF_TASK_HOST_MEMCPY,
	},
	{
		.title = "Host Init",
		.type = PROF_TASK_HOST_INIT,
	},
	{
		.title = "Host Verify",
		.type = PROF_TASK_HOST_VERIFY,
	},
	{
		.title = "Host Compute",
		.type = PROF_TASK_HOST_COMPUTE,
	},
	{
		.title = "Device Alloc",
		.type = PROF_TASK_DEVICE_ALLOC,
	},
	{
		.title = "Device Free",
		.type = PROF_TASK_DEVICE_FREE,
	},
	{
		.title = "Device Memcpy",
		.type = PROF_TASK_DEVICE_FREE,
	},
	{
		.title = "Device Compute",
		.type = PROF_TASK_DEVICE_COMPUTE,
	},
};

/**
 * @brief Start the profiling
 *
 * @param[in]	type		Type of task to profile
 * @param[in]	prof_enable	Boolean that indicates whether profiling is enabled
 */
static inline void
profile_record_start(enum profile_task_type type, int prof_enable)
{
	struct profile_ctx *ctx;

	if (!prof_enable)
		return;

	if (type >= PROF_TASK_MAX_TYPES) {
		printf("PROF START: Invalid profile task type %d\n", type);
		return;
	}

	ctx = &prof_ctx_grp[type];
	clock_gettime(CLOCK_REALTIME, &ctx->start);
}

/**
 * @brief Stop the profiling
 *
 * @param[in]	type		Type of task to profile
 * @param[in]	prof_enable	Boolean that indicates whether profiling is enabled
 */
static inline void
profile_record_stop(enum profile_task_type type, int prof_enable)
{
	struct profile_ctx *ctx;

	if (!prof_enable)
		return;

	if (type >= PROF_TASK_MAX_TYPES) {
		printf("PROF STOP: Invalid profile task type %d\n", type);
		return;
	}

	ctx = &prof_ctx_grp[type];
	clock_gettime(CLOCK_REALTIME, &ctx->finish);
}

/**
 * @brief Calculate the delta and record it
 *
 * @param[in]	ctx	Pointer to profile context
 */
static inline void
profile_record_calc_delta(struct profile_ctx *ctx)
{
	struct timespec *t1, *t2, *td;

	t1 = &ctx->start;
	t2 = &ctx->finish;
	td = &ctx->delta;

	td->tv_nsec = t2->tv_nsec - t1->tv_nsec;
	td->tv_sec = t2->tv_sec - t1->tv_sec;

	if (td->tv_sec > 0 && td->tv_nsec < 0) {
		td->tv_nsec += NS_PER_SECOND;
		td->tv_sec--;
	} else if (td->tv_sec < 0 && td->tv_nsec > 0) {
		td->tv_nsec -= NS_PER_SECOND;
		td->tv_sec++;
	}
}

/**
 * @brief Calculate the delta for all the profile task types
 *
 * @param[in]	prof_enable	Boolean that indicates whether profiling is enabled
 */
static inline void
profile_record_calc_delta_all(int prof_enable)
{
	int i;

	if (!prof_enable)
		return;

	for (i = 0; i < sizeof(prof_ctx_grp)/sizeof(prof_ctx_grp[0]); i++) {
		profile_record_calc_delta(&prof_ctx_grp[i]);
	}
}

/**
 * @brief Print summary of profiling for a given task type
 *
 * @param[in]	ctx	Pointer to profile context
 */
static inline void
profile_record_summarize(struct profile_ctx *ctx)
{
	char buf[SUMMARY_BUF_LEN];
	int len;
	int tab_stops, tabs = SUMMARY_NUM_TABS;

	if (!ctx->delta.tv_sec && !ctx->delta.tv_nsec)
		return;

	len = snprintf(buf, sizeof(buf), "\t%s time:", ctx->title);
	tab_stops = len/TABSPACES;

	if (len % TABSPACES)
		tab_stops++;

	tabs -= tab_stops;
	memset(buf + len, '\t', tabs);
	len += tabs;
	len = snprintf(buf + len, sizeof(buf) - len, "[%d.%.9ld seconds]\n",
			(int)ctx->delta.tv_sec,
			ctx->delta.tv_nsec);
	fprintf(stdout, "%s", buf);
}

/**
 * @brief Print summary all the profiling task types
 *
 * @param[in]	prof_enable	Boolean that indicates whether profiling is enabled
 */
static inline void
profile_record_summarize_all(int prof_enable)
{
	int i;

	if (!prof_enable)
		return;

	for (i = 0; i < sizeof(prof_ctx_grp)/sizeof(prof_ctx_grp[0]); i++) {
		profile_record_summarize(&prof_ctx_grp[i]);
	}
}

/**
 * @brief The kernel function that performs the parallel compute operation
 *	  in the GPU
 *
 * @param[in]		n Vector size
 * @param[in]		a Number to multiply with the vector x
 * @param[in]		x Pointer to the device memory holding x vector
 * @param[in, out]	y Pointer to the device memory holding y vector
 */
__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		y[i] = a * x[i] + y[i];
}

/**
 * @brief The kernel function that performs the parallel compute operation
 *	  in the Host CPU
 *
 * @param[in]		n Vector size
 * @param[in]		a Number to multiply with the vector x
 * @param[in]		x Pointer to the Host memory holding x vector
 * @param[in, out]	y Pointer to the Host memory holding y vector
 */
void saxpy_host(int n, float a, float *x, float *y)
{
	int i;

	for (i = 0; i < n; i++) {
		y[i] += a * x[i];
	}
}

/**
 * @brief The main function that allocates the vectors in the Host memory
 *	  and initializes them with constant values.
 *
 * Depending on the options passed, the vectors are either copied to the Device memory
 * or mapped from the Host memory and asks the Device to perform parallel computation
 * with the kernel function.
 *
 * @param[in] argc Number of command line arguments passed
 * @param[in] argv The arguments passed to the executable
 *
 * @return 0 for success, -1 for failure
 */
int main(int argc, char *argv[])
{
	int i, ret = 0, N = TEST_VECTOR_SIZE;
	int cuda_memcpy_enabled = true;
	int host_map_enabled = true;
	int profiling_enabled = true;
	int host_compute_mode = false;
	int c, option_index = 0;
	unsigned int dev_flags, host_flags;
	float *x, *y, *r, *d_x = NULL, *d_y = NULL, *p_x, *p_y;
	float maxError = 0.0f;
	cudaError_t err;

	/* Parse the command line arguments */
	while (1) {
		struct option long_options[] = {
			/* Options that set a flag */
			{ "no-cuda-memcpy", no_argument, &cuda_memcpy_enabled, 0 },
			{ "no-host-map", no_argument, &host_map_enabled, 0 },
			{ "no-profiling", no_argument, &profiling_enabled, 0 },
			{ "host-compute-mode", no_argument, &host_compute_mode, 1 },

			/* Options that don't set a flag */
			{ "vector-size", required_argument, 0, 's' },
			{ "help", no_argument, 0, 'h' },
		};

		c = getopt_long(argc, argv, "hs:", long_options, &option_index);
		if (c == -1)
			break;

		switch (c) {
		case 0:
			if (long_options[option_index].flag != 0)
				break;
			printf ("option %s", long_options[option_index].name);
			if (optarg)
				printf(" with arg %s", optarg);
			printf("\n");
			break;
		case 's':
			N = atoi(optarg);
			break;
		case 'h':
			/* fallthrough */
		default:
			printf("Usage: %s [--no-cuda-memcpy] [--no-host-map] "		\
					"[--no-profiling] [--host-compute-mode] "	\
					"[--vector-size <size>]\n", argv[0]);
			exit(0);
			break;
		}
	}

	if (host_compute_mode) {
		cuda_memcpy_enabled = false;
		host_map_enabled = false;
	}

	/* 8< Profile Setup time */
	profile_record_start(PROF_TASK_SETUP, profiling_enabled);

	/* Get the CUDA Device flags */
	err = cudaGetDeviceFlags(&dev_flags);
	if (err != cudaSuccess) {
		printf("Failed to get CUDA device flags: %s/n", cudaGetErrorString(err));
		ret = -1;
		goto err_flags;
	}

	/* Profile Setup time >8 */
	profile_record_stop(PROF_TASK_SETUP, profiling_enabled);

	/* Interrogate the Device flags and summarize the available support */
	dev_flags &= cudaDeviceMask;
	printf("CUDA Device Support Summary:\n");
	printf("\tMapped pinned allocations\t\t\t[%s]\n",
			(dev_flags & cudaDeviceMapHost) ? "Yes" : " No");
	printf("\tAutomatic Scheduling\t\t\t\t[%s]\n",
			(dev_flags & cudaDeviceScheduleAuto) ? "Yes" : " No");
	printf("\tUse blocking synchronization\t\t\t[%s]\n",
			(dev_flags & cudaDeviceScheduleBlockingSync) ? "Yes" : " No");
	printf("\tSpin default scheduling\t\t\t\t[%s]\n",
			(dev_flags & cudaDeviceScheduleSpin) ? "Yes" : " No");
	printf("\tYield default scheduling\t\t\t[%s]\n",
			(dev_flags & cudaDeviceScheduleYield) ? "Yes" : " No");
	printf("\tKeep local memory allocation after launch\t[%s]\n",
			(dev_flags & cudaDeviceLmemResizeToMax) ? "Yes" : " No");
	printf("\n");

	/* Show the compute options */
	printf("Compute options:\n");
	printf("\tCUDA Memcpy\t\t\t\t\t[%s]\n",
			cuda_memcpy_enabled ? " Enabled" : "Disabled");
	printf("\tHost Map\t\t\t\t\t[%s]\n",
			((dev_flags & cudaDeviceMapHost)
			 && host_map_enabled) ? " Enabled" : "Disabled");
	printf("\tProfiling Enabled\t\t\t\t[%s]\n",
			profiling_enabled ? " Enabled" : "Disabled");
	printf("\tHost Compute Mode\t\t\t\t[%s]\n",
			host_compute_mode ? " Enabled" : "Disabled");
	printf("\tVector Size\t\t\t\t\t[%d]\n\n", N);

	/* Setup Host allocation flags */
	host_flags = cudaHostAllocDefault;
	if ((dev_flags & cudaDeviceMapHost) && host_map_enabled) {
		host_flags |= (cudaHostAllocMapped | cudaHostAllocWriteCombined);
	}

	/* 8< Profile Host allocation time */
	profile_record_start(PROF_TASK_HOST_ALLOC, profiling_enabled);

	/* Allocate x in Host memory */
	err = cudaHostAlloc(&x, N * sizeof(float), host_flags);
	if (err != cudaSuccess) {
		printf("Failed to allocate memory for x: %s\n", cudaGetErrorString(err));
		ret = -1;
		goto err_alloc_x;
	}

	/* Allocate y in Host memory */
	err = cudaHostAlloc(&y, N * sizeof(float), host_flags);
	if (err != cudaSuccess) {
		printf("Failed to allocate memory for y: %s\n", cudaGetErrorString(err));
		ret = -1;
		goto err_alloc_y;
	}

	/* Allocate r in Host memory */
	if ((dev_flags & cudaDeviceMapHost) && host_map_enabled)
		host_flags &= ~cudaHostAllocWriteCombined;

	err = cudaHostAlloc(&r, N * sizeof(float), host_flags);
	if (err != cudaSuccess) {
		printf("Failed to allocate memory for r: %s\n", cudaGetErrorString(err));
		ret = -1;
		goto err_alloc_r;
	}

	/* Profile Host allocation time >8 */
	profile_record_stop(PROF_TASK_HOST_ALLOC, profiling_enabled);

	/* Allocate Device memory if CUDA memcpy is enabled */
	if (cuda_memcpy_enabled) {
		/* 8< Profile Device allocation time */
		profile_record_start(PROF_TASK_DEVICE_ALLOC, profiling_enabled);

		/* Allocate x in Device memory */
		err = cudaMalloc(&d_x, N * sizeof(float));
		if (err != cudaSuccess) {
			printf("Failed to CUDA Malloc for d_x: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_alloc_d_x;
		}

		/* Allocate y in Device memory */
		err = cudaMalloc(&d_y, N * sizeof(float));
		if (err != cudaSuccess) {
			printf("Failed to CUDA Malloc for d_y: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_alloc_d_y;
		}

		/* Profile Device allocation time >8 */
		profile_record_stop(PROF_TASK_DEVICE_ALLOC, profiling_enabled);
	}

	/* 8< Profile Host initialisation time */
	profile_record_start(PROF_TASK_HOST_INIT, profiling_enabled);

	/* Initialize the array in Host */
	for (i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	/* Profile Host initialisation time >8 */
	profile_record_stop(PROF_TASK_HOST_INIT, profiling_enabled);

	if (cuda_memcpy_enabled) {
		/* 8< Profile Device Memcpy time */
		profile_record_start(PROF_TASK_DEVICE_MEMCPY, profiling_enabled);

		/* Move x from Host to Device memory */
		err = cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Failed to copy x to d_x: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_copy;
		}

		/* Move y from Host to Device memory */
		err = cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Failed to copy y to d_y: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_copy;
		}

		/* Profile Device Memcpy time >8 */
		profile_record_stop(PROF_TASK_DEVICE_MEMCPY, profiling_enabled);

		p_x = d_x;
		p_y = d_y;
	} else {
		p_x = x;
		p_y = y;
	}

	if (!host_compute_mode) {
		/* 8< Profile Device Compute time */
		profile_record_start(PROF_TASK_DEVICE_COMPUTE, profiling_enabled);

		/* Perform SAXPY on the vectors present in the respective memory */
		saxpy<<<(N + 255)/256, 256>>>(N, 2.0f, p_x, p_y);

		/* Wait for the Device to finish */
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_sync;
		}

		/* Profile Device Compute time >8 */
		profile_record_stop(PROF_TASK_DEVICE_COMPUTE, profiling_enabled);
	} else {
		/* 8< Profile Host Compute time */
		profile_record_start(PROF_TASK_HOST_COMPUTE, profiling_enabled);

		saxpy_host(N, 2.0f, p_x, p_y);

		/* Profile Host Compute time >8 */
		profile_record_stop(PROF_TASK_HOST_COMPUTE, profiling_enabled);
	}

	if (cuda_memcpy_enabled) {
		/* Copy results from Device to Host memory */
		err = cudaMemcpy(r, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("Failed to copy d_y to r: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_copy;
		}
	} else {
		/* 8< Profile Host Memcpy time */
		profile_record_start(PROF_TASK_HOST_MEMCPY, profiling_enabled);

		/* Copy results from Host to Host memory */
		err = cudaMemcpy(r, y, N * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			printf("Failed to copy y to r: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_copy;
		}

		/* 8< Profile Host Memcpy time */
		profile_record_stop(PROF_TASK_HOST_MEMCPY, profiling_enabled);
	}

	/* 8< Profile Host Verification time */
	profile_record_start(PROF_TASK_HOST_VERIFY, profiling_enabled);

	/* Calculate errors and display the result */
	for (i = 0; i < N; i++)
		maxError = max(maxError, abs(r[i] - 4.0f));

	/* Profile Host Verification time >8 */
	profile_record_stop(PROF_TASK_HOST_VERIFY, profiling_enabled);

	printf("Compute summary:\n\tMax error:\t\t\t\t\t[%f]\n", maxError);

err_sync:
err_copy:
	if (d_y) {
		/* 8< Profile Device Free time */
		profile_record_start(PROF_TASK_DEVICE_FREE, profiling_enabled);

		cudaFree(d_y);
	}

err_alloc_d_y:
	if (d_x) {
		cudaFree(d_x);

		/* 8< Profile Device Free time */
		profile_record_stop(PROF_TASK_DEVICE_FREE, profiling_enabled);
	}

err_alloc_d_x:
	/* 8< Profile Host Free time */
	profile_record_start(PROF_TASK_HOST_FREE, profiling_enabled);

	cudaFreeHost(r);

err_alloc_r:
	cudaFreeHost(y);

err_alloc_y:
	cudaFreeHost(x);

	/* 8< Profile Host Free time */
	profile_record_stop(PROF_TASK_HOST_FREE, profiling_enabled);

err_alloc_x:
err_flags:
	if (!ret && profiling_enabled) {
		profile_record_calc_delta_all(profiling_enabled);
		printf("\nProfiling results:\n");
		profile_record_summarize_all(profiling_enabled);
	}

	return ret;
}
