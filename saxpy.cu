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
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <cuda.h>

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
 * @brief Number of Grids (Thread Blocks)
 */
#define GRID_COUNT 256

/**
 * @brief Number of Threads per Block
 */
#define THREAD_COUNT 256

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
	tab_stops = (len + TABSPACES - 1)/TABSPACES;

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
 * @brief Print summary of the CUDA device
 *
 * @param[in] device_id	CUDA Device ID
 */
void print_device_summary(int device_id)
{
	cudaDeviceProp prop;
	cudaError_t err;

	memset(&prop, 0, sizeof(prop));

	/* Get CUDA Device Poperties */
	err = cudaGetDeviceProperties(&prop, device_id);
	if (err != cudaSuccess) {
		printf("Error getting CUDA device properties: %s\n", cudaGetErrorString(err));
		return;
	}

	printf("CUDA Device Summary:\n");
	printf("\tDevice ID\t\t\t\t\t[%d]\n", device_id);
	printf("\tDevice Name\t\t\t\t\t[%s]\n", prop.name);
	printf("\tTotal Global Memory\t\t\t\t[%lu MBytes]\n", prop.totalGlobalMem >> 20);
	printf("\tTotal Constant Memory\t\t\t\t[%lu Bytes]\n", prop.totalConstMem);
	printf("\tShared Memory Per Block\t\t\t\t[%lu Bytes]\n", prop.sharedMemPerBlock);
	printf("\tL2 Cache Size\t\t\t\t\t[%d MBytes]\n", prop.l2CacheSize >> 20);
	printf("\tRegisters Per Block\t\t\t\t[%d]\n", prop.regsPerBlock);
	printf("\tWarp Size\t\t\t\t\t[%d]\n", prop.warpSize);
	printf("\tMax Threads Per Block\t\t\t\t[%d]\n", prop.maxThreadsPerBlock);
	printf("\tClock Rate\t\t\t\t\t[%d MHz]\n", prop.clockRate / 1000);
	printf("\tMemory Clock Rate\t\t\t\t[%d MHz]\n", prop.memoryClockRate / 1000);
	printf("\tMemory Bus Width\t\t\t\t[%d Bits]\n", prop.memoryBusWidth);
	printf("\tMulti Processor Count\t\t\t\t[%d]\n", prop.multiProcessorCount);
	printf("\tMax Threads Per Multiprocessor\t\t\t[%d]\n", prop.maxThreadsPerMultiProcessor);
	printf("\tDevice Overlap\t\t\t\t\t[%s]\n", prop.deviceOverlap ? "Yes" : "No");
	printf("\tAsynchronous Engine\t\t\t\t[%s]\n", (prop.asyncEngineCount == 0) ? "No" :
			(prop.asyncEngineCount == 1) ? "Single" : "Dual");
	printf("\tConcurrent Kernels\t\t\t\t[%s]\n", prop.concurrentKernels ? "Yes" : "No");
	printf("\tMap Host Memory\t\t\t\t\t[%s]\n", prop.canMapHostMemory ? "Yes" : "No");
	printf("\tIPC Event Supported\t\t\t\t[%s]\n", prop.ipcEventSupported ? "Yes" : "No");
	printf("\tUnified Addressing\t\t\t\t[%s]\n", prop.unifiedAddressing ? "Yes" : "No");
	printf("\tGPU Type\t\t\t\t\t[%s]\n", (prop.integrated == 1) ? "Integrated" : "Discrete");
	printf("\n");
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
void saxpy(unsigned int n, float a, float *x, float *y)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		y[i] += (a * x[i]);
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
void saxpy_host(unsigned int n, float a, float *x, float *y)
{
	unsigned int i;

	for (i = 0; i < n; i++) {
		y[i] += (a * x[i]);
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
	int ret = 0;
	int device_count = 0, device_id = 0;
	unsigned int i, N = TEST_VECTOR_SIZE;
	int cuda_memcpy_enabled = true;
	int host_map_enabled = true;
	int profiling_enabled = true;
	int host_compute_mode = false;
	int c, option_index = 0;
	unsigned int dev_flags, host_flags;
	int grid_count = GRID_COUNT, thread_count = THREAD_COUNT;
	float a_val = 2.0f, x_val = 1.0f, y_val = 2.0f, result;
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
			{ "a", required_argument, 0, 'a' },
			{ "x", required_argument, 0, 'x' },
			{ "y", required_argument, 0, 'y' },
			{ "vector-size", required_argument, 0, 's' },
			{ "grid-count", required_argument, 0, 'g' },
			{ "thread-count", required_argument, 0, 't' },
			{ "help", no_argument, 0, 'h' },
		};

		c = getopt_long(argc, argv, "axyhsgt:", long_options, &option_index);
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
		case 'a':
			a_val = (float)atof(optarg);
			break;
		case 'x':
			x_val = (float)atof(optarg);
			break;
		case 'y':
			y_val = (float)atof(optarg);
			break;
		case 's':
			N = (unsigned int)strtoul(optarg, NULL, 10);
			break;
		case 'g':
			grid_count = atoi(optarg);
			break;
		case 't':
			thread_count = atoi(optarg);
			break;
		case 'h':
			/* fallthrough */
		default:
			printf("Usage:\n\t%s [--no-cuda-memcpy] [--no-host-map] "		\
					"[--no-profiling] [--host-compute-mode] "		\
					"[--a <value>] [--x <value>] [--y <value>]\n"		\
					"\t[--vector-size <size>] [--grid-count <value>]"	\
					"[--thread-count <value>]\n",
					argv[0]);
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

	/* Check if CUDA capable devices are present */
	err = cudaGetDeviceCount(&device_count);
	if (err != cudaSuccess) {
		printf("Error getting CUDA device count (%d). Exiting.\n", err);
		goto err_no_cuda_dev;
	}

	if (!device_count) {
		printf("No CUDA capable devices present. Exiting.\n");
		goto err_no_cuda_dev;
	}

	/* Get Current CUDA device */
	err = cudaGetDevice(&device_id);
	if (err != cudaSuccess) {
		printf("Error getting CUDA device: %s\n", cudaGetErrorString(err));
		goto err_no_cuda_dev;
	}

	/* Print CUDA device summary */
	print_device_summary(device_id);

	/* Get the CUDA Device flags */
	err = cudaGetDeviceFlags(&dev_flags);
	if (err != cudaSuccess) {
		printf("Failed to get CUDA device flags: %s/n", cudaGetErrorString(err));
		ret = -1;
		goto err_flags;
	}

	/* Limit the Device flags */
	dev_flags &= cudaDeviceMask;

	/* Profile Setup time >8 */
	profile_record_stop(PROF_TASK_SETUP, profiling_enabled);

	/* Show the compute options */
	printf("Compute options:\n");
	printf("\tCUDA Memcpy\t\t\t\t\t[%s]\n",
			cuda_memcpy_enabled ? "Enabled" : "Disabled");
	printf("\tHost Map\t\t\t\t\t[%s]\n",
			((dev_flags & cudaDeviceMapHost)
			 && host_map_enabled) ? "Enabled" : "Disabled");
	printf("\tProfiling Enabled\t\t\t\t[%s]\n",
			profiling_enabled ? "Enabled" : "Disabled");
	printf("\tHost Compute Mode\t\t\t\t[%s]\n",
			host_compute_mode ? "Enabled" : "Disabled");
	printf("\tA value\t\t\t\t\t\t[%g]\n", a_val);
	printf("\tX value\t\t\t\t\t\t[%g]\n", x_val);
	printf("\tY value\t\t\t\t\t\t[%g]\n", y_val);
	printf("\tGrid Count\t\t\t\t\t[%d]\n", grid_count);
	printf("\tThread Count\t\t\t\t\t[%d]\n", thread_count);
	printf("\tVector Size\t\t\t\t\t[%u]\n\n", N);

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
		x[i] = x_val;
		y[i] = y_val;
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
		int M, T;

		/* 8< Profile Device Compute time */
		profile_record_start(PROF_TASK_DEVICE_COMPUTE, profiling_enabled);

		M = (N + grid_count - 1)/grid_count;
		T = thread_count;

		/* Perform SAXPY on the vectors present in the respective memory */
		saxpy<<<M, T>>>(N, a_val, p_x, p_y);

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

		saxpy_host(N, a_val, p_x, p_y);

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
	result = (a_val * x_val) + y_val;
	for (i = 0; i < N; i++)
		maxError = max(maxError, abs(r[i] - result));

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

err_no_cuda_dev:
	return ret;
}
