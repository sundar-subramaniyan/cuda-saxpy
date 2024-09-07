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

/**
 * @brief Default size of the vectors
 */
#define TEST_VECTOR_SIZE (1 << 30)

/**
 * @brief The kernel function that performs the parallel compute operation
 *	  in the GPU
 *
 * @param[in] n Vector size
 * @param[in] a Number to multiply with the vector x
 * @param[in] x Pointer to the device memory holding x vector
 * @param[in, out] y Pointer to the device memory holding y vector
 */
__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		y[i] = a * x[i] + y[i];
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
	int c, option_index = 0;
	unsigned int deviceFlags, hostFlags;
	float *x, *y, *d_x, *d_y, *r;
	float maxError = 0.0f;
	cudaError_t err;

	/* Parse the command line arguments */
	while (1) {
		struct option long_options[] = {
			/* Options that set a flag */
			{ "no-cuda-memcpy", no_argument, &cuda_memcpy_enabled, 0 },
			{ "no-host-map", no_argument, &host_map_enabled, 0 },

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
			printf("Usage: %s [--no-cuda-memcpy] [--no-host-map] "\
					"[--vector-size <size>]\n", argv[0]);
			exit(0);
			break;
		}
	}

	/* Get the CUDA Device flags */
	err = cudaGetDeviceFlags(&deviceFlags);
	if (err != cudaSuccess) {
		printf("Failed to get CUDA device flags: %s/n", cudaGetErrorString(err));
		ret = -1;
		goto err_flags;
	}

	/* Interrogate the Device flags and summarize the available support */
	deviceFlags &= cudaDeviceMask;
	printf("CUDA Device Support Summary:\n");
	printf("\tMapped pinned allocations\t\t\t[%s]\n",
			(deviceFlags & cudaDeviceMapHost) ? "Yes" : " No");
	printf("\tAutomatic Scheduling\t\t\t\t[%s]\n",
			(deviceFlags & cudaDeviceScheduleAuto) ? "Yes" : " No");
	printf("\tUse blocking synchronization\t\t\t[%s]\n",
			(deviceFlags & cudaDeviceScheduleBlockingSync) ? "Yes" : " No");
	printf("\tSpin default scheduling\t\t\t\t[%s]\n",
			(deviceFlags & cudaDeviceScheduleSpin) ? "Yes" : " No");
	printf("\tYield default scheduling\t\t\t[%s]\n",
			(deviceFlags & cudaDeviceScheduleYield) ? "Yes" : " No");
	printf("\tKeep local memory allocation after launch\t[%s]\n",
			(deviceFlags & cudaDeviceLmemResizeToMax) ? "Yes" : " No");
	printf("\n");

	/* Show the compute options */
	printf("Compute options:\n");
	printf("\tUsing CUDA Memcpy\t\t\t\t[%s]\n",
			cuda_memcpy_enabled ? "Yes" : " No");
	printf("\tUsing Host Map\t\t\t\t\t[%s]\n",
			((deviceFlags & cudaDeviceMapHost)
			 && host_map_enabled) ? "Yes" : " No");
	printf("\tUsing Vector Size\t\t\t\t[%d]\n\n", N);

	/* Setup Host allocation flags */
	hostFlags = cudaHostAllocDefault;
	if ((deviceFlags & cudaDeviceMapHost) && host_map_enabled) {
		hostFlags |= (cudaHostAllocMapped | cudaHostAllocWriteCombined);
	}

	/* Allocate x in Host memory */
	err = cudaHostAlloc(&x, N * sizeof(float), hostFlags);
	if (err != cudaSuccess) {
		printf("Failed to allocate memory for x: %s\n", cudaGetErrorString(err));
		ret = -1;
		goto err_alloc_x;
	}

	/* Allocate y in Host memory */
	err = cudaHostAlloc(&y, N * sizeof(float), hostFlags);
	if (err != cudaSuccess) {
		printf("Failed to allocate memory for y: %s\n", cudaGetErrorString(err));
		ret = -1;
		goto err_alloc_y;
	}

	/* Allocate r in Host memory */
	if ((deviceFlags & cudaDeviceMapHost) && host_map_enabled)
		hostFlags &= ~cudaHostAllocWriteCombined;

	err = cudaHostAlloc(&r, N * sizeof(float), hostFlags);
	if (err != cudaSuccess) {
		printf("Failed to allocate memory for r: %s\n", cudaGetErrorString(err));
		ret = -1;
		goto err_alloc_r;
	}

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

	/* Initialize the array in Host */
	for (i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	if (cuda_memcpy_enabled) {
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

		/* Perform SAXPY on the vectors copied to the Device memory */
		saxpy<<<(N + 255)/256, 256>>>(N, 2.0f, d_x, d_y);
	} else {
		/* Perform SAXPY on the vectors present in the Host memory */
		saxpy<<<(N + 255)/256, 256>>>(N, 2.0f, x, y);
	}

	/* Wait for the Device to finish */
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
		ret = -1;
		goto err_sync;
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
		/* Copy results from Host to Host memory */
		err = cudaMemcpy(r, y, N * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			printf("Failed to copy y to r: %s\n", cudaGetErrorString(err));
			ret = -1;
			goto err_copy;
		}
	}

	/* Calculate errors and display the result */
	for (i = 0; i < N; i++)
		maxError = max(maxError, abs(r[i] - 4.0f));

	printf("Compute complete. Max error: %f\n", maxError);

err_sync:
err_copy:
	cudaFree(d_y);

err_alloc_d_y:
	cudaFree(d_x);

err_alloc_d_x:
	cudaFreeHost(r);

err_alloc_r:
	cudaFreeHost(y);

err_alloc_y:
	cudaFreeHost(x);

err_alloc_x:
err_flags:
	return ret;
}
