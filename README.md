This is a basic CUDA example that does Single-precision A*X Plus Y (SAXPY) computation on the Nvidia GPU.

**Reference:**
```
  https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c
```

**Build:**
```
$ make
CC saxpy.cu
LD saxpy.o
EXEC saxpy
DOXYGEN HTML
```

 **Clean:**
 ```
$ make clean
CLEAN saxpy.o
CLEAN saxpy
CLEAN html
 ```

**Usage:**
```
$ saxpy --help
Usage:
	saxpy [--no-cuda-memcpy] [--no-host-map] [--no-profiling] [--host-compute-mode] [--a <value>] [--x <value>] [--y <value>]
	[--vector-size <size>] [--grid-count <value>][--thread-count <value>]
```

**Example:**
```
$ saxpy 
CUDA Device Summary:
	Device ID					[0]
	Device Name					[NVIDIA GeForce RTX 4090]
	Total Global Memory				[24118 MBytes]
	Total Constant Memory				[65536 Bytes]
	Shared Memory Per Block				[49152 Bytes]
	L2 Cache Size					[72 MBytes]
	Registers Per Block				[65536]
	Warp Size					[32]
	Max Threads Per Block				[1024]
	Clock Rate					[2520 MHz]
	Memory Clock Rate				[10501 MHz]
	Memory Bus Width				[384 Bits]
	Multi Processor Count				[128]
	Max Threads Per Multiprocessor			[1536]
	Device Overlap					[Yes]
	Asynchronous Engine				[Dual]
	Concurrent Kernels				[Yes]
	Map Host Memory					[Yes]
	IPC Event Supported				[Yes]
	Unified Addressing				[Yes]
	GPU Type					[Discrete]

Compute options:
	CUDA Memcpy					[Enabled]
	Host Map					[Enabled]
	Profiling Enabled				[Enabled]
	Host Compute Mode				[Disabled]
	A value						[2]
	X value						[1]
	Y value						[2]
	Grid Count					[256]
	Thread Count					[256]
	Vector Size					[1073741824]

Compute summary:
	Max error:					[0.000000]

Profiling results:
	Host Setup time:				[0.018164440 seconds]
	Host Alloc time:				[3.966379875 seconds]
	Host Free time:					[0.967037230 seconds]
	Host Init time:					[0.703828498 seconds]
	Host Verify time:				[0.764632830 seconds]
	Device Alloc time:				[0.000254140 seconds]
	Device Free time:				[0.003523977 seconds]
	Device Memcpy time:				[0.360076698 seconds]
	Device Compute time:				[0.013531125 seconds]
```
