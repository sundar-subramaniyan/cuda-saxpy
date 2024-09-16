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
Usage: saxpy [--no-cuda-memcpy] [--no-host-map] [--no-profiling] [--host-compute-mode] [--a <value>] [--x <value>] [--y <value>] [--vector-size <size>]
```

**Example:**
```
$ saxpy 
CUDA Device Support Summary:
	Mapped pinned allocations			[Yes]
	Automatic Scheduling				[ No]
	Use blocking synchronization			[ No]
	Spin default scheduling				[ No]
	Yield default scheduling			[ No]
	Keep local memory allocation after launch	[ No]

Compute options:
	CUDA Memcpy					[ Enabled]
	Host Map					[ Enabled]
	Profiling Enabled				[ Enabled]
	Host Compute Mode				[Disabled]
	A value						[2]
	X value						[1]
	Y value						[2]
	Vector Size					[1073741824]

Compute summary:
	Max error:					[0.000000]

Profiling results:
	Host Setup time:				[0.016196367 seconds]
	Host Alloc time:				[4.024780518 seconds]
	Host Free time:					[1.025804307 seconds]
	Host Init time:					[0.483573437 seconds]
	Host Verify time:				[0.763202876 seconds]
	Device Alloc time:				[0.000254980 seconds]
	Device Free time:				[0.003375982 seconds]
	Device Memcpy time:				[0.359966050 seconds]
	Device Compute time:				[0.013518742 seconds]
```
