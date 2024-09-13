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

**Run:**
```
$ saxpy --help
Usage: saxpy [--no-cuda-memcpy] [--no-host-map] [--no-profiling] [--host-compute-mode] [--vector-size <size>]
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
	Vector Size					[1073741824]

Compute summary:
	Max error:					[0.000000]

Profiling results:
	Host Setup time:				[0.016320759 seconds]
	Host Alloc time:				[3.983245582 seconds]
	Host Free time:					[0.977180777 seconds]
	Host Init time:					[0.696118777 seconds]
	Host Verify time:				[0.770025566 seconds]
	Device Alloc time:				[0.000242508 seconds]
	Device Free time:				[0.003560735 seconds]
	Device Memcpy time:				[0.359980237 seconds]
	Device Compute time:				[0.013579677 seconds]
```
