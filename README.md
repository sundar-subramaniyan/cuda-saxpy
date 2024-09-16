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
	Grid Count					[256]
	Thread Count					[256]
	Vector Size					[1073741824]

Compute summary:
	Max error:					[0.000000]

Profiling results:
	Host Setup time:				[0.016225946 seconds]
	Host Alloc time:				[2.338147570 seconds]
	Host Free time:					[0.978175473 seconds]
	Host Init time:					[0.484248015 seconds]
	Host Verify time:				[0.771969269 seconds]
	Device Alloc time:				[0.000239118 seconds]
	Device Free time:				[0.003376936 seconds]
	Device Memcpy time:				[0.360199580 seconds]
	Device Compute time:				[0.013518014 seconds]
```
