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
Usage: saxpy [--no-cuda-memcpy] [--no-host-map] [--no-profiling] [--vector-size <size>]
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
	Vector Size					[1073741824]

Compute summary:
	Max error:					[0.000000]

Profiling results:
	Setup time:					[3.024503611 seconds]
	Compute time:					[0.536040190 seconds]
	Verification time:				[0.776172142 seconds]
```
