This is a basic CUDA example that does Single-precision A*X Plus Y (SAXPY) computation on the Nvidia GPU.

Reference:
  https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c

**Build:**
```
$ make
```

 **Clean:**
 ```
$ make clean
 ```

**Run:**
```
$ saxpy --help
Usage: saxpy [--no-cuda-memcpy] [--no-host-map] [--vector-size <size>]
```
**Example:**
```
$ saxpy 
CUDA Device Support Summary:
	Mapped pinned allocations				[Yes]
	Automatic Scheduling					[ No]
	Use blocking synchronization				[ No]
	Spin default scheduling					[ No]
	Yield default scheduling				[ No]
	Keep local memory allocation after launch		[ No]
    
Compute options:
	Using CUDA Memcpy					[Yes]
	Using Host Map						[Yes]
	Using Vector Size					[1073741824]
	    
Compute complete. Max error: 0.000000
```
