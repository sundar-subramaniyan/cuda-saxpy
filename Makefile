CC := nvcc
CFLAGS += "-O3"
LDFLAGS := "-lcuda"
OBJS := saxpy.o
EXEC := saxpy
DOXYGEN_DIR := html

ifneq (, $(shell which nvcc))
	CUDA_TOOLCHAIN_INSTALLED := true
else
	CUDA_TOOLCHAIN_INSTALLED := false
endif

ifneq (, $(shell which doxygen))
	DOXYGEN_INSTALLED := true
else
	DOXYGEN_INSTALLED := false
endif

all: nvcc_check $(EXEC) $(DOXYGEN_DIR)

nvcc_check:
	@$(CUDA_TOOLCHAIN_INSTALLED) || \
	(echo "Nvidia CUDA Toolchain is not installed. Aborting build."; exit 1)

$(EXEC): $(OBJS)
	@echo "LD $<"
	@echo "EXEC $(EXEC)"
	@$(CC) $(OBJS) $(LDFLAGS) -o $(EXEC)

%.o: %.cu
	@echo "CC $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(DOXYGEN_DIR):
	@if [ $(DOXYGEN_INSTALLED) = "true" ]; then \
	echo "DOXYGEN HTML"; \
	doxygen -q; \
	else \
	echo "Doxygen not installed. Skipping document generation."; \
	fi

.PHONY: clean
clean:
	@[ -f $(OBJS) ] && echo "CLEAN $(OBJS)" && rm -rf $(OBJS) || true
	@[ -f $(EXEC) ] && echo "CLEAN $(EXEC)" && rm -rf $(EXEC) || true
	@[ -d $(DOXYGEN_DIR) ] && echo "CLEAN $(DOXYGEN_DIR)" && rm -rf $(DOXYGEN_DIR) || true
