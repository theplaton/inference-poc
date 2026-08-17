IMAGE ?= gpu-validator
# How GPUs are exposed. Use --device nvidia.com/gpu=all on hosts where the
# --gpus shorthand misdetects the vendor (seen on Docker 29 + CDI).
GPU_FLAG ?= --gpus all

.PHONY: build run clean

build:
	docker build -t $(IMAGE) .

# Development defaults: inventory checks off, relative-only compute check.
run:
	docker run --rm $(GPU_FLAG) --ipc=host \
		-e EXPECTED_GPU_COUNT="$(EXPECTED_GPU_COUNT)" \
		-e EXPECTED_GPU_NAME="$(EXPECTED_GPU_NAME)" \
		-e MIN_GEMM_TFLOPS="$(MIN_GEMM_TFLOPS)" \
		-e MIN_NCCL_BUSBW_GBPS="$(MIN_NCCL_BUSBW_GBPS)" \
		-e RELATIVE_GPU_FLOOR="$(or $(RELATIVE_GPU_FLOOR),0.90)" \
		-v "$(CURDIR)/results:/results" \
		$(IMAGE)

clean:
	rm -f results/report.txt results/nvidia-smi.txt results/topology.txt \
		results/gemm.txt results/nccl.txt
