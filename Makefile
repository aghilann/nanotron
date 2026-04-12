NGPU ?= 2

.PHONY: test-tp test-layers test-convergence test-gpt test-single-gpu benchmark test-all train-single train-tp

test-tp:
	torchrun --nproc_per_node=$(NGPU) tests/test_tp_communication.py

test-layers:
	torchrun --nproc_per_node=$(NGPU) tests/test_tp_layers.py

test-convergence:
	torchrun --nproc_per_node=$(NGPU) tests/test_convergence.py

test-gpt:
	torchrun --nproc_per_node=$(NGPU) tests/test_gpt.py

test-single-gpu:
	python tests/test_single_gpu.py

benchmark:
	@echo "========================================"
	@echo " 2-GPU  Tensor Parallel"
	@echo "========================================"
	torchrun --nproc_per_node=2 tests/test_e2e.py
	@echo ""
	@echo "========================================"
	@echo " 1-GPU  Single GPU"
	@echo "========================================"
	python tests/test_e2e.py

test-all: test-tp test-layers test-convergence test-gpt

train-single:
	python tests/test_e2e.py

train-tp:
	torchrun --nproc_per_node=$(NGPU) tests/test_e2e.py
