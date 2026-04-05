NGPU ?= 2

.PHONY: test-tp

test-tp:
	torchrun --nproc_per_node=$(NGPU) tests/test_tp_communication.py
