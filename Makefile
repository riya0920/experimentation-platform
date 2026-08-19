.PHONY: test validate-aa validate-power validate-peek validate-cuped validate-seq readout validate-all
test:
	pytest
validate-aa:
	PYTHONPATH=src python -u -m expkit.validation aa --runs 1000
validate-power:
	PYTHONPATH=src python -u -m expkit.validation power --runs 300
validate-peek:
	PYTHONPATH=src python -u -m expkit.validation peeking --runs 400
validate-cuped:
	PYTHONPATH=src python -u -m expkit.validation cuped --runs 200
validate-seq:
	PYTHONPATH=src python -u -m expkit.validation sequential --runs 400
validate-all: validate-aa validate-power validate-peek validate-cuped validate-seq
readout:
	PYTHONPATH=src python -m expkit.readout --lift 0.10 --latency-delta 30
