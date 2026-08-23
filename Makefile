.PHONY: test validate-aa validate-power validate-peek validate-cuped validate-seq readout validate-all dashboard interference interference-sweep interference-mechanism interference-regimes interference-coverage
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
dashboard:
	PYTHONPATH=src python -m expkit.dashboard --lift 0.04 --latency-delta 8
	PYTHONPATH=src python -m expkit.dashboard --lift 0.04 --latency-delta 8 --drop-treatment 0.06 --out results/dashboard_srm.html
interference:
	PYTHONPATH=src python -u -m expkit.interference compare --worlds 30
interference-sweep:
	PYTHONPATH=src python -u -m expkit.interference sweep --worlds 30
interference-mechanism:
	PYTHONPATH=src python -u -m expkit.interference mechanism --worlds 20
interference-regimes:
	PYTHONPATH=src python -u -m expkit.interference regimes --worlds 15
interference-coverage:
	PYTHONPATH=src python -u -m expkit.interference coverage --worlds 150
