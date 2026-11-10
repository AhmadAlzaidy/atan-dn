# Thin wrapper around tasks.py (works with GNU make on Linux/macOS/Git Bash/Windows).
# Without make on Windows: py -3 tasks.py <task> MISSION=m1
# Variables: MISSION=m1|m2|ops_sat  CONFIG=configs/<mission>.yaml  ARGS="--n-trials 5"
PYTHON ?= python
MISSION ?= m1
CONFIG ?= configs/$(MISSION).yaml
ARGS ?=

.PHONY: install test lint prepare tune train evaluate complexity pipeline

install test lint prepare tune train evaluate complexity pipeline:
	$(PYTHON) tasks.py $@ MISSION=$(MISSION) CONFIG=$(CONFIG) "ARGS=$(ARGS)"
