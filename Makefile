PYTHON ?= python3
BUILD_DIR ?= build
.PHONY: help configure build build-host build-worker build-runtime build-dsp check check-model check-layout check-docs audit run
help:
	@$(PYTHON) tools/dev.py help
configure:
	cmake -S . -B $(BUILD_DIR)
build build-host: configure
	cmake --build $(BUILD_DIR) --parallel
build-worker: configure
	cmake --build $(BUILD_DIR) --parallel --target frame_depth_worker
build-runtime: build-worker
	$(MAKE) -C runtime
build-dsp:
	$(MAKE) -C hexecutor/native
check: build-host
	ctest --test-dir $(BUILD_DIR) --output-on-failure
	$(PYTHON) tools/dev.py check
check-model:
	$(PYTHON) tools/dev.py check-model
check-layout:
	$(PYTHON) tools/dev.py check-layout
check-docs:
	$(PYTHON) tools/dev.py check-docs
audit:
	$(PYTHON) tools/model_artifact_policy.py --history
run:
	$(PYTHON) tools/dev.py run $(SCRIPT) $(ARGS)
