HEX_CC ?= clang-14
HEX_LD ?= ld.lld-14
HOST_CC ?= cc
CODE_BASE ?= 0x87605000
HEX_FLAGS = --target=hexagon -mcpu=hexagonv68 -mhvx -mhvx-length=128b -O2 -ffreestanding -fno-stack-protector -fno-pic -fno-builtin -Wall -Wextra -Werror

.PHONY: all clean
all: build/qd_hvx.elf build/qd_loader.elf build/qd_conv.elf build/bench_hvx.elf

build:
	mkdir -p build
build/qd_hvx.o: qd_hvx.c qd_protocol.h | build
	$(HEX_CC) $(HEX_FLAGS) -c qd_hvx.c -o $@
build/qd_hvx.elf: build/qd_hvx.o qd_link.ld
	$(HEX_LD) --defsym=QD_LINK_BASE=$(CODE_BASE) -T qd_link.ld $< -o $@
build/libqd_host.so: qd_hvx.c qd_protocol.h | build
	$(HOST_CC) -O2 -std=c11 -fPIC -shared -DQD_HOST_TEST -Wall -Wextra -Werror qd_hvx.c -o $@
clean:
	rm -f build/qd_hvx.o build/qd_hvx.elf build/libqd_host.so
build/qd_loader.o: qd_loader.S | build
	$(HEX_CC) --target=hexagon -mcpu=hexagonv68 -Wno-everything -c $< -o $@
build/qd_loader.elf: build/qd_loader.o qd_link.ld
	$(HEX_LD) --defsym=QD_LINK_BASE=0x87606000 -e loader -T qd_link.ld $< -o $@
build/%.o: bench/%.c | build
	$(HEX_CC) $(HEX_FLAGS) -c $< -o $@
build/bench_hvx.elf: build/bench_hvx.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/qd_conv.o: qd_conv.c qd_conv.h qd_protocol.h qd_axim.h qd_hmx_layout.h qd_hmx_kernel.h qd_hmx_power.h | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c qd_conv.c -o $@
build/qd_hmx_ops.o: qd_hmx_ops_encoded.S | build
	$(HEX_CC) --target=hexagon -mcpu=hexagonv68 -c $< -o $@
build/qd_conv.elf: build/qd_conv.o build/qd_hmx_ops.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld build/qd_conv.o build/qd_hmx_ops.o -o $@
build/libqd_conv_host.so: qd_conv.c qd_conv.h qd_protocol.h | build
	$(HOST_CC) -O2 -std=c11 -fPIC -shared -DQD_HOST_TEST -Wall -Wextra -Werror qd_conv.c -o $@
build/qd_resident.o: qd_resident.c qd_resident.h qd_resident_schema.h | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
build/qd_resident_probe.o: bench/qd_resident_probe.c qd_resident.h qd_resident_schema.h qd_protocol.h qd_conv.h | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
build/qd_resident_probe.elf: build/qd_resident_probe.o build/qd_resident.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld build/qd_resident_probe.o build/qd_resident.o -o $@
build/qd_owned_mem_probe.o: bench/qd_owned_mem_probe.c qd_owned_memory.h | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
build/qd_owned_mem_probe.elf: build/qd_owned_mem_probe.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/libqd_resident_host.so: qd_resident.c qd_resident.h qd_resident_schema.h | build
	$(HOST_CC) -O2 -std=c11 -fPIC -shared -Wall -Wextra -Werror qd_resident.c -o $@
build/hmx_probe.elf: build/hmx_probe.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/rsc_dump.elf: build/rsc_dump.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/hmx_bringup.elf: build/hmx_bringup.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/hmx_bringup.o: bench/hmx_bringup.c | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
build/hmx_exec.o: bench/hmx_bringup.c | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -DHMX_EXEC -c $< -o $@
build/hmx_ops.o: bench/hmx_ops.S | build
	$(HEX_CC) --target=hexagon -mcpu=hexagonv68 -c $< -o $@
build/hmx_exec.elf: build/hmx_exec.o build/hmx_ops.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld build/hmx_exec.o build/hmx_ops.o -o $@
build/hmx_tune.o: bench/hmx_tune.c bench/hmx_power.h | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
build/hmx_tune_asm.o: bench/hmx_tune_encoded.S | build
	$(HEX_CC) --target=hexagon -mcpu=hexagonv68 -mhvx -mhvx-length=128b -c $< -o $@
build/hmx_tune.elf: build/hmx_tune.o build/hmx_tune_asm.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld build/hmx_tune.o build/hmx_tune_asm.o -o $@
build/hmx_q4_matvec.o: llm/hmx_q4_matvec.c bench/hmx_power.h | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
build/hmx_q4_matvec.elf: build/hmx_q4_matvec.o build/hmx_tune_asm.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld build/hmx_q4_matvec.o build/hmx_tune_asm.o -o $@
build/bench_dma.elf: build/bench_dma.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/bench_dma.o: bench/bench_dma.c | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
HEX73_FLAGS = --target=hexagon -mcpu=hexagonv73 -mhvx=v73 -mhvx-length=128b -O2 -ffreestanding -fno-stack-protector -fno-pic -fno-builtin -Wall -Wextra -Werror -G0
build/probe_qf.o: bench/probe_qf.c | build
	$(HEX_CC) $(HEX73_FLAGS) -c $< -o $@
build/probe_qf.elf: build/probe_qf.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/qwen_dsp.o: llm/qwen_dsp.c | build
	$(HEX_CC) $(HEX73_FLAGS) -c $< -o $@
build/qwen_dsp.elf: build/qwen_dsp.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/clk_dump.elf: build/clk_dump.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/axim_tune.o: bench/axim_tune.c | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -c $< -o $@
build/axim_tune.elf: build/axim_tune.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/axim_tune_1000.o: bench/axim_tune.c | build
	$(HEX_CC) $(HEX_FLAGS) -G0 -DAXIM_PLL_L=104u -DAXIM_PLL_ALPHA=0x2aaau -c $< -o $@
build/axim_tune_1000.elf: build/axim_tune_1000.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/qwen_dsp_1000.o: llm/qwen_dsp.c | build
	$(HEX_CC) $(HEX73_FLAGS) -DAXIM_PLL_L=104u -DAXIM_PLL_ALPHA=0x2aaau -c $< -o $@
build/qwen_dsp_1000.elf: build/qwen_dsp_1000.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
build/qwen_ops_test.o: bench/qwen_ops_test.c llm/qwen_dsp.c | build
	$(HEX_CC) $(HEX73_FLAGS) -Wno-unused-function -c $< -o $@
build/qwen_ops_test.elf: build/qwen_ops_test.o qd_payload.ld
	$(HEX_LD) -T qd_payload.ld $< -o $@
