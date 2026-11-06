CC = clang
LD = ld.lld
CFLAGS = --target=hexagon -mcpu=hexagonv73 -O2 -ffreestanding -fno-stack-protector -fno-pic
HVXFLAGS = -mhvx -mhvx-length=128b

.PHONY: all
all: callback.elf callback_hvx.elf callback_vector.elf callback_matmul.elf

callback.o: callback.c
	$(CC) $(CFLAGS) -c $< -o $@

callback_hvx.o: callback_hvx.c
	$(CC) $(CFLAGS) -mhvx -mhvx-length=64b -c $< -o $@

callback_vector.o: callback_vector.c
	$(CC) $(CFLAGS) $(HVXFLAGS) -c $< -o $@

callback_matmul.o: callback_matmul.c
	$(CC) $(CFLAGS) $(HVXFLAGS) -c $< -o $@

callback.elf: callback.o
	$(LD) -Ttext=0x87600000 -e callback $< -o $@

callback_hvx.elf: callback_hvx.o
	$(LD) -Ttext=0x87601000 -e callback $< -o $@

callback_vector.elf: callback_vector.o
	$(LD) -Ttext=0x87602000 -e callback $< -o $@

callback_matmul.elf: callback_matmul.o
	$(LD) -Ttext=0x87603000 -e callback $< -o $@
