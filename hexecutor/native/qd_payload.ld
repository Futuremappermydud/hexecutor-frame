/* SPDX-License-Identifier: LGPL-3.0-or-later */
ENTRY(payload)
PHDRS { code PT_LOAD FLAGS(5); }
SECTIONS {
    . = 0x87610000;
    .text : { KEEP(*(.text.entry)) *(.text*) } :code
    .rodata : { *(.rodata*) } :code
    .data : { *(.data*) } :code
    .bss : { *(.bss*) *(COMMON) } :code
    ASSERT(. <= 0x87618000, "payload exceeds its 32 KB region")
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
