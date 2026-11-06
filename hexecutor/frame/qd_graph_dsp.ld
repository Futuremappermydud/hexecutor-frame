ENTRY(resident_entry)
PHDRS { code PT_LOAD FLAGS(7); }
SECTIONS {
    . = 0xa0000000;
    .text : { KEEP(*(.text.resident_entry)) *(.text*) } :code
    .rodata : { *(.rodata*) } :code
    .data : { *(.data*) *(.sdata*) } :code
    .bss : { *(.bss*) *(.sbss*) *(COMMON) } :code
    ASSERT(. <= 0xa0100000, "resident engine exceeds its private code region")
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
