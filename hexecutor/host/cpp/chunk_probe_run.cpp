#include "detail.hpp"
#include <iostream>
// Runs the read-only chunk-map probe once. Usage: hexecutor_chunk_probe LOADER PROBE_ELF
// (set HEXECUTOR_CHUNK_WINDOW_MB for windows above the shipped 128 MiB).
int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: hexecutor_chunk_probe LOADER PROBE_ELF\n";
        return 2;
    }
    try {
        hexecutor::Artifacts a;
        a.loader = argv[1];
        a.chunk_probe = argv[2];
        hexecutor::detail::verify_chunk_map(a);
        std::cout << "chunk map clean\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "chunk map failed: " << e.what() << '\n';
        return 1;
    }
}
