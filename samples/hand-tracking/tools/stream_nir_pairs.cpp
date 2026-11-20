// Read-only XRService tracking-camera stream, based on plaice/runtime/src/ir_pair_capture.cpp.
// Binary stdout: magic FRNIR002, RAW timestamps (int64), sequences (uint32),
// JPEG byte lengths (uint32), grayscale JPEGs. --raw retains FRNIR001 pixels.
// Build with -std=c++17 -O2 -lpthread -ljpeg. All integers are little endian.
#include "ir_pair_capture.hpp"
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <jpeglib.h>

static std::vector<uint8_t> jpeg(const std::vector<uint8_t>& pixels, int quality) {
    jpeg_compress_struct c{};
    jpeg_error_mgr error{};
    c.err = jpeg_std_error(&error);
    jpeg_create_compress(&c);
    unsigned char* data = nullptr;
    unsigned long size = 0;
    jpeg_mem_dest(&c, &data, &size);
    c.image_width = 1056; c.image_height = 1024;
    c.input_components = 1; c.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&c);
    jpeg_set_quality(&c, quality, TRUE);
    jpeg_start_compress(&c, TRUE);
    while (c.next_scanline < c.image_height) {
        JSAMPROW row = const_cast<JSAMPROW>(pixels.data() + c.next_scanline * 1056);
        jpeg_write_scanlines(&c, &row, 1);
    }
    jpeg_finish_compress(&c);
    std::vector<uint8_t> result(data, data + size);
    std::free(data);
    jpeg_destroy_compress(&c);
    return result;
}

static volatile std::sig_atomic_t running = 1;
static void stop(int) { running = 0; }

int main(int argc, char** argv) {
    int limit = 0, quality = 70;
    bool raw = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "--raw")) raw = true;
        else if ((!std::strcmp(argv[i], "--max-pairs") || !std::strcmp(argv[i], "--jpeg-quality")) && i+1<argc) {
            const bool q = !std::strcmp(argv[i], "--jpeg-quality");
            const int n = std::atoi(argv[++i]);
            if (q) quality = n; else limit = n;
        } else { std::cerr << "usage: stream_nir_pairs [--max-pairs N] [--jpeg-quality 1..100] [--raw]\n"; return 2; }
    }
    if (quality < 1 || quality > 100 || limit < 0) return 2;
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    FrameSpatial::IrPairCapture capture;
    if (!capture.initialize()) { std::cerr << capture.lastError() << '\n'; return 1; }
    int emitted = 0;
    int misses = 0;
    while (running && (!limit || emitted < limit)) {
        FrameSpatial::IrPairCapture::Pair pair;
        if (!capture.grab(pair, 500) || pair.left.size() != 1056u*1024u ||
            pair.right.size() != 1056u*1024u) {
            if (++misses % 4 == 0) std::cerr << "NIR capture waiting: " << capture.lastError() << '\n';
            continue;
        }
        misses = 0;
        if (!raw) { pair.left = jpeg(pair.left,quality); pair.right = jpeg(pair.right,quality); }
        const uint32_t leftSize = uint32_t(pair.left.size()), rightSize = uint32_t(pair.right.size());
        const char magic[8] = {'F','R','N','I','R','0','0',raw ? '1' : '2'};
        if (std::fwrite(magic,1,8,stdout)!=8 ||
            std::fwrite(&pair.leftTsNs,8,1,stdout)!=1 ||
            std::fwrite(&pair.rightTsNs,8,1,stdout)!=1 ||
            std::fwrite(&pair.leftSeq,4,1,stdout)!=1 ||
            std::fwrite(&pair.rightSeq,4,1,stdout)!=1 ||
            (!raw && (std::fwrite(&leftSize,4,1,stdout)!=1 || std::fwrite(&rightSize,4,1,stdout)!=1)) ||
            std::fwrite(pair.left.data(),1,pair.left.size(),stdout)!=pair.left.size() ||
            std::fwrite(pair.right.data(),1,pair.right.size(),stdout)!=pair.right.size()) break;
        std::fflush(stdout);
        ++emitted;
    }
    return 0; // RAII closes duplicates and mappings; XRService was never stopped.
}
