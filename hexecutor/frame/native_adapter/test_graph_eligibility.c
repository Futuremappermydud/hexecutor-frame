#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "qd_native_1x1_graph_eligibility.h"

#define MAGIC 0x31454e47u
#ifndef QD_N1X1_FIXTURE_DIR
#define QD_N1X1_FIXTURE_DIR "/src/native_adapter"
#endif
typedef struct { uint32_t words[20]; } Record;

static int run_file(const char *path, uint32_t wanted_ci, uint32_t wanted_native,
                    int wanted_padded) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    uint32_t header[2]; Record records[4];
    int ok = fread(header, sizeof(header), 1, f) == 1 &&
             header[0] == MAGIC &&
             fread(records, sizeof(records), 1, f) == 1 && fgetc(f) == EOF;
    fclose(f);
    if (!ok) { fprintf(stderr, "bad metadata packet %s\n", path); return 1; }
    QdN1x1GraphTensorMeta meta[4];
    memcpy(meta, records, sizeof(meta));
    QdN1x1GraphCandidate candidate;
    int accepted = qd_native_1x1_graph_candidate(&meta[0], &meta[1], &meta[2],
                                                  &meta[3], &candidate);
    if (wanted_native == 0) {
        if (accepted) { fprintf(stderr, "%s: disabled candidate accepted\n", path); return 1; }
        return 0;
    }
    if (!accepted || candidate.raw_ci != wanted_ci ||
        candidate.native_ci != wanted_native ||
        candidate.padded_ci != (uint32_t)wanted_padded) {
        fprintf(stderr, "%s: candidate mismatch accepted=%d raw=%u native=%u pad=%u\n",
                path, accepted, candidate.raw_ci, candidate.native_ci,
                candidate.padded_ci);
        return 1;
    }
    QdN1x1GraphTensorMeta bad = meta[0];
    bad.dtype = QD_N1X1_G_I8;
    if (qd_native_1x1_graph_candidate(&bad, &meta[1], &meta[2], &meta[3], &candidate)) {
        fprintf(stderr, "%s: bad activation dtype accepted\n", path); return 1;
    }
    bad = meta[1]; bad.m[2] += 4; bad.d[2] = bad.m[2];
    if (qd_native_1x1_graph_candidate(&meta[0], &bad, &meta[2], &meta[3], &candidate)) {
        fprintf(stderr, "%s: unsupported weight layout accepted\n", path); return 1;
    }
    return 0;
}

int main(void) {
#if QDS_NATIVE_PADDED_1X1
    if (run_file(QD_N1X1_FIXTURE_DIR "/graph-eligibility-115.bin", 12, 32, 1) ||
        run_file(QD_N1X1_FIXTURE_DIR "/graph-eligibility-3416.bin", 20, 32, 1))
        return 1;
#else
    if (run_file(QD_N1X1_FIXTURE_DIR "/graph-eligibility-115.bin", 12, 0, 0) ||
        run_file(QD_N1X1_FIXTURE_DIR "/graph-eligibility-3416.bin", 20, 0, 0))
        return 1;
#endif
    if (run_file(QD_N1X1_FIXTURE_DIR "/graph-eligibility-1195.bin", 128, 128, 0))
        return 1;
    puts("PASS actual-QDRX C bridge eligibility predicates and negatives");
    return 0;
}
