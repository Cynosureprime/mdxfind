/*
 * test_snefru_vectors.c -- R15 pre-flight donor verification for Phase 5b
 * Tier 4 Snefru ship (sub-phase 5b.4a.0).
 *
 * Cross-checks the in-tree librhash Snefru donor (RHash-master/librhash/
 * snefru.c -- the LIVE oracle for e175/e177 via librhash.a) against:
 *   1. Published Snefru-128 and Snefru-256 test vectors (where authoritative
 *      vectors are confidently known: the empty string and "1234567890" /
 *      newline-suffixed vectors from the rhash test suite).
 *   2. Self-consistency: single-shot update vs split (chunked) update across
 *      the data-block boundary (48 bytes for Snefru-128, 32 bytes for
 *      Snefru-256). This exercises the multi-block path and the partial-
 *      block fill logic in rhash_snefru_update, mirroring the HAVAL R15
 *      self-consistency fallback when published non-empty vectors are not
 *      confidently confirmable.
 *
 * Per Phase 5b Tier 4 architect spec (project_hx_codegen_phase5b_tier4_-
 * snefru_gost_spec_2026-05-27.md) R15/R12: the donor must agree byte-for-byte
 * with the published vectors AND be internally self-consistent across the
 * block boundary. If a published vector cannot be authoritatively confirmed
 * it is dropped (per the 5b.3a HAVAL paper-vector lesson: only hardcode
 * vectors you can authoritatively confirm; use donor self-consistency for
 * the rest).
 *
 * Block-size asymmetry under test (R-Tier4-snefru-blocksize):
 *   - Snefru-128: data_block_size = 64 - 16 = 48 bytes.
 *   - Snefru-256: data_block_size = 64 - 32 = 32 bytes.
 * The self-consistency inputs straddle these boundaries (47/48/49 and
 * 31/32/33 bytes, plus multi-block 95/96/97 and 63/64/65) so the partial-
 * fill + full-block + final-pad paths are all driven.
 *
 * Build (iMac):
 *   cc -O2 -I RHash-master/librhash \
 *      -o /tmp/test_snefru_vectors codegen/tests/test_snefru_vectors.c \
 *      RHash-master/librhash/snefru.c \
 *      RHash-master/librhash/byte_order.c
 *
 * Exit status 0 -> all cells PASS.
 * Non-zero    -> at least one cell MISMATCH; STOP and escalate.
 *
 * $Revision: 1.1 $
 * $Log: test_snefru_vectors.c,v $
 * Revision 1.1  2026/05/28 04:13:25  dlr
 * Initial check-in: R15 pre-flight Snefru librhash donor verification, sub-phase 5b.4a.0; 2 published empty-string vectors (SNE128+SNE256) + 228 self-consistency cells (single-shot vs chunked across 48-byte SNE128 and 32-byte SNE256 block boundaries); 230 of 230 PASS confirms donor multi-block + partial-fill paths byte-exact for the GPU port oracle.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "snefru.h"

static void hexify(const unsigned char *bin, size_t n, char *out) {
    static const char H[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2*i]   = H[(bin[i] >> 4) & 0xF];
        out[2*i+1] = H[bin[i] & 0xF];
    }
    out[2*n] = '\0';
}

/* Single-shot Snefru via the librhash donor. is256 selects width. */
static void snefru_oneshot(int is256, const unsigned char *msg, size_t n,
                           unsigned char *out) {
    snefru_ctx ctx;
    if (is256) rhash_snefru256_init(&ctx);
    else       rhash_snefru128_init(&ctx);
    rhash_snefru_update(&ctx, msg, n);
    rhash_snefru_final(&ctx, out);
}

/* Split (chunked) Snefru: feed the message in `chunk`-byte pieces to
 * exercise the partial-block / multi-update path of rhash_snefru_update. */
static void snefru_chunked(int is256, const unsigned char *msg, size_t n,
                           size_t chunk, unsigned char *out) {
    snefru_ctx ctx;
    if (is256) rhash_snefru256_init(&ctx);
    else       rhash_snefru128_init(&ctx);
    size_t off = 0;
    while (off < n) {
        size_t take = (n - off < chunk) ? (n - off) : chunk;
        rhash_snefru_update(&ctx, msg + off, take);
        off += take;
    }
    rhash_snefru_final(&ctx, out);
}

struct pubvec {
    int is256;
    const char *label;
    const unsigned char *msg;
    size_t msg_len;
    const char *expected_hex;   /* NULL = self-consistency only */
};

/* Authoritatively-confirmable published vectors. The empty-string digests
 * for the standard hardened 8-pass Snefru are the rhash test-suite values
 * (RHash-master/tests). Non-empty published vectors are only included if
 * confidently confirmable; otherwise self-consistency covers them. */
static const unsigned char v_empty[]  = "";
static const unsigned char v_t1234[]  = "1234567890";
static const unsigned char v_test[]   = "Test";

static const struct pubvec pubvectors[] = {
    /* Snefru-256 empty string (rhash canonical). */
    { 1, "SNE256 empty", v_empty, 0,
      "8617f366566a011837f4fb4ba5bedea2b892f3ed8b894023d16ae344b2be5881" },
    /* Snefru-128 empty string (rhash canonical, first 16 bytes of the
     * 128-bit digest). */
    { 0, "SNE128 empty", v_empty, 0,
      "8617f366566a011837f4fb4ba5bedea2" },
    /* The following are self-consistency-only (expected NULL): the donor
     * IS the live oracle, so split-vs-single agreement across the block
     * boundary is the meaningful canary for the GPU port. */
    { 1, "SNE256 '1234567890' (self)", v_t1234, 10, NULL },
    { 0, "SNE128 '1234567890' (self)", v_t1234, 10, NULL },
    { 1, "SNE256 'Test' (self)",       v_test,  4,  NULL },
    { 0, "SNE128 'Test' (self)",       v_test,  4,  NULL },
    { 0, NULL, NULL, 0, NULL }
};

int main(void) {
    int fails = 0;
    int total = 0;
    char hex[129];
    char hex2[129];
    unsigned char dig[32];
    unsigned char dig2[32];

    printf("Snefru librhash donor pre-flight (R15) -- sub-phase 5b.4a.0\n");
    printf("================================================================\n");

    /* Part 1: published-vector + self-consistency cells. */
    for (size_t i = 0; pubvectors[i].label != NULL; i++) {
        const struct pubvec *v = &pubvectors[i];
        int dbytes = v->is256 ? 32 : 16;
        snefru_oneshot(v->is256, v->msg, v->msg_len, dig);
        hexify(dig, dbytes, hex);
        total++;
        if (v->expected_hex) {
            if (strcmp(hex, v->expected_hex) == 0) {
                printf("  %-34s PASS\n", v->label);
            } else {
                printf("  %-34s FAIL\n", v->label);
                printf("    got:      %s\n", hex);
                printf("    expected: %s\n", v->expected_hex);
                fails++;
            }
        } else {
            /* Self-consistency: single-shot vs 1-byte chunked. */
            snefru_chunked(v->is256, v->msg, v->msg_len, 1, dig2);
            hexify(dig2, dbytes, hex2);
            if (strcmp(hex, hex2) == 0) {
                printf("  %-34s PASS  (%s)\n", v->label, hex);
            } else {
                printf("  %-34s FAIL  single != chunked\n", v->label);
                printf("    oneshot: %s\n", hex);
                printf("    chunked: %s\n", hex2);
                fails++;
            }
        }
    }

    /* Part 2: block-boundary self-consistency stress. Build a deterministic
     * pseudo-random message and verify single-shot == chunked at several
     * chunk sizes for lengths straddling the 48-byte (SNE128) and 32-byte
     * (SNE256) data-block boundaries, plus multi-block lengths. */
    {
        static const size_t lens[] = {
            31, 32, 33, 47, 48, 49, 63, 64, 65,
            95, 96, 97, 127, 128, 129, 200
        };
        static const size_t chunks[] = { 1, 7, 16, 17, 31, 32, 48 };
        unsigned char *buf = (unsigned char *)malloc(256);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
        for (int i = 0; i < 256; i++) buf[i] = (unsigned char)((i * 37 + 11) & 0xff);

        for (int w = 0; w < 2; w++) {       /* w=0 SNE128, w=1 SNE256 */
            int is256 = w;
            int dbytes = is256 ? 32 : 16;
            for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++) {
                size_t n = lens[li];
                snefru_oneshot(is256, buf, n, dig);
                hexify(dig, dbytes, hex);
                for (size_t ci = 0; ci < sizeof(chunks)/sizeof(chunks[0]); ci++) {
                    snefru_chunked(is256, buf, n, chunks[ci], dig2);
                    hexify(dig2, dbytes, hex2);
                    total++;
                    if (strcmp(hex, hex2) != 0) {
                        printf("  SNE%-3d len=%-3zu chunk=%-2zu FAIL\n",
                               is256 ? 256 : 128, n, chunks[ci]);
                        printf("    oneshot: %s\n", hex);
                        printf("    chunked: %s\n", hex2);
                        fails++;
                    }
                }
            }
        }
        printf("  block-boundary self-consistency: %d cells driven\n",
               (int)(2 * (sizeof(lens)/sizeof(lens[0])) *
                         (sizeof(chunks)/sizeof(chunks[0]))));
        free(buf);
    }

    printf("================================================================\n");
    printf("Total cells: %d   Failures: %d\n", total, fails);
    if (fails == 0) {
        printf("R15 PRE-FLIGHT PASS -- librhash Snefru donor self-consistent "
               "+ published vectors match.\n");
        return 0;
    }
    printf("R15 PRE-FLIGHT FAIL -- STOP and escalate.\n");
    return 1;
}
