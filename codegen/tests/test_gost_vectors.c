/*
 * test_gost_vectors.c -- R15 / R12 pre-flight donor verification for Phase 5b
 * Tier 4 GOST ship (sub-phase 5b.4b.0).
 *
 * Cross-checks the in-tree gosthash() donor (gosthash/gosthash.c -- the LIVE
 * oracle for e125 GOSTMD5PASS via gosthash.o, called directly from
 * mdxfind.c:29076) against:
 *   1. Published GOST R 34.11-94 TEST S-box set vectors (Saarinen 1998 /
 *      RFC 4357 test set). These are the canonical values produced by the
 *      sbox[8][16] table at gosthash.c:32-42. The CryptoPro S-box set
 *      (RHASH_GOST_CRYPTOPRO, e14 GOST-CRYPTO -- a DIFFERENT, out-of-scope
 *      job) produces DIFFERENT digests; if gosthash() matched the CryptoPro
 *      values the donor/S-box would be WRONG and we MUST stop
 *      (R-Tier4-gost-sbox, HIGH).
 *   2. rhash RHASH_GOST (the librhash default GOST, which is ALSO the test
 *      S-box set per hashpipe.c:3096 MAKE_RHASH(gost, RHASH_GOST)). Byte-exact
 *      agreement across multi-block inputs proves gosthash() == RHASH_GOST,
 *      the Tier-2 Whirlpool (OpenSSL vs librhash) byte-equivalence precedent
 *      applied to GOST.
 *
 * If gosthash() matches the published TEST-set vectors AND agrees with
 * RHASH_GOST across all inputs: proceed to the C-mirror pre-port (5b.4b.1).
 * If it matches CryptoPro values or fails: STOP and escalate (wrong donor or
 * wrong S-box).
 *
 * Build (iMac):
 *   cc -O2 -I gosthash -I RHash-master/librhash \
 *      -o /tmp/test_gost_vectors codegen/tests/test_gost_vectors.c \
 *      gosthash/gosthash.c \
 *      RHash-master/librhash/gost.c
 *
 * Exit status 0 -> all cells PASS.
 * Non-zero    -> at least one cell MISMATCH; STOP and escalate.
 *
 * $Revision: 1.1 $
 * $Log: test_gost_vectors.c,v $
 * Revision 1.1  2026/05/28 04:39:13  dlr
 * Initial revision
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gosthash.h"
#include "gost.h"

extern void gosthash(unsigned char *src, int len, unsigned char *digest);

static void hexify(const unsigned char *bin, size_t n, char *out) {
    static const char H[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2*i]   = H[(bin[i] >> 4) & 0xF];
        out[2*i+1] = H[bin[i] & 0xF];
    }
    out[2*n] = '\0';
}

/* The in-tree donor (the production CPU oracle for e125). */
static void gost_intree(const unsigned char *msg, size_t n, unsigned char *out) {
    gosthash((unsigned char *)msg, (int)n, out);
}

/* The rhash default GOST (= TEST S-box set). Used for cross-validation only. */
static void gost_rhash(const unsigned char *msg, size_t n, unsigned char *out) {
    gost_ctx ctx;
    rhash_gost_init(&ctx);
    rhash_gost_update(&ctx, msg, n);
    rhash_gost_final(&ctx, out);
}

/* The rhash CryptoPro GOST -- the WRONG set for e125. We compute it ONLY to
 * prove gosthash() does NOT match it (the negative control for
 * R-Tier4-gost-sbox). */
static void gost_rhash_cryptopro(const unsigned char *msg, size_t n,
                                 unsigned char *out) {
    gost_ctx ctx;
    rhash_gost_cryptopro_init(&ctx);
    rhash_gost_update(&ctx, msg, n);
    rhash_gost_final(&ctx, out);
}

struct pubvec {
    const char *label;
    const char *msg;       /* NUL-terminated ASCII */
    const char *expected_hex;
};

/* Canonical GOST R 34.11-94 TEST S-box set vectors (Saarinen reference impl /
 * the values produced by gosthash.c's sbox[8][16]). These match the values
 * supplied in the sub-phase brief. */
static const struct pubvec pubvectors[] = {
    { "empty string",   "",
      "ce85b99cc46752fffee35cab9a7b0278abb4c2d2055cff685af4912c49490f8d" },
    { "\"a\"",          "a",
      "d42c539e367c66e9c88a801f6649349c21871b4344c6a573f849fdce62f314dd" },
    { "\"abc\"",        "abc",
      "f3134348c44fb1b2a277729e2285ebb5cb5e0f29c975bc753b70497c06a4d51d" },
    { "\"message digest\"", "message digest",
      "ad4434ecb18f2c99b60cbe59ec3d2469582b65273f48de72db2fde16a4889a4d" },
    { NULL, NULL, NULL }
};

int main(void) {
    int fails = 0;
    int total = 0;
    int cryptopro_collisions = 0;
    char hex[65], hex2[65], hexc[65];
    unsigned char dig[32], dig2[32], digc[32];

    /* The in-tree donor precomputes its 4 derived S-box tables in
     * gosthash_init(); call it once (mdxfind.c:45885 does the same). */
    gosthash_init();

    printf("GOST gosthash() donor pre-flight (R15/R12) -- sub-phase 5b.4b.0\n");
    printf("================================================================\n");

    /* Part 1: published TEST-set vectors via gosthash() (the live donor). */
    printf("-- Part 1: published GOST R 34.11-94 TEST-set vectors --\n");
    for (size_t i = 0; pubvectors[i].label != NULL; i++) {
        const struct pubvec *v = &pubvectors[i];
        size_t n = strlen(v->msg);
        gost_intree((const unsigned char *)v->msg, n, dig);
        hexify(dig, 32, hex);
        total++;
        if (strcmp(hex, v->expected_hex) == 0) {
            printf("  %-22s PASS  %s\n", v->label, hex);
        } else {
            printf("  %-22s FAIL\n", v->label);
            printf("    got:      %s\n", hex);
            printf("    expected: %s\n", v->expected_hex);
            fails++;
        }
        /* Negative control: prove gosthash() != CryptoPro for this input. */
        gost_rhash_cryptopro((const unsigned char *)v->msg, n, digc);
        hexify(digc, 32, hexc);
        if (strcmp(hex, hexc) == 0) {
            printf("    !! gosthash() MATCHES CryptoPro for %s -- WRONG S-box!\n",
                   v->label);
            cryptopro_collisions++;
        }
    }

    /* Part 2: gosthash() == rhash RHASH_GOST (both test set) across a range of
     * multi-block lengths straddling the 32-byte GOST block boundary. */
    printf("-- Part 2: gosthash() == rhash RHASH_GOST (multi-block) --\n");
    {
        static const size_t lens[] = {
            0, 1, 2, 7, 16, 31, 32, 33, 47, 48, 49, 63, 64, 65,
            95, 96, 97, 127, 128, 129, 200, 255
        };
        unsigned char *buf = (unsigned char *)malloc(256);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
        for (int i = 0; i < 256; i++)
            buf[i] = (unsigned char)((i * 37 + 11) & 0xff);

        for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++) {
            size_t n = lens[li];
            gost_intree(buf, n, dig);
            gost_rhash(buf, n, dig2);
            hexify(dig, 32, hex);
            hexify(dig2, 32, hex2);
            total++;
            if (strcmp(hex, hex2) != 0) {
                printf("  len=%-3zu FAIL  gosthash != RHASH_GOST\n", n);
                printf("    gosthash:   %s\n", hex);
                printf("    RHASH_GOST: %s\n", hex2);
                fails++;
            }
            /* Negative control across the full length sweep: gosthash() must
             * NOT equal CryptoPro. */
            gost_rhash_cryptopro(buf, n, digc);
            hexify(digc, 32, hexc);
            if (strcmp(hex, hexc) == 0) {
                printf("  len=%-3zu !! gosthash() MATCHES CryptoPro -- WRONG!\n",
                       n);
                cryptopro_collisions++;
            }
        }
        printf("  multi-block cross-check: %d lengths driven\n",
               (int)(sizeof(lens)/sizeof(lens[0])));
        free(buf);
    }

    printf("================================================================\n");
    printf("Total cells: %d   Failures: %d   CryptoPro-collisions: %d\n",
           total, fails, cryptopro_collisions);
    if (cryptopro_collisions > 0) {
        printf("R15 PRE-FLIGHT FAIL -- gosthash() matches the CryptoPro S-box "
               "set; e125 needs the TEST set. STOP and escalate.\n");
        return 1;
    }
    if (fails == 0) {
        printf("R15 PRE-FLIGHT PASS -- gosthash() matches the GOST R 34.11-94 "
               "TEST-set vectors and rhash RHASH_GOST byte-for-byte; distinct "
               "from CryptoPro. Donor confirmed for the GPU port oracle.\n");
        return 0;
    }
    printf("R15 PRE-FLIGHT FAIL -- STOP and escalate.\n");
    return 1;
}
