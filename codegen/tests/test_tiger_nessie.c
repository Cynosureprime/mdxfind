/*
 * test_tiger_nessie.c -- R12 pre-flight donor verification for Phase 5b Tier 2
 * Tiger ship (sub-phase 5b.2b.0).
 *
 * Cross-checks two CPU Tiger donors against published NESSIE test vectors:
 *   - sph_tiger from libsph        (D16.1.a -- mdxfind's CPU oracle)
 *   - rhash_tiger from librhash    (D16.1.c -- in-tree secondary reference)
 *
 * Per Phase 5b Tier 2 architect spec R12: any donor that fails ANY published
 * NESSIE vector is escalated and the port is paused until resolved. Both
 * donors must agree byte-for-byte with the canonical NESSIE vectors below.
 *
 * Build (iMac):
 *   cc -O2 -I /opt/local/include -I RHash-master/librhash \
 *      -o /tmp/test_tiger_nessie codegen/tests/test_tiger_nessie.c \
 *      RHash-master/librhash/tiger.c \
 *      RHash-master/librhash/tiger_sbox.c \
 *      RHash-master/librhash/byte_order.c \
 *      libsph.a
 *
 * Exit status 0 -> all vectors PASS on both donors.
 * Non-zero    -> at least one vector MISMATCH; STOP and escalate.
 *
 * $Revision: 1.1 $
 * $Log: test_tiger_nessie.c,v $
 * Revision 1.1  2026/05/27 22:57:46  dlr
 * Initial check-in: R12 pre-flight donor verification, sub-phase 5b.2b.0.
 *
 * Revision 1.1  2026/05/27 23:00:00  dlr
 * Initial check-in: R12 pre-flight donor verification, sub-phase 5b.2b.0.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "sph_tiger.h"
#include "tiger.h"

/* NESSIE-canonical Tiger/192 (3-pass) test vectors.
 * Hex digests are 48 chars (24 bytes) each.
 */
struct vec {
    const char *label;
    const unsigned char *msg;
    size_t msg_len;
    const char *expected_hex;
};

static const unsigned char v_empty[]   = "";
static const unsigned char v_a[]       = "a";
static const unsigned char v_abc[]     = "abc";
static const unsigned char v_msgdig[]  = "message digest";
static const unsigned char v_az[]      = "abcdefghijklmnopqrstuvwxyz";
static const unsigned char v_alnum[]   =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
static const unsigned char v_8x1234[]  =
    "12345678901234567890123456789012345678901234567890"
    "1234567890123456789012345678901234567890";

static const struct vec vectors[] = {
    { "empty (0 bytes)",        v_empty,  0,
      "3293ac630c13f0245f92bbb1766e16167a4e58492dde73f3" },
    { "\"a\"",                  v_a,      1,
      "77befbef2e7ef8ab2ec8f93bf587a7fc613e247f5f247809" },
    { "\"abc\"",                v_abc,    3,
      "2aab1484e8c158f2bfb8c5ff41b57a525129131c957b5f93" },
    { "\"message digest\"",     v_msgdig, 14,
      "d981f8cb78201a950dcf3048751e441c517fca1aa55a29f6" },
    { "a..z (26 bytes)",        v_az,     26,
      "1714a472eee57d30040412bfcc55032a0b11602ff37beee9" },
    { "A..Z+a..z+0..9 (62)",    v_alnum,  62,
      "8dcea680a17583ee502ba38a3c368651890ffbccdc49a8cc" },
    { "8x'1234567890' (80)",    v_8x1234, 80,
      "1c14795529fd9f207a958f84c52f11e887fa0cabdfd91bfd" },
    { NULL, NULL, 0, NULL }
};

static const char *million_a_expected =
    "6db0e2729cbead93d715c6a7d36302e9b3cee0d2bc314b41";

static void hexify(const unsigned char *bin, size_t n, char *out) {
    static const char H[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2*i]   = H[(bin[i] >> 4) & 0xF];
        out[2*i+1] = H[bin[i] & 0xF];
    }
    out[2*n] = '\0';
}

static int compare_hex(const char *label, const char *donor,
                       const char *got, const char *expected) {
    if (strcmp(got, expected) == 0) {
        printf("  %-10s %-32s PASS\n", donor, label);
        return 0;
    }
    printf("  %-10s %-32s FAIL\n", donor, label);
    printf("    got:      %s\n", got);
    printf("    expected: %s\n", expected);
    return 1;
}

static int run_sph(const unsigned char *msg, size_t n,
                   const char *label, const char *expected) {
    sph_tiger_context ctx;
    unsigned char digest[24];
    char hex[49];
    sph_tiger_init(&ctx);
    sph_tiger(&ctx, msg, n);
    sph_tiger_close(&ctx, digest);
    hexify(digest, 24, hex);
    return compare_hex(label, "sph_tiger", hex, expected);
}

static int run_rhash(const unsigned char *msg, size_t n,
                     const char *label, const char *expected) {
    tiger_ctx ctx;
    unsigned char digest[24];
    char hex[49];
    rhash_tiger_init(&ctx);
    rhash_tiger_update(&ctx, msg, n);
    rhash_tiger_final(&ctx, digest);
    hexify(digest, 24, hex);
    return compare_hex(label, "rhash", hex, expected);
}

int main(void) {
    int fails = 0;
    int total = 0;

    printf("Tiger NESSIE donor pre-flight (R12) -- sub-phase 5b.2b.0\n");
    printf("================================================================\n");

    for (size_t i = 0; vectors[i].label != NULL; i++) {
        printf("%s\n", vectors[i].label);
        fails += run_sph(vectors[i].msg, vectors[i].msg_len,
                         vectors[i].label, vectors[i].expected_hex);
        fails += run_rhash(vectors[i].msg, vectors[i].msg_len,
                           vectors[i].label, vectors[i].expected_hex);
        total += 2;
    }

    /* Stress vector: 1,000,000 'a' bytes. */
    {
        size_t n = 1000000;
        unsigned char *buf = (unsigned char *)malloc(n);
        if (!buf) {
            fprintf(stderr, "alloc failed for 1M-a stress vector\n");
            return 2;
        }
        memset(buf, 'a', n);
        printf("1,000,000 x 'a' (stress)\n");
        fails += run_sph(buf, n, "1M-a", million_a_expected);
        fails += run_rhash(buf, n, "1M-a", million_a_expected);
        total += 2;
        free(buf);
    }

    printf("================================================================\n");
    printf("Total cells: %d   Failures: %d\n", total, fails);
    if (fails == 0) {
        printf("R12 PRE-FLIGHT PASS -- both donors standard-conformant.\n");
        return 0;
    }
    printf("R12 PRE-FLIGHT FAIL -- STOP and escalate.\n");
    return 1;
}
