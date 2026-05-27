/*
 * test_rmd160_bosselaers.c - Bosselaers RIPEMD-160 test-vector regression
 *
 * Standalone single-source-file harness that links against the in-tree
 * rmd160.c and verifies the 8 canonical Bosselaers test vectors. Added
 * 2026-05-27 as a permanent regression to detect any future regression
 * of the long-standing RIPEMD-160 length-encoding bug fixed in
 * rmd160.c rev 1.1 (the bug encoded post-loop residual bytes instead
 * of total message length, breaking digests of any input > 63 bytes).
 *
 * The crucial vectors that exercise the previously-broken multi-block
 * path are the 80-byte "1234567890" x8 input and the 1,000,000-byte
 * "a" stress input. Single-block vectors (empty, "a", "abc", etc.) are
 * included for completeness but never exercised the buggy code path.
 *
 * Build (from /Users/dlr/src/mdfind):
 *   cc -O2 -o /tmp/test_rmd160_bosselaers \
 *       codegen/tests/test_rmd160_bosselaers.c rmd160.c -I.
 * Run:
 *   /tmp/test_rmd160_bosselaers
 *
 * Exit code 0 = all 8 vectors PASS; non-zero = at least one FAIL.
 *
 * $Revision: 1.1 $
 * $Log: test_rmd160_bosselaers.c,v $
 * Revision 1.1  2026/05/27 19:17:58  dlr
 * initial check-in sub-phase 5b1c Bosselaers RIPEMD-160 test-vector regression harness all 8 canonical vectors verified PASS against in-tree rmd160.c rev 1.1 after standard-conformance length-encoding bug fix the 80-byte 1234567890 x8 and 1M a vectors exercise the previously-broken multi-block path mirror of test_rmd128_bosselaers.c rev 1.1
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern void RIPEMD160(unsigned char *src, int len, unsigned char *dest);

static void hex20(const unsigned char *b, char *out) {
    for (int i = 0; i < 20; i++) sprintf(out + i*2, "%02x", b[i]);
    out[40] = 0;
}

struct tv {
    const char *label;
    const char *msg;
    int repeat;
    const char *expect;
};

int main(void) {
    struct tv vectors[] = {
        {"empty",                "",                                              1,       "9c1185a5c5e9fc54612808977ee8f548b2258d31"},
        {"\"a\"",                "a",                                             1,       "0bdc9d2d256b3ee9daae347be6f4dc835a467ffe"},
        {"\"abc\"",              "abc",                                           1,       "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"},
        {"\"message digest\"",   "message digest",                                1,       "5d0689ef49d2fae572b881b123a85ffa21595f36"},
        {"a..z (26)",            "abcdefghijklmnopqrstuvwxyz",                    1,       "f71c27109c692c1b56bbdceb5b9d2865b3708dbc"},
        {"A..Z+a..z+0..9 (62)",  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 1, "b0e20b6e3116640286ed3a87a5713079b21f5189"},
        {"8x\"1234567890\" (80)","1234567890",                                    8,       "9b752e45573d4b39f4dbd3323cab82bf63326bfb"},
        {"1M \"a\"",             "a",                                             1000000, "52783243c1697bdbe16d37f97f68f08325dc1528"},
        {NULL, NULL, 0, NULL}
    };
    int fails = 0;
    for (int v = 0; vectors[v].label; v++) {
        size_t mlen = strlen(vectors[v].msg);
        size_t tot = mlen * (size_t)vectors[v].repeat;
        unsigned char *buf = malloc(tot ? tot : 1);
        for (int r = 0; r < vectors[v].repeat; r++)
            memcpy(buf + r*mlen, vectors[v].msg, mlen);
        unsigned char digest[20];
        RIPEMD160(buf, (int)tot, digest);
        char got[41];
        hex20(digest, got);
        int ok = strcmp(got, vectors[v].expect) == 0;
        printf("%-30s len=%-8zu got=%s expect=%s %s\n",
               vectors[v].label, tot, got, vectors[v].expect,
               ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        free(buf);
    }
    printf("\n%d FAILS (0 = all Bosselaers RIPEMD-160 vectors PASS)\n", fails);
    return fails ? 1 : 0;
}
