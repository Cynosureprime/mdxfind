/*
 * test_rmd128_bosselaers.c - Bosselaers RIPEMD-128 test-vector regression
 *
 * Standalone single-source-file harness that links against the in-tree
 * rmd128.c and verifies the 8 canonical Bosselaers test vectors. Added
 * 2026-05-27 as a permanent regression to detect any future regression
 * of the long-standing RIPEMD-128 length-encoding bug fixed in
 * rmd128.c rev 1.1 (the bug encoded post-loop residual bytes instead
 * of total message length, breaking digests of any input > 63 bytes).
 *
 * The crucial vectors that exercise the previously-broken multi-block
 * path are the 80-byte "1234567890" x8 input and the 1,000,000-byte
 * "a" stress input. Single-block vectors (empty, "a", "abc", etc.) are
 * included for completeness but never exercised the buggy code path.
 *
 * Build (from /Users/dlr/src/mdfind):
 *   cc -O2 -o /tmp/test_rmd128_bosselaers \
 *       codegen/tests/test_rmd128_bosselaers.c rmd128.c -I.
 * Run:
 *   /tmp/test_rmd128_bosselaers
 *
 * Exit code 0 = all 8 vectors PASS; non-zero = at least one FAIL.
 *
 * $Revision: 1.1 $
 * $Log: test_rmd128_bosselaers.c,v $
 * Revision 1.1  2026/05/27 18:50:35  dlr
 * initial check-in sub-phase 5b1b7 Bosselaers RIPEMD-128 test-vector regression harness all 8 canonical vectors verified PASS against in-tree rmd128.c rev 1.1 after standard-conformance length-encoding bug fix the 80-byte 1234567890 x8 and 1M a vectors exercise the previously-broken multi-block path
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern void RIPEMD128(unsigned char *src, int len, unsigned char *dest);

static void hex16(const unsigned char *b, char *out) {
    for (int i = 0; i < 16; i++) sprintf(out + i*2, "%02x", b[i]);
    out[32] = 0;
}

struct tv {
    const char *label;
    const char *msg;
    int repeat;
    const char *expect;
};

int main(void) {
    struct tv vectors[] = {
        {"empty",                "",                                              1,       "cdf26213a150dc3ecb610f18f6b38b46"},
        {"\"a\"",                "a",                                             1,       "86be7afa339d0fc7cfc785e72f578d33"},
        {"\"abc\"",              "abc",                                           1,       "c14a12199c66e4ba84636b0f69144c77"},
        {"\"message digest\"",   "message digest",                                1,       "9e327b3d6e523062afc1132d7df9d1b8"},
        {"a..z (26)",            "abcdefghijklmnopqrstuvwxyz",                    1,       "fd2aa607f71dc8f510714922b371834e"},
        {"A..Z+a..z+0..9 (62)",  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 1, "d1e959eb179c911faea4624c60c5c702"},
        {"8x\"1234567890\" (80)","1234567890",                                    8,       "3f45ef194732c2dbb2c4a2c769795fa3"},
        {"1M \"a\"",             "a",                                             1000000, "4a7f5723f954eba1216c9d8f6320431f"},
        {NULL, NULL, 0, NULL}
    };
    int fails = 0;
    for (int v = 0; vectors[v].label; v++) {
        size_t mlen = strlen(vectors[v].msg);
        size_t tot = mlen * (size_t)vectors[v].repeat;
        unsigned char *buf = malloc(tot ? tot : 1);
        for (int r = 0; r < vectors[v].repeat; r++)
            memcpy(buf + r*mlen, vectors[v].msg, mlen);
        unsigned char digest[16];
        RIPEMD128(buf, (int)tot, digest);
        char got[33];
        hex16(digest, got);
        int ok = strcmp(got, vectors[v].expect) == 0;
        printf("%-30s len=%-8zu got=%s expect=%s %s\n",
               vectors[v].label, tot, got, vectors[v].expect,
               ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        free(buf);
    }
    printf("\n%d FAILS (0 = all Bosselaers RIPEMD-128 vectors PASS)\n", fails);
    return fails ? 1 : 0;
}
