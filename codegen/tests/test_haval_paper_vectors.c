/*
 * test_haval_paper_vectors.c -- R15 pre-flight donor verification for
 * Phase 5b Tier 3 HAVAL ship (sub-phase 5b.3a.0).
 *
 * Verifies the sph_haval donor (libsph; mdxfind's live CPU oracle) for
 * all 15 variants (5 digest widths x 3 pass counts) in two layers:
 *
 *   Layer 1 -- 15 published HAVAL paper-canonical empty-input vectors
 *              (HAVAL(empty) for every (passes, bits) pair). These are
 *              the most-cited published HAVAL vectors and have been
 *              independently reproduced across multiple HAVAL reference
 *              implementations. If sph_haval matches all 15, its (passes,
 *              digest_bits) parameter substitution is correct.
 *
 *   Layer 2 -- self-consistency. For each variant: split-update vs
 *              single-update equivalence on the 6 sample inputs. This
 *              verifies the donor's state machine is internally
 *              consistent across multi-block inputs without needing
 *              ground-truth expected hex per input. Any internal donor
 *              bug surfaces here.
 *
 * The Layer 1 empty-input vectors come from the HAVAL paper (Zheng-
 * Pieprzyk-Seberry 1993) and are reproduced verbatim in the haval-test.c
 * reference distribution shipped with the HAVAL public-domain code.
 *
 * R15 + R13 pre-flight: a donor that passes Layer 1 has correct
 * variant-parameter encoding (the block[118..119] substitution) and
 * correct per-width digest-fold algebra. A donor that passes Layer 2
 * has a self-consistent state machine. Together these establish trust
 * for the GPU port's CPU oracle.
 *
 * Build (iMac):
 *   cc -O2 -I /opt/local/include -o /tmp/test_haval_paper_vectors \
 *      codegen/tests/test_haval_paper_vectors.c libsph.a
 *
 * Exit status 0 -> all layers PASS on sph_haval; donor trustworthy.
 * Non-zero    -> at least one MISMATCH; STOP and escalate.
 *
 * $Revision: 1.1 $
 * $Log: test_haval_paper_vectors.c,v $
 * Revision 1.1  2026/05/28 02:03:40  dlr
 * R15 pre-flight donor verification for Phase 5b Tier 3 HAVAL sub-phase 5b3a0 cross-checks sph_haval donor libsph mdxfind live CPU oracle against 15 published empty-input HAVAL paper-canonical vectors all 5 widths x 3 pass counts plus 105 self-consistency split-update vs single-update cells across multi-block boundary inputs 120 of 120 cells PASS confirmed 2026-05-27 unblocks 5b.3a port
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "sph_haval.h"

/* Layer 1: published empty-input HAVAL vectors, all 15 variants.
 * Source: HAVAL reference distribution haval-test.c expected output
 * for the empty-string vector (-string="" test mode). Cross-verified
 * against cryptopp HAVAL test vectors and the HAVAL paper itself.
 */
struct empty_vec {
    int         passes;
    int         bits;
    const char *expected;
};

static const struct empty_vec empty_vectors[] = {
    { 3, 128, "c68f39913f901f3ddf44c707357a7d70" },
    { 3, 160, "d353c3ae22a25401d257643836d7231a9a95f953" },
    { 3, 192, "e9c48d7903eaf2a91c5b350151efcb175c0fc82de2289a4e" },
    { 3, 224, "c5aae9d47bffcaaf84a8c6e7ccacd60a"
              "0dd1932be7b1a192b9214b6d" },
    { 3, 256, "4f6938531f0bc8991f62da7bbd6f7de3"
              "fad44562b8c6f4ebf146d5b4e46f7c17" },
    { 4, 128, "ee6bbf4d6a46a679b3a856c88538bb98" },
    { 4, 160, "1d33aae1be4146dbaaca0b6e70d7a11f10801525" },
    { 4, 192, "4a8372945afa55c7dead800311272523ca19d42ea47b72da" },
    { 4, 224, "3e56243275b3b81561750550e36fcd67"
              "6ad2f5dd9e15f2e89e6ed78e" },
    { 4, 256, "c92b2e23091e80e375dadce26982482d"
              "197b1a2521be82da819f8ca2c579b99b" },
    { 5, 128, "184b8482a0c050dca54b59c7f05bf5dd" },
    { 5, 160, "255158cfc1eed1a7be7c55ddd64d9790415b933b" },
    { 5, 192, "4839d0626f95935e17ee2fc4509387bbe2cc46cb382ffe85" },
    { 5, 224, "4a0513c032754f5582a758d35917ac9a"
              "df3854219b39e3ac77d1837e" },
    { 5, 256, "be417bb4dd5cfb76c7126f4f8eeb1553"
              "a449039307b1a3cd451dbfdc0fbbe330" },
};

#define N_EMPTY (sizeof(empty_vectors)/sizeof(empty_vectors[0]))

/* Layer 2: self-consistency inputs for split-update vs single-update. */
struct sc_input {
    const char *data;
    size_t      len;
};

static const struct sc_input sc_inputs[] = {
    { "a", 1 },
    { "HAVAL", 5 },
    { "0123456789", 10 },
    { "abcdefghijklmnopqrstuvwxyz", 26 },
    { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
      62 },
    /* 128-byte boundary input (exactly one HAVAL block). */
    { "0123456789abcdef0123456789abcdef"
      "0123456789abcdef0123456789abcdef"
      "0123456789abcdef0123456789abcdef"
      "0123456789abcdef0123456789abcdef", 128 },
    /* 200-byte input (spans two blocks; tail < 128). */
    { "abcdefghijklmnopqrstuvwxyz" "0123456789"
      "abcdefghijklmnopqrstuvwxyz" "0123456789"
      "abcdefghijklmnopqrstuvwxyz" "0123456789"
      "abcdefghijklmnopqrstuvwxyz" "0123456789"
      "abcdefghijklmnopqrstuvwxyz" "0123456789"
      "abcdefghijklmnop", 200 },
};

#define N_SC_INPUTS (sizeof(sc_inputs)/sizeof(sc_inputs[0]))

struct variant {
    int   passes;
    int   bits;
    void (*init_fn)(void *);
    void (*update_fn)(void *, const void *, size_t);
    void (*close_fn)(void *, void *);
};

#define VARIANT_ROW(P, BITS) \
    { P, BITS, \
      sph_haval##BITS##_##P##_init, \
      sph_haval##BITS##_##P, \
      sph_haval##BITS##_##P##_close }

static const struct variant variants[] = {
    VARIANT_ROW(3, 128),
    VARIANT_ROW(3, 160),
    VARIANT_ROW(3, 192),
    VARIANT_ROW(3, 224),
    VARIANT_ROW(3, 256),
    VARIANT_ROW(4, 128),
    VARIANT_ROW(4, 160),
    VARIANT_ROW(4, 192),
    VARIANT_ROW(4, 224),
    VARIANT_ROW(4, 256),
    VARIANT_ROW(5, 128),
    VARIANT_ROW(5, 160),
    VARIANT_ROW(5, 192),
    VARIANT_ROW(5, 224),
    VARIANT_ROW(5, 256),
};

#define N_VARIANTS (sizeof(variants)/sizeof(variants[0]))

static void hexify(const unsigned char *bin, size_t n, char *out)
{
    static const char H[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2*i]   = H[(bin[i] >> 4) & 0xF];
        out[2*i+1] = H[bin[i] & 0xF];
    }
    out[2*n] = '\0';
}

static const struct variant *find_variant(int passes, int bits)
{
    for (size_t i = 0; i < N_VARIANTS; i++) {
        if (variants[i].passes == passes && variants[i].bits == bits)
            return &variants[i];
    }
    return NULL;
}

int main(void)
{
    int fails = 0;
    int total = 0;

    printf("HAVAL paper-vectors + self-consistency donor pre-flight\n");
    printf("(R15) -- sub-phase 5b.3a.0\n");
    printf("================================================================\n");

    /* Layer 1: published empty-input vectors. */
    printf("\nLayer 1: published empty-input HAVAL vectors (15 of 15)\n");
    printf("----------------------------------------------------------------\n");
    for (size_t v = 0; v < N_EMPTY; v++) {
        const struct empty_vec *ev = &empty_vectors[v];
        const struct variant *V = find_variant(ev->passes, ev->bits);
        if (!V) {
            printf("  HAVAL-%d/%d: NO VARIANT (test bug)\n",
                   ev->bits, ev->passes);
            fails++;
            total++;
            continue;
        }
        sph_haval_context ctx;
        unsigned char digest[32];
        char hex[65];
        int bytes = ev->bits / 8;

        V->init_fn(&ctx);
        V->update_fn(&ctx, "", 0);
        V->close_fn(&ctx, digest);
        hexify(digest, bytes, hex);
        total++;
        if (strcmp(hex, ev->expected) == 0) {
            printf("  HAVAL-%d/%d(empty): PASS\n", ev->bits, ev->passes);
        } else {
            printf("  HAVAL-%d/%d(empty): FAIL\n", ev->bits, ev->passes);
            printf("    got:      %s\n", hex);
            printf("    expected: %s\n", ev->expected);
            fails++;
        }
    }

    /* Layer 2: split-update vs single-update self-consistency. */
    printf("\nLayer 2: split-update vs single-update consistency\n");
    printf("(15 variants x %zu inputs = %zu cells)\n",
           N_SC_INPUTS, N_VARIANTS * N_SC_INPUTS);
    printf("----------------------------------------------------------------\n");
    for (size_t v = 0; v < N_VARIANTS; v++) {
        const struct variant *V = &variants[v];
        int bytes = V->bits / 8;
        int variant_fails = 0;
        for (size_t i = 0; i < N_SC_INPUTS; i++) {
            sph_haval_context ctx_one;
            sph_haval_context ctx_split;
            unsigned char d_one[32], d_split[32];
            char hex_one[65], hex_split[65];

            /* Single-shot update. */
            V->init_fn(&ctx_one);
            V->update_fn(&ctx_one, sc_inputs[i].data, sc_inputs[i].len);
            V->close_fn(&ctx_one, d_one);

            /* Split update: byte-by-byte for first 3 bytes, then bulk
             * for the rest. Exercises both the buffered and the bulk
             * paths of havalUpdate(). */
            V->init_fn(&ctx_split);
            size_t L = sc_inputs[i].len;
            size_t first = (L >= 3) ? 3 : L;
            for (size_t k = 0; k < first; k++) {
                V->update_fn(&ctx_split,
                             sc_inputs[i].data + k, 1);
            }
            if (L > first) {
                V->update_fn(&ctx_split,
                             sc_inputs[i].data + first, L - first);
            }
            V->close_fn(&ctx_split, d_split);

            hexify(d_one,   bytes, hex_one);
            hexify(d_split, bytes, hex_split);
            total++;
            if (strcmp(hex_one, hex_split) == 0) {
                /* Silent pass to keep output readable. */
            } else {
                printf("  HAVAL-%d/%d in[%zu] (L=%zu): SPLIT FAIL\n",
                       V->bits, V->passes, i, sc_inputs[i].len);
                printf("    one:   %s\n", hex_one);
                printf("    split: %s\n", hex_split);
                fails++;
                variant_fails++;
            }
        }
        if (variant_fails == 0) {
            printf("  HAVAL-%d/%d: %zu of %zu split-update cells PASS\n",
                   V->bits, V->passes, N_SC_INPUTS, N_SC_INPUTS);
        } else {
            printf("  HAVAL-%d/%d: %d of %zu split-update cells FAIL\n",
                   V->bits, V->passes, variant_fails, N_SC_INPUTS);
        }
    }

    printf("\n================================================================\n");
    printf("Total cells: %d   Failures: %d\n", total, fails);
    if (fails == 0) {
        printf("R15 PRE-FLIGHT PASS -- sph_haval is standard-conformant on\n");
        printf("Layer 1 empty-input vectors (15/15) AND self-consistent on\n");
        printf("Layer 2 multi-block boundary tests. Donor trustworthy for\n");
        printf("the GPU port's CPU oracle.\n");
        return 0;
    }
    printf("R15 PRE-FLIGHT FAIL -- STOP and escalate. If Layer 1 fails the\n");
    printf("variant-parameter substitution or digest fold is wrong; if\n");
    printf("Layer 2 fails the donor state machine has a bug.\n");
    return 1;
}
