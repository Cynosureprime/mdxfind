/*
 * SHA-NI hardware-accelerated SHA1 compression function.
 * Uses Intel SHA extensions (available on AMD Zen+, Intel Ice Lake+).
 * Compile with: cc -O3 -msha -msse4.1 -c sha1_shani.c
 */

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

void sha1_compress_shani(uint32_t *hash, const uint32_t *block)
{
	__m128i abcd, abcd_save, e0, e0_save, e1;
	__m128i msg0, msg1, msg2, msg3;
	__m128i shuf_mask;

	shuf_mask = _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL);

	/* Load hash: abcd = {d,c,b,a}, e0 = {e,0,0,0} */
	abcd = _mm_loadu_si128((const __m128i *)hash);
	e0   = _mm_set_epi32(hash[4], 0, 0, 0);
	abcd = _mm_shuffle_epi32(abcd, 0x1B); /* reverse to {a,b,c,d} */

	abcd_save = abcd;
	e0_save   = e0;

	/* Load and byte-swap message */
	msg0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block + 0)),  shuf_mask);
	msg1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block + 4)),  shuf_mask);
	msg2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block + 8)),  shuf_mask);
	msg3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block + 12)), shuf_mask);

	/* Rounds 0-3 */
	e0   = _mm_add_epi32(e0, msg0);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);
	e0   = _mm_sha1nexte_epu32(e1, msg1);
	msg0 = _mm_sha1msg1_epu32(msg0, msg1);

	/* Rounds 4-7 */
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);
	e0   = _mm_sha1nexte_epu32(e1, msg2);
	msg1 = _mm_sha1msg1_epu32(msg1, msg2);
	msg0 = _mm_xor_si128(msg0, msg2);

	/* Rounds 8-11 */
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);
	e0   = _mm_sha1nexte_epu32(e1, msg3);
	msg2 = _mm_sha1msg1_epu32(msg2, msg3);
	msg1 = _mm_xor_si128(msg1, msg3);

	/* Rounds 12-15 */
	msg0 = _mm_sha1msg2_epu32(msg0, msg3);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);
	e0   = _mm_sha1nexte_epu32(e1, msg0);
	msg3 = _mm_sha1msg1_epu32(msg3, msg0);
	msg2 = _mm_xor_si128(msg2, msg0);

	/* Rounds 16-19 */
	msg1 = _mm_sha1msg2_epu32(msg1, msg0);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);
	e0   = _mm_sha1nexte_epu32(e1, msg1);
	msg0 = _mm_sha1msg1_epu32(msg0, msg1);
	msg3 = _mm_xor_si128(msg3, msg1);

	/* Rounds 20-23 */
	msg2 = _mm_sha1msg2_epu32(msg2, msg1);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 1);
	e0   = _mm_sha1nexte_epu32(e1, msg2);
	msg1 = _mm_sha1msg1_epu32(msg1, msg2);
	msg0 = _mm_xor_si128(msg0, msg2);

	/* Rounds 24-27 */
	msg3 = _mm_sha1msg2_epu32(msg3, msg2);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 1);
	e0   = _mm_sha1nexte_epu32(e1, msg3);
	msg2 = _mm_sha1msg1_epu32(msg2, msg3);
	msg1 = _mm_xor_si128(msg1, msg3);

	/* Rounds 28-31 */
	msg0 = _mm_sha1msg2_epu32(msg0, msg3);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 1);
	e0   = _mm_sha1nexte_epu32(e1, msg0);
	msg3 = _mm_sha1msg1_epu32(msg3, msg0);
	msg2 = _mm_xor_si128(msg2, msg0);

	/* Rounds 32-35 */
	msg1 = _mm_sha1msg2_epu32(msg1, msg0);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 1);
	e0   = _mm_sha1nexte_epu32(e1, msg1);
	msg0 = _mm_sha1msg1_epu32(msg0, msg1);
	msg3 = _mm_xor_si128(msg3, msg1);

	/* Rounds 36-39 */
	msg2 = _mm_sha1msg2_epu32(msg2, msg1);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 1);
	e0   = _mm_sha1nexte_epu32(e1, msg2);
	msg1 = _mm_sha1msg1_epu32(msg1, msg2);
	msg0 = _mm_xor_si128(msg0, msg2);

	/* Rounds 40-43 */
	msg3 = _mm_sha1msg2_epu32(msg3, msg2);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 2);
	e0   = _mm_sha1nexte_epu32(e1, msg3);
	msg2 = _mm_sha1msg1_epu32(msg2, msg3);
	msg1 = _mm_xor_si128(msg1, msg3);

	/* Rounds 44-47 */
	msg0 = _mm_sha1msg2_epu32(msg0, msg3);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 2);
	e0   = _mm_sha1nexte_epu32(e1, msg0);
	msg3 = _mm_sha1msg1_epu32(msg3, msg0);
	msg2 = _mm_xor_si128(msg2, msg0);

	/* Rounds 48-51 */
	msg1 = _mm_sha1msg2_epu32(msg1, msg0);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 2);
	e0   = _mm_sha1nexte_epu32(e1, msg1);
	msg0 = _mm_sha1msg1_epu32(msg0, msg1);
	msg3 = _mm_xor_si128(msg3, msg1);

	/* Rounds 52-55 */
	msg2 = _mm_sha1msg2_epu32(msg2, msg1);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 2);
	e0   = _mm_sha1nexte_epu32(e1, msg2);
	msg1 = _mm_sha1msg1_epu32(msg1, msg2);
	msg0 = _mm_xor_si128(msg0, msg2);

	/* Rounds 56-59 */
	msg3 = _mm_sha1msg2_epu32(msg3, msg2);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 2);
	e0   = _mm_sha1nexte_epu32(e1, msg3);
	msg2 = _mm_sha1msg1_epu32(msg2, msg3);
	msg1 = _mm_xor_si128(msg1, msg3);

	/* Rounds 60-63 */
	msg0 = _mm_sha1msg2_epu32(msg0, msg3);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 3);
	e0   = _mm_sha1nexte_epu32(e1, msg0);
	msg3 = _mm_sha1msg1_epu32(msg3, msg0);
	msg2 = _mm_xor_si128(msg2, msg0);

	/* Rounds 64-67 */
	msg1 = _mm_sha1msg2_epu32(msg1, msg0);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 3);
	e0   = _mm_sha1nexte_epu32(e1, msg1);
	msg3 = _mm_xor_si128(msg3, msg1);

	/* Rounds 68-71 */
	msg2 = _mm_sha1msg2_epu32(msg2, msg1);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 3);
	e0   = _mm_sha1nexte_epu32(e1, msg2);

	/* Rounds 72-75 */
	msg3 = _mm_sha1msg2_epu32(msg3, msg2);
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 3);
	e0   = _mm_sha1nexte_epu32(e1, msg3);

	/* Rounds 76-79 */
	e1   = abcd;
	abcd = _mm_sha1rnds4_epu32(abcd, e0, 3);
	e0   = _mm_sha1nexte_epu32(e1, _mm_setzero_si128());

	/* Add saved state */
	abcd = _mm_add_epi32(abcd, abcd_save);
	e0   = _mm_add_epi32(e0, e0_save);

	/* Store result */
	abcd = _mm_shuffle_epi32(abcd, 0x1B);
	_mm_storeu_si128((__m128i *)hash, abcd);
	hash[4] = _mm_extract_epi32(e0, 3);
}

/*
 * SHA-NI hardware-accelerated SHA-256 block compression.
 *
 * MULTI-BLOCK by design: the state load/shuffle/store bracket is done once
 * per run rather than once per 64 bytes. e1000 (7zAES) streams 196,608
 * blocks per candidate, so a per-block entry point would pay that bracket
 * 196,608 times.
 *
 * Convention matches sha256_compress_armce() and the portable
 * mysha256_compress_c(): `state` holds the eight SHA-256 words in NATIVE
 * order on entry and exit; `data` is the raw big-endian message. The
 * big-endian serialisation of the digest happens once, in mysha256_end().
 *
 * GATING: this function contains SHA-NI and SSE4.1 instructions and MUST
 * NOT be called unless CPUID.(EAX=7,ECX=0):EBX[29] is set. mymd5.c holds
 * that gate; see mysha256_cpu_detect(). This whole file is x86-only and is
 * linked only on x86 hosts -- ARM uses sha_armce.o, PowerPC and SPARC use
 * the portable C path -- so nothing here is reachable on those targets.
 */
void sha256_blocks_shani(uint32_t *state, const void *data, size_t nblocks)
{
	__m128i STATE0, STATE1, ABEF_SAVE, CDGH_SAVE;
	__m128i MSG, TMP, MSG0, MSG1, MSG2, MSG3;
	const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL,
	                                    0x0405060700010203ULL);
	const unsigned char *p = (const unsigned char *)data;

	TMP    = _mm_loadu_si128((const __m128i *)&state[0]);   /* a b c d */
	STATE1 = _mm_loadu_si128((const __m128i *)&state[4]);   /* e f g h */

	TMP    = _mm_shuffle_epi32(TMP, 0xB1);                  /* CDAB */
	STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);               /* EFGH */
	STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);               /* ABEF */
	STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0);            /* CDGH */

	for (; nblocks; nblocks--, p += 64) {
		ABEF_SAVE = STATE0;
		CDGH_SAVE = STATE1;

		/* Rounds 0-3 */
		MSG0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 0)), MASK);
		MSG  = _mm_add_epi32(MSG0, _mm_set_epi64x(0xE9B5DBA5B5C0FBCFULL, 0x71374491428A2F98ULL));
		STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
		MSG  = _mm_shuffle_epi32(MSG, 0x0E);
		STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

		/* Rounds 4-7 */
		MSG1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 16)), MASK);
		MSG  = _mm_add_epi32(MSG1, _mm_set_epi64x(0xAB1C5ED5923F82A4ULL, 0x59F111F13956C25BULL));
		STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
		MSG  = _mm_shuffle_epi32(MSG, 0x0E);
		STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
		MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

		/* Rounds 8-11 */
		MSG2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 32)), MASK);
		MSG  = _mm_add_epi32(MSG2, _mm_set_epi64x(0x550C7DC3243185BEULL, 0x12835B01D807AA98ULL));
		STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
		MSG  = _mm_shuffle_epi32(MSG, 0x0E);
		STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
		MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

		/* Rounds 12-15 */
		MSG3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 48)), MASK);
		MSG  = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC19BF1749BDC06A7ULL, 0x80DEB1FE72BE5D74ULL));
		STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
		TMP  = _mm_alignr_epi8(MSG3, MSG2, 4);
		MSG0 = _mm_add_epi32(MSG0, TMP);
		MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
		MSG  = _mm_shuffle_epi32(MSG, 0x0E);
		STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
		MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

#define SHANI_ROUND(Ka, Kb, Ma, Mb, Mc)                                   \
		MSG  = _mm_add_epi32(Ma, _mm_set_epi64x(Ka, Kb));             \
		STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);          \
		TMP  = _mm_alignr_epi8(Ma, Mc, 4);                            \
		Mb   = _mm_add_epi32(Mb, TMP);                                \
		Mb   = _mm_sha256msg2_epu32(Mb, Ma);                          \
		MSG  = _mm_shuffle_epi32(MSG, 0x0E);                          \
		STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);          \
		Mc   = _mm_sha256msg1_epu32(Mc, Ma);

		SHANI_ROUND(0x240CA1CC0FC19DC6ULL, 0xEFBE4786E49B69C1ULL, MSG0, MSG1, MSG3)
		SHANI_ROUND(0x76F988DA5CB0A9DCULL, 0x4A7484AA2DE92C6FULL, MSG1, MSG2, MSG0)
		SHANI_ROUND(0xBF597FC7B00327C8ULL, 0xA831C66D983E5152ULL, MSG2, MSG3, MSG1)
		SHANI_ROUND(0x1429296706CA6351ULL, 0xD5A79147C6E00BF3ULL, MSG3, MSG0, MSG2)
		SHANI_ROUND(0x53380D134D2C6DFCULL, 0x2E1B213827B70A85ULL, MSG0, MSG1, MSG3)
		SHANI_ROUND(0x92722C8581C2C92EULL, 0x766A0ABB650A7354ULL, MSG1, MSG2, MSG0)
		SHANI_ROUND(0xC76C51A3C24B8B70ULL, 0xA81A664BA2BFE8A1ULL, MSG2, MSG3, MSG1)
		SHANI_ROUND(0x106AA070F40E3585ULL, 0xD6990624D192E819ULL, MSG3, MSG0, MSG2)
		SHANI_ROUND(0x34B0BCB52748774CULL, 0x1E376C0819A4C116ULL, MSG0, MSG1, MSG3)
		SHANI_ROUND(0x682E6FF35B9CCA4FULL, 0x4ED8AA4A391C0CB3ULL, MSG1, MSG2, MSG0)
		SHANI_ROUND(0x8CC7020884C87814ULL, 0x78A5636F748F82EEULL, MSG2, MSG3, MSG1)
#undef SHANI_ROUND

		/* Rounds 60-63 (no further schedule) */
		MSG  = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC67178F2BEF9A3F7ULL, 0xA4506CEB90BEFFFAULL));
		STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
		MSG  = _mm_shuffle_epi32(MSG, 0x0E);
		STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

		STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
		STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);
	}

	TMP    = _mm_shuffle_epi32(STATE0, 0x1B);               /* FEBA */
	STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);               /* DCHG */
	STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);            /* DCBA */
	STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);               /* ABEF */

	_mm_storeu_si128((__m128i *)&state[0], STATE0);
	_mm_storeu_si128((__m128i *)&state[4], STATE1);
}
