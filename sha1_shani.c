/*
 * SHA-NI hardware-accelerated SHA1 compression function.
 * Uses Intel SHA extensions (available on AMD Zen+, Intel Ice Lake+).
 * Compile with: cc -O3 -msha -msse4.1 -c sha1_shani.c
 */

#include <stdint.h>
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
