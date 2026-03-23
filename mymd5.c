/* 
 * $Log: mymd5.c,v $
 * Revision 1.22  2026/02/28 16:22:20  dlr
 * hash_exists() fast-path probe, adaptive high-water-mark SSE buffer zero, mymd5salt_pre/post partial-state MD5, Livesalts dynamically allocated with JOB_DONE
 *
 * Revision 1.21  2026/02/22 07:33:04  dlr
 * *** empty log message ***
 *
 * Revision 1.20  2026/02/22 04:37:08  dlr
 * *** empty log message ***
 *
 * Revision 1.19  2026/02/21 21:07:54  dlr
 * Add SHA-NI runtime dispatch to mysha1() via CPUID function pointer
 *
 * Revision 1.18  2026/02/20 07:14:17  dlr
 * Add missing #include <stdlib.h> for exit() — fixes ARM build
 *
 * Revision 1.17  2026/02/20 06:48:28  dlr
 * No changes
 *
 * Revision 1.16  2017/11/02 13:42:55  dlr
 * cosmetic fix for arm
 *
 * Revision 1.15  2017/11/02 13:19:55  dlr
 * minor change for ARM
 *
 * Revision 1.14  2017/11/01 16:23:42  dlr
 * Add pbk code
 *
 * Revision 1.13  2017/10/23 06:18:14  dlr
 * Better handling of big endian power8
 *
 * Revision 1.12  2017/09/28 16:08:45  dlr
 * Freebsd changes
 *
 * Revision 1.11  2017/08/25 04:17:31  dlr
 * port for ARM SIMD and PowerPC SIMD
 *
 * Revision 1.10  2017/06/30 13:31:18  dlr
 * Additional work on bb3
 *
 * Revision 1.9  2017/06/30 06:46:35  dlr
 * fix log
 *
 * MD5 hash in C
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef ARM
#define NOTINTEL 1
#endif
#ifdef POWERPC
#define NOTINTEL 1
#endif

#ifndef NOTINTEL

#include <emmintrin.h>
#include <xmmintrin.h>
#include <cpuid.h>
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 5)
static __inline int my_cpuid_count(unsigned int l, unsigned int s,
    unsigned int *a, unsigned int *b, unsigned int *c, unsigned int *d) {
    __asm__ __volatile__("cpuid" : "=a"(*a),"=b"(*b),"=c"(*c),"=d"(*d) : "0"(l),"2"(s));
    return 1;
}
#define __get_cpuid_count my_cpuid_count
#endif

#endif
#ifdef POWERPC
#include <altivec.h>
#endif

#ifdef __FreeBSD__
#include <sys/endian.h>
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#else
#define bswap_32(x) __builtin_bswap32(x)
#define bswap_64(x) __builtin_bswap64(x)
#endif

#include "mdxfind.h"

#ifdef ARM
#if ARM > 6
#include <arm_neon.h>
extern int Neon;
#endif

#if ARM <8
#define MDX_BIT32 1
#endif

#if ARM >= 8 && defined(__aarch64__)
extern void sha1_compress_armce(uint32_t *, const uint32_t *);
extern void sha256_compress_armce(uint32_t *, const uint32_t *);
#ifdef HAVE_SHA512_CE
extern void sha512_compress_armce(uint64_t *, const uint64_t *);
#endif
#if defined(_WIN32)
#include <windows.h>
#elif !defined(MACOSX)
#include <sys/auxv.h>
#endif
#endif

#endif

#ifndef NOTINTEL
extern void sha1_update_intel(uint32_t *hash, uint32_t *block);
extern void sha1_compress_shani(uint32_t *hash, const uint32_t *block);
#endif


#define ROUND0(a, b, c, d, k, s, t)  ROUND_TAIL(a, b, d ^ (b & (c ^ d)), k, s, t)
#define ROUND1(a, b, c, d, k, s, t)  ROUND_TAIL(a, b, c ^ (d & (b ^ c)), k, s, t)
#define ROUND2(a, b, c, d, k, s, t)  ROUND_TAIL(a, b, b ^ c ^ d        , k, s, t)
#define ROUND3(a, b, c, d, k, s, t)  ROUND_TAIL(a, b, c ^ (b | ~d)     , k, s, t)

#ifdef AIX
#define ROUND_TAIL(a, b, expr, k, s, t)  \
  a += (expr) + t + bswap_32(block[k]);          \
  a = b + (a << s | a >> (32 - s));
#else
#define ROUND_TAIL(a, b, expr, k, s, t)  \
  a += (expr) + t + block[k];          \
  a = b + (a << s | a >> (32 - s));
#endif

void md5_compress(uint32_t *state, uint32_t *block) {
  uint32_t a, b, c, d;
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  ROUND0(a, b, c, d, 0, 7, 0xD76AA478)
  ROUND0(d, a, b, c, 1, 12, 0xE8C7B756)
  ROUND0(c, d, a, b, 2, 17, 0x242070DB)
  ROUND0(b, c, d, a, 3, 22, 0xC1BDCEEE)
  ROUND0(a, b, c, d, 4, 7, 0xF57C0FAF)
  ROUND0(d, a, b, c, 5, 12, 0x4787C62A)
  ROUND0(c, d, a, b, 6, 17, 0xA8304613)
  ROUND0(b, c, d, a, 7, 22, 0xFD469501)
  ROUND0(a, b, c, d, 8, 7, 0x698098D8)
  ROUND0(d, a, b, c, 9, 12, 0x8B44F7AF)
  ROUND0(c, d, a, b, 10, 17, 0xFFFF5BB1)
  ROUND0(b, c, d, a, 11, 22, 0x895CD7BE)
  ROUND0(a, b, c, d, 12, 7, 0x6B901122)
  ROUND0(d, a, b, c, 13, 12, 0xFD987193)
  ROUND0(c, d, a, b, 14, 17, 0xA679438E)
  ROUND0(b, c, d, a, 15, 22, 0x49B40821)
  ROUND1(a, b, c, d, 1, 5, 0xF61E2562)
  ROUND1(d, a, b, c, 6, 9, 0xC040B340)
  ROUND1(c, d, a, b, 11, 14, 0x265E5A51)
  ROUND1(b, c, d, a, 0, 20, 0xE9B6C7AA)
  ROUND1(a, b, c, d, 5, 5, 0xD62F105D)
  ROUND1(d, a, b, c, 10, 9, 0x02441453)
  ROUND1(c, d, a, b, 15, 14, 0xD8A1E681)
  ROUND1(b, c, d, a, 4, 20, 0xE7D3FBC8)
  ROUND1(a, b, c, d, 9, 5, 0x21E1CDE6)
  ROUND1(d, a, b, c, 14, 9, 0xC33707D6)
  ROUND1(c, d, a, b, 3, 14, 0xF4D50D87)
  ROUND1(b, c, d, a, 8, 20, 0x455A14ED)
  ROUND1(a, b, c, d, 13, 5, 0xA9E3E905)
  ROUND1(d, a, b, c, 2, 9, 0xFCEFA3F8)
  ROUND1(c, d, a, b, 7, 14, 0x676F02D9)
  ROUND1(b, c, d, a, 12, 20, 0x8D2A4C8A)
  ROUND2(a, b, c, d, 5, 4, 0xFFFA3942)
  ROUND2(d, a, b, c, 8, 11, 0x8771F681)
  ROUND2(c, d, a, b, 11, 16, 0x6D9D6122)
  ROUND2(b, c, d, a, 14, 23, 0xFDE5380C)
  ROUND2(a, b, c, d, 1, 4, 0xA4BEEA44)
  ROUND2(d, a, b, c, 4, 11, 0x4BDECFA9)
  ROUND2(c, d, a, b, 7, 16, 0xF6BB4B60)
  ROUND2(b, c, d, a, 10, 23, 0xBEBFBC70)
  ROUND2(a, b, c, d, 13, 4, 0x289B7EC6)
  ROUND2(d, a, b, c, 0, 11, 0xEAA127FA)
  ROUND2(c, d, a, b, 3, 16, 0xD4EF3085)
  ROUND2(b, c, d, a, 6, 23, 0x04881D05)
  ROUND2(a, b, c, d, 9, 4, 0xD9D4D039)
  ROUND2(d, a, b, c, 12, 11, 0xE6DB99E5)
  ROUND2(c, d, a, b, 15, 16, 0x1FA27CF8)
  ROUND2(b, c, d, a, 2, 23, 0xC4AC5665)
  ROUND3(a, b, c, d, 0, 6, 0xF4292244)
  ROUND3(d, a, b, c, 7, 10, 0x432AFF97)
  ROUND3(c, d, a, b, 14, 15, 0xAB9423A7)
  ROUND3(b, c, d, a, 5, 21, 0xFC93A039)
  ROUND3(a, b, c, d, 12, 6, 0x655B59C3)
  ROUND3(d, a, b, c, 3, 10, 0x8F0CCC92)
  ROUND3(c, d, a, b, 10, 15, 0xFFEFF47D)
  ROUND3(b, c, d, a, 1, 21, 0x85845DD1)
  ROUND3(a, b, c, d, 8, 6, 0x6FA87E4F)
  ROUND3(d, a, b, c, 15, 10, 0xFE2CE6E0)
  ROUND3(c, d, a, b, 6, 15, 0xA3014314)
  ROUND3(b, c, d, a, 13, 21, 0x4E0811A1)
  ROUND3(a, b, c, d, 4, 6, 0xF7537E82)
  ROUND3(d, a, b, c, 11, 10, 0xBD3AF235)
  ROUND3(c, d, a, b, 2, 15, 0x2AD7D2BB)
  ROUND3(b, c, d, a, 9, 21, 0xEB86D391)

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
}

/* Full message hasher */

void mymd5(unsigned char *message, int len, uint32_t *hash) {
  SVAL block[4];
  int i, rem, remcnt;
#ifndef NOTINTEL
  __m128i *p, r1;
#endif

  uint8_t *byteBlock;

  hash[0] = 0x67452301;
  hash[1] = 0xEFCDAB89;
  hash[2] = 0x98BADCFE;
  hash[3] = 0x10325476;

  remcnt = 0;
  for (i = 0; i + 64 <= len; i += 64)
    md5_compress(hash, (uint32_t * )(message + i));


  rem = len - i;
  remcnt = (rem < (64 - 8)) ? 8 : 0;

#ifndef NOTINTEL
  p = &block[0].sse;
  r1 = _mm_setzero_si128();
  p[0] = r1;
  p[1] = r1;
  p[2] = r1;
  p[3] = r1;
  memcpy(&block[0].sse, message + i, rem);
#else
  memcpy(&block[0].sse,message+i,rem);
  memset(&block[0].raw8[rem],0,64-rem-remcnt);
#endif

  byteBlock = (uint8_t * ) & block[0].sse;

  byteBlock[rem] = 0x80;
  rem++;
  if (remcnt == 0) {
    md5_compress(hash, (uint32_t * ) & block[0].sse);
#ifndef NOTINTEL
    r1 = _mm_setzero_si128();
    p[0] = r1;
    p[1] = r1;
    p[2] = r1;
    p[3] = r1;
#else
    memset(&block[0].words[0],0,64-8);
#endif

  }
#ifdef AIX
  block[3].longs[1] = bswap_64(((uint64_t)len) << 3);
#else
  block[3].longs[1] = ((uint64_t) len) << 3;
#endif
  md5_compress(hash, (uint32_t * ) & block[0].sse);
#ifdef AIX
  hash[1] = bswap_32(hash[1]);
  hash[2] = bswap_32(hash[2]);
  hash[3] = bswap_32(hash[3]);
  hash[0] = bswap_32(hash[0]);
#endif
}

#ifdef POWERPC
void init_md5sse(unsigned char *message, int len, unsigned char *block) {
  int i,j;
  uint32_t *s,*d, temp;
  vector unsigned int *p, r1;

  d = (uint32_t *)block;
  s = (uint32_t *)message;
  p = (vector unsigned int *) d;
  for (i=0; i < (len/4) + 1; i++) {
    p[i] = (vector unsigned int){s[i],s[i],s[i],s[i]};
  }
  r1 = (vector unsigned int){0,0,0,0};
  for (; i < 14; i++)
    p[i] = r1;
#ifdef AIX
  temp = bswap_32(len << 3);
#else
  temp = len << 3;
#endif
  p[14] = (vector unsigned int){temp,temp,temp,temp};
#ifdef AIX
  temp = bswap_32(len >> 29);
#else
  temp = len >> 29;
#endif
  p[15] = (vector unsigned int){temp,temp,temp,temp};
}



/* rol: (v << s) | (v >> (32 - s)) */
static inline vector unsigned int rol_vec32(vector unsigned int v, int s){
  vector unsigned int v1 = v << s;
  vector unsigned int v2 = v >> (32-s);
  return (v1 | v2);
}

#define AND(x, y) (x & y)
#define OR(x, y) (x | y)
#define ANDNOT(x, y) (x & ~y)
#define XOR(x,y) (x ^ y)

#define ADD(x, y) (x + y)
#define SHR(x, y) (x >> y)
#define SHL(x, y) (x << y)
#define SET1(x) ((vector unsigned int){x,x,x,x})

#define ROL(x,s) rol_vec32(x,s)

#define AA (unsigned int)0x67452301
#define BB (unsigned int)0xefcdab89
#define CC (unsigned int)0x98badcfe
#define DD (unsigned int)0x10325476

/* F: ((X & Y) | ((~X) & Z)) */
/* Fa: (z ^ (x & (y ^ z))) */
#define F(x,y,z) XOR(z, AND(x, XOR(y,z)))

/* G: ((X & Z) | (Y & (~Z))) */
/* Ga: (y ^ (z & (x ^ y))) */
#define G(x,y,z) XOR(y,AND(z,XOR(x,y))) 
/* #define G(x,y,z) OR(AND(x, z), ANDNOT(z, y)) */

/* H: (X ^ Y ^ Z) */
#define H(x,y,z) XOR(x, XOR(y, z))

/* I: (Y ^ (X | (~Z))) */
#define I(x,y,z) XOR(y, OR(x, XOR(z, SET1(-1))))

/* a = b + rol(a + fn(b, c, d) + X[k] + T_i, s); */
#ifdef AIX
#define swap_128(xx) (vec_perm(xx,xx,pv))
#define OP(fn, a, b, c, d, k, s, T_i) \
        a = ADD(b, ROL(ADD(ADD(ADD(SET1(T_i), a), swap_128(X[k])), fn(b, c, d)), s))
#else
#define OP(fn, a, b, c, d, k, s, T_i) \
        a = ADD(b, ROL(ADD(ADD(ADD(SET1(T_i), a), X[k]), fn(b, c, d)), s))
#endif

#ifdef AIX
#define FINAL(idx, val, old) hash[idx] =  swap_128(ADD(val, SET1(old)))
#define FINAL2(idx, val) hash[idx]= swap_128(ADD(val, hash[idx]))
#else
#define FINAL(idx, val, old) hash[idx] =  ADD(val, SET1(old))
#define FINAL2(idx, val) hash[idx]=ADD(val, hash[idx])
#endif


void mymd5salt(unsigned char *block, vector unsigned int *hash) {
  vector unsigned int a,b,c,d;
#ifdef AIX
  vector unsigned char pv = {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
#endif
  const vector unsigned int *X = (vector unsigned int *)block;

  a = (vector unsigned int){AA,AA,AA,AA};
  b = (vector unsigned int){BB,BB,BB,BB};
  c = (vector unsigned int){CC,CC,CC,CC};
  d = (vector unsigned int){DD,DD,DD,DD};

  OP(F, a, b, c, d, 0, 7, 0xd76aa478);
  OP(F, d, a, b, c, 1, 12, 0xe8c7b756);
  OP(F, c, d, a, b, 2, 17, 0x242070db);
  OP(F, b, c, d, a, 3, 22, 0xc1bdceee);
  OP(F, a, b, c, d, 4, 7, 0xf57c0faf);
  OP(F, d, a, b, c, 5, 12, 0x4787c62a);
  OP(F, c, d, a, b, 6, 17, 0xa8304613);
  OP(F, b, c, d, a, 7, 22, 0xfd469501);
  OP(F, a, b, c, d, 8, 7, 0x698098d8);
  OP(F, d, a, b, c, 9, 12, 0x8b44f7af);
  OP(F, c, d, a, b, 10, 17, 0xffff5bb1);
  OP(F, b, c, d, a, 11, 22, 0x895cd7be);
  OP(F, a, b, c, d, 12, 7, 0x6b901122);
  OP(F, d, a, b, c, 13, 12, 0xfd987193);
  OP(F, c, d, a, b, 14, 17, 0xa679438e);
  OP(F, b, c, d, a, 15, 22, 0x49b40821);
  /* Round 2. */
  OP(G, a, b, c, d, 1, 5, 0xf61e2562);
  OP(G, d, a, b, c, 6, 9, 0xc040b340);
  OP(G, c, d, a, b, 11, 14, 0x265e5a51);
  OP(G, b, c, d, a, 0, 20, 0xe9b6c7aa);
  OP(G, a, b, c, d, 5, 5, 0xd62f105d);
  OP(G, d, a, b, c, 10, 9, 0x02441453);
  OP(G, c, d, a, b, 15, 14, 0xd8a1e681);
  OP(G, b, c, d, a, 4, 20, 0xe7d3fbc8);
  OP(G, a, b, c, d, 9, 5, 0x21e1cde6);
  OP(G, d, a, b, c, 14, 9, 0xc33707d6);
  OP(G, c, d, a, b, 3, 14, 0xf4d50d87);
  OP(G, b, c, d, a, 8, 20, 0x455a14ed);
  OP(G, a, b, c, d, 13, 5, 0xa9e3e905);
  OP(G, d, a, b, c, 2, 9, 0xfcefa3f8);
  OP(G, c, d, a, b, 7, 14, 0x676f02d9);
  OP(G, b, c, d, a, 12, 20, 0x8d2a4c8a);
  /* Round 3. */
  OP(H, a, b, c, d, 5, 4, 0xfffa3942);
  OP(H, d, a, b, c, 8, 11, 0x8771f681);
  OP(H, c, d, a, b, 11, 16, 0x6d9d6122);
  OP(H, b, c, d, a, 14, 23, 0xfde5380c);
  OP(H, a, b, c, d, 1, 4, 0xa4beea44);
  OP(H, d, a, b, c, 4, 11, 0x4bdecfa9);
  OP(H, c, d, a, b, 7, 16, 0xf6bb4b60);
  OP(H, b, c, d, a, 10, 23, 0xbebfbc70);
  OP(H, a, b, c, d, 13, 4, 0x289b7ec6);
  OP(H, d, a, b, c, 0, 11, 0xeaa127fa);
  OP(H, c, d, a, b, 3, 16, 0xd4ef3085);
  OP(H, b, c, d, a, 6, 23, 0x04881d05);
  OP(H, a, b, c, d, 9, 4, 0xd9d4d039);
  OP(H, d, a, b, c, 12, 11, 0xe6db99e5);
  OP(H, c, d, a, b, 15, 16, 0x1fa27cf8);
  OP(H, b, c, d, a, 2, 23, 0xc4ac5665);
  /* Round 4. */
  OP(I, a, b, c, d, 0, 6, 0xf4292244);
  OP(I, d, a, b, c, 7, 10, 0x432aff97);
  OP(I, c, d, a, b, 14, 15, 0xab9423a7);
  OP(I, b, c, d, a, 5, 21, 0xfc93a039);
  OP(I, a, b, c, d, 12, 6, 0x655b59c3);
  OP(I, d, a, b, c, 3, 10, 0x8f0ccc92);
  OP(I, c, d, a, b, 10, 15, 0xffeff47d);
  OP(I, b, c, d, a, 1, 21, 0x85845dd1);
  OP(I, a, b, c, d, 8, 6, 0x6fa87e4f);
  OP(I, d, a, b, c, 15, 10, 0xfe2ce6e0);
  OP(I, c, d, a, b, 6, 15, 0xa3014314);
  OP(I, b, c, d, a, 13, 21, 0x4e0811a1);
  OP(I, a, b, c, d, 4, 6, 0xf7537e82);
  OP(I, d, a, b, c, 11, 10, 0xbd3af235);
  OP(I, c, d, a, b, 2, 15, 0x2ad7d2bb);
  OP(I, b, c, d, a, 9, 21, 0xeb86d391);

  FINAL(0, a, AA);
  FINAL(1, b, BB);
  FINAL(2, c, CC);
  FINAL(3, d, DD);
}
#endif


#if ARM > 6

void init_md5sse(unsigned char *message, int len, unsigned char *block) {
  int i,j;
  uint32_t *s,*d, temp;
  uint32x4_t *p, r1;

  if (!Neon) {
   fprintf(stderr,"Neon detection error\n");
    exit(1);
  }
  d = (uint32_t *)block;
  s = (uint32_t *)message;
  p = (uint32x4_t *) d;
  for (i=0; i < (len/4) + 1; i++) {
    p[i] = vdupq_n_u32(s[i]);
  }
  r1 = vdupq_n_u32(0);
  for (; i < 14; i++)
    p[i] = r1;
  temp = len << 3;
  p[14] = vdupq_n_u32(temp);
  temp = len >> 29;
  p[15] = vdupq_n_u32(temp);
}

static inline uint32x4_t sval_load(const SVAL *sval) { return vld1q_u32((uint32_t *)&sval->sse); }
static inline void sval_store(SVAL *sval, uint32x4_t v) { vst1q_u32((uint32_t *)&sval->sse, v); }

/* rol: (v << s) | (v >> (32 - s)) */
static inline uint32x4_t rol_epi32(uint32x4_t v, int s){
  int32x4_t sv = vdupq_n_s32(s);
  int32x4_t sv2 = vdupq_n_s32(s - 32);
  uint32x4_t v1 = vshlq_u32(v, sv);
  uint32x4_t v2 = vshlq_u32(v, sv2);
  return vorrq_u32(v1, v2);
}
#define ror_epi32(v,s) rol_epi32(v, 32 - (s))

#define AND(x, y) vandq_u32(x, y)
#define OR(x, y) vorrq_u32(x, y)
#define ANDNOT(x, y) vbicq_u32(y, x)
#define XOR(x,y) veorq_u32(x,y)

#define ADD(x, y) vaddq_u32(x, y)
#define SHR(x, y) vshrq_n_u32(x, y)
#define SHL(x, y) vshlq_n_u32(x, y)
#define SET1(x) vdupq_n_u32(x)

#define ROR(x,s) ror_epi32(x,s)
#define ROL(x,s) rol_epi32(x,s)

#define AA 0x67452301
#define BB 0xefcdab89
#define CC 0x98badcfe
#define DD 0x10325476

/* F: ((X & Y) | ((~X) & Z)) */
/* Fa: (z ^ (x & (y ^ z))) */
#define F(x,y,z) XOR(z, AND(x, XOR(y,z)))

/* G: ((X & Z) | (Y & (~Z))) */
/* Ga: (y ^ (z & (x ^ y))) */
/* #define G(x,y,z) XOR(y,AND(z,XOR(x,y))) */
#define G(x,y,z) OR(AND(x, z), ANDNOT(z, y))

/* H: (X ^ Y ^ Z) */
#define H(x,y,z) XOR(x, XOR(y, z))

/* I: (Y ^ (X | (~Z))) */
#define I(x,y,z) XOR(y, OR(x, XOR(z, SET1(-1))))

/* a = b + rol(a + fn(b, c, d) + X[k] + T_i, s); */
#define OP(fn, a, b, c, d, k, s, T_i) \
        a = ADD(b, ROL(ADD(ADD(ADD(SET1(T_i), a), sval_load(&X[k])), fn(b, c, d)), s))

#define FINAL(idx, val, old) sval_store(&hash[idx], ADD(val, SET1(old)))
#define FINAL2(idx, val) sval_store(&hash[idx], ADD(val, hash[idx].sse))


void mymd5salt(unsigned char *block, SVAL  *hash) {
  SVAL T;
  uint32x4_t a,b,c,d;
  const SVAL *X = (SVAL *)block;

  a = vdupq_n_u32(AA);
  b = vdupq_n_u32(BB);
  c = vdupq_n_u32(CC);
  d = vdupq_n_u32(DD);

  OP(F, a, b, c, d, 0, 7, 0xd76aa478);
  OP(F, d, a, b, c, 1, 12, 0xe8c7b756);
  OP(F, c, d, a, b, 2, 17, 0x242070db);
  OP(F, b, c, d, a, 3, 22, 0xc1bdceee);
  OP(F, a, b, c, d, 4, 7, 0xf57c0faf);
  OP(F, d, a, b, c, 5, 12, 0x4787c62a);
  OP(F, c, d, a, b, 6, 17, 0xa8304613);
  OP(F, b, c, d, a, 7, 22, 0xfd469501);
  OP(F, a, b, c, d, 8, 7, 0x698098d8);
  OP(F, d, a, b, c, 9, 12, 0x8b44f7af);
  OP(F, c, d, a, b, 10, 17, 0xffff5bb1);
  OP(F, b, c, d, a, 11, 22, 0x895cd7be);
  OP(F, a, b, c, d, 12, 7, 0x6b901122);
  OP(F, d, a, b, c, 13, 12, 0xfd987193);
  OP(F, c, d, a, b, 14, 17, 0xa679438e);
  OP(F, b, c, d, a, 15, 22, 0x49b40821);
  /* Round 2. */
  OP(G, a, b, c, d, 1, 5, 0xf61e2562);
  OP(G, d, a, b, c, 6, 9, 0xc040b340);
  OP(G, c, d, a, b, 11, 14, 0x265e5a51);
  OP(G, b, c, d, a, 0, 20, 0xe9b6c7aa);
  OP(G, a, b, c, d, 5, 5, 0xd62f105d);
  OP(G, d, a, b, c, 10, 9, 0x02441453);
  OP(G, c, d, a, b, 15, 14, 0xd8a1e681);
  OP(G, b, c, d, a, 4, 20, 0xe7d3fbc8);
  OP(G, a, b, c, d, 9, 5, 0x21e1cde6);
  OP(G, d, a, b, c, 14, 9, 0xc33707d6);
  OP(G, c, d, a, b, 3, 14, 0xf4d50d87);
  OP(G, b, c, d, a, 8, 20, 0x455a14ed);
  OP(G, a, b, c, d, 13, 5, 0xa9e3e905);
  OP(G, d, a, b, c, 2, 9, 0xfcefa3f8);
  OP(G, c, d, a, b, 7, 14, 0x676f02d9);
  OP(G, b, c, d, a, 12, 20, 0x8d2a4c8a);
  /* Round 3. */
  OP(H, a, b, c, d, 5, 4, 0xfffa3942);
  OP(H, d, a, b, c, 8, 11, 0x8771f681);
  OP(H, c, d, a, b, 11, 16, 0x6d9d6122);
  OP(H, b, c, d, a, 14, 23, 0xfde5380c);
  OP(H, a, b, c, d, 1, 4, 0xa4beea44);
  OP(H, d, a, b, c, 4, 11, 0x4bdecfa9);
  OP(H, c, d, a, b, 7, 16, 0xf6bb4b60);
  OP(H, b, c, d, a, 10, 23, 0xbebfbc70);
  OP(H, a, b, c, d, 13, 4, 0x289b7ec6);
  OP(H, d, a, b, c, 0, 11, 0xeaa127fa);
  OP(H, c, d, a, b, 3, 16, 0xd4ef3085);
  OP(H, b, c, d, a, 6, 23, 0x04881d05);
  OP(H, a, b, c, d, 9, 4, 0xd9d4d039);
  OP(H, d, a, b, c, 12, 11, 0xe6db99e5);
  OP(H, c, d, a, b, 15, 16, 0x1fa27cf8);
  OP(H, b, c, d, a, 2, 23, 0xc4ac5665);
  /* Round 4. */
  OP(I, a, b, c, d, 0, 6, 0xf4292244);
  OP(I, d, a, b, c, 7, 10, 0x432aff97);
  OP(I, c, d, a, b, 14, 15, 0xab9423a7);
  OP(I, b, c, d, a, 5, 21, 0xfc93a039);
  OP(I, a, b, c, d, 12, 6, 0x655b59c3);
  OP(I, d, a, b, c, 3, 10, 0x8f0ccc92);
  OP(I, c, d, a, b, 10, 15, 0xffeff47d);
  OP(I, b, c, d, a, 1, 21, 0x85845dd1);
  OP(I, a, b, c, d, 8, 6, 0x6fa87e4f);
  OP(I, d, a, b, c, 15, 10, 0xfe2ce6e0);
  OP(I, c, d, a, b, 6, 15, 0xa3014314);
  OP(I, b, c, d, a, 13, 21, 0x4e0811a1);
  OP(I, a, b, c, d, 4, 6, 0xf7537e82);
  OP(I, d, a, b, c, 11, 10, 0xbd3af235);
  OP(I, c, d, a, b, 2, 15, 0x2ad7d2bb);
  OP(I, b, c, d, a, 9, 21, 0xeb86d391);

  FINAL(0, a, AA);
  FINAL(1, b, BB);
  FINAL(2, c, CC);
  FINAL(3, d, DD);
}


#endif
#ifndef NOTINTEL

void init_md5sse(unsigned char *message, int len, unsigned char *block) {
  int i, j;
  uint32_t *s, *d, temp;
  __m128i *p, r1;

  d = (uint32_t *) block;
  s = (uint32_t *) message;
  p = (__m128i *) d;
  for (i = 0; i < (len / 4) + 1; i++) {
    p[i] = _mm_set1_epi32(s[i]);
  }
  r1 = _mm_setzero_si128();
  for (; i < 14; i++)
    p[i] = r1;
  temp = len << 3;
  p[14] = _mm_set1_epi32(temp);
  temp = len >> 29;
  p[15] = _mm_set1_epi32(temp);
}


#define C32(x) { .words = {x,x,x,x} }
#define C64(x) { .longs = {x,x} }

static inline __m128i sval_load(const SVAL *sval) { return _mm_load_si128(&sval->sse); }

static inline void sval_store(SVAL *sval, __m128i v) { _mm_store_si128(&sval->sse, v); }

/* rol: (v << s) | (v >> (32 - s)) */
static inline __m128i rol_epi32(__m128i v, int s) {
  __m128i v1 = _mm_slli_epi32(v, s);
  __m128i v2 = _mm_srli_epi32(v, 32 - s);
  return _mm_or_si128(v1, v2);
}

#define ror_epi32(v, s) rol_epi32(v, 32 - (s))

static inline __m128i rol_epi64(__m128i v, int s) {
  __m128i v1 = _mm_slli_epi64(v, s);
  __m128i v2 = _mm_srli_epi64(v, 64 - s);
  return _mm_or_si128(v1, v2);
}

#define ror_epi64(v, s) rol_epi64(v, 64 - (s))
/*
 * readable ops
 */
#ifndef SSE2BITS
#define SSE2BITS 32
#endif
#if SSE2BITS != 32 && SSE2BITS != 64
#error bad value for SSE2BITS
#endif

#define XOR(x, y) _mm_xor_si128(x, y)
#define AND(x, y) _mm_and_si128(x, y)
#define OR(x, y) _mm_or_si128(x, y)
#define ANDNOT(x, y) _mm_andnot_si128(x, y)

#if SSE2BITS == 64
#define ADD(x, y) _mm_add_epi64(x, y)
#define SHR(x, y) _mm_srli_epi64(x, y)
#define SHL(x, y) _mm_slli_epi64(x, y)
#define SET1(x) _mm_set1_epi64(x)

#if 1
#define ROR(x,s) ror_epi64(x,s)
#define ROL(x,s) rol_epi64(x,s)
#else
#define ROR(x, s) XOR(SHR(x, s), SHL(x, 64-(s)))
#define ROL(x, s) XOR(SHL(x, s), SHR(x, 64-(s)))
#endif

#else // SSE2BITS == 32

#define ADD(x, y) _mm_add_epi32(x, y)
#define SHR(x, y) _mm_srli_epi32(x, y)
#define SHL(x, y) _mm_slli_epi32(x, y)
#define SET1(x) _mm_set1_epi32(x)

#if 1
#define ROR(x, s) ror_epi32(x,s)
#define ROL(x, s) rol_epi32(x,s)
#else
#define ROR(x, s) XOR(SHR(x, s), SHL(x, 32-(s)))
#define ROL(x, s) XOR(SHL(x, s), SHR(x, 32-(s)))
#endif

#endif

#define AA 0x67452301
#define BB 0xefcdab89
#define CC 0x98badcfe
#define DD 0x10325476

/* F: ((X & Y) | ((~X) & Z)) */
/* Fa: (z ^ (x & (y ^ z))) */
#define F(x, y, z) XOR(z, AND(x, XOR(y,z)))

/* G: ((X & Z) | (Y & (~Z))) */
/* Ga: (y ^ (z & (x ^ y))) */
/* #define G(x,y,z) XOR(y,AND(z,XOR(x,y))) */
#define G(x, y, z) OR(AND(x, z), ANDNOT(z, y))

/* H: (X ^ Y ^ Z) */
#define H(x, y, z) XOR(x, XOR(y, z))

/* I: (Y ^ (X | (~Z))) */
#define I(x, y, z) XOR(y, OR(x, XOR(z, SET1(-1))))

/* a = b + rol(a + fn(b, c, d) + X[k] + T_i, s); */
#define OP(fn, a, b, c, d, k, s, T_i) \
        a = ADD(b, ROL(ADD(ADD(ADD(SET1(T_i), a), sval_load(&X[k])), fn(b, c, d)), s))

#define FINAL(idx, val, old) sval_store(&hash[idx], ADD(val, SET1(old)))
#define FINAL2(idx, val) sval_store(&hash[idx], ADD(val, hash[idx].sse))


void mymd5salt(unsigned char *block, SVAL *hash) {
  SVAL T;
  __m128i a, b, c, d;
  const SVAL *X = (SVAL *) block;

  a = _mm_set1_epi32(AA);
  b = _mm_set1_epi32(BB);
  c = _mm_set1_epi32(CC);
  d = _mm_set1_epi32(DD);

  OP(F, a, b, c, d, 0, 7, 0xd76aa478);
  OP(F, d, a, b, c, 1, 12, 0xe8c7b756);
  OP(F, c, d, a, b, 2, 17, 0x242070db);
  OP(F, b, c, d, a, 3, 22, 0xc1bdceee);
  OP(F, a, b, c, d, 4, 7, 0xf57c0faf);
  OP(F, d, a, b, c, 5, 12, 0x4787c62a);
  OP(F, c, d, a, b, 6, 17, 0xa8304613);
  OP(F, b, c, d, a, 7, 22, 0xfd469501);
  OP(F, a, b, c, d, 8, 7, 0x698098d8);
  OP(F, d, a, b, c, 9, 12, 0x8b44f7af);
  OP(F, c, d, a, b, 10, 17, 0xffff5bb1);
  OP(F, b, c, d, a, 11, 22, 0x895cd7be);
  OP(F, a, b, c, d, 12, 7, 0x6b901122);
  OP(F, d, a, b, c, 13, 12, 0xfd987193);
  OP(F, c, d, a, b, 14, 17, 0xa679438e);
  OP(F, b, c, d, a, 15, 22, 0x49b40821);
  /* Round 2. */
  OP(G, a, b, c, d, 1, 5, 0xf61e2562);
  OP(G, d, a, b, c, 6, 9, 0xc040b340);
  OP(G, c, d, a, b, 11, 14, 0x265e5a51);
  OP(G, b, c, d, a, 0, 20, 0xe9b6c7aa);
  OP(G, a, b, c, d, 5, 5, 0xd62f105d);
  OP(G, d, a, b, c, 10, 9, 0x02441453);
  OP(G, c, d, a, b, 15, 14, 0xd8a1e681);
  OP(G, b, c, d, a, 4, 20, 0xe7d3fbc8);
  OP(G, a, b, c, d, 9, 5, 0x21e1cde6);
  OP(G, d, a, b, c, 14, 9, 0xc33707d6);
  OP(G, c, d, a, b, 3, 14, 0xf4d50d87);
  OP(G, b, c, d, a, 8, 20, 0x455a14ed);
  OP(G, a, b, c, d, 13, 5, 0xa9e3e905);
  OP(G, d, a, b, c, 2, 9, 0xfcefa3f8);
  OP(G, c, d, a, b, 7, 14, 0x676f02d9);
  OP(G, b, c, d, a, 12, 20, 0x8d2a4c8a);
  /* Round 3. */
  OP(H, a, b, c, d, 5, 4, 0xfffa3942);
  OP(H, d, a, b, c, 8, 11, 0x8771f681);
  OP(H, c, d, a, b, 11, 16, 0x6d9d6122);
  OP(H, b, c, d, a, 14, 23, 0xfde5380c);
  OP(H, a, b, c, d, 1, 4, 0xa4beea44);
  OP(H, d, a, b, c, 4, 11, 0x4bdecfa9);
  OP(H, c, d, a, b, 7, 16, 0xf6bb4b60);
  OP(H, b, c, d, a, 10, 23, 0xbebfbc70);
  OP(H, a, b, c, d, 13, 4, 0x289b7ec6);
  OP(H, d, a, b, c, 0, 11, 0xeaa127fa);
  OP(H, c, d, a, b, 3, 16, 0xd4ef3085);
  OP(H, b, c, d, a, 6, 23, 0x04881d05);
  OP(H, a, b, c, d, 9, 4, 0xd9d4d039);
  OP(H, d, a, b, c, 12, 11, 0xe6db99e5);
  OP(H, c, d, a, b, 15, 16, 0x1fa27cf8);
  OP(H, b, c, d, a, 2, 23, 0xc4ac5665);
  /* Round 4. */
  OP(I, a, b, c, d, 0, 6, 0xf4292244);
  OP(I, d, a, b, c, 7, 10, 0x432aff97);
  OP(I, c, d, a, b, 14, 15, 0xab9423a7);
  OP(I, b, c, d, a, 5, 21, 0xfc93a039);
  OP(I, a, b, c, d, 12, 6, 0x655b59c3);
  OP(I, d, a, b, c, 3, 10, 0x8f0ccc92);
  OP(I, c, d, a, b, 10, 15, 0xffeff47d);
  OP(I, b, c, d, a, 1, 21, 0x85845dd1);
  OP(I, a, b, c, d, 8, 6, 0x6fa87e4f);
  OP(I, d, a, b, c, 15, 10, 0xfe2ce6e0);
  OP(I, c, d, a, b, 6, 15, 0xa3014314);
  OP(I, b, c, d, a, 13, 21, 0x4e0811a1);
  OP(I, a, b, c, d, 4, 6, 0xf7537e82);
  OP(I, d, a, b, c, 11, 10, 0xbd3af235);
  OP(I, c, d, a, b, 2, 15, 0x2ad7d2bb);
  OP(I, b, c, d, a, 9, 21, 0xeb86d391);

  /* FINAL(0, a, AA); */
  a = ADD(a, SET1(AA));
  sval_store(&hash[0], a);

  FINAL(1, b, BB);
  FINAL(2, c, CC);
  FINAL(3, d, DD);

}

/* Precompute first 8 F-rounds from password words X[0]-X[7].
 * Stores intermediate a,b,c,d in state[0..3]. */
void mymd5salt_pre(unsigned char *block, SVAL *state) {
  __m128i a, b, c, d;
  const SVAL *X = (SVAL *) block;

  a = _mm_set1_epi32(AA);
  b = _mm_set1_epi32(BB);
  c = _mm_set1_epi32(CC);
  d = _mm_set1_epi32(DD);

  OP(F, a, b, c, d, 0, 7, 0xd76aa478);
  OP(F, d, a, b, c, 1, 12, 0xe8c7b756);
  OP(F, c, d, a, b, 2, 17, 0x242070db);
  OP(F, b, c, d, a, 3, 22, 0xc1bdceee);
  OP(F, a, b, c, d, 4, 7, 0xf57c0faf);
  OP(F, d, a, b, c, 5, 12, 0x4787c62a);
  OP(F, c, d, a, b, 6, 17, 0xa8304613);
  OP(F, b, c, d, a, 7, 22, 0xfd469501);

  sval_store(&state[0], a);
  sval_store(&state[1], b);
  sval_store(&state[2], c);
  sval_store(&state[3], d);
}

/* Complete MD5 from saved partial state. Loads a,b,c,d from state[0..3],
 * runs remaining 56 rounds using full block X[0]-X[15]. */
void mymd5salt_post(unsigned char *block, SVAL *state, SVAL *hash) {
  __m128i a, b, c, d;
  const SVAL *X = (SVAL *) block;

  a = sval_load(&state[0]);
  b = sval_load(&state[1]);
  c = sval_load(&state[2]);
  d = sval_load(&state[3]);

  OP(F, a, b, c, d, 8, 7, 0x698098d8);
  OP(F, d, a, b, c, 9, 12, 0x8b44f7af);
  OP(F, c, d, a, b, 10, 17, 0xffff5bb1);
  OP(F, b, c, d, a, 11, 22, 0x895cd7be);
  OP(F, a, b, c, d, 12, 7, 0x6b901122);
  OP(F, d, a, b, c, 13, 12, 0xfd987193);
  OP(F, c, d, a, b, 14, 17, 0xa679438e);
  OP(F, b, c, d, a, 15, 22, 0x49b40821);
  /* Round 2. */
  OP(G, a, b, c, d, 1, 5, 0xf61e2562);
  OP(G, d, a, b, c, 6, 9, 0xc040b340);
  OP(G, c, d, a, b, 11, 14, 0x265e5a51);
  OP(G, b, c, d, a, 0, 20, 0xe9b6c7aa);
  OP(G, a, b, c, d, 5, 5, 0xd62f105d);
  OP(G, d, a, b, c, 10, 9, 0x02441453);
  OP(G, c, d, a, b, 15, 14, 0xd8a1e681);
  OP(G, b, c, d, a, 4, 20, 0xe7d3fbc8);
  OP(G, a, b, c, d, 9, 5, 0x21e1cde6);
  OP(G, d, a, b, c, 14, 9, 0xc33707d6);
  OP(G, c, d, a, b, 3, 14, 0xf4d50d87);
  OP(G, b, c, d, a, 8, 20, 0x455a14ed);
  OP(G, a, b, c, d, 13, 5, 0xa9e3e905);
  OP(G, d, a, b, c, 2, 9, 0xfcefa3f8);
  OP(G, c, d, a, b, 7, 14, 0x676f02d9);
  OP(G, b, c, d, a, 12, 20, 0x8d2a4c8a);
  /* Round 3. */
  OP(H, a, b, c, d, 5, 4, 0xfffa3942);
  OP(H, d, a, b, c, 8, 11, 0x8771f681);
  OP(H, c, d, a, b, 11, 16, 0x6d9d6122);
  OP(H, b, c, d, a, 14, 23, 0xfde5380c);
  OP(H, a, b, c, d, 1, 4, 0xa4beea44);
  OP(H, d, a, b, c, 4, 11, 0x4bdecfa9);
  OP(H, c, d, a, b, 7, 16, 0xf6bb4b60);
  OP(H, b, c, d, a, 10, 23, 0xbebfbc70);
  OP(H, a, b, c, d, 13, 4, 0x289b7ec6);
  OP(H, d, a, b, c, 0, 11, 0xeaa127fa);
  OP(H, c, d, a, b, 3, 16, 0xd4ef3085);
  OP(H, b, c, d, a, 6, 23, 0x04881d05);
  OP(H, a, b, c, d, 9, 4, 0xd9d4d039);
  OP(H, d, a, b, c, 12, 11, 0xe6db99e5);
  OP(H, c, d, a, b, 15, 16, 0x1fa27cf8);
  OP(H, b, c, d, a, 2, 23, 0xc4ac5665);
  /* Round 4. */
  OP(I, a, b, c, d, 0, 6, 0xf4292244);
  OP(I, d, a, b, c, 7, 10, 0x432aff97);
  OP(I, c, d, a, b, 14, 15, 0xab9423a7);
  OP(I, b, c, d, a, 5, 21, 0xfc93a039);
  OP(I, a, b, c, d, 12, 6, 0x655b59c3);
  OP(I, d, a, b, c, 3, 10, 0x8f0ccc92);
  OP(I, c, d, a, b, 10, 15, 0xffeff47d);
  OP(I, b, c, d, a, 1, 21, 0x85845dd1);
  OP(I, a, b, c, d, 8, 6, 0x6fa87e4f);
  OP(I, d, a, b, c, 15, 10, 0xfe2ce6e0);
  OP(I, c, d, a, b, 6, 15, 0xa3014314);
  OP(I, b, c, d, a, 13, 21, 0x4e0811a1);
  OP(I, a, b, c, d, 4, 6, 0xf7537e82);
  OP(I, d, a, b, c, 11, 10, 0xbd3af235);
  OP(I, c, d, a, b, 2, 15, 0x2ad7d2bb);
  OP(I, b, c, d, a, 9, 21, 0xeb86d391);

  a = ADD(a, SET1(AA));
  sval_store(&hash[0], a);

  FINAL(1, b, BB);
  FINAL(2, c, CC);
  FINAL(3, d, DD);
}

void mymd5salt2(unsigned char *block, SVAL *hash) {
  __m128i a, b, c, d;
  const SVAL *X = (SVAL *) block;


  a = _mm_set1_epi32(AA);
  b = _mm_set1_epi32(BB);
  c = _mm_set1_epi32(CC);
  d = _mm_set1_epi32(DD);

  OP(F, a, b, c, d, 0, 7, 0xd76aa478);
  OP(F, d, a, b, c, 1, 12, 0xe8c7b756);
  OP(F, c, d, a, b, 2, 17, 0x242070db);
  OP(F, b, c, d, a, 3, 22, 0xc1bdceee);
  OP(F, a, b, c, d, 4, 7, 0xf57c0faf);
  OP(F, d, a, b, c, 5, 12, 0x4787c62a);
  OP(F, c, d, a, b, 6, 17, 0xa8304613);
  OP(F, b, c, d, a, 7, 22, 0xfd469501);
  OP(F, a, b, c, d, 8, 7, 0x698098d8);
  OP(F, d, a, b, c, 9, 12, 0x8b44f7af);
  OP(F, c, d, a, b, 10, 17, 0xffff5bb1);
  OP(F, b, c, d, a, 11, 22, 0x895cd7be);
  OP(F, a, b, c, d, 12, 7, 0x6b901122);
  OP(F, d, a, b, c, 13, 12, 0xfd987193);
  OP(F, c, d, a, b, 14, 17, 0xa679438e);
  OP(F, b, c, d, a, 15, 22, 0x49b40821);
  /* Round 2. */
  OP(G, a, b, c, d, 1, 5, 0xf61e2562);
  OP(G, d, a, b, c, 6, 9, 0xc040b340);
  OP(G, c, d, a, b, 11, 14, 0x265e5a51);
  OP(G, b, c, d, a, 0, 20, 0xe9b6c7aa);
  OP(G, a, b, c, d, 5, 5, 0xd62f105d);
  OP(G, d, a, b, c, 10, 9, 0x02441453);
  OP(G, c, d, a, b, 15, 14, 0xd8a1e681);
  OP(G, b, c, d, a, 4, 20, 0xe7d3fbc8);
  OP(G, a, b, c, d, 9, 5, 0x21e1cde6);
  OP(G, d, a, b, c, 14, 9, 0xc33707d6);
  OP(G, c, d, a, b, 3, 14, 0xf4d50d87);
  OP(G, b, c, d, a, 8, 20, 0x455a14ed);
  OP(G, a, b, c, d, 13, 5, 0xa9e3e905);
  OP(G, d, a, b, c, 2, 9, 0xfcefa3f8);
  OP(G, c, d, a, b, 7, 14, 0x676f02d9);
  OP(G, b, c, d, a, 12, 20, 0x8d2a4c8a);
  /* Round 3. */
  OP(H, a, b, c, d, 5, 4, 0xfffa3942);
  OP(H, d, a, b, c, 8, 11, 0x8771f681);
  OP(H, c, d, a, b, 11, 16, 0x6d9d6122);
  OP(H, b, c, d, a, 14, 23, 0xfde5380c);
  OP(H, a, b, c, d, 1, 4, 0xa4beea44);
  OP(H, d, a, b, c, 4, 11, 0x4bdecfa9);
  OP(H, c, d, a, b, 7, 16, 0xf6bb4b60);
  OP(H, b, c, d, a, 10, 23, 0xbebfbc70);
  OP(H, a, b, c, d, 13, 4, 0x289b7ec6);
  OP(H, d, a, b, c, 0, 11, 0xeaa127fa);
  OP(H, c, d, a, b, 3, 16, 0xd4ef3085);
  OP(H, b, c, d, a, 6, 23, 0x04881d05);
  OP(H, a, b, c, d, 9, 4, 0xd9d4d039);
  OP(H, d, a, b, c, 12, 11, 0xe6db99e5);
  OP(H, c, d, a, b, 15, 16, 0x1fa27cf8);
  OP(H, b, c, d, a, 2, 23, 0xc4ac5665);
  /* Round 4. */
  OP(I, a, b, c, d, 0, 6, 0xf4292244);
  OP(I, d, a, b, c, 7, 10, 0x432aff97);
  OP(I, c, d, a, b, 14, 15, 0xab9423a7);
  OP(I, b, c, d, a, 5, 21, 0xfc93a039);
  OP(I, a, b, c, d, 12, 6, 0x655b59c3);
  OP(I, d, a, b, c, 3, 10, 0x8f0ccc92);
  OP(I, c, d, a, b, 10, 15, 0xffeff47d);
  OP(I, b, c, d, a, 1, 21, 0x85845dd1);
  OP(I, a, b, c, d, 8, 6, 0x6fa87e4f);
  OP(I, d, a, b, c, 15, 10, 0xfe2ce6e0);
  OP(I, c, d, a, b, 6, 15, 0xa3014314);
  OP(I, b, c, d, a, 13, 21, 0x4e0811a1);
  OP(I, a, b, c, d, 4, 6, 0xf7537e82);
  OP(I, d, a, b, c, 11, 10, 0xbd3af235);
  OP(I, c, d, a, b, 2, 15, 0x2ad7d2bb);
  OP(I, b, c, d, a, 9, 21, 0xeb86d391);

  X += 16;
  FINAL(0, a, AA);
  FINAL(1, b, BB);
  FINAL(2, c, CC);
  FINAL(3, d, DD);
  a = sval_load(&hash[0]);
  b = sval_load(&hash[1]);
  c = sval_load(&hash[2]);
  d = sval_load(&hash[3]);

  OP(F, a, b, c, d, 0, 7, 0xd76aa478);
  OP(F, d, a, b, c, 1, 12, 0xe8c7b756);
  OP(F, c, d, a, b, 2, 17, 0x242070db);
  OP(F, b, c, d, a, 3, 22, 0xc1bdceee);
  OP(F, a, b, c, d, 4, 7, 0xf57c0faf);
  OP(F, d, a, b, c, 5, 12, 0x4787c62a);
  OP(F, c, d, a, b, 6, 17, 0xa8304613);
  OP(F, b, c, d, a, 7, 22, 0xfd469501);
  OP(F, a, b, c, d, 8, 7, 0x698098d8);
  OP(F, d, a, b, c, 9, 12, 0x8b44f7af);
  OP(F, c, d, a, b, 10, 17, 0xffff5bb1);
  OP(F, b, c, d, a, 11, 22, 0x895cd7be);
  OP(F, a, b, c, d, 12, 7, 0x6b901122);
  OP(F, d, a, b, c, 13, 12, 0xfd987193);
  OP(F, c, d, a, b, 14, 17, 0xa679438e);
  OP(F, b, c, d, a, 15, 22, 0x49b40821);
  /* Round 2. */
  OP(G, a, b, c, d, 1, 5, 0xf61e2562);
  OP(G, d, a, b, c, 6, 9, 0xc040b340);
  OP(G, c, d, a, b, 11, 14, 0x265e5a51);
  OP(G, b, c, d, a, 0, 20, 0xe9b6c7aa);
  OP(G, a, b, c, d, 5, 5, 0xd62f105d);
  OP(G, d, a, b, c, 10, 9, 0x02441453);
  OP(G, c, d, a, b, 15, 14, 0xd8a1e681);
  OP(G, b, c, d, a, 4, 20, 0xe7d3fbc8);
  OP(G, a, b, c, d, 9, 5, 0x21e1cde6);
  OP(G, d, a, b, c, 14, 9, 0xc33707d6);
  OP(G, c, d, a, b, 3, 14, 0xf4d50d87);
  OP(G, b, c, d, a, 8, 20, 0x455a14ed);
  OP(G, a, b, c, d, 13, 5, 0xa9e3e905);
  OP(G, d, a, b, c, 2, 9, 0xfcefa3f8);
  OP(G, c, d, a, b, 7, 14, 0x676f02d9);
  OP(G, b, c, d, a, 12, 20, 0x8d2a4c8a);
  /* Round 3. */
  OP(H, a, b, c, d, 5, 4, 0xfffa3942);
  OP(H, d, a, b, c, 8, 11, 0x8771f681);
  OP(H, c, d, a, b, 11, 16, 0x6d9d6122);
  OP(H, b, c, d, a, 14, 23, 0xfde5380c);
  OP(H, a, b, c, d, 1, 4, 0xa4beea44);
  OP(H, d, a, b, c, 4, 11, 0x4bdecfa9);
  OP(H, c, d, a, b, 7, 16, 0xf6bb4b60);
  OP(H, b, c, d, a, 10, 23, 0xbebfbc70);
  OP(H, a, b, c, d, 13, 4, 0x289b7ec6);
  OP(H, d, a, b, c, 0, 11, 0xeaa127fa);
  OP(H, c, d, a, b, 3, 16, 0xd4ef3085);
  OP(H, b, c, d, a, 6, 23, 0x04881d05);
  OP(H, a, b, c, d, 9, 4, 0xd9d4d039);
  OP(H, d, a, b, c, 12, 11, 0xe6db99e5);
  OP(H, c, d, a, b, 15, 16, 0x1fa27cf8);
  OP(H, b, c, d, a, 2, 23, 0xc4ac5665);
  /* Round 4. */
  OP(I, a, b, c, d, 0, 6, 0xf4292244);
  OP(I, d, a, b, c, 7, 10, 0x432aff97);
  OP(I, c, d, a, b, 14, 15, 0xab9423a7);
  OP(I, b, c, d, a, 5, 21, 0xfc93a039);
  OP(I, a, b, c, d, 12, 6, 0x655b59c3);
  OP(I, d, a, b, c, 3, 10, 0x8f0ccc92);
  OP(I, c, d, a, b, 10, 15, 0xffeff47d);
  OP(I, b, c, d, a, 1, 21, 0x85845dd1);
  OP(I, a, b, c, d, 8, 6, 0x6fa87e4f);
  OP(I, d, a, b, c, 15, 10, 0xfe2ce6e0);
  OP(I, c, d, a, b, 6, 15, 0xa3014314);
  OP(I, b, c, d, a, 13, 21, 0x4e0811a1);
  OP(I, a, b, c, d, 4, 6, 0xf7537e82);
  OP(I, d, a, b, c, 11, 10, 0xbd3af235);
  OP(I, c, d, a, b, 2, 15, 0x2ad7d2bb);
  OP(I, b, c, d, a, 9, 21, 0xeb86d391);

  FINAL2(0, a);
  FINAL2(1, b);
  FINAL2(2, c);
  FINAL2(3, d);
}


#define ROTL32(a, s) (((a) << (s)) | ((a) >> (32 - (s))))
#define W0R(a) (ROTL32(W0,(a)))


static inline uint32_t myf00(uint32_t x, uint32_t y, uint32_t z) {
  return ((y ^ z) & x) ^ z;
}


static inline uint32_t myf20(uint32_t x, uint32_t y, uint32_t z) {
  return (x ^ z) ^ y;
}


static inline uint32_t myf40(uint32_t x, uint32_t y, uint32_t z) {
  return (x & z) | ((x | z) & y);
}


static inline uint32_t myf60(uint32_t x, uint32_t y, uint32_t z) {
  return myf20(x, y, z);
}


#define mystep(nn, xa, xb, xc, xd, xe, xt, input, cons) do {          \
    (xt) = (input) + myf##nn((xb), (xc), (xd)) + (cons);        \
    (xb) = ROTL32((xb), 30);              \
    (xt) += ((xe) + ROTL32((xa), 5));            \
  } while(0)


static void sha1_compress(uint32_t* hash, const uint32_t *block){
  uint32_t W[80], W0, A, B, C, D, E, T;
  uint32_t temp;
  __m128i r1, r2, *p, *o, xr1, xr2, xr3, xr4;
  SVAL mask, v1, v2, v3, v4;

  A = hash[0];
  B = hash[1];
  C = hash[2];
  D = hash[3];
  E = hash[4];
  p = (__m128i *) block;
  o = (__m128i * ) & W[0];
  r1 = *p++;
  r1 = _mm_shufflehi_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r1 = _mm_shufflelo_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r2 = _mm_slli_epi16(r1, 8);
  r1 = _mm_srli_epi16(r1, 8);
  r1 = _mm_or_si128(r1, r2);
  *o++ = r1;
  W0 = W[0];
  W[0] = 0;
  mystep(00, A, B, C, D, E, T, W0, 0x5A827999);
  mystep(00, T, A, B, C, D, E, W[1], 0x5A827999);
  mystep(00, E, T, A, B, C, D, W[2], 0x5A827999);
  mystep(00, D, E, T, A, B, C, W[3], 0x5A827999);
  r1 = *p++;
  r1 = _mm_shufflehi_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r1 = _mm_shufflelo_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r2 = _mm_slli_epi16(r1, 8);
  r1 = _mm_srli_epi16(r1, 8);
  r1 = _mm_or_si128(r1, r2);
  *o++ = r1;
  mystep(00, C, D, E, T, A, B, W[4], 0x5A827999);
  mystep(00, B, C, D, E, T, A, W[5], 0x5A827999);
  mystep(00, A, B, C, D, E, T, W[6], 0x5A827999);
  mystep(00, T, A, B, C, D, E, W[7], 0x5A827999);
  r1 = *p++;
  r1 = _mm_shufflehi_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r1 = _mm_shufflelo_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r2 = _mm_slli_epi16(r1, 8);
  r1 = _mm_srli_epi16(r1, 8);
  r1 = _mm_or_si128(r1, r2);
  *o++ = r1;
  mystep(00, E, T, A, B, C, D, W[8], 0x5A827999);
  mystep(00, D, E, T, A, B, C, W[9], 0x5A827999);
  mystep(00, C, D, E, T, A, B, W[10], 0x5A827999);
  mystep(00, B, C, D, E, T, A, W[11], 0x5A827999);
  r1 = *p++;
  r1 = _mm_shufflehi_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r1 = _mm_shufflelo_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));
  r2 = _mm_slli_epi16(r1, 8);
  r1 = _mm_srli_epi16(r1, 8);
  r1 = _mm_or_si128(r1, r2);
  *o++ = r1;

  mystep(00, A, B, C, D, E, T, W[12], 0x5A827999);
  mystep(00, T, A, B, C, D, E, W[13], 0x5A827999);
  mystep(00, E, T, A, B, C, D, W[14], 0x5A827999);
  mystep(00, D, E, T, A, B, C, W[15], 0x5A827999);

  W[16] = ROTL32((W[13] ^ W[8] ^ W[2]), 1);
  W[17] = ROTL32((W[14] ^ W[9] ^ W[3] ^ W[1]), 1);
  W[18] = ROTL32((W[15] ^ W[10] ^ W[4] ^ W[2]), 1);
  W[19] = ROTL32((W[16] ^ W[11] ^ W[5] ^ W[3]), 1);

  mystep(00, C, D, E, T, A, B, (W[16] ^ W0R(1)), 0x5A827999);
  mystep(00, B, C, D, E, T, A, W[17], 0x5A827999);
  mystep(00, A, B, C, D, E, T, W[18], 0x5A827999);
  mystep(00, T, A, B, C, D, E, W[19] ^ W0R(2), 0x5A827999);

  W[20] = ROTL32((W[17] ^ W[12] ^ W[6] ^ W[4]), 1);
  mystep(20, E, T, A, B, C, D, W[20], 0x6ED9EBA1);
  W[21] = ROTL32((W[18] ^ W[13] ^ W[7] ^ W[5]), 1);
  mystep(20, D, E, T, A, B, C, W[21], 0x6ED9EBA1);
  W[22] = ROTL32((W[19] ^ W[14] ^ W[8] ^ W[6]), 1);
  mystep(20, C, D, E, T, A, B, (W[22] ^ W0R(3)), 0x6ED9EBA1);
  W[23] = ROTL32((W[20] ^ W[15] ^ W[9] ^ W[7]), 1);
  mystep(20, B, C, D, E, T, A, W[23], 0x6ED9EBA1);
  W[24] = ROTL32((W[21] ^ W[16] ^ W[10] ^ W[8]), 1);
  mystep(20, A, B, C, D, E, T, (W[24] ^ W0R(2)), 0x6ED9EBA1);
  W[25] = ROTL32((W[22] ^ W[17] ^ W[11] ^ W[9]), 1);
  mystep(20, T, A, B, C, D, E, (W[25] ^ W0R(4)), 0x6ED9EBA1);
  W[26] = ROTL32((W[23] ^ W[18] ^ W[12] ^ W[10]), 1);
  mystep(20, E, T, A, B, C, D, W[26], 0x6ED9EBA1);
  W[27] = ROTL32((W[24] ^ W[19] ^ W[13] ^ W[11]), 1);
  mystep(20, D, E, T, A, B, C, W[27], 0x6ED9EBA1);
  W[28] = ROTL32((W[25] ^ W[20] ^ W[14] ^ W[12]), 1);
  mystep(20, C, D, E, T, A, B, (W[28] ^ W0R(5)), 0x6ED9EBA1);
  W[29] = ROTL32((W[26] ^ W[21] ^ W[15] ^ W[13]), 1);
  mystep(20, B, C, D, E, T, A, W[29], 0x6ED9EBA1);
  W[30] = ROTL32((W[27] ^ W[22] ^ W[16] ^ W[14]), 1);
  mystep(20, A, B, C, D, E, T, (W[30] ^ W0R(4) ^ W0R(2)), 0x6ED9EBA1);
  W[31] = ROTL32((W[28] ^ W[23] ^ W[17] ^ W[15]), 1);
  mystep(20, T, A, B, C, D, E, (W[31] ^ W0R(6)), 0x6ED9EBA1);
  W[32] = ROTL32((W[29] ^ W[24] ^ W[18] ^ W[16]), 1);
  mystep(20, E, T, A, B, C, D, (W[32] ^ W0R(2) ^ W0R(3)), 0x6ED9EBA1);
  W[33] = ROTL32((W[30] ^ W[25] ^ W[19] ^ W[17]), 1);
  mystep(20, D, E, T, A, B, C, W[33], 0x6ED9EBA1);
  W[34] = ROTL32((W[31] ^ W[26] ^ W[20] ^ W[18]), 1);
  mystep(20, C, D, E, T, A, B, (W[34] ^ W0R(7)), 0x6ED9EBA1);
  W[35] = ROTL32((W[32] ^ W[27] ^ W[21] ^ W[19]), 1);
  mystep(20, B, C, D, E, T, A, (W[35] ^ W0R(4)), 0x6ED9EBA1);
  W[36] = ROTL32((W[33] ^ W[28] ^ W[22] ^ W[20]), 1);
  mystep(20, A, B, C, D, E, T, (W[36] ^ W0R(4) ^ W0R(6)), 0x6ED9EBA1);
  W[37] = ROTL32((W[34] ^ W[29] ^ W[23] ^ W[21]), 1);
  mystep(20, T, A, B, C, D, E, (W[37] ^ W0R(8)), 0x6ED9EBA1);
  W[38] = ROTL32((W[35] ^ W[30] ^ W[24] ^ W[22]), 1);
  mystep(20, E, T, A, B, C, D, (W[38] ^ W0R(4)), 0x6ED9EBA1);
  W[39] = ROTL32((W[36] ^ W[31] ^ W[25] ^ W[23]), 1);
  mystep(20, D, E, T, A, B, C, W[39], 0x6ED9EBA1);

  W[40] = ROTL32((W[37] ^ W[32] ^ W[26] ^ W[24]), 1);
  mystep(40, C, D, E, T, A, B, (W[40] ^ W0R(4) ^ W0R(9)), 0x8F1BBCDC);
  W[41] = ROTL32((W[38] ^ W[33] ^ W[27] ^ W[25]), 1);
  mystep(40, B, C, D, E, T, A, W[41], 0x8F1BBCDC);
  W[42] = ROTL32((W[39] ^ W[34] ^ W[28] ^ W[26]), 1);
  mystep(40, A, B, C, D, E, T, (W[42] ^ W0R(6) ^ W0R(8)), 0x8F1BBCDC);
  W[43] = ROTL32((W[40] ^ W[35] ^ W[29] ^ W[27]), 1);
  mystep(40, T, A, B, C, D, E, (W[43] ^ W0R(10)), 0x8F1BBCDC);
  W[44] = ROTL32((W[41] ^ W[36] ^ W[30] ^ W[28]), 1);
  mystep(40, E, T, A, B, C, D, (W[44] ^ W0R(3) ^ W0R(6) ^ W0R(7)), 0x8F1BBCDC);
  W[45] = ROTL32((W[42] ^ W[37] ^ W[31] ^ W[29]), 1);
  mystep(40, D, E, T, A, B, C, W[45], 0x8F1BBCDC);
  W[46] = ROTL32((W[43] ^ W[38] ^ W[32] ^ W[30]), 1);
  mystep(40, C, D, E, T, A, B, (W[46] ^ W0R(4) ^ W0R(11)), 0x8F1BBCDC);
  W[47] = ROTL32((W[44] ^ W[39] ^ W[33] ^ W[31]), 1);
  mystep(40, B, C, D, E, T, A, (W[47] ^ W0R(4) ^ W0R(8)), 0x8F1BBCDC);
  W[48] = ROTL32((W[45] ^ W[40] ^ W[34] ^ W[32]), 1);
  mystep(40, A, B, C, D, E, T, (W[48] ^ W0R(3) ^ W0R(4) ^ W0R(5) ^ W0R(8) ^ W0R(10)), 0x8F1BBCDC);
  W[49] = ROTL32((W[46] ^ W[41] ^ W[35] ^ W[33]), 1);
  mystep(40, T, A, B, C, D, E, (W[49] ^ W0R(12)), 0x8F1BBCDC);
  W[50] = ROTL32((W[47] ^ W[42] ^ W[36] ^ W[34]), 1);
  mystep(40, E, T, A, B, C, D, (W[50] ^ W0R(8)), 0x8F1BBCDC);
  W[51] = ROTL32((W[48] ^ W[43] ^ W[37] ^ W[35]), 1);
  mystep(40, D, E, T, A, B, C, (W[51] ^ W0R(4) ^ W0R(6)), 0x8F1BBCDC);
  W[52] = ROTL32((W[49] ^ W[44] ^ W[38] ^ W[36]), 1);
  mystep(40, C, D, E, T, A, B, (W[52] ^ W0R(4) ^ W0R(8) ^ W0R(13)), 0x8F1BBCDC);
  W[53] = ROTL32((W[50] ^ W[45] ^ W[39] ^ W[37]), 1);
  mystep(40, B, C, D, E, T, A, W[53], 0x8F1BBCDC);
  W[54] = ROTL32((W[51] ^ W[46] ^ W[40] ^ W[38]), 1);
  mystep(40, A, B, C, D, E, T, (W[54] ^ W0R(7) ^ W0R(10) ^ W0R(12)), 0x8F1BBCDC);
  W[55] = ROTL32((W[52] ^ W[47] ^ W[41] ^ W[39]), 1);
  mystep(40, T, A, B, C, D, E, (W[55] ^ W0R(14)), 0x8F1BBCDC);
  W[56] = ROTL32((W[53] ^ W[48] ^ W[42] ^ W[40]), 1);
  mystep(40, E, T, A, B, C, D, (W[56] ^ W0R(4) ^ W0R(6) ^ W0R(7) ^ W0R(10) ^ W0R(11)), 0x8F1BBCDC);
  W[57] = ROTL32((W[54] ^ W[49] ^ W[43] ^ W[41]), 1);
  mystep(40, D, E, T, A, B, C, (W[57] ^ W0R(8)), 0x8F1BBCDC);
  W[58] = ROTL32((W[55] ^ W[50] ^ W[44] ^ W[42]), 1);
  mystep(40, C, D, E, T, A, B, (W[58] ^ W0R(4) ^ W0R(8) ^ W0R(15)), 0x8F1BBCDC);
  W[59] = ROTL32((W[56] ^ W[51] ^ W[45] ^ W[43]), 1);
  mystep(40, B, C, D, E, T, A, (W[59] ^ W0R(8) ^ W0R(12)), 0x8F1BBCDC);

  W[60] = ROTL32((W[57] ^ W[52] ^ W[46] ^ W[44]), 1);
  mystep(60, A, B, C, D, E, T, (W[60] ^ W0R(4) ^ W0R(7) ^ W0R(8) ^ W0R(12) ^ W0R(14)), 0xCA62C1D6);
  W[61] = ROTL32((W[58] ^ W[53] ^ W[47] ^ W[45]), 1);
  mystep(60, T, A, B, C, D, E, (W[61] ^ W0R(16)), 0xCA62C1D6);
  W[62] = ROTL32((W[59] ^ W[54] ^ W[48] ^ W[46]), 1);
  mystep(60, E, T, A, B, C, D, (W[62] ^ W0R(4) ^ W0R(6) ^ W0R(8) ^ W0R(12)), 0xCA62C1D6);
  W[63] = ROTL32((W[60] ^ W[55] ^ W[49] ^ W[47]), 1);
  mystep(60, D, E, T, A, B, C, (W[63] ^ W0R(8)), 0xCA62C1D6);
  W[64] = ROTL32((W[61] ^ W[56] ^ W[50] ^ W[48]), 1);
  mystep(60, C, D, E, T, A, B, (W[64] ^ W0R(4) ^ W0R(6) ^ W0R(7) ^ W0R(8) ^ W0R(12) ^ W0R(17)), 0xCA62C1D6);
  W[65] = ROTL32((W[62] ^ W[57] ^ W[51] ^ W[49]), 1);
  mystep(60, B, C, D, E, T, A, W[65], 0xCA62C1D6);
  W[66] = ROTL32((W[63] ^ W[58] ^ W[52] ^ W[50]), 1);
  mystep(60, A, B, C, D, E, T, (W[66] ^ W0R(14) ^ W0R(16)), 0xCA62C1D6);
  W[67] = ROTL32((W[64] ^ W[59] ^ W[53] ^ W[51]), 1);
  mystep(60, T, A, B, C, D, E, (W[67] ^ W0R(8) ^ W0R(18)), 0xCA62C1D6);
  W[68] = ROTL32((W[65] ^ W[60] ^ W[54] ^ W[52]), 1);
  mystep(60, E, T, A, B, C, D, (W[68] ^ W0R(11) ^ W0R(14) ^ W0R(15)), 0xCA62C1D6);
  W[69] = ROTL32((W[66] ^ W[61] ^ W[55] ^ W[53]), 1);
  mystep(60, D, E, T, A, B, C, W[69], 0xCA62C1D6);
  W[70] = ROTL32((W[67] ^ W[62] ^ W[56] ^ W[54]), 1);
  mystep(60, C, D, E, T, A, B, (W[70] ^ W0R(12) ^ W0R(19)), 0xCA62C1D6);
  W[71] = ROTL32((W[68] ^ W[63] ^ W[57] ^ W[55]), 1);
  mystep(60, B, C, D, E, T, A, (W[71] ^ W0R(12) ^ W0R(16)), 0xCA62C1D6);
  W[72] = ROTL32((W[69] ^ W[64] ^ W[58] ^ W[56]), 1);
  mystep(60, A, B, C, D, E, T, (W[72] ^ W0R(5) ^ W0R(11) ^ W0R(12) ^ W0R(13) ^ W0R(16) ^ W0R(18)), 0xCA62C1D6);
  W[73] = ROTL32((W[70] ^ W[65] ^ W[59] ^ W[57]), 1);
  mystep(60, T, A, B, C, D, E, (W[73] ^ W0R(20)), 0xCA62C1D6);
  W[74] = ROTL32((W[71] ^ W[66] ^ W[60] ^ W[58]), 1);
  mystep(60, E, T, A, B, C, D, (W[74] ^ W0R(8) ^ W0R(16)), 0xCA62C1D6);
  W[75] = ROTL32((W[72] ^ W[67] ^ W[61] ^ W[59]), 1);
  mystep(60, D, E, T, A, B, C, (W[75] ^ W0R(6) ^ W0R(12) ^ W0R(14)), 0xCA62C1D6);
  W[76] = ROTL32((W[73] ^ W[68] ^ W[62] ^ W[60]), 1);
  mystep(60, C, D, E, T, A, B, (W[76] ^ W0R(7) ^ W0R(8) ^ W0R(12) ^ W0R(16) ^ W0R(21)), 0xCA62C1D6);
  W[77] = ROTL32((W[74] ^ W[69] ^ W[63] ^ W[61]), 1);
  mystep(60, B, C, D, E, T, A, W[77], 0xCA62C1D6);
  W[78] = ROTL32((W[75] ^ W[70] ^ W[64] ^ W[62]), 1);
  mystep(60, A, B, C, D, E, T, (W[78] ^ W0R(7) ^ W0R(8) ^ W0R(15) ^ W0R(18) ^ W0R(20)), 0xCA62C1D6);
  W[79] = ROTL32((W[76] ^ W[71] ^ W[65] ^ W[63]), 1);
  mystep(60, T, A, B, C, D, E, (W[79] ^ W0R(8) ^ W0R(22)), 0xCA62C1D6);



  hash[0] += E;
  hash[1] += T;
  hash[2] += A;
  hash[3] += B;
  hash[4] += C;
}
static void sha1_compress_orig(uint32_t * hash, const uint32_t *block){
  int t;                 /* Loop counter */
  uint32_t temp;              /* Temporary word value */
  uint32_t W[80];             /* Word sequence */
  uint32_t A, B, C, D, E;     /* Word buffers */

  /* initialize the first 16 words in the array W */
  for (t = 0; t < 16; t++) {
    /* note: it is much faster to apply be2me here, then using be32_ copy */
    W[t] = bswap_32(block[t]);
  }

  /* initialize the rest */
  for (t = 16; t < 80; t++) {
    W[t] = ROTL32(W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16], 1);
  }

  A = hash[0];
  B = hash[1];
  C = hash[2];
  D = hash[3];
  E = hash[4];

  for (t = 0; t < 20; t++) {
    /* the following is faster than ((B & C) | ((~B) & D)) */
    temp = ROTL32(A, 5) + (((C ^ D) & B) ^ D) + E + W[t] + 0x5A827999;
    E = D;
    D = C;
    C = ROTL32(B, 30);
    B = A;
    A = temp;
  }

  for (t = 20; t < 40; t++) {
    temp = ROTL32(A, 5) + (B ^ C ^ D) + E + W[t] + 0x6ED9EBA1;
    E = D;
    D = C;
    C = ROTL32(B, 30);
    B = A;
    A = temp;
  }

  for (t = 40; t < 60; t++) {
    temp = ROTL32(A, 5) + ((B & C) | (B & D) | (C & D)) + E + W[t] + 0x8F1BBCDC;
    E = D;
    D = C;
    C = ROTL32(B, 30);
    B = A;
    A = temp;
  }

  for (t = 60; t < 80; t++) {
    temp = ROTL32(A, 5) + (B ^ C ^ D) + E + W[t] + 0xCA62C1D6;
    E = D;
    D = C;
    C = ROTL32(B, 30);
    B = A;
    A = temp;
  }

  hash[0] += A;
  hash[1] += B;
  hash[2] += C;
  hash[3] += D;
  hash[4] += E;
}


#include <stddef.h>

#define SHA1_HASH_SIZE  (5)
#define SHA1_STEP_SIZE  (16)


/* this code is public domain.
 *
 * dean gaudet <dean@arctic.org>
 *
 * this code was inspired by this paper:
 *
 *     SHA: A Design for Parallel Architectures?
 *     Antoon Bosselaers, Ren´e Govaerts and Joos Vandewalle
 *     <http://www.esat.kuleuven.ac.be/~cosicart/pdf/AB-9700.pdf>
 *
 * more information available on this implementation here:
 *
 * 	http://arctic.org/~dean/crypto/sha1.html
 *
 * version: 2
 */


typedef union {
  uint32_t u32[4];
  __m128i u128;
} v4si __attribute__((aligned(16)));

static const v4si K00_19 = {.u32 = {0x5a827999, 0x5a827999, 0x5a827999, 0x5a827999}};
static const v4si K20_39 = {.u32 = {0x6ed9eba1, 0x6ed9eba1, 0x6ed9eba1, 0x6ed9eba1}};
static const v4si K40_59 = {.u32 = {0x8f1bbcdc, 0x8f1bbcdc, 0x8f1bbcdc, 0x8f1bbcdc}};
static const v4si K60_79 = {.u32 = {0xca62c1d6, 0xca62c1d6, 0xca62c1d6, 0xca62c1d6}};

#define LOC_UNALIGNED 1
#if LOC_UNALIGNED
#define load(p)  _mm_loadu_si128(p)
#else
#define load(p) (*p)
#endif


/*
	the first 16 bytes only need byte swapping

	prepared points to 4x uint32_t, 16-byte aligned

	W points to the 4 dwords which need preparing --
	and is overwritten with the swapped bytes
*/
#define prep00_15(prep, W)  do {          \
    __m128i r1, r2;            \
                  \
    r1 = (W);            \
    if (1) {            \
    r1 = _mm_shufflehi_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));  \
    r1 = _mm_shufflelo_epi16(r1, _MM_SHUFFLE(2, 3, 0, 1));  \
    r2 = _mm_slli_epi16(r1, 8);        \
    r1 = _mm_srli_epi16(r1, 8);        \
    r1 = _mm_or_si128(r1, r2);        \
    (W) = r1;            \
    }              \
    (prep).u128 = _mm_add_epi32(K00_19.u128, r1);    \
  } while(0)



/*
	for each multiple of 4, t, we want to calculate this:

	W[t+0] = rol(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16], 1);
	W[t+1] = rol(W[t-2] ^ W[t-7] ^ W[t-13] ^ W[t-15], 1);
	W[t+2] = rol(W[t-1] ^ W[t-6] ^ W[t-12] ^ W[t-14], 1);
	W[t+3] = rol(W[t]   ^ W[t-5] ^ W[t-11] ^ W[t-13], 1);

	we'll actually calculate this:

	W[t+0] = rol(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16], 1);
	W[t+1] = rol(W[t-2] ^ W[t-7] ^ W[t-13] ^ W[t-15], 1);
	W[t+2] = rol(W[t-1] ^ W[t-6] ^ W[t-12] ^ W[t-14], 1);
	W[t+3] = rol(  0    ^ W[t-5] ^ W[t-11] ^ W[t-13], 1);
	W[t+3] ^= rol(W[t+0], 1);

	the parameters are:

	W0 = &W[t-16];
	W1 = &W[t-12];
	W2 = &W[t- 8];
	W3 = &W[t- 4];

	and on output:
		prepared = W0 + K
		W0 = W[t]..W[t+3]
*/

/* note that there is a step here where i want to do a rol by 1, which
 * normally would look like this:
 *
 * r1 = psrld r0,$31
 * r0 = pslld r0,$1
 * r0 = por r0,r1
 *
 * but instead i do this:
 *
 * r1 = pcmpltd r0,zero
 * r0 = paddd r0,r0
 * r0 = psub r0,r1
 *
 * because pcmpltd and paddd are availabe in both MMX units on
 * efficeon, pentium-m, and opteron but shifts are available in
 * only one unit.
 */
#define prep(prep, XW0, XW1, XW2, XW3, K) do {          \
    __m128i r0, r1, r2, r3;            \
                    \
    /* load W[t-4] 16-byte aligned, and shift */      \
    r3 = _mm_srli_si128((XW3), 4);          \
    r0 = (XW0);              \
    /* get high 64-bits of XW0 into low 64-bits */      \
    r1 = _mm_shuffle_epi32((XW0), _MM_SHUFFLE(1,0,3,2));    \
    /* load high 64-bits of r1 */          \
    r1 = _mm_unpacklo_epi64(r1, (XW1));        \
    r2 = (XW2);              \
                    \
    r0 = _mm_xor_si128(r1, r0);          \
    r2 = _mm_xor_si128(r3, r2);          \
    r0 = _mm_xor_si128(r2, r0);          \
    /* unrotated W[t]..W[t+2] in r0 ... still need W[t+3] */  \
                    \
    r2 = _mm_slli_si128(r0, 12);          \
    r1 = _mm_cmplt_epi32(r0, _mm_setzero_si128());      \
    r0 = _mm_add_epi32(r0, r0);  /* shift left by 1 */    \
    r0 = _mm_sub_epi32(r0, r1);  /* r0 has W[t]..W[t+2] */  \
                    \
    r3 = _mm_srli_epi32(r2, 30);          \
    r2 = _mm_slli_epi32(r2, 2);          \
                    \
    r0 = _mm_xor_si128(r0, r3);          \
    r0 = _mm_xor_si128(r0, r2);  /* r0 now has W[t+3] */    \
                    \
    (XW0) = r0;              \
    (prep).u128 = _mm_add_epi32(r0, (K).u128);      \
  } while(0)


static inline uint32_t rol(uint32_t src, uint32_t amt){
  /* gcc and icc appear to turn this into a rotate */
  return (src << amt) | (src >> (32 - amt));
}


static inline uint32_t f00_19(uint32_t x, uint32_t y, uint32_t z) {
  /* FIPS 180-2 says this: (x & y) ^ (~x & z)
   * but we can calculate it in fewer steps.
   */
  return ((y ^z) & x) ^z;
}


static inline uint32_t f20_39(uint32_t x, uint32_t y, uint32_t z) {
  return (x ^z) ^y;
}


static inline uint32_t f40_59(uint32_t x, uint32_t y, uint32_t z) {
  /* FIPS 180-2 says this: (x & y) ^ (x & z) ^ (y & z)
   * but we can calculate it in fewer steps.
   */
  return (x &z) | ((x | z) & y);
}


static inline uint32_t f60_79(uint32_t x, uint32_t y, uint32_t z) {
  return f20_39(x, y, z);
}


#define step(nn_mm, xa, xb, xc, xd, xe, xt, input) do {          \
    (xt) = (input) + f##nn_mm((xb), (xc), (xd));        \
    (xb) = rol((xb), 30);              \
    (xt) += ((xe) + rol((xa), 5));            \
  } while(0)

void sha1_step(uint32_t * H, const uint32_t *inputu) {
  const __m128i *input = (const __m128i *) inputu;
  __m128i W0, W1, W2, W3;
  v4si prep0, prep1, prep2;
  uint32_t a, b, c, d, e, t;

  a = H[0];
  b = H[1];
  c = H[2];
  d = H[3];
  e = H[4];

  /* i've tried arranging the SSE2 code to be 4, 8, 12, and 16
   * steps ahead of the integer code.  12 steps ahead seems
   * to produce the best performance. -dean
   */
  W0 = load(&input[0]);
  prep00_15(prep0, W0);        /* prepare for 00 through 03 */
  W1 = load(&input[1]);
  prep00_15(prep1, W1);        /* prepare for 04 through 07 */
  W2 = load(&input[2]);
  prep00_15(prep2, W2);        /* prepare for 08 through 11 */
  W3 = load(&input[3]);
  step(00_19, a, b, c, d, e, t, prep0.u32[0]);  /* 00 */
  step(00_19, t, a, b, c, d, e, prep0.u32[1]);  /* 01 */
  step(00_19, e, t, a, b, c, d, prep0.u32[2]);  /* 02 */
  step(00_19, d, e, t, a, b, c, prep0.u32[3]);  /* 03 */
  prep00_15(prep0, W3);
  step(00_19, c, d, e, t, a, b, prep1.u32[0]);  /* 04 */
  step(00_19, b, c, d, e, t, a, prep1.u32[1]);  /* 05 */
  step(00_19, a, b, c, d, e, t, prep1.u32[2]);  /* 06 */
  step(00_19, t, a, b, c, d, e, prep1.u32[3]);  /* 07 */
  prep(prep1, W0, W1, W2, W3, K00_19);    /* prepare for 16 through 19 */
  step(00_19, e, t, a, b, c, d, prep2.u32[0]);  /* 08 */
  step(00_19, d, e, t, a, b, c, prep2.u32[1]);  /* 09 */
  step(00_19, c, d, e, t, a, b, prep2.u32[2]);  /* 10 */
  step(00_19, b, c, d, e, t, a, prep2.u32[3]);  /* 11 */
  prep(prep2, W1, W2, W3, W0, K20_39);    /* prepare for 20 through 23 */
  step(00_19, a, b, c, d, e, t, prep0.u32[0]);  /* 12 */
  step(00_19, t, a, b, c, d, e, prep0.u32[1]);  /* 13 */
  step(00_19, e, t, a, b, c, d, prep0.u32[2]);  /* 14 */
  step(00_19, d, e, t, a, b, c, prep0.u32[3]);  /* 15 */
  prep(prep0, W2, W3, W0, W1, K20_39);
  step(00_19, c, d, e, t, a, b, prep1.u32[0]);  /* 16 */
  step(00_19, b, c, d, e, t, a, prep1.u32[1]);  /* 17 */
  step(00_19, a, b, c, d, e, t, prep1.u32[2]);  /* 18 */
  step(00_19, t, a, b, c, d, e, prep1.u32[3]);  /* 19 */

  prep(prep1, W3, W0, W1, W2, K20_39);
  step(20_39, e, t, a, b, c, d, prep2.u32[0]);  /* 20 */
  step(20_39, d, e, t, a, b, c, prep2.u32[1]);  /* 21 */
  step(20_39, c, d, e, t, a, b, prep2.u32[2]);  /* 22 */
  step(20_39, b, c, d, e, t, a, prep2.u32[3]);  /* 23 */
  prep(prep2, W0, W1, W2, W3, K20_39);
  step(20_39, a, b, c, d, e, t, prep0.u32[0]);  /* 24 */
  step(20_39, t, a, b, c, d, e, prep0.u32[1]);  /* 25 */
  step(20_39, e, t, a, b, c, d, prep0.u32[2]);  /* 26 */
  step(20_39, d, e, t, a, b, c, prep0.u32[3]);  /* 27 */
  prep(prep0, W1, W2, W3, W0, K20_39);
  step(20_39, c, d, e, t, a, b, prep1.u32[0]);  /* 28 */
  step(20_39, b, c, d, e, t, a, prep1.u32[1]);  /* 29 */
  step(20_39, a, b, c, d, e, t, prep1.u32[2]);  /* 30 */
  step(20_39, t, a, b, c, d, e, prep1.u32[3]);  /* 31 */
  prep(prep1, W2, W3, W0, W1, K40_59);
  step(20_39, e, t, a, b, c, d, prep2.u32[0]);  /* 32 */
  step(20_39, d, e, t, a, b, c, prep2.u32[1]);  /* 33 */
  step(20_39, c, d, e, t, a, b, prep2.u32[2]);  /* 34 */
  step(20_39, b, c, d, e, t, a, prep2.u32[3]);  /* 35 */
  prep(prep2, W3, W0, W1, W2, K40_59);
  step(20_39, a, b, c, d, e, t, prep0.u32[0]);  /* 36 */
  step(20_39, t, a, b, c, d, e, prep0.u32[1]);  /* 37 */
  step(20_39, e, t, a, b, c, d, prep0.u32[2]);  /* 38 */
  step(20_39, d, e, t, a, b, c, prep0.u32[3]);  /* 39 */

  prep(prep0, W0, W1, W2, W3, K40_59);
  step(40_59, c, d, e, t, a, b, prep1.u32[0]);  /* 40 */
  step(40_59, b, c, d, e, t, a, prep1.u32[1]);  /* 41 */
  step(40_59, a, b, c, d, e, t, prep1.u32[2]);  /* 42 */
  step(40_59, t, a, b, c, d, e, prep1.u32[3]);  /* 43 */
  prep(prep1, W1, W2, W3, W0, K40_59);
  step(40_59, e, t, a, b, c, d, prep2.u32[0]);  /* 44 */
  step(40_59, d, e, t, a, b, c, prep2.u32[1]);  /* 45 */
  step(40_59, c, d, e, t, a, b, prep2.u32[2]);  /* 46 */
  step(40_59, b, c, d, e, t, a, prep2.u32[3]);  /* 47 */
  prep(prep2, W2, W3, W0, W1, K40_59);
  step(40_59, a, b, c, d, e, t, prep0.u32[0]);  /* 48 */
  step(40_59, t, a, b, c, d, e, prep0.u32[1]);  /* 49 */
  step(40_59, e, t, a, b, c, d, prep0.u32[2]);  /* 50 */
  step(40_59, d, e, t, a, b, c, prep0.u32[3]);  /* 51 */
  prep(prep0, W3, W0, W1, W2, K60_79);
  step(40_59, c, d, e, t, a, b, prep1.u32[0]);  /* 52 */
  step(40_59, b, c, d, e, t, a, prep1.u32[1]);  /* 53 */
  step(40_59, a, b, c, d, e, t, prep1.u32[2]);  /* 54 */
  step(40_59, t, a, b, c, d, e, prep1.u32[3]);  /* 55 */
  prep(prep1, W0, W1, W2, W3, K60_79);
  step(40_59, e, t, a, b, c, d, prep2.u32[0]);  /* 56 */
  step(40_59, d, e, t, a, b, c, prep2.u32[1]);  /* 57 */
  step(40_59, c, d, e, t, a, b, prep2.u32[2]);  /* 58 */
  step(40_59, b, c, d, e, t, a, prep2.u32[3]);  /* 59 */

  prep(prep2, W1, W2, W3, W0, K60_79);
  step(60_79, a, b, c, d, e, t, prep0.u32[0]);  /* 60 */
  step(60_79, t, a, b, c, d, e, prep0.u32[1]);  /* 61 */
  step(60_79, e, t, a, b, c, d, prep0.u32[2]);  /* 62 */
  step(60_79, d, e, t, a, b, c, prep0.u32[3]);  /* 63 */
  prep(prep0, W2, W3, W0, W1, K60_79);
  step(60_79, c, d, e, t, a, b, prep1.u32[0]);  /* 64 */
  step(60_79, b, c, d, e, t, a, prep1.u32[1]);  /* 65 */
  step(60_79, a, b, c, d, e, t, prep1.u32[2]);  /* 66 */
  step(60_79, t, a, b, c, d, e, prep1.u32[3]);  /* 67 */
  prep(prep1, W3, W0, W1, W2, K60_79);
  step(60_79, e, t, a, b, c, d, prep2.u32[0]);  /* 68 */
  step(60_79, d, e, t, a, b, c, prep2.u32[1]);  /* 69 */
  step(60_79, c, d, e, t, a, b, prep2.u32[2]);  /* 70 */
  step(60_79, b, c, d, e, t, a, prep2.u32[3]);  /* 71 */

  /* no more input to prepare */
  step(60_79, a, b, c, d, e, t, prep0.u32[0]);  /* 72 */
  step(60_79, t, a, b, c, d, e, prep0.u32[1]);  /* 73 */
  step(60_79, e, t, a, b, c, d, prep0.u32[2]);  /* 74 */
  step(60_79, d, e, t, a, b, c, prep0.u32[3]);  /* 75 */
  /* no more input to prepare */
  step(60_79, c, d, e, t, a, b, prep1.u32[0]);  /* 76 */
  step(60_79, b, c, d, e, t, a, prep1.u32[1]);  /* 77 */
  step(60_79, a, b, c, d, e, t, prep1.u32[2]);  /* 78 */
  step(60_79, t, a, b, c, d, e, prep1.u32[3]);  /* 79 */
  /* e, t, a, b, c, d */
  H[0] += e;
  H[1] += t;
  H[2] += a;
  H[3] += b;
  H[4] += c;
}
#ifdef MDX_BIT32
#define SHAUP sha1_step
#else
/* Runtime dispatch: SHA-NI > SSSE3 assembly > C fallback */
static void sha1_block_init(uint32_t *hash, uint32_t *block);
static void (*sha1_block_fn)(uint32_t *, uint32_t *) = sha1_block_init;

static void sha1_block_init(uint32_t *hash, uint32_t *block) {
    unsigned int eax, ebx, ecx, edx;
    sha1_block_fn = sha1_update_intel;   /* default: existing SSSE3/sha1_step asm dispatch */
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1u << 29)))
        sha1_block_fn = (void (*)(uint32_t *, uint32_t *))sha1_compress_shani;
    sha1_block_fn(hash, block);
}
#define SHAUP sha1_block_fn
#endif

void mysha1(unsigned char *message, int len, uint32_t *hash) {
  SVAL block[4];
  __m128i *p, r1, r2;
  int i, rem;
  uint8_t *byteBlock;

  hash[0] = 0x67452301;
  hash[1] = 0xefcdab89;
  hash[2] = 0x98badcfe;
  hash[3] = 0x10325476;
  hash[4] = 0xc3d2e1f0;

  for (i = 0; i + 64 <= len; i += 64)
    SHAUP(hash, (uint32_t * )(message + i));


  rem = len - i;
  r1 = _mm_setzero_si128();
  p = &block[0].sse;
  p[0] = r1;
  p[1] = r1;
  p[2] = r1;
  p[3] = r1;
  memcpy(&block[0].sse, message + i, rem);

  byteBlock = (uint8_t * ) & block[0].sse;

  byteBlock[rem] = 0x80;
  rem++;
  if (64 - rem < 8) {
    SHAUP(hash, (uint32_t * ) & block[0].sse);

    r1 = _mm_setzero_si128();
    p[0] = r1;
    p[1] = r1;
    p[2] = r1;
    p[3] = r1;
  }
  block[3].longs[1] = bswap_64((uint64_t) len << 3);
  SHAUP(hash, (uint32_t * ) & block[0].sse);
  hash[0] = bswap_32(hash[0]);
  hash[1] = bswap_32(hash[1]);
  hash[2] = bswap_32(hash[2]);
  hash[3] = bswap_32(hash[3]);
  hash[4] = bswap_32(hash[4]);
}
extern int checkhashbb(union HashU *curin, int len, char *salt, struct job *job);
extern int NoMarkSalt;

void procsaltbb(__m128i *SSEBUF, struct job *job, int pcnt, char *sbuf[], int myiter) {
  __m128i a, b, c, d;
  union HashU curin;
  const SVAL *X = (SVAL *) SSEBUF;
  int x, y;

  if (pcnt <= 0 || pcnt > 4) return;
  curin.x[0] = _mm_setzero_si128();
  while (myiter--) {
    a = _mm_set1_epi32(AA);
    b = _mm_set1_epi32(BB);
    c = _mm_set1_epi32(CC);
    d = _mm_set1_epi32(DD);

    OP(F, a, b, c, d, 0, 7, 0xd76aa478);
    OP(F, d, a, b, c, 1, 12, 0xe8c7b756);
    OP(F, c, d, a, b, 2, 17, 0x242070db);
    OP(F, b, c, d, a, 3, 22, 0xc1bdceee);
    OP(F, a, b, c, d, 4, 7, 0xf57c0faf);
    OP(F, d, a, b, c, 5, 12, 0x4787c62a);
    OP(F, c, d, a, b, 6, 17, 0xa8304613);
    OP(F, b, c, d, a, 7, 22, 0xfd469501);
    OP(F, a, b, c, d, 8, 7, 0x698098d8);
    OP(F, d, a, b, c, 9, 12, 0x8b44f7af);
    OP(F, c, d, a, b, 10, 17, 0xffff5bb1);
    OP(F, b, c, d, a, 11, 22, 0x895cd7be);
    OP(F, a, b, c, d, 12, 7, 0x6b901122);
    OP(F, d, a, b, c, 13, 12, 0xfd987193);
    OP(F, c, d, a, b, 14, 17, 0xa679438e);
    OP(F, b, c, d, a, 15, 22, 0x49b40821);
    /* Round 2. */
    OP(G, a, b, c, d, 1, 5, 0xf61e2562);
    OP(G, d, a, b, c, 6, 9, 0xc040b340);
    OP(G, c, d, a, b, 11, 14, 0x265e5a51);
    OP(G, b, c, d, a, 0, 20, 0xe9b6c7aa);
    OP(G, a, b, c, d, 5, 5, 0xd62f105d);
    OP(G, d, a, b, c, 10, 9, 0x02441453);
    OP(G, c, d, a, b, 15, 14, 0xd8a1e681);
    OP(G, b, c, d, a, 4, 20, 0xe7d3fbc8);
    OP(G, a, b, c, d, 9, 5, 0x21e1cde6);
    OP(G, d, a, b, c, 14, 9, 0xc33707d6);
    OP(G, c, d, a, b, 3, 14, 0xf4d50d87);
    OP(G, b, c, d, a, 8, 20, 0x455a14ed);
    OP(G, a, b, c, d, 13, 5, 0xa9e3e905);
    OP(G, d, a, b, c, 2, 9, 0xfcefa3f8);
    OP(G, c, d, a, b, 7, 14, 0x676f02d9);
    OP(G, b, c, d, a, 12, 20, 0x8d2a4c8a);
    /* Round 3. */
    OP(H, a, b, c, d, 5, 4, 0xfffa3942);
    OP(H, d, a, b, c, 8, 11, 0x8771f681);
    OP(H, c, d, a, b, 11, 16, 0x6d9d6122);
    OP(H, b, c, d, a, 14, 23, 0xfde5380c);
    OP(H, a, b, c, d, 1, 4, 0xa4beea44);
    OP(H, d, a, b, c, 4, 11, 0x4bdecfa9);
    OP(H, c, d, a, b, 7, 16, 0xf6bb4b60);
    OP(H, b, c, d, a, 10, 23, 0xbebfbc70);
    OP(H, a, b, c, d, 13, 4, 0x289b7ec6);
    OP(H, d, a, b, c, 0, 11, 0xeaa127fa);
    OP(H, c, d, a, b, 3, 16, 0xd4ef3085);
    OP(H, b, c, d, a, 6, 23, 0x04881d05);
    OP(H, a, b, c, d, 9, 4, 0xd9d4d039);
    OP(H, d, a, b, c, 12, 11, 0xe6db99e5);
    OP(H, c, d, a, b, 15, 16, 0x1fa27cf8);
    OP(H, b, c, d, a, 2, 23, 0xc4ac5665);
    /* Round 4. */
    OP(I, a, b, c, d, 0, 6, 0xf4292244);
    OP(I, d, a, b, c, 7, 10, 0x432aff97);
    OP(I, c, d, a, b, 14, 15, 0xab9423a7);
    OP(I, b, c, d, a, 5, 21, 0xfc93a039);
    OP(I, a, b, c, d, 12, 6, 0x655b59c3);
    OP(I, d, a, b, c, 3, 10, 0x8f0ccc92);
    OP(I, c, d, a, b, 10, 15, 0xffeff47d);
    OP(I, b, c, d, a, 1, 21, 0x85845dd1);
    OP(I, a, b, c, d, 8, 6, 0x6fa87e4f);
    OP(I, d, a, b, c, 15, 10, 0xfe2ce6e0);
    OP(I, c, d, a, b, 6, 15, 0xa3014314);
    OP(I, b, c, d, a, 13, 21, 0x4e0811a1);
    OP(I, a, b, c, d, 4, 6, 0xf7537e82);
    OP(I, d, a, b, c, 11, 10, 0xbd3af235);
    OP(I, c, d, a, b, 2, 15, 0x2ad7d2bb);
    OP(I, b, c, d, a, 9, 21, 0xeb86d391);

    SSEBUF[0] = ADD(a, SET1(AA));
    SSEBUF[1] = ADD(b, SET1(BB));
    SSEBUF[2] = ADD(c, SET1(CC));
    SSEBUF[3] = ADD(d, SET1(DD));

  }
  for (x = 0; x < pcnt; x++) {
    curin.i[0] = X[0].words[x];
    curin.i[1] = X[1].words[x];
    curin.i[2] = X[2].words[x];
    curin.i[3] = X[3].words[x];
    if (checkhashbb(&curin, 32, sbuf[x], job) && NoMarkSalt == 0)
      sbuf[x][3] = 0xfe;
  }
}

#endif

#ifdef POWERPC
#define mysha1 SHA1
extern void SHA1(unsigned char *cur, int len,uint32_t *hash);
#endif
#if defined(ARM) && ARM >= 8 && defined(__aarch64__)
/* ARM Crypto Extensions dispatch */
static void (*sha1_arm_fn)(uint32_t *, const uint32_t *);
static void (*sha256_arm_fn)(uint32_t *, const uint32_t *);
#ifdef HAVE_SHA512_CE
static void (*sha512_arm_fn)(uint64_t *, const uint64_t *);
#endif
void arm_ce_detect(void) {
#ifdef MACOSX
    sha1_arm_fn = sha1_compress_armce;
    sha256_arm_fn = sha256_compress_armce;
#ifdef HAVE_SHA512_CE
    sha512_arm_fn = sha512_compress_armce;
#endif
#elif defined(_WIN32)
    if (IsProcessorFeaturePresent(30)) { /* PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE */
        sha1_arm_fn = sha1_compress_armce;
        sha256_arm_fn = sha256_compress_armce;
    }
#ifdef HAVE_SHA512_CE
    if (IsProcessorFeaturePresent(65)) /* PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE */
        sha512_arm_fn = sha512_compress_armce;
#endif
#else
    unsigned long hw = getauxval(16); /* AT_HWCAP */
    if (hw & (1 << 5))  sha1_arm_fn   = sha1_compress_armce;   /* HWCAP_SHA1 */
    if (hw & (1 << 6))  sha256_arm_fn = sha256_compress_armce;  /* HWCAP_SHA2 */
#ifdef HAVE_SHA512_CE
    if (hw & (1UL << 21)) sha512_arm_fn = sha512_compress_armce; /* HWCAP_SHA512 */
#endif
#endif
    if (sha1_arm_fn || sha256_arm_fn
#ifdef HAVE_SHA512_CE
        || sha512_arm_fn
#endif
    ) {
        fprintf(stderr, "ARM CE acceleration enabled:");
        if (sha1_arm_fn)   fprintf(stderr, " SHA1");
        if (sha256_arm_fn) fprintf(stderr, " SHA256");
#ifdef HAVE_SHA512_CE
        if (sha512_arm_fn) fprintf(stderr, " SHA512");
#endif
        fprintf(stderr, "\n");
    }
}

void mysha1(unsigned char *message, int len, uint32_t *hash) {
  uint8_t block[64];
  int i, rem;

  if (!sha1_arm_fn) {
      extern void SHA1(unsigned char *, int, uint32_t *);
      SHA1(message, len, hash);
      return;
  }

  hash[0] = 0x67452301;
  hash[1] = 0xefcdab89;
  hash[2] = 0x98badcfe;
  hash[3] = 0x10325476;
  hash[4] = 0xc3d2e1f0;

  for (i = 0; i + 64 <= len; i += 64)
    sha1_arm_fn(hash, (const uint32_t *)(message + i));

  rem = len - i;
  memset(block, 0, 64);
  memcpy(block, message + i, rem);
  block[rem] = 0x80;
  if (64 - rem - 1 < 8) {
    sha1_arm_fn(hash, (const uint32_t *)block);
    memset(block, 0, 64);
  }
  {
    uint64_t bits = (uint64_t)len << 3;
    block[56] = bits >> 56; block[57] = bits >> 48;
    block[58] = bits >> 40; block[59] = bits >> 32;
    block[60] = bits >> 24; block[61] = bits >> 16;
    block[62] = bits >> 8;  block[63] = bits;
  }
  sha1_arm_fn(hash, (const uint32_t *)block);
  hash[0] = bswap_32(hash[0]);
  hash[1] = bswap_32(hash[1]);
  hash[2] = bswap_32(hash[2]);
  hash[3] = bswap_32(hash[3]);
  hash[4] = bswap_32(hash[4]);
}

void mysha256(char *cur, int len, unsigned char *dest) {
  uint8_t block[64];
  uint32_t *hash = (uint32_t *)dest;
  int i, rem;

  if (!sha256_arm_fn) {
      extern void SHA256(char *, int, unsigned char *);
      SHA256(cur, len, dest);
      return;
  }

  hash[0] = 0x6a09e667; hash[1] = 0xbb67ae85;
  hash[2] = 0x3c6ef372; hash[3] = 0xa54ff53a;
  hash[4] = 0x510e527f; hash[5] = 0x9b05688c;
  hash[6] = 0x1f83d9ab; hash[7] = 0x5be0cd19;

  for (i = 0; i + 64 <= len; i += 64)
    sha256_arm_fn(hash, (const uint32_t *)((unsigned char *)cur + i));

  rem = len - i;
  memset(block, 0, 64);
  memcpy(block, (unsigned char *)cur + i, rem);
  block[rem] = 0x80;
  if (64 - rem - 1 < 8) {
    sha256_arm_fn(hash, (const uint32_t *)block);
    memset(block, 0, 64);
  }
  {
    uint64_t bits = (uint64_t)len << 3;
    block[56] = bits >> 56; block[57] = bits >> 48;
    block[58] = bits >> 40; block[59] = bits >> 32;
    block[60] = bits >> 24; block[61] = bits >> 16;
    block[62] = bits >> 8;  block[63] = bits;
  }
  sha256_arm_fn(hash, (const uint32_t *)block);
  for (i = 0; i < 8; i++) hash[i] = bswap_32(hash[i]);
}

void mysha512(char *cur, int len, unsigned char *dest) {
#ifdef HAVE_SHA512_CE
  uint8_t block[128];
  uint64_t *hash = (uint64_t *)dest;
  int i, rem;

  if (!sha512_arm_fn) {
      extern void SHA512(char *, int, unsigned char *);
      SHA512(cur, len, dest);
      return;
  }

  hash[0] = 0x6a09e667f3bcc908ULL; hash[1] = 0xbb67ae8584caa73bULL;
  hash[2] = 0x3c6ef372fe94f82bULL; hash[3] = 0xa54ff53a5f1d36f1ULL;
  hash[4] = 0x510e527fade682d1ULL; hash[5] = 0x9b05688c2b3e6c1fULL;
  hash[6] = 0x1f83d9abfb41bd6bULL; hash[7] = 0x5be0cd19137e2179ULL;

  for (i = 0; i + 128 <= len; i += 128)
    sha512_arm_fn(hash, (const uint64_t *)((unsigned char *)cur + i));

  rem = len - i;
  memset(block, 0, 128);
  memcpy(block, (unsigned char *)cur + i, rem);
  block[rem] = 0x80;
  if (128 - rem - 1 < 16) {
    sha512_arm_fn(hash, (const uint64_t *)block);
    memset(block, 0, 128);
  }
  {
    uint64_t bits = (uint64_t)len << 3;
    /* big-endian 128-bit length: upper 64 bits = 0 for messages < 2^64 bits */
    memset(block + 112, 0, 8);
    block[120] = bits >> 56; block[121] = bits >> 48;
    block[122] = bits >> 40; block[123] = bits >> 32;
    block[124] = bits >> 24; block[125] = bits >> 16;
    block[126] = bits >> 8;  block[127] = bits;
  }
  sha512_arm_fn(hash, (const uint64_t *)block);
  for (i = 0; i < 8; i++) hash[i] = bswap_64(hash[i]);
#else
  extern void SHA512(char *, int, unsigned char *);
  SHA512(cur, len, dest);
#endif
}

#elif defined(ARM)
#define mysha1 SHA1
extern void SHA1(unsigned char *cur, int len,uint32_t *hash);
void mysha256(char *cur, int len, unsigned char *dest) {
  extern void SHA256(char *, int, unsigned char *);
  SHA256(cur, len, dest);
}
void mysha512(char *cur, int len, unsigned char *dest) {
  extern void SHA512(char *, int, unsigned char *);
  SHA512(cur, len, dest);
}
#endif

#include <sph_sha1.h>
#include <sph_sha2.h>
#include <sph_md5.h>

extern void SHA256(char *cur, int len, unsigned char *dest);
extern void SHA512(char *cur, int len, unsigned char *dest);

#ifndef ARM
void mysha256(char *cur, int len, unsigned char *dest) { SHA256(cur, len, dest); }
void mysha512(char *cur, int len, unsigned char *dest) { SHA512(cur, len, dest); }
#endif

void pbkdf2_md5(char *cur, int len, unsigned char *salt, int saltlen, int rounds, char *curin, int outlen)
{
  sph_md5_context ctx, ctx1, ctx2;
  unsigned char ipad[64],opad[64],out[32],tmp[32],num[4],loop,loops;
  int x,y, acc=0;
  char c;
  uint64_t *p1, *p2;

  p1 = (uint64_t *) out;
  p2 = (uint64_t *) tmp;
  if (len > 64) {
    mymd5((unsigned char *)cur,len,(uint32_t *)tmp);
    cur = (char *)tmp; len =16;
  }
  
  num[0] = num[1] = num[2] = 0; 
  memset(ipad,0x36,64);memset(opad,0x5c,64);
  for (x=0; x< len; x++) {
    c = cur[x]; ipad[x] ^= c; opad[x] ^= c;
  }
  sph_md5_init(&ctx1);
  sph_md5(&ctx1,ipad,64);
  sph_md5_init(&ctx2);
  sph_md5(&ctx2,opad,64);

  loops = (outlen+15) / 16;
  for (loop=1; loop <= loops; loop++) {
    num[3] = loop;
    memcpy(&ctx, &ctx1, sizeof(ctx));
    sph_md5(&ctx,salt,saltlen);
    sph_md5(&ctx,num,4);
    sph_md5_close(&ctx,tmp);

    memcpy(&ctx, &ctx2, sizeof(ctx));
    sph_md5(&ctx,tmp,16);
    sph_md5_close(&ctx,tmp);
    memcpy(out, tmp, 16);

    for (x=1; x < rounds; x++) {
      memcpy(&ctx,&ctx1,sizeof(ctx));
      sph_md5(&ctx,tmp,16);
      sph_md5_close(&ctx, tmp);
      memcpy(&ctx,&ctx2,sizeof(ctx));
      sph_md5(&ctx,tmp,16);
      sph_md5_close(&ctx, tmp);
      p1[0] ^= p2[0]; p1[1] ^= p2[1];
    }
    y = ((outlen - acc) < 16) ? (outlen - acc) : 16;
    memcpy(&curin[acc], out, y); acc += y;
  }
}

void pbkdf2_sha1(char *cur, int len, unsigned char *salt, int saltlen, int rounds, char *curin, int outlen)
{
  sph_sha1_context ctx, ctx1, ctx2;
  unsigned char ipad[64],opad[64],out[32],tmp[32],num[4],loop,loops;
  int x,y, acc=0;
  char c;
  uint64_t *p1, *p2;

  p1 = (uint64_t *) out;
  p2 = (uint64_t *) tmp;
  if (len > 64) {
    mysha1((unsigned char *)cur,len,(uint32_t *)tmp);
    cur = (char *)tmp; len =20;
  }
  
  num[0] = num[1] = num[2] = 0; 
  memset(ipad,0x36,64);memset(opad,0x5c,64);
  for (x=0; x< len; x++) {
    c = cur[x]; ipad[x] ^= c; opad[x] ^= c;
  }
  sph_sha1_init(&ctx1);
  sph_sha1(&ctx1,ipad,64);
  sph_sha1_init(&ctx2);
  sph_sha1(&ctx2,opad,64);

  loops = (outlen+19) / 20;
  for (loop=1; loop <= loops; loop++) {
    num[3] = loop;
    memcpy(&ctx, &ctx1, sizeof(ctx));
    sph_sha1(&ctx,salt,saltlen);
    sph_sha1(&ctx,num,4);
    sph_sha1_close(&ctx,tmp);

    memcpy(&ctx, &ctx2, sizeof(ctx));
    sph_sha1(&ctx,tmp,20);
    sph_sha1_close(&ctx,tmp);
    memcpy(out, tmp, 20);

    for (x=1; x < rounds; x++) {
      memcpy(&ctx,&ctx1,sizeof(ctx));
      sph_sha1(&ctx,tmp,20);
      sph_sha1_close(&ctx, tmp);
      memcpy(&ctx,&ctx2,sizeof(ctx));
      sph_sha1(&ctx,tmp,20);
      sph_sha1_close(&ctx, tmp);
      p1[0] ^= p2[0]; p1[1] ^= p2[1];
      p1[2] ^= p2[2];
    }
    y = ((outlen - acc) < 20) ? (outlen - acc) : 20;
    memcpy(&curin[acc], out, y); acc += y;
  }
}

void pbkdf2_sha256(char *cur, int len, unsigned char *salt, int saltlen, int rounds, char *curin, int outlen)
{
  sph_sha256_context ctx, ctx1, ctx2;
  unsigned char ipad[64],opad[64],out[32],tmp[32],num[4],loop,loops;
  int x,y, acc=0;
  char c;
  uint64_t *p1, *p2;

  p1 = (uint64_t *) out;
  p2 = (uint64_t *) tmp;
  if (len > 64) {
    mysha256(cur,len,tmp);
    cur = (char *)tmp; len =32;
  }
  
  num[0] = num[1] = num[2] = 0; 
  memset(ipad,0x36,64);memset(opad,0x5c,64);
  for (x=0; x< len; x++) {
    c = cur[x]; ipad[x] ^= c; opad[x] ^= c;
  }
  sph_sha256_init(&ctx1);
  sph_sha256(&ctx1,ipad,64);
  sph_sha256_init(&ctx2);
  sph_sha256(&ctx2,opad,64);

  loops = (outlen+31) / 32;
  for (loop=1; loop <= loops; loop++) {
    num[3] = loop;
    memcpy(&ctx, &ctx1, sizeof(ctx));
    sph_sha256(&ctx,salt,saltlen);
    sph_sha256(&ctx,num,4);
    sph_sha256_close(&ctx,tmp);

    memcpy(&ctx, &ctx2, sizeof(ctx));
    sph_sha256(&ctx,tmp,32);
    sph_sha256_close(&ctx,tmp);
    memcpy(out, tmp, 32);

    for (x=1; x < rounds; x++) {
      memcpy(&ctx,&ctx1,sizeof(ctx));
      sph_sha256(&ctx,tmp,32);
      sph_sha256_close(&ctx, tmp);
      memcpy(&ctx,&ctx2,sizeof(ctx));
      sph_sha256(&ctx,tmp,32);
      sph_sha256_close(&ctx, tmp);
      p1[0] ^= p2[0]; p1[1] ^= p2[1];
      p1[2] ^= p2[2]; p1[3] ^= p2[3];
    }
    y = ((outlen - acc) < 32) ? (outlen - acc) : 32;
    memcpy(&curin[acc], out, y); acc += y;
  }
}

void pbkdf2_sha512(char *cur, int len, unsigned char *salt, int saltlen, int rounds, char *curin, int outlen)
{
  sph_sha512_context ctx, ctx1, ctx2;
  unsigned char ipad[128],opad[128],out[64],tmp[64],num[4],loop,loops;
  int x,y, acc=0;
  char c;
  uint64_t *p1, *p2;

  p1 = (uint64_t *) out;
  p2 = (uint64_t *) tmp;
  if (len > 128) {
    mysha512(cur,len,tmp);
    cur = (char *)tmp; len =64;
  }
  
  num[0] = num[1] = num[2] = 0; 
  memset(ipad,0x36,128);memset(opad,0x5c,128);
  for (x=0; x< len; x++) {
    c = cur[x]; ipad[x] ^= c; opad[x] ^= c;
  }
  sph_sha512_init(&ctx1);
  sph_sha512(&ctx1,ipad,128);
  sph_sha512_init(&ctx2);
  sph_sha512(&ctx2,opad,128);

  loops = (outlen+63) / 64;
  for (loop=1; loop <= loops; loop++) {
    num[3] = loop;
    memcpy(&ctx, &ctx1, sizeof(ctx));
    sph_sha512(&ctx,salt,saltlen);
    sph_sha512(&ctx,num,4);
    sph_sha512_close(&ctx,tmp);

    memcpy(&ctx, &ctx2, sizeof(ctx));
    sph_sha512(&ctx,tmp,64);
    sph_sha512_close(&ctx,tmp);
    memcpy(out, tmp, 64);

    for (x=1; x < rounds; x++) {
      memcpy(&ctx,&ctx1,sizeof(ctx));
      sph_sha512(&ctx,tmp,64);
      sph_sha512_close(&ctx, tmp);
      memcpy(&ctx,&ctx2,sizeof(ctx));
      sph_sha512(&ctx,tmp,64);
      sph_sha512_close(&ctx, tmp);
      p1[0] ^= p2[0]; p1[1] ^= p2[1];
      p1[2] ^= p2[2]; p1[3] ^= p2[3];
      p1[4] ^= p2[4]; p1[5] ^= p2[5];
      p1[6] ^= p2[6]; p1[7] ^= p2[7];
    }
    y = ((outlen - acc) < 64) ? (outlen - acc) : 64;
    memcpy(&curin[acc], out, y); acc += y;
  }
}


     

