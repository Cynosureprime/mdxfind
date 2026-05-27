/********************************************************************\
 *
 *      FILE:     rmd128.c
 *
 *      CONTENTS: A sample C-implementation of the RIPEMD-128
 *                hash-function. This function is a plug-in substitute 
 *                for RIPEMD. A 160-bit hash result is obtained using
 *                RIPEMD-160.
 *      TARGET:   any computer with an ANSI C compiler
 *
 *      AUTHOR:   Antoon Bosselaers, ESAT-COSIC
 *      DATE:     1 March 1996
 *      VERSION:  1.0
 *
 *      Copyright (c) Katholieke Universiteit Leuven
 *      1996, All Rights Reserved
 *
\********************************************************************/

/*
 * $Revision: 1.1 $
 * $Log: rmd128.c,v $
 * Revision 1.1  2026/05/27 18:40:16  dlr
 * Standard-conformance fix in RIPEMD128 length encoding. Previously rmd128.c RIPEMD128 passed post-loop residual byte count to MDfinish as lswlen which made the encoded bit-length field equal to total_bytes mod 64 instead of total_bytes for any message longer than 63 bytes. Output disagreed with Bosselaers 1996 reference sph_ripemd128 openssl and every other conforming implementation. Fix preserves the ORIGINAL message length into total_lswlen plus total_mswlen before the per-block consumer loop then passes the total to MDfinish. No signature changes. MDfinish header docstring in rmd128.h already documented the intended semantics that lswlen is total bytes and only lswlen mod 64 remain in strptr the bug was solely in the caller. Verified PASS against all 8 Bosselaers test vectors including 80 char 1234567890 x8 canary and 1 million a stress vector. User confirmed no production solved hash records affected because rmd128 long messages were not exercised in archives. Also adds RCS keyword stanza to satisfy global rule for hand authored source.
 *
 */

/*  header files */
#include <string.h>
#include "rmd128.h"
#define RMDsize 128

/********************************************************************/

static void MDinit(dword *MDbuf)
{
   MDbuf[0] = 0x67452301UL;
   MDbuf[1] = 0xefcdab89UL;
   MDbuf[2] = 0x98badcfeUL;
   MDbuf[3] = 0x10325476UL;

   return;
}

/********************************************************************/

static void compress(dword *MDbuf, dword *X)
{
   dword aa = MDbuf[0],  bb = MDbuf[1],  cc = MDbuf[2],  dd = MDbuf[3];
   dword aaa = MDbuf[0], bbb = MDbuf[1], ccc = MDbuf[2], ddd = MDbuf[3];

   /* round 1 */
   FF(aa, bb, cc, dd, X[ 0], 11);
   FF(dd, aa, bb, cc, X[ 1], 14);
   FF(cc, dd, aa, bb, X[ 2], 15);
   FF(bb, cc, dd, aa, X[ 3], 12);
   FF(aa, bb, cc, dd, X[ 4],  5);
   FF(dd, aa, bb, cc, X[ 5],  8);
   FF(cc, dd, aa, bb, X[ 6],  7);
   FF(bb, cc, dd, aa, X[ 7],  9);
   FF(aa, bb, cc, dd, X[ 8], 11);
   FF(dd, aa, bb, cc, X[ 9], 13);
   FF(cc, dd, aa, bb, X[10], 14);
   FF(bb, cc, dd, aa, X[11], 15);
   FF(aa, bb, cc, dd, X[12],  6);
   FF(dd, aa, bb, cc, X[13],  7);
   FF(cc, dd, aa, bb, X[14],  9);
   FF(bb, cc, dd, aa, X[15],  8);
                             
   /* round 2 */
   GG(aa, bb, cc, dd, X[ 7],  7);
   GG(dd, aa, bb, cc, X[ 4],  6);
   GG(cc, dd, aa, bb, X[13],  8);
   GG(bb, cc, dd, aa, X[ 1], 13);
   GG(aa, bb, cc, dd, X[10], 11);
   GG(dd, aa, bb, cc, X[ 6],  9);
   GG(cc, dd, aa, bb, X[15],  7);
   GG(bb, cc, dd, aa, X[ 3], 15);
   GG(aa, bb, cc, dd, X[12],  7);
   GG(dd, aa, bb, cc, X[ 0], 12);
   GG(cc, dd, aa, bb, X[ 9], 15);
   GG(bb, cc, dd, aa, X[ 5],  9);
   GG(aa, bb, cc, dd, X[ 2], 11);
   GG(dd, aa, bb, cc, X[14],  7);
   GG(cc, dd, aa, bb, X[11], 13);
   GG(bb, cc, dd, aa, X[ 8], 12);

   /* round 3 */
   HH(aa, bb, cc, dd, X[ 3], 11);
   HH(dd, aa, bb, cc, X[10], 13);
   HH(cc, dd, aa, bb, X[14],  6);
   HH(bb, cc, dd, aa, X[ 4],  7);
   HH(aa, bb, cc, dd, X[ 9], 14);
   HH(dd, aa, bb, cc, X[15],  9);
   HH(cc, dd, aa, bb, X[ 8], 13);
   HH(bb, cc, dd, aa, X[ 1], 15);
   HH(aa, bb, cc, dd, X[ 2], 14);
   HH(dd, aa, bb, cc, X[ 7],  8);
   HH(cc, dd, aa, bb, X[ 0], 13);
   HH(bb, cc, dd, aa, X[ 6],  6);
   HH(aa, bb, cc, dd, X[13],  5);
   HH(dd, aa, bb, cc, X[11], 12);
   HH(cc, dd, aa, bb, X[ 5],  7);
   HH(bb, cc, dd, aa, X[12],  5);

   /* round 4 */
   II(aa, bb, cc, dd, X[ 1], 11);
   II(dd, aa, bb, cc, X[ 9], 12);
   II(cc, dd, aa, bb, X[11], 14);
   II(bb, cc, dd, aa, X[10], 15);
   II(aa, bb, cc, dd, X[ 0], 14);
   II(dd, aa, bb, cc, X[ 8], 15);
   II(cc, dd, aa, bb, X[12],  9);
   II(bb, cc, dd, aa, X[ 4],  8);
   II(aa, bb, cc, dd, X[13],  9);
   II(dd, aa, bb, cc, X[ 3], 14);
   II(cc, dd, aa, bb, X[ 7],  5);
   II(bb, cc, dd, aa, X[15],  6);
   II(aa, bb, cc, dd, X[14],  8);
   II(dd, aa, bb, cc, X[ 5],  6);
   II(cc, dd, aa, bb, X[ 6],  5);
   II(bb, cc, dd, aa, X[ 2], 12);

   /* parallel round 1 */
   III(aaa, bbb, ccc, ddd, X[ 5],  8); 
   III(ddd, aaa, bbb, ccc, X[14],  9);
   III(ccc, ddd, aaa, bbb, X[ 7],  9);
   III(bbb, ccc, ddd, aaa, X[ 0], 11);
   III(aaa, bbb, ccc, ddd, X[ 9], 13);
   III(ddd, aaa, bbb, ccc, X[ 2], 15);
   III(ccc, ddd, aaa, bbb, X[11], 15);
   III(bbb, ccc, ddd, aaa, X[ 4],  5);
   III(aaa, bbb, ccc, ddd, X[13],  7);
   III(ddd, aaa, bbb, ccc, X[ 6],  7);
   III(ccc, ddd, aaa, bbb, X[15],  8);
   III(bbb, ccc, ddd, aaa, X[ 8], 11);
   III(aaa, bbb, ccc, ddd, X[ 1], 14);
   III(ddd, aaa, bbb, ccc, X[10], 14);
   III(ccc, ddd, aaa, bbb, X[ 3], 12);
   III(bbb, ccc, ddd, aaa, X[12],  6);

   /* parallel round 2 */
   HHH(aaa, bbb, ccc, ddd, X[ 6],  9);
   HHH(ddd, aaa, bbb, ccc, X[11], 13);
   HHH(ccc, ddd, aaa, bbb, X[ 3], 15);
   HHH(bbb, ccc, ddd, aaa, X[ 7],  7);
   HHH(aaa, bbb, ccc, ddd, X[ 0], 12);
   HHH(ddd, aaa, bbb, ccc, X[13],  8);
   HHH(ccc, ddd, aaa, bbb, X[ 5],  9);
   HHH(bbb, ccc, ddd, aaa, X[10], 11);
   HHH(aaa, bbb, ccc, ddd, X[14],  7);
   HHH(ddd, aaa, bbb, ccc, X[15],  7);
   HHH(ccc, ddd, aaa, bbb, X[ 8], 12);
   HHH(bbb, ccc, ddd, aaa, X[12],  7);
   HHH(aaa, bbb, ccc, ddd, X[ 4],  6);
   HHH(ddd, aaa, bbb, ccc, X[ 9], 15);
   HHH(ccc, ddd, aaa, bbb, X[ 1], 13);
   HHH(bbb, ccc, ddd, aaa, X[ 2], 11);

   /* parallel round 3 */   
   GGG(aaa, bbb, ccc, ddd, X[15],  9);
   GGG(ddd, aaa, bbb, ccc, X[ 5],  7);
   GGG(ccc, ddd, aaa, bbb, X[ 1], 15);
   GGG(bbb, ccc, ddd, aaa, X[ 3], 11);
   GGG(aaa, bbb, ccc, ddd, X[ 7],  8);
   GGG(ddd, aaa, bbb, ccc, X[14],  6);
   GGG(ccc, ddd, aaa, bbb, X[ 6],  6);
   GGG(bbb, ccc, ddd, aaa, X[ 9], 14);
   GGG(aaa, bbb, ccc, ddd, X[11], 12);
   GGG(ddd, aaa, bbb, ccc, X[ 8], 13);
   GGG(ccc, ddd, aaa, bbb, X[12],  5);
   GGG(bbb, ccc, ddd, aaa, X[ 2], 14);
   GGG(aaa, bbb, ccc, ddd, X[10], 13);
   GGG(ddd, aaa, bbb, ccc, X[ 0], 13);
   GGG(ccc, ddd, aaa, bbb, X[ 4],  7);
   GGG(bbb, ccc, ddd, aaa, X[13],  5);

   /* parallel round 4 */
   FFF(aaa, bbb, ccc, ddd, X[ 8], 15);
   FFF(ddd, aaa, bbb, ccc, X[ 6],  5);
   FFF(ccc, ddd, aaa, bbb, X[ 4],  8);
   FFF(bbb, ccc, ddd, aaa, X[ 1], 11);
   FFF(aaa, bbb, ccc, ddd, X[ 3], 14);
   FFF(ddd, aaa, bbb, ccc, X[11], 14);
   FFF(ccc, ddd, aaa, bbb, X[15],  6);
   FFF(bbb, ccc, ddd, aaa, X[ 0], 14);
   FFF(aaa, bbb, ccc, ddd, X[ 5],  6);
   FFF(ddd, aaa, bbb, ccc, X[12],  9);
   FFF(ccc, ddd, aaa, bbb, X[ 2], 12);
   FFF(bbb, ccc, ddd, aaa, X[13],  9);
   FFF(aaa, bbb, ccc, ddd, X[ 9], 12);
   FFF(ddd, aaa, bbb, ccc, X[ 7],  5);
   FFF(ccc, ddd, aaa, bbb, X[10], 15);
   FFF(bbb, ccc, ddd, aaa, X[14],  8);

   /* combine results */
   ddd += cc + MDbuf[1];               /* final result for MDbuf[0] */
   MDbuf[1] = MDbuf[2] + dd + aaa;
   MDbuf[2] = MDbuf[3] + aa + bbb;
   MDbuf[3] = MDbuf[0] + bb + ccc;
   MDbuf[0] = ddd;

   return;
}

/********************************************************************/

static void MDfinish(dword *MDbuf, byte *strptr, dword lswlen, dword mswlen)
{
   unsigned int i;                                 /* counter       */
   dword        X[16];                             /* message words */

   memset(X, 0, 16*sizeof(dword));

   /* put bytes from strptr into X */
   for (i=0; i<(lswlen&63); i++) {
      /* byte i goes into word X[i div 4] at pos.  8*(i mod 4)  */
      X[i>>2] ^= (dword) *strptr++ << (8 * (i&3));
   }

   /* append the bit m_n == 1 */
   X[(lswlen>>2)&15] ^= (dword)1 << (8*(lswlen&3) + 7);

   if ((lswlen & 63) > 55) {
      /* length goes to next block */
      compress(MDbuf, X);
      memset(X, 0, 16*sizeof(dword));
   }

   /* append length in bits*/
   X[14] = lswlen << 3;
   X[15] = (lswlen >> 29) | (mswlen << 3);
   compress(MDbuf, X);

   return;
}

void RIPEMD128(unsigned char *src, int len, unsigned char *dest)
{
      dword MDbuf[RMDsize/8], X[16];
      int i;
      /* Standard-conformance fix (2026-05-27): preserve the ORIGINAL
       * message length so MDfinish() can encode the correct total
       * bit-length suffix. The legacy code passed the post-loop
       * residual to MDfinish() which made the length-field encode
       * (total_bytes mod 64) instead of total_bytes for any message
       * longer than 63 bytes -- producing non-standard digests that
       * disagree with Bosselaers's 1996 reference, sph_ripemd128,
       * and every other conforming implementation. The MDfinish()
       * header comment in rmd128.h already documented the intended
       * semantics: lswlen is the TOTAL byte length and only
       * (lswlen mod 64) bytes remain in strptr. The bug was solely
       * in this caller. */
      dword total_lswlen = (dword)len;
      dword total_mswlen = 0;

      MDinit(MDbuf);

   /* process message in 16-word chunks */
   for (; len > 63; len-=64) {
      for (i=0; i<16; i++) {
         X[i] = BYTES_TO_DWORD(src);
         src += 4;
      }
      compress(MDbuf, X);
   }                                    /* length mod 64 bytes left */

   /* finish: pass TOTAL length, not residual. */
   MDfinish(MDbuf, src, total_lswlen, total_mswlen);

   for (i=0; i<RMDsize/8; i+=4) {
      dest[i]   =  MDbuf[i>>2];         /* implicit cast to byte  */
      dest[i+1] = (MDbuf[i>>2] >>  8);  /*  extracts the 8 least  */
      dest[i+2] = (MDbuf[i>>2] >> 16);  /*  significant bits.     */
      dest[i+3] = (MDbuf[i>>2] >> 24);
   }
}




/************************ end of file rmd128.c **********************/

