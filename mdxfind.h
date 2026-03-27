/*
 * $Log: mdxfind.h,v $
 * Revision 1.11  2026/03/25 23:11:05  dlr
 * Move Hashchain struct to header
 *
 * Revision 1.10  2026/03/23 02:51:54  dlr
 * Replace -n digit hack with mask-based hybrid attack: -n "?l?d" append, -N prepend, ?[0-9a-f] custom classes
 *
 * Revision 1.9  2025/08/24 22:08:56  dlr
 * changes for atomic
 *
 * Revision 1.8  2025/08/23 22:26:25  dlr
 * Move to new outbuf
 *
 * Revision 1.7  2020/03/11 02:49:29  dlr
 * SSSE modifications complete.  About to start on fastrule
 *
 * Revision 1.6  2017/10/19 03:38:44  dlr
 * Add rule counter
 *
 * Revision 1.5  2017/08/25 05:09:54  dlr
 * minor change for ARM6
 *
 * Revision 1.4  2017/08/25 04:16:03  dlr
 * Porting for ARM/POWERPC.  Fix SQL5
 *
 * Revision 1.3  2017/06/30 13:35:32  dlr
 * fix for ARM
 *
 * Revision 1.2  2017/06/30 13:23:13  dlr
 * Added SVAL
 *
 * Revision 1.1  2017/06/29 14:09:29  dlr
 * Initial revision
 *
 *
 */
#if ARM > 6
#include <arm_neon.h>
#endif

#define MAXLINE (40*1024)
struct job {
    struct job *next;
    char *readbuf,*outbuf,*pass;
    unsigned int *found;
    struct LineInfo *readindex;
    int op,len,clen,flags;
    int Ruleindex,digits,outlen;
    unsigned int startline,numline;
    unsigned long long Numbers;
    unsigned long long MaskIndex;
    unsigned long long MaskCount;
    char *filename;
    int *doneprint;
    char prefix[MAXLINE],line[MAXLINE+MAXLINE];
};
#define JOBFLAG_PRINT 1
#define JOBFLAG_HEX 2
#define JOBFLAG_NUMBERS 4
#define JOBFLAG_IP 8
#define JOBFLAG_PREPEND 16

union HashU {
    unsigned char h[256];
    uint32_t i[64];
    unsigned long long v[32];
#ifndef NOTINTEL
    __m128i x[16];
#endif
#if ARM > 6
    uint32x4_t x[16];
#endif
#ifdef POWERPC
    vector unsigned int x[16];
#endif
};

struct Hashchain {
    struct Hashchain *next;
    unsigned short int flags, len;
    unsigned char hash[1];
};

#ifdef ARM
union sse_value {
#if ARM > 6
    uint32x4_t sse;
#else
    uint64_t sse,sse1;
#endif
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif
#ifdef POWERPC
union sse_value {
   vector unsigned int sse;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

#ifdef SPARC
union sse_value {
   uint64_t sse,sse1;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

#ifndef NOTINTEL
union sse_value {
    __m128i sse;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

#define BCRYPT_HASHSIZE 64
#define MAXVECSIZE 2000000  /* Maximum test vector size */

#define MAXTHREADS 8

#define LDAP_MAX_UTF8_LEN  ( sizeof(wchar_t) * 3/2 )
#define FLOOR_LOG2(x) (31 - __builtin_clz((x) | 1))
static inline int log2i(uint64_t n) {
#define S(k) if (n >= ((uint64_t)1 << k)) { i += k; n >>= k; }
    int i = -(n == 0); S(32); S(16); S(8); S(4); S(2); S(1); return i;
#undef S
}

/* MAXCHUNK sets the maximum amount of memory used for each chunk.
   As I write this, typical hard drive speeds are 100 Mbytes/sec, so
   100M represents about 1 seconds of data.  Increase as appropriate.
*/
#ifdef ARM
/* INPUTCHUNK - maximum number of hashes to process at once from stdin */
#define INPUTCHUNK (100000)
#define MAXCHUNK (5*1024*1024)
#else
/* INPUTCHUNK - maximum number of hashes to process at once from stdin */
#define INPUTCHUNK (10000000)
#define MAXCHUNK (50*1024*1024)
#endif

#define MAXLINEPERCHUNK (MAXCHUNK/2/8)

#define MAXLJOB (32)



