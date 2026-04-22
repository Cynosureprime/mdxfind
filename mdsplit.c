#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <dirent.h>
#include <stdint.h>
#include <errno.h>
#include <Judy.h>

#ifdef ARM
#define NOTINTEL 1
#endif
#ifdef POWERPC
#define NOTINTEL 1
#endif

#ifndef NOTINTEL

#include <emmintrin.h>
#include <xmmintrin.h>

#endif

static char *Version = "$Header: /Users/dlr/src/mdfind/RCS/mdsplit.c,v 1.30 2026/04/21 17:11:40 dlr Exp dlr $";
/*
 * $Log: mdsplit.c,v $
 * Revision 1.30  2026/04/21 17:11:40  dlr
 * Fix partial-match false positive: reject matches where boundary char is still hex. Previously fell through to accept any best_len2 > 0 even when no entry passed the boundary check, causing 16-char prefix collisions on reversed hashes to be accepted as valid matches.
 *
 * Revision 1.29  2026/04/21 13:35:55  dlr
 * Fix SolnOff overflow: uint32_t overflows at 4GB of solution data (~93M entries). Changed to size_t for correct indexing at 2B+ entries. Increase COMPACT_MAX_PROBE to 256. Add overflow diagnostics.
 *
 * Revision 1.28  2026/04/21 05:37:14  dlr
 * Phase 2 I/O optimizations: increase default buffer from 500KB to 4MB (256KB on ARM), add posix_fadvise(SEQUENTIAL) on Linux for hash file reads, write buffer pool (64x64KB) for type output files, prefetch compact table slot on probe
 *
 * Revision 1.27  2026/04/21 05:27:09  dlr
 * Replace hard-coded text hash type list with full hash content scan; any non-hex character in the hash portion triggers texthash classification. Future-proof against new crypt/structured types.
 *
 * Revision 1.26  2026/04/21 05:12:53  dlr
 * Phase 1: Replace JudySL Hash with compact hash table for hex hashes. Flat SolnBuf storage with parallel index arrays. Text hashes remain in separate TextHash JudySL. Preserves partial prefix matching, reverse lookup, all file management features.
 *
 * Revision 1.25  2026/02/21 01:58:48  dlr
 * *** empty log message ***
 *
 * Revision 1.24  2025/08/28 23:04:10  dlr
 * fix logic error in mdsplit on non-prefix files.
 *
 * Revision 1.23  2025/08/28 17:58:58  dlr
 * strip basename on input file, if adding a prefix
 *
 * Revision 1.22  2025/08/27 15:14:18  dlr
 * Add usage for -V and -?
 *
 * Revision 1.21  2025/08/26 22:19:55  dlr
 * Add -p, -i and -s prefix/infix/suffix to the solution filenames.
 *
 * Revision 1.20  2025/08/19 13:38:55  dlr
 * Increase max line length to 80k
 *
 * Revision 1.19  2023/08/31 05:00:23  dlr
 * prep to change to public version
 *
 * Revision 1.18  2017/11/02 23:06:53  dlr
 * Support longer sha256/pbkdf2 hash types
 *
 * Revision 1.17  2017/09/15 20:04:04  dlr
 * Error if there is unlabled input - fixed.
 *
 * Revision 1.16  2017/09/05 14:45:09  dlr
 * Improve text-based hash processing
 *
 * Revision 1.15  2017/08/25 04:18:19  dlr
 * Porting for powerpc and arm
 *
 * Revision 1.14  2017/08/16 03:35:00  dlr
 * allow for unlocked file manipulation
 *
 * Revision 1.13  2017/08/13 15:29:28  dlr
 * Add multiple BOM types
 *
 * Revision 1.12  2017/08/04 18:51:08  dlr
 * Improve detection of type/end of hash
 *
 * Revision 1.11  2017/08/04 18:39:49  dlr
 * Add -t option
 *
 * Revision 1.10  2017/07/16 17:11:09  dlr
 * Add reverse, and new hash types
 *
 * Revision 1.9  2017/07/15 13:55:37  dlr
 * Add reverse hash support
 *
 * Revision 1.8  2017/07/10 13:59:58  dlr
 * Add more extensive txthash detection
 *
 * Revision 1.7  2017/07/08 15:37:54  dlr
 * close lock file before unlinking it.
 *
 * Revision 1.6  2017/07/08 15:24:11  dlr
 * windows does not like to rename open files.
 *
 * Revision 1.5  2017/07/08 14:36:25  dlr
 * Fix broken windows UCS2/UTF16LE powershell botch
 *
 * Revision 1.4  2017/07/08 05:50:01  dlr
 * Upper/lower case hash issues
 *
 * Revision 1.3  2017/07/07 19:35:50  dlr
 * signed char for arm
 *
 * Revision 1.2  2017/07/07 19:32:39  dlr
 * looks like it works
 *
 * Revision 1.1  2017/07/07 14:00:12  dlr
 * Initial revision
 *
 *
 */

#ifdef AIX
#define _WIN32 1
#endif

#define MAXLINE (81*1024)
#define MDMAXPATHLEN 5000

union HashU {
  unsigned char h[256];
  uint32_t i[64];
  unsigned long long v[32];
#ifndef NOTINTEL
  __m128i x[16];
#endif
};

char *prefix, *infix, *suffix;

#define FNInfo_hexhash 1
#define FNInfo_texthash 2
#define FNInfo_3sep 4
#define FNInfo_4sep 8
struct FNInfo {
  unsigned long long hitcount, incount;
  int iter;
  int flags;
  char *type;
  FILE *fh;
  char filename[MDMAXPATHLEN + 16];
};
struct Resinfo {
  struct FNInfo *hashtype;
  char *found;
  struct Resinfo *next;
  int hlen, len;
};


char Sep = ':';
int Exact = 0;
int Longmatch;
#ifdef ARM
#define FILEBUFSIZE (256*1024)
#else
#define FILEBUFSIZE (4*1024*1024)
#endif

/* Pool of write buffers for type output files */
#define HITBUF_COUNT 64
#define HITBUF_SIZE  (64*1024)
static char *HitBufPool[HITBUF_COUNT];
static int   HitBufUsed = 0;


int memsize = FILEBUFSIZE;
int NoReverse = 0, NoLock = 0;

#ifndef _WIN32
char *Inbuf, *Outbuf;
#endif

Pvoid_t Hfn, Bhash, Hash, Specfn;
Pvoid_t TextHash = NULL;  /* JudySL for text-based hashes */
int _dowildcard = -1; /* enable wildcard expansion for Windows */

int Maxsuf = 0;
pid_t Mypid;

/* Compact hash table for hex hash solutions */
#define COMPACT_MAX_PROBE 256

static uint32_t *CompactFP;      /* fingerprint array */
static uint32_t *CompactIdx;     /* index into Soln arrays */
static uint64_t CompactSize;     /* power of 2 */
static uint64_t CompactMask;     /* CompactSize - 1 */
static uint32_t CompactUsed;

/* Flat solution storage for hex hashes */
static char     *SolnBuf;        /* concatenated solution strings */
static size_t   SolnBufUsed, SolnBufCap;
static size_t   *SolnOff;        /* offset into SolnBuf for each solution */
static uint16_t *SolnHlen;       /* hash portion length */
static uint16_t *SolnTotalLen;   /* total found string length */
static struct FNInfo **SolnType; /* pointer to type info */
static uint64_t *SolnKey;        /* decoded uint64_t key for building compact table */
static uint32_t SolnCount, SolnCap;

/* Forward declarations */
extern unsigned char trhex[];
char *commify(unsigned long long n);

static inline uint64_t compact_mix(uint64_t k) {
    return k ^ (k >> 32);
}

/* Decode first 16 hex chars (or fewer) of a string into a uint64_t key.
 * Uses trhex[] for case-insensitive hex decoding.
 * Returns the key, or 0 if fewer than 2 hex chars available. */
static inline uint64_t decode_hex_key(const char *s, int len) {
    uint64_t key = 0;
    int i, nibbles;
    unsigned char c;

    nibbles = (len > 16) ? 16 : len;
    for (i = 0; i < nibbles; i++) {
        c = trhex[((unsigned char)s[i]) & 0xff];
        if (c > 15) break;
        key = (key << 4) | c;
    }
    /* Pad remaining nibbles with 0 if we got fewer than 16 */
    for (; i < 16; i++)
        key <<= 4;
    return key;
}

/* Add a hex solution to the flat arrays. Returns the index. */
static uint32_t soln_add(const char *found, int len, int hlen, struct FNInfo *ftype, uint64_t key) {
    uint32_t idx;

    if (SolnCount >= SolnCap) {
        uint32_t newcap = SolnCap ? SolnCap * 2 : 1024 * 1024;
        SolnOff = realloc(SolnOff, newcap * sizeof(size_t));
        SolnHlen = realloc(SolnHlen, newcap * sizeof(uint16_t));
        SolnTotalLen = realloc(SolnTotalLen, newcap * sizeof(uint16_t));
        SolnType = realloc(SolnType, newcap * sizeof(struct FNInfo *));
        SolnKey = realloc(SolnKey, newcap * sizeof(uint64_t));
        if (!SolnOff || !SolnHlen || !SolnTotalLen || !SolnType || !SolnKey) {
            fprintf(stderr, "Out of memory growing Soln arrays at %u entries\n", newcap);
            exit(1);
        }
        SolnCap = newcap;
    }
    /* Grow SolnBuf if needed */
    if (SolnBufUsed + len + 1 > SolnBufCap) {
        size_t newcap = SolnBufCap ? SolnBufCap * 2 : 64 * 1024 * 1024;
        while (newcap < SolnBufUsed + len + 1) newcap *= 2;
        SolnBuf = realloc(SolnBuf, newcap);
        if (!SolnBuf) {
            fprintf(stderr, "Out of memory growing SolnBuf to %zu bytes\n", newcap);
            exit(1);
        }
        SolnBufCap = newcap;
    }

    idx = SolnCount++;
    SolnOff[idx] = SolnBufUsed;
    SolnHlen[idx] = (uint16_t)hlen;
    SolnTotalLen[idx] = (uint16_t)len;
    SolnType[idx] = ftype;
    SolnKey[idx] = key;
    memmove(SolnBuf + SolnBufUsed, found, len + 1);
    SolnBufUsed += len + 1;
    return idx;
}

/* Build the compact hash table from SolnKey[]/SolnCount.
 * Called after all solutions are loaded. */
static void build_compact(void) {
    uint64_t tsize;
    uint32_t i;

    if (SolnCount == 0) return;

    tsize = 1;
    while (tsize <= SolnCount) tsize <<= 1;
    tsize <<= 1;

    CompactSize = tsize;
    CompactMask = tsize - 1;
    CompactUsed = 0;
    CompactFP = calloc(tsize, sizeof(uint32_t));
    CompactIdx = malloc(tsize * sizeof(uint32_t));
    if (!CompactFP || !CompactIdx) {
        fprintf(stderr, "Failed to allocate compact table (%llu slots)\n", (unsigned long long)tsize);
        exit(1);
    }

    for (i = 0; i < SolnCount; i++) {
        uint64_t key = SolnKey[i];
        uint32_t fp = (uint32_t)(key >> 32);
        uint64_t pos = compact_mix(key) & CompactMask;
        int p;

        /* Prefetch lookahead of 8 */
        if (i + 8 < SolnCount) {
            uint64_t nkey = SolnKey[i + 8];
            uint64_t npos = compact_mix(nkey) & CompactMask;
            __builtin_prefetch(&CompactFP[npos], 1, 1);
            __builtin_prefetch(&CompactIdx[npos], 1, 1);
        }

        if (!fp) fp = 1;

        for (p = 0; p < COMPACT_MAX_PROBE; p++) {
            if (!CompactFP[pos]) {
                CompactFP[pos] = fp;
                CompactIdx[pos] = i;
                CompactUsed++;
                break;
            }
            /* On collision with same fp, keep existing - we'll handle
             * multiple entries with same key in the lookup (longest match) */
            if (CompactFP[pos] == fp) {
                uint32_t eidx = CompactIdx[pos];
                if (SolnKey[eidx] == key) {
                    /* Same key - chain: store as linked via next index.
                     * For simplicity, just keep the first one in the table.
                     * The match function will scan all SolnCount entries
                     * with matching key via linear probe. We allow duplicates
                     * in adjacent probe slots. */
                }
            }
            pos = (pos + 1) & CompactMask;
        }
        if (p == COMPACT_MAX_PROBE) {
            /* Overflow beyond max probe — should be exceedingly rare at <50% load.
             * Log count for diagnostics. */
            static uint32_t overflow_count = 0;
            overflow_count++;
            if (overflow_count <= 5)
                fprintf(stderr, "Warning: compact table overflow at entry %u (key %016llx)\n",
                        i, (unsigned long long)key);
            /* Place in next available slot — match_compact won't find it
             * within COMPACT_MAX_PROBE, but this keeps the table consistent */
            uint64_t opos = pos;
            while (CompactFP[opos]) opos = (opos + 1) & CompactMask;
            CompactFP[opos] = fp;
            CompactIdx[opos] = i;
            CompactUsed++;
        }
    }
    fprintf(stderr, "Compact table: %s hex entries in %llu slots\n",
            commify(SolnCount), (unsigned long long)tsize);
}

char *DefaultType;


#define MEMCHUNK ((sizeof(struct Resinfo))*1024)
char *Mybuf;
int Mybufindex;


/*
 * mymalloc allocates a fixed-size, non-returnable space which
 * is aligned in memory according to the align value.
 * It allocates pre-zeroed memory, originally obtained from
 * calloc in MEMCHUNK-sized units.  It will always return
 * a valid memory pointer, and if there is not enough memory,
 * or the block is too large, will abend the program rather
 * than return a failure code.
 * Not thread safe - call from only the main program.
 */
void *mymalloc(int size, int align) {
  char *t;
  if ((Mybufindex % align) != 0)
    Mybufindex += align - (Mybufindex % align);
  if (Mybuf == NULL || (MEMCHUNK - Mybufindex) < (size + 2)) {
    Mybuf = calloc(1, MEMCHUNK);
    if (!Mybuf) {
      fprintf(stderr, "Out of memory in mymalloc\n");
      exit(1);
    }
    Mybufindex = 0;
  }
  if (Mybuf == NULL || (MEMCHUNK - Mybufindex) < (size + 2)) {
    fprintf(stderr, "mymalloc: You want too much space, %d bytes", size);
    exit(1);
  }
  t = &Mybuf[Mybufindex];
  Mybufindex += size;
  return (t);
}

/*
 * findchr() is identical in purpose to strchr(3).  It's just
 * about 16 times faster.
 */

#ifdef NOTINTEL
inline char *findchr(char *s, unsigned char c) {
    return(strchr(s,c));
}
#else
/* char *findchr(char *s, unsigned char c); */

char *findchr(char *s, unsigned char c);

inline char *findchr(char *s, unsigned char c) {
  unsigned int align;
  unsigned int res, resz;
  __m128i cur, seek, zero;

  align = ((unsigned long long) s) & 0xf;
  seek = _mm_set1_epi8(c);
  zero = _mm_setzero_si128();
  s = (char *) (((unsigned long long) s) & 0xfffffffffffffff0);
  cur = _mm_load_si128((__m128i const *) s);
  resz = _mm_movemask_epi8(_mm_cmpeq_epi8(zero, cur)) >> align;
  res = (_mm_movemask_epi8(_mm_cmpeq_epi8(seek, cur)) >> align) & ~resz & (resz
                                                                           - 1);
  res <<= align;

  while (1) {
    if (res)
      return (s + ffs(res) - 1);
    if (resz)
      return NULL;
    s += 16;
    cur = _mm_load_si128((__m128i const *) s);
    resz = _mm_movemask_epi8(_mm_cmpeq_epi8(zero, cur));
    res = _mm_movemask_epi8(_mm_cmpeq_epi8(seek, cur)) & ~resz & (resz - 1);
  }
}

#endif

/*
 * mystrlen is identical in function to strlen, but it can
 * read up to 3 bytes past the null.  Not an issue in this
 * code, as I know where all the buffers are, and there is ample
 * extra space at the end of each of them.
 *
 * This code would go faster if the pointers were aligned.
 */

static inline size_t mystrlen(const void *v) {
  const unsigned int *s1;
  const unsigned char *s;
  unsigned int t;
  size_t len = 0;
  s1 = (unsigned int *) v;

  while ((t = *s1)) {
    if ((t - 0x01010101) & (~t) & 0x80808080)
      break;
    len += 4;
    s1++;
  }
  s = (unsigned char *) s1;
  while (*s++) len++;
  return (len);
}


unsigned char i64hex[] = {
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 17, 255, 255, 17, 255, 255,   /* 00-0f */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* 10-1f */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1,     /* 20-2f */
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 255, 255, 255, 255, 255,                   /* 30-3f */
    255, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,                /* 40-4f */
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 255, 255, 255, 255, 255,            /* 50-5f */
    255, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,                /* 60-6f */
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 255, 255, 255, 255, 255,            /* 70-7f */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* 80-8f */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* 90-9f */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* a0-af */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* b0-bf */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* c0-cf */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* d0-df */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, /* e0-ef */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};/* f0-ff */

unsigned char trhex[] = {
    17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 16, 17, 16, 16, /* 00-0f */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 10-1f */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 20-2f */
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 16, 16, 16, 16, 16,           /* 30-3f */
    16, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 40-4f */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 50-5f */
    16, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 60-6f */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 70-7f */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 80-8f */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* 90-9f */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* a0-af */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* b0-bf */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* c0-cf */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* d0-df */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, /* e0-ef */
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16};/* f0-ff */



static int geti64(char *iline, unsigned char *dest, int len) {
  unsigned char c, c1, c2, c3, c4, *line = iline;
  unsigned int cas;
  int cnt;

  cnt = 0;
  while ((cnt + 3) <= len && (c1 = i64hex[*line++]) < 64) {
    c2 = i64hex[*line++];
    c3 = i64hex[*line++];
    c4 = i64hex[*line++];
    cas = c1 | ((c2 & 0x3f) << 6) | ((c3 & 0x3f) << 12) | ((c4 & 0x3f) << 18);
    *dest++ = (cas & 0xff);
    cnt++;
    if (c2 > 63)
      break;
    *dest++ = (cas >> 8) & 0xff;
    cnt++;
    if (c3 > 63)
      break;
    *dest++ = (cas >> 16) & 0xff;
    cnt++;
    if (c4 > 63)
      break;
  }
  return (cnt);
}


static int get32(char *iline, unsigned char *dest, int len) {
  unsigned char c, c1, c2, *line = iline;
  int cnt;
  unsigned char *tdest;
  uint64_t *curi, i;

  cnt = 0;
  while ((c = *line++)) {
    c1 = trhex[c];
    c2 = trhex[*line];
    if (c1 > 16 || c2 > 16)
      break;
    if (c1 < 16 && c2 < 16) {
      tdest = dest;
      cnt = 1;
      *tdest++ = (c1 << 4) + c2;
      line++;
      curi = (uint64_t *) line;
      while (1) {
        i = *curi++;
        c1 = trhex[(i & 255)];
        c2 = trhex[(i >> 8) & 255];
        if (c1 > 15 || c2 > 15 || cnt >= len)
          goto get32_exit;
        *tdest++ = (c1 << 4) + c2;
        cnt++;
        i >>= 16;
        c1 = trhex[(i & 255)];
        c2 = trhex[(i >> 8) & 255];
        if (c1 > 15 || c2 > 15 || cnt >= len)
          goto get32_exit;
        *tdest++ = (c1 << 4) + c2;
        cnt++;
        i >>= 16;
        c1 = trhex[(i & 255)];
        c2 = trhex[(i >> 8) & 255];
        if (c1 > 15 || c2 > 15 || cnt >= len)
          goto get32_exit;
        *tdest++ = (c1 << 4) + c2;
        cnt++;
        i >>= 16;
        c1 = trhex[(i & 255)];
        c2 = trhex[(i >> 8) & 255];
        if (c1 > 15 || c2 > 15 || cnt >= len)
          goto get32_exit;
        *tdest++ = (c1 << 4) + c2;
        cnt++;
      }
    }
  }
  get32_exit:
  return (cnt);
}


/*
 * commify will return a static pointer to a buffer containing
 * a null-terminated string, with a comma separting each group
 * of thousands.
 * For example, a supplied value of 123456789 will be returned
 * in a char * buffer like "123,456,789" with a null termination
 * ready for printing with %s.
 * You need to copy it, print it, or otherwise deal with it prior
 * to calling it again.  Not thread safe.
 */
static char Commify[128];

char *commify(unsigned long long source) {
  char temp[128];
  char *s, *d;
  int len, targlen, x;

  sprintf(temp, "%llu", source);
  len = mystrlen(temp);
  targlen = len + ((len - 1) / 3);
  d = &Commify[targlen];
  s = &temp[len];
  *d-- = *s--;
  for (x = 1; x <= len && d >= Commify; x++) {
    *d-- = *s--;
    if ((x % 3) == 0 && x && d >= Commify)
      *d-- = ',';
  }
  return (Commify);
}


char *prmd5REV(unsigned char *md5, char *out, int len) {
  char *ob;
  static char hextab[16] = "0123456789abcdef";
  int x;
  unsigned char v;

  ob = out + len;
  *ob-- = 0;
  for (x = 0; x < len / 2; x++) {
    v = *md5++;
    *ob-- = hextab[(v >> 4) & 0xf];
    *ob-- = hextab[v & 0xf];
  }
  return (out);
}

char *prmd5(unsigned char *md5, char *out, int len) {
  char *ob;
  static char hextab[16] = "0123456789abcdef";
  int x;
  unsigned char v;

  ob = out;
  for (x = 0; x < len / 2; x++) {
    v = *md5++;
    *ob++ = hextab[(v >> 4) & 0xf];
    *ob++ = hextab[v & 0xf];
  }
  *ob = 0;
  return (out);
}

char *prmd5UCREV(unsigned char *md5, char *out, int len) {
  char *ob;
  static char hextab[16] = "0123456789ABCDEF";
  int x;
  unsigned char v;

  ob = out + len;
  *ob-- = 0;
  for (x = 0; x < len / 2; x++) {
    v = *md5++;
    *ob-- = hextab[(v >> 4) & 0xf];
    *ob-- = hextab[v & 0xf];
  }
  return (out);
}

char *prmd5UC(unsigned char *md5, char *out, int len) {
  char *ob;
  static char hextab[16] = "0123456789ABCDEF";
  int x;
  unsigned char v;

  ob = out;
  for (x = 0; x < len / 2; x++) {
    v = *md5++;
    *ob++ = hextab[(v >> 4) & 0xf];
    *ob++ = hextab[v & 0xf];
  }
  *ob = 0;
  return (out);
}

void safecat(char *dest,char *src) {
  int dlen, slen;
  dlen = strlen(dest);
  slen = strlen(src);
  if (dlen > MDMAXPATHLEN) {
    fprintf(stderr,"Base string too long in safecat (%d bytes)\n",dlen);
    exit (1);
  }
  if (slen > MDMAXPATHLEN) {
    fprintf(stderr,"Source string too long in safecat (%d bytes)\n",slen);
    exit (1);
  }
  if ((dlen+slen) > MDMAXPATHLEN) {
    fprintf(stderr,"Strings too long in safecat (%d, %d bytes)\n",dlen,slen);
    exit (1);
  }
  strcat(dest,src);
}
    

Pvoid_t Files = NULL;
Word_t Filecount = 0;

void addfiles(char *path) {
  struct stat bstat;
  DIR *dir;
  struct dirent *result;
  char cur[MDMAXPATHLEN * 3], *new;
  int bufpathlen, len;
  PWord_t PV;

  bufpathlen = strlen(path);
  if (bufpathlen > MDMAXPATHLEN)
    return;

  if (stat(path, &bstat)) {
    fprintf(stderr, "Can't stat %s: ", path);
    perror(NULL);
    return;
  }
  if (bstat.st_mode & S_IFDIR) {
    dir = opendir(path);
    if (dir == NULL) {
      fprintf(stderr, "Can't open directory %s: ", path);
      perror(NULL);
      return;
    }
    strncpy(cur, path, bufpathlen);
    cur[bufpathlen] = 0;
    if (bufpathlen && cur[bufpathlen - 1] == '\\')
      cur[bufpathlen - 1] = '/';
    if (bufpathlen && cur[bufpathlen - 1] != '/') {
      cur[bufpathlen++] = '/';
      cur[bufpathlen] = 0;
    }
    while ((result = readdir(dir)) != NULL) {
      len = strlen(result->d_name);
      if (result->d_name[0] == '.')
        continue;
      if ((len + bufpathlen + 1) > MDMAXPATHLEN) {
        fprintf(stderr, "New path too long at %s, ignored\n", result->d_name);
      } else {
        strncpy(cur + bufpathlen, result->d_name, len);
        cur[len + bufpathlen] = 0;
        addfiles(cur);
      }
    }
    closedir(dir);
  } else {
    new = strdup(path);
    JLI(PV, Files, Filecount);
    *PV = (Word_t) new;
    Filecount++;
  }
}


char **dirprocess(int *argcp, char **argvp) {
  int myfiles = 0, myc;
  char cur[10240], **ret;
  Word_t filecount = 0, RC;
  PWord_t PV;
  struct stat bstat;
  DIR *dir;

  Filecount = 0;

  for (myc = 0; myc < *argcp; myc++)
    addfiles(argvp[myc]);
  fprintf(stderr, "Found %d files to process\n", (int) Filecount);
  ret = NULL;
  if (Filecount > 0) {
    ret = malloc(sizeof(char **) * Filecount + 16);
    if (!ret) {
      fprintf(stderr, "Out of memory adding files\n");
      exit(1);
    }
    for (myc = 0, filecount = 0; filecount < Filecount; filecount++) {
      JLG(PV, Files, filecount);
      if (PV && *PV)
        ret[myc++] = (char *) (*PV);
    }
    *argcp = myc;
    return (ret);
  } else {
    return (argvp);
  }
}


void closeall() {
  char line[MAXLINE + 16];
  Word_t *PV;
  struct FNInfo *curfn;


  line[0] = 0;
  JSLF(PV, Hfn, line);
  while (PV) {
    curfn = (struct FNInfo *) (*PV);
    if (curfn->fh) {
      if (fclose(curfn->fh)) {
        fprintf(stderr, "Can't close %s\n", curfn->filename);
        perror(curfn->filename);
      }
    }
    curfn->fh = NULL;
    JSLN(PV, Hfn, line);
  }
}

/* match_compact: look up a hex hash in the compact table.
 * line is the original (possibly mixed-case) hash from the file.
 * len is the length of the hash portion to match.
 * Returns a static Resinfo with match info, or NULL on miss. */
static struct Resinfo _compact_result;

struct Resinfo *match_compact(const char *line, int len) {
    uint64_t key;
    uint32_t fp;
    uint64_t pos;
    int p, best_match = -1, best_len2 = 0;
    uint32_t best_idx = 0;

    if (!CompactFP || !SolnCount) return NULL;

    key = decode_hex_key(line, len);
    fp = (uint32_t)(key >> 32);
    if (!fp) fp = 1;
    pos = compact_mix(key) & CompactMask;

    /* Prefetch the compact table slot — hides DRAM latency when table exceeds L3 */
    __builtin_prefetch(&CompactFP[pos], 0, 1);
    __builtin_prefetch(&CompactIdx[pos], 0, 1);

    /* First pass: find longest prefix match among all entries with matching fp */
    for (p = 0; p < COMPACT_MAX_PROBE; p++) {
        uint64_t cpos = (pos + p) & CompactMask;
        uint32_t idx;
        int x, hlen, matchlen;
        const char *found;
        signed char ch;

        if (!CompactFP[cpos]) break;
        if (CompactFP[cpos] != fp) continue;

        idx = CompactIdx[cpos];
        found = SolnBuf + SolnOff[idx];
        hlen = SolnHlen[idx];

        /* Compare hash portion case-insensitively */
        matchlen = (len < hlen) ? len : hlen;
        for (x = 0; x < matchlen; x++) {
            ch = line[x];
            if (isupper(ch)) ch = tolower(ch);
            if (ch != found[x]) break;
        }
        if (x > best_len2) {
            best_len2 = x;
            best_idx = idx;
            best_match = p;
        }
    }

    if (best_match < 0) return NULL;

    /* Second pass: among entries with the same best_len2, find the right boundary match */
    for (p = 0; p < COMPACT_MAX_PROBE; p++) {
        uint64_t cpos = (pos + p) & CompactMask;
        uint32_t idx;
        int x, hlen, matchlen;
        const char *found;
        signed char ch;

        if (!CompactFP[cpos]) break;
        if (CompactFP[cpos] != fp) continue;

        idx = CompactIdx[cpos];
        found = SolnBuf + SolnOff[idx];
        hlen = SolnHlen[idx];

        matchlen = (len < hlen) ? len : hlen;
        for (x = 0; x < matchlen; x++) {
            ch = line[x];
            if (isupper(ch)) ch = tolower(ch);
            if (ch != found[x]) break;
        }
        if (x == best_len2) {
            if (x >= len) {
                best_idx = idx;
                goto found_it;
            }
            /* Check if next char in file line is non-hex (hash boundary) */
            ch = line[x];
            if (trhex[((unsigned int)ch) & 0xff] > 15) {
                best_idx = idx;
                goto found_it;
            }
        }
    }
    /* No entry passed the boundary check — not a valid match */
    return NULL;

found_it:
    _compact_result.hashtype = SolnType[best_idx];
    _compact_result.found = SolnBuf + SolnOff[best_idx];
    _compact_result.hlen = SolnHlen[best_idx];
    _compact_result.len = SolnTotalLen[best_idx];
    _compact_result.next = NULL;
    return &_compact_result;
}

/* match_text: look up a text hash in the TextHash JudySL.
 * Preserves original linked-list collision chain semantics. */
struct Resinfo *match_text(char *tline, char *line, int len) {
    int x, len2;
    Word_t *PV;
    struct Resinfo *resi1;
    signed char ch;

    JSLG(PV, TextHash, tline);
    if (PV) {
        resi1 = (struct Resinfo *) (*PV);
        /* Find longest match */
        len2 = 0;
        do {
            for (x = 0; x < len && x < resi1->hlen; x++) {
                ch = line[x];
                if (resi1->hashtype->flags & FNInfo_hexhash)
                    if (isupper(ch)) ch = tolower(ch);
                if (ch != resi1->found[x]) break;
            }
            if (x > len2) len2 = x;
            resi1 = resi1->next;
        } while (resi1);
        resi1 = (struct Resinfo *) (*PV);
        do {
            for (x = 0; x < len && x < resi1->hlen; x++) {
                ch = line[x];
                if (resi1->hashtype->flags & FNInfo_hexhash)
                    if (isupper(ch)) ch = tolower(ch);
                if (ch != resi1->found[x]) break;
            }
            if (x == len2) {
                if (x >= len) break;
                if (resi1->hashtype->flags & FNInfo_texthash) {
                    ch = line[x];
                    if (ch == 0 || ch == Sep || ch <= ' ' || ch > 126) break;
                }
            }
            resi1 = resi1->next;
        } while (resi1);
        return (resi1);
    }
    return (NULL);
}

/* match: unified dispatch - hex hashes go to compact table, text to JudySL */
struct Resinfo *match(char *tline, char *line, int len) {
    struct Resinfo *r;

    /* Try compact table for hex hashes first */
    r = match_compact(line, len);
    if (r) return r;

    /* Try text hash JudySL */
    r = match_text(tline, line, len);
    return r;
}

char *stripprefix(char *src) {
  char *s = src + strlen(src);

  while (s > src) {
    s--;
    if (*s < ' ' || *s == '/' || *s ==':' || *s =='\\' || *s == '<' ||
        *s == '>' || *s == '|' || *s == '?' || *s == '*' || *s == '"') 
          return(s+1);
  }
  return(src);
}

  

unsigned long long process(char *initfi, int Procall, int Simulate) {
  char file1[MDMAXPATHLEN + 16], file2[MDMAXPATHLEN + 16], basename[MDMAXPATHLEN + 16], hashname[MDMAXPATHLEN + 16], lockname[MDMAXPATHLEN + 16];
  char line[MAXLINE + 16], tline[MAXLINE + 16], mdbuf[MAXLINE], rline[MAXLINE + 16], oline[MAXLINE + 16];
  char *s1, *s2, c, *cur;
  signed char ch;
  unsigned long long hitcount = 0;
  FILE *fi, *fo, *fh;
  int len, len1, len2, fl, x, gothit;
  struct stat bstat;
  Word_t *PV, RC, tr1;
  struct FNInfo *curfn;
  struct Resinfo *resi1, *resi2, *resil;
  union HashU curin, testin;

  len1 = mystrlen(initfi);
  if (len1 == 0)
    return (hitcount);
  if (len1 == 0 || len1 > (MDMAXPATHLEN - Maxsuf - 1)) {
    fprintf(stderr, "Filename too long: %s\n", initfi);
    exit(1);
  }
  strcpy(file1, initfi);
  fi = fopen(file1, "r");
  if (!fi) {
    safecat(file1, ".txt");
    fi = fopen(file1, "r");
    if (!fi) {
      fprintf(stderr, "Can't open %s or %s!\n", initfi, file1);
      perror(file1);
      return (hitcount);
    }
  }
  fclose(fi);
  strcpy(basename, file1);
  s1 = &basename[strlen(basename)];
  for (; s1 != basename; s1--)
    if (*s1 == '.')
      break;
  if (s1 == basename) {
    fprintf(stderr, "Can't find basename in %s\n", file1);
    return (hitcount);
  }
  *++s1 = 0;

  if (NoLock == 0) {
    strcpy(lockname, basename);
    safecat(lockname, "lock");
    len = 1;
    for (x = 0; x < 10; x++) {
#ifdef _WIN32
      fl = open(lockname,O_EXCL+O_CREAT,0644);
#else
      fl = open(lockname, O_CREAT, 0644);
#endif
      if (fl != -1)
        break;

      fprintf(stderr, "Cannot create lock file for %s - sleeping for %d\n", lockname, len);
      perror(lockname);
      sleep(len);
      len += x;
    }
    if (fl == -1)
      return (hitcount);

#ifndef _WIN32
    for (x = 0, len = 1; x < 10; x++) {
      if (flock(fl, LOCK_EX)) {
        fprintf(stderr, "Can't exclusive lock file %s\n", lockname);
        perror(lockname);
        if (x == 9) return (hitcount);
        sleep(len);
        len += x;
      } else {
        break;
      }
    }
#endif
  }
  fi = fopen(file1, "rb");
  if (!fi) {
    fprintf(stderr, "Can't re-open %s\n", file1);
    perror(file1);
    goto unlock;
  }
#ifndef _WIN32
  setbuffer(fi, Inbuf, memsize);
#endif
  sprintf(file2, "%s.w%d", file1, Mypid);
  if (!Simulate) {
    fo = fopen(file2, "wb");
    if (!fo) {
      fprintf(stderr, "Can't open temp file %s\n", file2);
      perror(file2);
      goto unlock;
    }
#ifndef _WIN32
    setbuffer(fo, Outbuf, memsize);
#endif
    fclose(fi);
    sprintf(file2, "%s.t%d", file1, Mypid);
    if (rename(file1, file2)) {
      fprintf(stderr, "Can't rename %s\n", file1);
      perror(file2);
      goto unlock;
    }
    fi = fopen(file2, "rb");
    if (!fi) {
      fprintf(stderr, "Can't re-open %s\n", file2);
      perror(file2);
      goto unlock;
    }
#ifndef _WIN32
    setbuffer(fi, Inbuf, memsize);
#endif
#ifdef __linux__
    posix_fadvise(fileno(fi), 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

    sprintf(file2, "%s.w%d", file1, Mypid);
  }
  while (fgets(line, MAXLINE, fi)) {
    cur = findchr(line, 13);
    if (cur)
      *cur = 0;
    cur = findchr(line, 10);
    if (cur)
      *cur = 0;
    len = mystrlen(line);
    memmove(oline, line, len + 1);
    gothit = 0;
    if (len > 7) {
      len1 = (len > 16) ? 16 : len;
      for (x = 0; x < len1; x++) {
        ch = line[x];
        if (trhex[(ch & 0xff)] > 15) 
          len1 = 30;
        if (Longmatch == 0 && ch == Sep) {
          len1 = x;
          break;
        }
        if (isupper(ch))
          ch = tolower(ch);
        tline[x] = ch;
      }
      tline[len1] = 0;
      resi1 = match(tline, line, len);
      if (resi1) {
        gothit++;
        hitcount++;
        curfn = resi1->hashtype;
        curfn->hitcount++;
        if (len >= resi1->hlen) {
	  if (strlen(prefix)) {
	    strcpy(hashname,prefix);
	    safecat(hashname, stripprefix(basename));
	  } else {
	    strcpy(hashname,basename);
	  }
	  safecat(hashname, infix);
          safecat(hashname, curfn->type);
 	  safecat(hashname, suffix);
          if (!Simulate) {
            if (curfn->fh && strcmp(hashname, curfn->filename) != 0) {
              if (fclose(curfn->fh)) {
                curfn->fh = 0;
                fprintf(stderr, "Can't close %s\n", curfn->filename);
                perror(curfn->filename);
                goto unlock;
              }
            }
            fh = curfn->fh;
            if (!fh) {
              fh = fopen(hashname, "ab");
              if (!fh) {
                closeall();
                fh = fopen(hashname, "ab");
              }
              if (!fh) {
                fprintf(stderr, "Can't append to %s\n", hashname);
                perror(hashname);
                goto unlock;
              }
              strncpy(curfn->filename, hashname, MDMAXPATHLEN);
              curfn->fh = fh;
#ifndef _WIN32
              /* Assign a write buffer from the pool for better I/O batching */
              if (HitBufUsed < HITBUF_COUNT) {
                if (!HitBufPool[HitBufUsed])
                  HitBufPool[HitBufUsed] = malloc(HITBUF_SIZE);
                if (HitBufPool[HitBufUsed])
                  setbuffer(fh, HitBufPool[HitBufUsed++], HITBUF_SIZE);
              }
#endif
            }
            if (fprintf(fh, "%s\n", resi1->found) < 0) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
          }
        } else {
          len &= -2;
          sprintf(tline, "%schop%d", resi1->hashtype->type, len);
          line[len] = 0;
	  if (strlen(prefix)) {
	    strcpy(hashname,prefix);
	    safecat(hashname, stripprefix(basename));
	  } else {
	    strcpy(hashname,basename);
	  }
	  safecat(hashname, infix);
          safecat(hashname, tline);
 	  safecat(hashname, suffix);
          if (!Simulate) {
            fh = fopen(hashname, "ab");
            if (!fh) {
              closeall();
              fh = fopen(hashname, "ab");
            }
            if (!fh) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
            if (fprintf(fh, "%s%s\n", line, &resi1->found[resi1->hlen]) < 0) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
            if (fclose(fh)) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
          }

        }
      }
      for (x = 0; x < len; x++) {
        if (line[x] == Sep) {
          len = x;
          break;
        }
      }
      line[len] = 0;
      for (x = 0; x < len; x++)
        rline[x] = line[len - x - 1];
      rline[len] = 0;
      len1 = (len > 16) ? 16 : len;
      for (x = 0; x < len1; x++) {
        ch = rline[x];
        if (trhex[(ch & 0xff)] > 15) 
          len1 = 30;
        if (Longmatch == 0 && ch == Sep) {
          len1 = x;
          break;
        }
        if (isupper(ch))
          ch = tolower(ch);
        tline[x] = ch;
      }
      tline[len1] = 0;
      resi1 = match(tline, rline, len);
      if (resi1 && NoReverse == 0) {
        gothit++;
        hitcount++;
        curfn = resi1->hashtype;
        curfn->hitcount++;
        if (len >= resi1->hlen) {
	  if (strlen(prefix)) {
	    strcpy(hashname,prefix);
	    safecat(hashname, stripprefix(basename));
	  } else {
	    strcpy(hashname,basename);
	  }
	  safecat(hashname, infix);
          safecat(hashname, "r");
          safecat(hashname, curfn->type);
 	  safecat(hashname, suffix);
          if (!Simulate) {
            fh = fopen(hashname, "ab");
            if (!fh) {
              closeall();
              fh = fopen(hashname, "ab");
            }
            if (!fh) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
            line[resi1->hlen] = 0;
            if (fprintf(fh, "%s%s\n", line, &resi1->found[resi1->hlen]) < 0) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
            if (fclose(fh)) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
          }
        } else {
          len &= -2;
          sprintf(tline, "r%schop%d", resi1->hashtype->type, len);
          rline[len] = 0;
          line[len] = 0;
	  if (strlen(prefix)) {
	    strcpy(hashname,prefix);
	    safecat(hashname, stripprefix(basename));
	  } else {
	    strcpy(hashname,basename);
	  }
	  safecat(hashname, infix);
          safecat(hashname, tline);
 	  safecat(hashname, suffix);
          if (!Simulate) {
            fh = fopen(hashname, "ab");
            if (!fh) {
              closeall();
              fh = fopen(hashname, "ab");
            }
            if (!fh) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
            if (fprintf(fh, "%s%s\n", line, &resi1->found[resi1->hlen]) < 0) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
            if (fclose(fh)) {
              fprintf(stderr, "Can't append to %s\n", hashname);
              perror(hashname);
              goto unlock;
            }
          }
        }
      }
      if (gothit)
        continue;
    }
    if (!Simulate) {
      if (fprintf(fo, "%s\n", oline) < 0) {
        fprintf(stderr, "Can't write output file!");
        perror(file2);
        exit(1);
      }
    }
  }
  if (fclose(fi)) {
    fprintf(stderr, "Can't close input file %s\n", file1);
    perror(file1);
    exit(1);
  }
  closeall();
  if (!Simulate) {
    fi = fopen(file1, "rb");
    if (fi) {
      while (fgets(line, MAXLINE, fi)) {
        cur = findchr(line, 13);
        if (cur)
          *cur = 0;
        cur = findchr(line, 10);
        if (cur)
          *cur = 0;
        if (fprintf(fo, "%s\n", line) < 0) {
          fprintf(stderr, "Can't write output file!");
          perror(file2);
          exit(1);
        }
      }
      fclose(fi);
      unlink(file1);
    }

    if (fclose(fo)) {
      fprintf(stderr, "Can't close output file %s\n", file2);
      perror(file2);
      exit(1);
    }
    if (hitcount) {
      sprintf(file2, "%s.t%d", file1, Mypid);
      unlink(file2);
      sprintf(file2, "%s.w%d", file1, Mypid);
    } else {
      sprintf(file2, "%s.w%d", file1, Mypid);
      unlink(file2);
      sprintf(file2, "%s.t%d", file1, Mypid);
    }
    if (rename(file2, file1)) {
      fprintf(stderr, "Can't rename %s to %s!", file2, file1);
      perror(file2);
    }
  }
  unlock:
  closeall();
  close(fl);
  if (NoLock == 0)
    unlink(lockname);
  if (hitcount)
    printf("%s had %llu hits\n", file1, hitcount);
  return (hitcount);
}


int main(int argc, char **argv) {
  FILE *fi;
  Word_t *PV, RC, tr1;
  char line[MAXLINE + 16], tline[MAXLINE + 16];
  char *cur, *hp, *eoh, *type, *Prog;
  signed char ch;
  struct FNInfo *curfn;
  struct Resinfo *resi1, *resi2, *resil;
  union HashU curin, testin;
  int Simulate = 0, Procall = 0, len, Exthash = 0, hlen, tlen, x, y;
  int Hashcount;
  int Detail = 0, first = 1, doutf16 = 0;
  unsigned long long Inres = 0, Infound = 0;

  Prog = argv[0];
  prefix = infix = suffix = "";
  Mypid = getpid();
  fi = stdin;
  while ((ch = getopt(argc, argv, "?adehlrVb:f:i:p:t:s:S:")) != -1) {
    switch (ch) {
      case 'V':
	fprintf(stderr,"%s: %s\n",Prog,Version);
	exit(1);
	break;
      case 'a':
        Procall = 1;
        fprintf(stderr, "Processing all files matching basename\n");
        break;
      case 'b':
#ifdef _WIN32
        fprintf(stderr,"-b unsupported in Windows\n");
        break;
#else
        memsize = atoi(optarg);
        if (memsize < 1) {
          fprintf(stderr, "size %d is not supported on this platform\n", memsize);
          memsize = FILEBUFSIZE;
          break;
        }
        ch = 'k';
        if (optarg[0])
          ch = optarg[strlen(optarg) - 1];
        switch (ch) {
          default:
          case 'k':
          case 'K':
            memsize *= 1024;
            break;
          case 'M':
          case 'm':
            memsize *= 1024 * 1024;
            break;
          case 'G':
          case 'g':
            memsize *= 1024 * 1024 * 1024;
            break;
          case 'b':
          case 'B':
            break;
        }
        fprintf(stderr, "Memory size for buffers set to %d bytes\n", memsize);
        break;
#endif

      case 'd':
        Detail = 1;
        fprintf(stderr, "Detail stats enabled\n");
        break;
      case 'e':
        Exact = 1;
	fprintf(stderr,"Exact match required\n");
	break;
      case 'l':
        NoLock = 1;
        fprintf(stderr, "No locking of files - good for shared filesystems.\n");
        break;
      case 'r':
        NoReverse = 1;
        break;
      case 'f':
        fi = fopen(optarg, "rb");
        if (!fi) {
          fprintf(stderr, "Can't open file %s to read results\n", optarg);
          perror(optarg);
          exit(1);
        }
        break;
      case 't':
        if (mystrlen(optarg) < 3 || mystrlen(optarg) > 64) {
          fprintf(stderr, "length of optional type is invalid: %d\n", (int) strlen(optarg));
          exit(1);
        }
        DefaultType = strdup(optarg);
        if (!DefaultType) {
          fprintf(stderr, "Out of memory for type\n");
          exit(1);
        }
        fprintf(stderr, "Default type set to %s\n", DefaultType);
        break;

      case '?':
      case 'h':
      help:
        printf("%s\n", Version);
        printf("mdsplit - processes MDXfind result files\n");
	printf("-V        Display version and exit\n");
	printf("-h or -?  Display this usage\n");
        printf("-a        Process all files, not just .txt, looking for matches\n");
        printf("-l        Do not lock files. If your filesystem can't lock...\n");
        printf("-r        No reverse hashes scanned (default is to scan for reverse)\n");
        printf("-b 500K   Buffer size 500K (or M or G) - default is %dK\n", FILEBUFSIZE / 1024);
        printf("-f file   Reads result file from filename, instead of stdin\n");
        printf("-t MD5x01 Uses the type MD5x01 for results that have no type\n");
	printf("-p str    Prepends the string to each solution filename\n");  
	printf("-i str    Adds the string just after the . in the solution filename\n");
	printf("-s str    Adds the string to the end of the solution filename");
        printf("\nmdsplit processes MDXfind result files, separating hash lists into\nsolved files. Unsolved hashes should be in .txt files\n\nTypical usage:\ncat *.txt | mdxfind -i 3 /tmp/words | mdsplit *.txt\ncat foo.res bar.res | mdsplit /other/unsolved\n\nmdsplit -f foo.res biglist\n");
        exit(0);
        break;
      case 'i':
	infix = strdup(optarg);
	fprintf(stderr,"Infix set to %s\n",infix);
	break;
      case 'p':
	prefix = strdup(optarg);
	fprintf(stderr,"Prefix set to %s\n",prefix);
	break;
      case 's':
        suffix = strdup(optarg);
	fprintf(stderr,"Suffix set to %s\n",suffix);
        break;
      case 'S':
        Sep = optarg[0];
	fprintf(stderr,"Separator set to %c\n",Sep);
	break;
      default:
        break;
    }
  }

  argc -= optind;
  argv += optind;
  if (argc <= 0) goto help;
#ifndef _WIN32
  Inbuf = malloc(memsize);
  Outbuf = malloc(memsize);
  if (!Inbuf || !Outbuf) {
    fprintf(stderr, "Cannot allocate %d bytes for file buffers!\n", memsize);
    exit(1);
  }
#endif


#ifndef _WIN32
  setbuffer(fi, Inbuf, memsize);
#endif
  Hashcount = Inres = 0;
  first = 1;
  doutf16 = 0;
  while (fgets(line, MAXLINE, fi)) {
    line[MAXLINE] = 10;
    if (first) {
      unsigned short int *si1;
      first = 0;
      /* 000000    fffe 5300 4800 4100  */
      /* 000000    efbb bf53 */
      si1 = (unsigned short int *) line;
      switch (*si1) {
        case 0xfeff:
          si1++;
          for (x = 0; (ch = (*si1++) & 0xff); x++)
            line[x] = ch;
          line[x] = 0;
          doutf16 = 1;
          fgetc(fi);
          break;
        case 0xfffe:
          si1++;
          for (x = 0; ch = ((*si1++) & 0xff00) >> 8; x++)
            line[x] = ch;
          line[x] = 0;
          doutf16 = 2;
          fgetc(fi);
          break;
        case 0xbbef:
          if ((line[2] & 0xff) == 0xbf) {
            memmove(line, line + 3, MAXLINE - 3);
          } else {
            fprintf(stderr, "Unknown BOM encoding on result file: %02x %02x %02x\n", line[0] & 0xff, line[1] & 0xff, line[2] & 0xff);
            exit(1);
          }
          break;
        case 0x3184:
          if ((line[2] & 0xff) == 0x95 && line[3] == 0x33) {
            memmove(line, line + 4, MAXLINE - 4);
          } else {
            fprintf(stderr, "Unknown BOM encoding on result file: %02x %02x %02x %02x\n", line[0] & 0xff, line[1] & 0xff, line[2] & 0xff, line[3] & 0xff);
            exit(1);
          }
          break;

        case 0x2f2b:
          if (line[2] == 0x76 && (line[3] == 0x38 ||
                                  line[3] == 0x39 ||
                                  line[3] == 0x2b ||
                                  line[3] == 0x2f ||
                                  line[3] == 0x38)) {
            x = 4;
            if (line[3] == 0x38 && line[4] == 0x2d)
              x = 5;
            memmove(line, line + x, MAXLINE - x);
          } else {
            fprintf(stderr, "Unknown BOM encoding on result file: %02x %02x %02x %02x\n", line[0] & 0xff, line[1] & 0xff, line[2] & 0xff, line[3] & 0xff);
            exit(1);
          }
          break;

        default:
          break;
      }


    } else if (doutf16) {
      unsigned short int *si1;
      si1 = (unsigned short int *) line;
      if (doutf16 == 1)
        for (x = 0; (ch = (*si1++) & 0xff); x++)
          line[x] = ch;
      else
        for (x = 0; ch = ((*si1++) & 0xff00) >> 8; x++)
          line[x] = ch;
      line[x] = 0;
      fgetc(fi);
    }
    cur = findchr(line, 13);
    if (cur)
      *cur = 0;
    cur = findchr(line, 10);
    if (cur)
      *cur = 0;
    len = mystrlen(line);
    if (len > MAXLINE)
      len = MAXLINE - 1;
    line[len] = 0;
    cur = findchr(line, ' ');
    eoh = findchr(line, Sep);
    if (!eoh)
      continue;
    if (!DefaultType && cur && eoh && (eoh-line) < len && eoh < cur)
       eoh = findchr(cur,Sep);
    if (cur && cur < eoh && len > 12 && (cur - line) < 50) {
      type = line;
      *cur++ = 0;
      len -= (cur - line);
    } else {
      if (!DefaultType) {
        fprintf(stderr, "No type in input, and no default type requested at line %s.\n", commify(Inres + 1));
        fprintf(stderr, "%s\n", line);
        exit(1);
      }
      type = DefaultType;
      cur = line;
    }
    hlen = eoh - cur;
    if (hlen < 8) { /* Try for longer match */
      int nm;
      char *lasteoh;
      for (nm=0; nm <2; nm++) {
	lasteoh = eoh;
	eoh = findchr(eoh+1,Sep);
	if (!eoh) {
	  eoh = lasteoh;
	  break;
	 }
      }
      if (!eoh) continue;
      hlen = eoh - cur;
      if (hlen < 8) 
        continue;
    }

    Inres++;
    JSLG(PV, Hfn, type);
    if (!PV) {
      long iter = 0;
      char *it;
      Hashcount++;
      it = findchr(type, 'x');
      if (it)
        iter = atol(&it[1]);
      JSLI(PV, Hfn, type);
      if (!PV) {
        fprintf(stderr, "Judy add error on HFN\n");
        exit(1);
      }

      curfn = mymalloc(sizeof(struct FNInfo), 8);
      /* Classify as text hash if any non-hex character appears in the hash.
       * Scan the full hash portion — catches all structured formats ($2b$, $6$,
       * $argon2id$, etc.) without needing a hard-coded type list. */
      curfn->flags = FNInfo_hexhash;
      for (x = 0; x < hlen; x++) {
        if (trhex[((unsigned char)cur[x]) & 0xff] > 15) {
          curfn->flags = FNInfo_texthash;
          break;
        }
      }
      curfn->hitcount = curfn->incount = 0;
      curfn->iter = iter;
      curfn->type = strdup(type);
      *PV = (Word_t) curfn;
    }
    curfn = (struct FNInfo *) (*PV);
    curfn->incount++;

    if (curfn->flags & FNInfo_hexhash) {
      /* Hex hash: lowercase the found string's hash portion, store in compact arrays */
      char *lowered = mymalloc(len + 1, 1);
      uint64_t key;
      memmove(lowered, cur, len + 1);
      for (x = 0; x < hlen; x++)
        if (isupper(lowered[x]))
          lowered[x] = tolower(lowered[x]);
      key = decode_hex_key(lowered, hlen);
      soln_add(lowered, len, hlen, curfn, key);
    } else {
      /* Text hash: insert into TextHash JudySL with Resinfo chain */
      tlen = (hlen > 30) ? 30 : hlen;
      for (x = 0; x < tlen; x++) {
        ch = cur[x];
        if (Longmatch == 0 && ch == Sep) {
          tlen = x;
          break;
        }
        if (isupper(ch))
          ch = tolower(ch);
        tline[x] = ch;
      }
      tline[tlen] = 0;
      resi1 = mymalloc(sizeof(struct Resinfo), 8);
      resi1->next = NULL;
      resi1->hashtype = curfn;
      resi1->hlen = hlen;
      resi1->len = len;
      resi1->found = mymalloc(len + 1, 1);
      memmove(resi1->found, cur, len + 1);
      JSLG(PV, TextHash, tline);
      if (PV) {
        resi2 = (struct Resinfo *) (*PV);
        do {
          resil = resi2;
          if (memcmp(resi2->found, cur, hlen) == 0) {
            if (len < resi2->len || (len == resi2->len && curfn->iter > resi2->hashtype->iter)) {
              resi2->hashtype = resi1->hashtype;
              resi2->hlen = resi1->hlen;
              resi2->len = resi1->len;
              resi2->found = resi1->found;
            }
            break;
          }
          resi2 = resi2->next;
        } while (resi2);
        if (!resi2)
          resil->next = resi1;
      } else {
        JSLI(PV, TextHash, tline);
        if (!PV) {
          fprintf(stderr, "TextHash add fail\n");
          exit(1);
        }
        *PV = (Word_t) resi1;
      }
    }
  }
  fclose(fi);
  printf("%s result lines processed, %d type%s found\n", commify(Inres), Hashcount, (Hashcount == 1) ? "" : "s");
  if (SolnCount)
    build_compact();
  if (Hashcount == 0) exit(0);

  line[0] = 0;
  JSLF(PV, Hfn, line);
  while (PV) {
    x = mystrlen(line);
    if (Maxsuf < x) Maxsuf = x;
    curfn = (struct FNInfo *) (*PV);
    printf("%s ", line);
    JSLN(PV, Hfn, line);
  }
  printf("\n");

  for (x = 0; x < argc; x++)
    Infound += process(argv[x], Procall, Simulate);

  if (Detail) {
    line[0] = 0;
    JSLF(PV, Hfn, line);
    if (Maxsuf < 9) Maxsuf = 9;
    printf("\n%*s %s\n", Maxsuf, "Hash Type", "Hit count");
    while (PV) {
      curfn = (struct FNInfo *) (*PV);
      if (curfn->hitcount)
        printf("%*s %s\n", Maxsuf, line, commify(curfn->hitcount));
      JSLN(PV, Hfn, line);
    }
    printf("\n");
  }

  printf("\nTotal %s hashes found\n", commify(Infound));
  return (0);

}
