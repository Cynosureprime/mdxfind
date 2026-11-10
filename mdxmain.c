/*
 * mdxmain.c -- main() and option parsing, the hash-list and wordlist loaders,
 * the readers for compressed input, the compact-table build and statistics.
 * The hash engine, procjob and everything it calls, is in mdxfind.c; what
 * the two share is declared in mdxint.h.
 *
 * $Revision: 1.5 $
 * $Log: mdxmain.c,v $
 * Revision 1.5  2026/11/09 23:00:16  dlr
 * Include md6.h in mdxfind.c only. md6.h defines compression_hook instead of declaring it, and since the split both mdxfind.c and mdxmain.c included it through mdxint.h. gcc merges the two as a common symbol, so Linux, FreeBSD and macOS linked; llvm-mingw compiles with -fno-common and the Windows link failed with duplicate symbol compression_hook. mdxint.h now includes md6.h unless MDX_NO_MD6 is defined, and mdxmain.c, which uses only the JOB_MD6 constants, defines it. Verified with the Windows x86_64 release objects: mdxmain.o no longer defines the symbol and the release link line produces a 13,590,528-byte PE32+ executable.
 *
 * Revision 1.4  2026/11/09 22:32:22  dlr
 * Include pthread.h for the ReportStats condition variable. The Windows mingw cross-build does not reach it through the other headers and failed with PTHREAD_COND_INITIALIZER undeclared; Linux, FreeBSD and macOS reached it indirectly. Verified with the x86_64 Windows rule on the cross host: the unfixed file fails with the same errors as the release build, the fixed one compiles.
 *
 * Revision 1.3  2026/11/09 20:54:20  dlr
 * Read pipeline: four read buffers, a reader thread, and an indexer that handles every line ending in a 64-byte block in one loop. A wordlist or hash file now passes through three stages, each owning one buffer of the NREADBUF=4 ring at a time: rb_reader reads, cacheline indexes on the caller thread, and the workers hash. On a cheap hash the run goes at the speed of the slowest stage rather than the sum of reading and indexing. Four rather than three because hashing usually holds two buffers, one draining its last jobs and one active. The reader hands cacheline whole lines only: after each read rb_cut finds the cut just past the last newline, or past the RINDEXSIZE-1 th newline if that comes first, and the rest is carried to the front of the next buffer, so the index cannot overflow. The cut applies at end of file too; a buffer is the last only when all that remains fits. As before, a newline in the last byte read waits for the next read, an unterminated tail of MAXLINE or more is dropped, and the last line of a file needs no newline. A buffer returns to the reader only when cacheline has moved past it and its job count, ReadBufL[i], is zero. ReadBuf0 and ReadBuf1 are replaced by the ReadBufL array and readbuf_lock(). gzread errors are now FATAL with file:line. Measured on mmt, reader-only e426 runs: 29m.pass x8 from NVMe, 232M lines, 2.31-2.48 s to 1.66-1.82 s at 36 threads and 2.92-3.04 s to 2.15-2.17 s at 72; hashmob.net_2026.found from ZFS, 2.49G lines, 51.9 s to 39.6-40.2 s, where the reader thread is now the limit and 83 percent of its time is ZFS LZ4 decompression and copy in the kernel; MD5 best64 on 10m.pass x3, 9.94-10.71 s to 9.24-9.35 s, 581,687 found both. -z output and line counts identical to the previous reader on arm64 and x86 across no trailing newline, empty, a lone newline, CRLF, $HEX[] edge cases, 300,000 lines of length 0-1000, lines at and past MAXLINE, a 30 MB line with and without a final newline, 20M lines plain and gzipped, stdin, and 30M one-byte lines that force cuts at the index cap. The one difference is -d on the one-byte files, 20 against 21 lines, which is expected: -d deduplicates within a buffer, so its output depends on where buffers are cut. ALSO: seconds hashing is no longer quantised to whole seconds. ReportStats waited out a sleep(1) before seeing the stop signal, and shutdown joins it before taking the end time; it now waits on a condition variable with a one-second timeout that stats_wake ends at once. MD4 skips the hex conversion after the last iteration, 18.15 s to 17.15 s on dev3 with identical results at x01-x04. The 1.666 log sentence in the $Log comment is corrected to match the rewritten RCS log.
 *
 * Revision 1.2  2026/11/09 16:10:46  dlr
 * -e takes 0, 1 or 2 and is applied after loading, since addhash undid the old parse-time setting. Before the compact table is built, main compares the shortest loaded hash with the largest TypeDigest of the selected types, announces extended matching once when it applies, keeps the GPU off for extended matching, the XOR key or rotation, and folds the pre-check validity into Precheck_maxlen. The usage text now lists -e.
 *
 * Revision 1.1  2026/11/09 04:21:36  dlr
 * New. main(), option parsing, the hash and wordlist loaders, the compressed-input readers, the compact-table build and statistics, moved unchanged from mdxfind.c 1.664.
 *
 */
#define MDX_NO_MD6 1    /* see mdxint.h: only mdxfind.c may include md6.h */
#include "mdxint.h"

static int Extmatch_set;   /* -e was given explicitly */

const struct Benchrates { int idx; long long rate; } bench_rates[] = {
#include "bench_rates.h"
,{INT_MAX,0}
};

/*
 * Dual-channel hashcat modes.
 *
 * hashcat has one trailing operand, calls it $salt, and has no userid concept
 * at all.  mdxfind has both channels, and for these four constructions the
 * catalog carries the SAME expression twice -- once reading the trailing
 * operand from -u/-U, once from -s/-S.  The digest is identical either way;
 * only the channel the operator feeds differs.  Maphashcat can name just one
 * of each pair, so `-m <mode> -s saltfile` used to select the userid member,
 * discard the salt file, invent a userid pool, and report a clean run that
 * found nothing.
 *
 * -m now picks the member matching the channel actually supplied.  -M is
 * deliberately NOT routed through this table: -M goes with -F, the structured
 * loader lifts the trailing field off the hash line into whichever channel the
 * selected type owns, and the input and output lines are then identical for
 * either member -- there is nothing to disambiguate, so the long-standing
 * mapping is the right one to keep.
 *
 * Idea: @osh.
 */
static const struct {
  unsigned int   hc;    /* hashcat mode                                      */
  unsigned short user;  /* member reading the operand from -u/-U             */
  unsigned short salt;  /* member reading it from -s/-S; 0 if none exists    */
} Mapdualchannel[] = {
    { 3610, 356, 347},  /* md5(md5(md5(pass)) . X)                           */
    { 4510, 310, 579},  /* sha1(sha1(pass) . X)                              */
    { 4711, 308, 471},  /* Huawei sha1(md5(pass) . X).  e578 and e587 are
                         * historical duplicates of e471; hx_dedup_check
                         * reports e471, so that is the one dispatched to.    */
    {32420, 510,   0},  /* sha512(sha512_bin(pass) . X) -- the operand is
                         * appended to the RAW digest.  e387 appends to the
                         * HEX one, so it is not a twin and there is no
                         * salt-channel member to dispatch to.                */
    {0, 0, 0}};



unsigned int matchhash(char *src) {
  char *s, *t;
  int x;
  for (x = 1; Types[x]; x++) {
    s = src;
    t = Types[x];
    while (*s && *t) {
      if (toupper(*s) != toupper(*t))
        break;
      s++;
      t++;
      if (*s == 0 && *t == 0)
        return (x);
    }
  }
  return (0);
}

int rulecmp(const void *a, const void *b) {
  struct RuleHist *a1, *b1;
  a1 = (struct RuleHist *) a;
  b1 = (struct RuleHist *) b;
  if (a1->val > b1->val)
    return (-1);
  if (a1->val < b1->val)
    return (1);
  return (strcmp(a1->rule, b1->rule));
}

static unsigned char *Localreadbuf;

int linecmp(const void *a, const void *b) {
  struct LineInfo *a1, *b1;
  unsigned char *s1, *s2;
  int len;

  a1 = (struct LineInfo *) a;
  b1 = (struct LineInfo *) b;

  s1 = Localreadbuf + a1->offset;
  s2 = Localreadbuf + b1->offset;

  len = a1->len;
  if (len > b1->len) len = b1->len;

  while (len) {
    if (*s1 != *s2)
      return ((int) (*s1) - (int) (*s2));
    len--;
    s1++;
    s2++;
  }
  return ((int) (*s1) - (int) (*s2));
}



/* ---- Compressed file reader ----
 * Auto-detects XZ, bzip2, and ZIP formats on first read from gzread().
 * gzip and plaintext are handled transparently by zlib.
 * When a non-gzip compressed format is detected, subsequent reads
 * use the appropriate library decoder instead of gzread(). */

enum compress_type { CMP_GZ = 0, CMP_XZ, CMP_BZ2, CMP_ZIP, CMP_ZSTD };

static struct {
    enum compress_type type;
    int detected;          /* 1 = first read done, format known */
    int gz_closed;         /* 1 = gzFile was closed (non-gzip format detected) */
    int fd;                /* file descriptor for library reads */
    char filename[4096];   /* saved filename for reopen */
    int eof;               /* decoder reached end of stream */
    /* XZ state */
    lzma_stream xz;        /* LZMA_STREAM_INIT */
    unsigned char xz_inbuf[256*1024];
    /* bzip2 state — low-level streaming API */
    bz_stream bz;
    unsigned char bz_inbuf[256*1024];
    /* ZIP state — uses zlib inflate, liblzma, libbz2 or zstd depending on method */
    z_stream zip;
    int zip_init;          /* 1=deflate, 2=lzma, 3=bzip2, 4=zstd in ZIP */
    int zip_ended;         /* deflate reached Z_STREAM_END */
    uint64_t zip_remaining;  /* compressed bytes remaining; 64-bit for zip64 on 32-bit hosts */
    unsigned char zip_inbuf[256*1024];
    /* zstd state */
    ZSTD_DStream *zstd_ds;
    ZSTD_inBuffer zstd_in;
    unsigned char zstd_inbuf[256*1024];
} Creader;

/* Read from the active decoder. Returns bytes read, 0 on EOF. */
static size_t creader_read(void *buf, size_t len) {
    if (Creader.type == CMP_XZ) {
        Creader.xz.next_out = (uint8_t *)buf;
        Creader.xz.avail_out = len;
        while (Creader.xz.avail_out > 0) {
            if (Creader.xz.avail_in == 0 && !Creader.eof) {
                ssize_t n = read(Creader.fd, Creader.xz_inbuf, sizeof(Creader.xz_inbuf));
                if (n <= 0) Creader.eof = 1;
                else { Creader.xz.next_in = Creader.xz_inbuf; Creader.xz.avail_in = n; }
            }
            lzma_ret r = lzma_code(&Creader.xz, Creader.eof ? LZMA_FINISH : LZMA_RUN);
            if (r == LZMA_STREAM_END) { Creader.eof = 1; break; }
            if (r != LZMA_OK) { Creader.eof = 1; break; }
            if (Creader.xz.avail_in == 0 && Creader.eof) break;
        }
        return len - Creader.xz.avail_out;
    } else if (Creader.type == CMP_BZ2) {
        Creader.bz.next_out = (char *)buf;
        Creader.bz.avail_out = (unsigned)len;
        while (Creader.bz.avail_out > 0) {
            if (Creader.bz.avail_in == 0 && !Creader.eof) {
                ssize_t n = read(Creader.fd, Creader.bz_inbuf, sizeof(Creader.bz_inbuf));
                if (n <= 0) Creader.eof = 1;
                else { Creader.bz.next_in = (char *)Creader.bz_inbuf; Creader.bz.avail_in = n; }
            }
            int r = BZ2_bzDecompress(&Creader.bz);
            if (r == BZ_STREAM_END) { Creader.eof = 1; break; }
            if (r != BZ_OK) { Creader.eof = 1; break; }
            if (Creader.bz.avail_in == 0 && Creader.eof) break;
        }
        return len - Creader.bz.avail_out;
    } else if (Creader.type == CMP_ZIP) {
        /* ZIP: dispatch based on zip_init (0=stored, 1=deflate, 2=lzma, 3=bzip2, 4=zstd) */
        if (Creader.zip_init == 0) {
            /* Stored — raw read */
            size_t want = len;
            if (want > Creader.zip_remaining) want = Creader.zip_remaining;
            if (want == 0) { Creader.eof = 1; return 0; }
            ssize_t n = read(Creader.fd, buf, want);
            if (n <= 0) { Creader.eof = 1; return 0; }
            Creader.zip_remaining -= n;
            if (Creader.zip_remaining == 0) Creader.eof = 1;
            return (size_t)n;
        }
        /* Read compressed data from fd into zip_inbuf as needed */
        #define ZIP_FILL_INPUT() do { \
            if (Creader.zip_remaining > 0) { \
                size_t want = sizeof(Creader.zip_inbuf); \
                if (want > Creader.zip_remaining) want = Creader.zip_remaining; \
                ssize_t n = read(Creader.fd, Creader.zip_inbuf, want); \
                if (n <= 0) { Creader.eof = 1; break; } \
                Creader.zip_remaining -= n; \
                zip_got = (size_t)n; \
            } else { Creader.eof = 1; } \
        } while(0)

        if (Creader.zip_init == 1) {
            /* Deflate.  The stream's own end is the only clean end: input that
             * runs out first, or an inflate error, means the wordlist was not
             * read whole, and treating either as EOF reports a partial run as
             * a clean one. */
            if (Creader.zip_ended) return 0;
            Creader.zip.next_out = (unsigned char *)buf;
            Creader.zip.avail_out = (unsigned)len;
            while (Creader.zip.avail_out > 0) {
                if (Creader.zip.avail_in == 0 && Creader.zip_remaining > 0) {
                    size_t zip_got = 0; ZIP_FILL_INPUT();
                    Creader.zip.next_in = Creader.zip_inbuf;
                    Creader.zip.avail_in = (unsigned)zip_got;
                }
                if (Creader.zip.avail_in == 0) {
                    fprintf(stderr, "FATAL: %s:%d zip input %s ended before its deflate stream did (%llu compressed bytes unread)\n",
                            __FILE__, __LINE__, Creader.filename,
                            (unsigned long long)Creader.zip_remaining);
                    exit(1);
                }
                int r = inflate(&Creader.zip, Z_NO_FLUSH);
                if (r == Z_STREAM_END) { Creader.zip_ended = Creader.eof = 1; break; }
                if (r != Z_OK) {
                    fprintf(stderr, "FATAL: %s:%d inflate failed on zip input %s (ret=%d%s%s)\n",
                            __FILE__, __LINE__, Creader.filename, r,
                            Creader.zip.msg ? ": " : "", Creader.zip.msg ? Creader.zip.msg : "");
                    exit(1);
                }
            }
            return len - Creader.zip.avail_out;
        } else if (Creader.zip_init == 2) {
            /* LZMA in ZIP */
            Creader.xz.next_out = (uint8_t *)buf;
            Creader.xz.avail_out = len;
            while (Creader.xz.avail_out > 0) {
                if (Creader.xz.avail_in == 0 && Creader.zip_remaining > 0) {
                    size_t zip_got = 0; ZIP_FILL_INPUT();
                    Creader.xz.next_in = Creader.xz_inbuf;
                    Creader.xz.avail_in = zip_got;
                    /* Copy to xz_inbuf since zip_inbuf may differ */
                    memcpy(Creader.xz_inbuf, Creader.zip_inbuf, zip_got);
                }
                if (Creader.xz.avail_in == 0) break;
                lzma_ret r = lzma_code(&Creader.xz, Creader.eof ? LZMA_FINISH : LZMA_RUN);
                if (r == LZMA_STREAM_END) { Creader.eof = 1; break; }
                if (r != LZMA_OK) { Creader.eof = 1; break; }
            }
            return len - Creader.xz.avail_out;
        } else if (Creader.zip_init == 3) {
            /* bzip2 in ZIP */
            Creader.bz.next_out = (char *)buf;
            Creader.bz.avail_out = (unsigned)len;
            while (Creader.bz.avail_out > 0) {
                if (Creader.bz.avail_in == 0 && Creader.zip_remaining > 0) {
                    size_t zip_got = 0; ZIP_FILL_INPUT();
                    Creader.bz.next_in = (char *)Creader.zip_inbuf;
                    Creader.bz.avail_in = zip_got;
                }
                if (Creader.bz.avail_in == 0) break;
                int r = BZ2_bzDecompress(&Creader.bz);
                if (r == BZ_STREAM_END) { Creader.eof = 1; break; }
                if (r != BZ_OK) { Creader.eof = 1; break; }
            }
            return len - Creader.bz.avail_out;
        } else if (Creader.zip_init == 4) {
            /* zstd in ZIP */
            ZSTD_outBuffer out = { buf, len, 0 };
            while (out.pos < out.size) {
                if (Creader.zstd_in.pos >= Creader.zstd_in.size && Creader.zip_remaining > 0) {
                    size_t zip_got = 0; ZIP_FILL_INPUT();
                    Creader.zstd_in.src = Creader.zip_inbuf;
                    Creader.zstd_in.size = zip_got;
                    Creader.zstd_in.pos = 0;
                }
                if (Creader.zstd_in.pos >= Creader.zstd_in.size) break;
                size_t r = ZSTD_decompressStream(Creader.zstd_ds, &out, &Creader.zstd_in);
                if (ZSTD_isError(r) || r == 0) { Creader.eof = 1; break; }
            }
            return out.pos;
        }
        return 0;
    } else if (Creader.type == CMP_ZSTD) {
        ZSTD_outBuffer out = { buf, len, 0 };
        while (out.pos < out.size) {
            if (Creader.zstd_in.pos >= Creader.zstd_in.size && !Creader.eof) {
                ssize_t n = read(Creader.fd, Creader.zstd_inbuf, sizeof(Creader.zstd_inbuf));
                if (n <= 0) { Creader.eof = 1; break; }
                Creader.zstd_in.src = Creader.zstd_inbuf;
                Creader.zstd_in.size = n;
                Creader.zstd_in.pos = 0;
            }
            size_t r = ZSTD_decompressStream(Creader.zstd_ds, &out, &Creader.zstd_in);
            if (ZSTD_isError(r)) { Creader.eof = 1; break; }
            if (r == 0) { Creader.eof = 1; break; }  /* frame complete */
            if (Creader.zstd_in.pos >= Creader.zstd_in.size && Creader.eof) break;
        }
        return out.pos;
    }
    return 0;
}

/* Detect format from first gzread() output and initialize decoder.
 * For non-gzip formats, reopens the file from Creader.filename.
 * Returns 1 if non-gzip format detected, 0 if gzip/plain. */
/* Compressed size of the zip entry whose 30-byte local header is h, read from
 * fd.  The 32-bit field is not always the size.  A zip64 entry writes
 * 0xFFFFFFFF there and keeps the real sizes in extra-field record 0x0001,
 * which holds ONLY the fields that are 0xFFFFFFFF, uncompressed first.  Taking
 * the field as written bounded every read at 4,294,967,295 bytes and ended a
 * larger wordlist early with exit 0: 1,071,478,623 of 1,797,692,209 lines.  A
 * streamed entry (flag bit 3) may write 0 and put the sizes after the data.
 * When the size cannot be read here, the rest of the file bounds the read and
 * the decoder's own end of stream stops it.  The extra field is read from fd
 * rather than from the caller's buffer, which a long entry name can overrun.
 * Leaves fd at the start of the entry data. */
static uint64_t zip_entry_comp_size(int fd, const unsigned char *h)
{
  uint32_t comp   = h[18] | (h[19] << 8) | (h[20] << 16) | ((uint32_t)h[21] << 24);
  uint32_t uncomp = h[22] | (h[23] << 8) | (h[24] << 16) | ((uint32_t)h[25] << 24);
  unsigned name_len = h[26] | (h[27] << 8), extra_len = h[28] | (h[29] << 8);
  uint64_t data = 30 + (uint64_t)name_len + extra_len, size = comp;
  int known = 1;

  if (comp == 0xFFFFFFFFu || uncomp == 0xFFFFFFFFu) {
    unsigned char *ex = malloc(extra_len + 1);
    known = 0;
    if (ex && lseek(fd, 30 + name_len, SEEK_SET) >= 0 &&
        read(fd, ex, extra_len) == (ssize_t)extra_len) {
      unsigned p = 0;
      while (p + 4 <= extra_len) {
        unsigned id = ex[p] | (ex[p + 1] << 8), sz = ex[p + 2] | (ex[p + 3] << 8);
        if (p + 4 + sz > extra_len) break;
        if (id == 1) {
          unsigned q = p + 4, i;
          if (uncomp == 0xFFFFFFFFu) q += 8;
          if (comp != 0xFFFFFFFFu) known = 1;
          else if (q + 8 <= p + 4 + sz) {
            for (size = 0, i = 8; i-- > 0; ) size = (size << 8) | ex[q + i];
            known = 1;
          }
          break;
        }
        p += 4 + sz;
      }
    }
    free(ex);
  }
  if (!known || size == 0) {
    struct stat st;
    size = (fstat(fd, &st) == 0 && (uint64_t)st.st_size > data)
           ? (uint64_t)st.st_size - data : 0;
  }
  lseek(fd, (off_t)data, SEEK_SET);
  return size;
}

static int creader_detect(char *buf, size_t buflen) {
    unsigned char *b = (unsigned char *)buf;

    /* Can only handle non-gzip formats if we can reopen the file */
    if (!Creader.filename[0]) return 0;  /* pipe/stdin — gzip only */

    /* If gzread decompressed gzip data, buf contains plaintext — no magic.
     * If file was not gzip, gzread passed raw bytes through.
     * Check for non-gzip compressed format magic in the raw bytes. */

    if (buflen >= 6 && b[0] == 0xfd && b[1] == 0x37 && b[2] == 0x7a &&
        b[3] == 0x58 && b[4] == 0x5a && b[5] == 0x00) {
        /* XZ format — reopen file from beginning */
        Creader.type = CMP_XZ;
        Creader.fd = open(Creader.filename, O_RDONLY | O_BINARY);
        if (Creader.fd < 0) return 0;
        memset(&Creader.xz, 0, sizeof(Creader.xz));
        {
            lzma_ret r = lzma_stream_decoder(&Creader.xz, UINT64_MAX, LZMA_CONCATENATED);
            if (r != LZMA_OK) {
                fprintf(stderr, "FATAL: %s:%d lzma_stream_decoder failed (ret=%d) for %s\n",
                        __FILE__, __LINE__, (int)r, Creader.filename);
                exit(1);
            }
        }
        Creader.xz.avail_in = 0;
        fprintf(stderr, "Detected XZ compressed input\n");
        return 1;
    }
    if (buflen >= 3 && b[0] == 'B' && b[1] == 'Z' && b[2] == 'h') {
        /* bzip2 format — reopen file from beginning */
        Creader.type = CMP_BZ2;
        Creader.fd = open(Creader.filename, O_RDONLY | O_BINARY);
        if (Creader.fd < 0) return 0;
        memset(&Creader.bz, 0, sizeof(Creader.bz));
        BZ2_bzDecompressInit(&Creader.bz, 0, 0);
        Creader.bz.avail_in = 0;
        fprintf(stderr, "Detected bzip2 compressed input\n");
        return 1;
    }
    if (buflen >= 4 && b[0] == 0x28 && b[1] == 0xb5 && b[2] == 0x2f && b[3] == 0xfd) {
        /* Zstandard format */
        Creader.type = CMP_ZSTD;
        Creader.fd = open(Creader.filename, O_RDONLY | O_BINARY);
        if (Creader.fd < 0) return 0;
        Creader.zstd_ds = ZSTD_createDStream();
        ZSTD_initDStream(Creader.zstd_ds);
        Creader.zstd_in.src = NULL;
        Creader.zstd_in.size = 0;
        Creader.zstd_in.pos = 0;
        fprintf(stderr, "Detected Zstandard compressed input\n");
        return 1;
    }
    if (buflen >= 4 && b[0] == 'P' && b[1] == 'K' && b[2] == 3 && b[3] == 4) {
        /* ZIP format — parse local file header, inflate first file */
        if (buflen >= 30) {
            unsigned short method = b[8] | (b[9] << 8);
            Creader.type = CMP_ZIP;
            Creader.fd = open(Creader.filename, O_RDONLY | O_BINARY);
            if (Creader.fd < 0) return 0;
            Creader.zip_remaining = zip_entry_comp_size(Creader.fd, b);

            if (method == 8) {  /* deflate */
                memset(&Creader.zip, 0, sizeof(Creader.zip));
                inflateInit2(&Creader.zip, -MAX_WBITS);
                Creader.zip.avail_in = 0;
                Creader.zip_init = 1;
                fprintf(stderr, "Detected ZIP input (deflate)\n");
                return 1;
            } else if (method == 12) {  /* bzip2 */
                memset(&Creader.bz, 0, sizeof(Creader.bz));
                BZ2_bzDecompressInit(&Creader.bz, 0, 0);
                Creader.bz.avail_in = 0;
                Creader.zip_init = 3;
                fprintf(stderr, "Detected ZIP input (bzip2)\n");
                return 1;
            } else if (method == 14) {  /* LZMA */
                memset(&Creader.xz, 0, sizeof(Creader.xz));
                {
                    lzma_ret r = lzma_alone_decoder(&Creader.xz, UINT64_MAX);
                    if (r != LZMA_OK) {
                        fprintf(stderr, "FATAL: %s:%d lzma_alone_decoder failed (ret=%d) for %s\n",
                                __FILE__, __LINE__, (int)r, Creader.filename);
                        exit(1);
                    }
                }
                Creader.xz.avail_in = 0;
                Creader.zip_init = 2;
                fprintf(stderr, "Detected ZIP input (LZMA)\n");
                return 1;
            } else if (method == 93) {  /* Zstandard */
                Creader.zstd_ds = ZSTD_createDStream();
                ZSTD_initDStream(Creader.zstd_ds);
                Creader.zstd_in.src = NULL;
                Creader.zstd_in.size = 0;
                Creader.zstd_in.pos = 0;
                Creader.zip_init = 4;
                fprintf(stderr, "Detected ZIP input (zstd)\n");
                return 1;
            } else if (method == 0) {  /* stored */
                Creader.zip_init = 0;
                Creader.eof = 0;
                fprintf(stderr, "Detected ZIP input (stored)\n");
                return 1;
            } else {
                fprintf(stderr, "ZIP: unsupported method %d\n", method);
                close(Creader.fd);
                Creader.fd = 0;
                Creader.type = CMP_GZ;
                return 0;
            }
        }
    }
    return 0;
}

static void creader_reset(void) {
    if (Creader.type == CMP_XZ) lzma_end(&Creader.xz);
    else if (Creader.type == CMP_BZ2) BZ2_bzDecompressEnd(&Creader.bz);
    else if (Creader.type == CMP_ZSTD && Creader.zstd_ds) ZSTD_freeDStream(Creader.zstd_ds);
    else if (Creader.type == CMP_ZIP) {
        if (Creader.zip_init == 1) inflateEnd(&Creader.zip);
        else if (Creader.zip_init == 2) lzma_end(&Creader.xz);
        else if (Creader.zip_init == 3) BZ2_bzDecompressEnd(&Creader.bz);
        else if (Creader.zip_init == 4 && Creader.zstd_ds) ZSTD_freeDStream(Creader.zstd_ds);
    }
    if (Creader.fd > 0) close(Creader.fd);
    memset(&Creader, 0, sizeof(Creader));
}

#define RINDEXSIZE (MAXLINEPERCHUNK)

/*
 * Read pipeline.  A wordlist (or hash file) passes through three stages:
 * READ the bytes, INDEX the lines, and HASH them.  A reader thread does the
 * first, cacheline() the second on the caller's thread, and the workers the
 * third.  Each stage owns a buffer of the NREADBUF ring at a time, so on a
 * cheap hash the run goes at the speed of the slowest stage rather than the
 * sum of reading and indexing.
 *
 * The reader hands cacheline() whole lines only.  After each read it cuts the
 * buffer after its last newline, or after its RINDEXSIZE-1'th newline if that
 * comes first, and carries the rest to the front of the next buffer, so the
 * index can never overflow and the indexer never has to look ahead.  As
 * before:
 *   - a newline in the last byte read is not trusted until the next read, so
 *     a file appended to during the read keeps its line whole;
 *   - an unterminated tail of MAXLINE bytes or more is dropped;
 *   - at end of file the last line needs no newline.
 *
 * A buffer goes back to the reader only when cacheline() has moved on to the
 * next one (RBready) AND every job using it has finished (ReadBufL).  The
 * indexer writes only inside the lines it hands out (the CR, the NUL, the
 * in-place $HEX[] decode), never in the carried tail.
 *
 * Newlines are found 64 bytes at a time: nlmask() compares a whole aligned
 * block and returns the newline positions as a 64-bit mask.  Aligned loads
 * never cross a page; bits before the buffer or at or past the bound are
 * discarded.
 */
#if !defined(NOTINTEL)
static inline unsigned long long nlmask(const char *p) {
  __m128i n = _mm_set1_epi8('\n');
  unsigned long long a = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(n, _mm_load_si128((const __m128i *)p)));
  unsigned long long b = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(n, _mm_load_si128((const __m128i *)(p + 16))));
  unsigned long long c = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(n, _mm_load_si128((const __m128i *)(p + 32))));
  unsigned long long d = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(n, _mm_load_si128((const __m128i *)(p + 48))));
  return a | (b << 16) | (c << 32) | (d << 48);
}
#define NLSCAN_SIMD 1
#elif defined(ARM) && defined(__aarch64__)
static inline unsigned long long nlmask(const char *p) {
  const uint8x16_t n = vdupq_n_u8('\n');
  const uint8x16_t bit = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  uint8x16_t a = vandq_u8(vceqq_u8(vld1q_u8((const uint8_t *)p), n), bit);
  uint8x16_t b = vandq_u8(vceqq_u8(vld1q_u8((const uint8_t *)p + 16), n), bit);
  uint8x16_t c = vandq_u8(vceqq_u8(vld1q_u8((const uint8_t *)p + 32), n), bit);
  uint8x16_t d = vandq_u8(vceqq_u8(vld1q_u8((const uint8_t *)p + 48), n), bit);
  uint8x16_t s = vpaddq_u8(vpaddq_u8(a, b), vpaddq_u8(c, d));
  s = vpaddq_u8(s, s);
  return vgetq_lane_u64(vreinterpretq_u64_u8(s), 0);
}
#define NLSCAN_SIMD 1
#endif

static lock *RBready[NREADBUF];      /* 0: free for the reader, 1: read, awaiting cacheline */
static unsigned int RBlen[NREADBUF]; /* bytes of whole lines in the buffer */
static int RBeof[NREADBUF];          /* the buffer ends the file */
static gzFile RBfile;                /* file the reader thread is reading */
static thread *RBthread;             /* NULL when no file is being read */
static int RBnext;                   /* next buffer cacheline() indexes */
static int RBheld = -1;              /* buffer cacheline() last handed out */
static int RBdone;                   /* the end-of-file buffer has been handed out */

/* Where to cut buf[0..limit): just past the k'th newline if there are that
 * many (*capped = 1), else just past the last one, else 0. */
static unsigned int rb_cut(const char *buf, unsigned int limit, unsigned int k, int *capped) {
  unsigned int last = 0, seen = 0;
  *capped = 0;
#ifdef NLSCAN_SIMD
  const char *blk = (const char *)((uintptr_t)buf & ~(uintptr_t)63);
  const char *end = buf + limit;
  unsigned long long m;
  if (limit == 0) return 0;
  m = nlmask(blk) & (~0ULL << ((uintptr_t)buf - (uintptr_t)blk));
  for (;;) {
    if (blk + 64 > end)
      m &= ~0ULL >> (64 - (end - blk));
    if (m) {
      unsigned int pc = (unsigned int)__builtin_popcountll(m);
      if (seen + pc >= k) {
        while (++seen < k) m &= m - 1;
        *capped = 1;
        return (unsigned int)(blk + __builtin_ctzll(m) - buf) + 1;
      }
      seen += pc;
      last = (unsigned int)(blk + 63 - __builtin_clzll(m) - buf) + 1;
    }
    blk += 64;
    if (blk >= end) return last;
    m = nlmask(blk);
  }
#else
  const char *p = buf, *end = buf + limit;
  while (p < end && (p = memchr(p, '\n', end - p))) {
    last = (unsigned int)(++p - buf);
    if (++seen >= k) { *capped = 1; return last; }
  }
  return last;
#endif
}

static int rb_read(gzFile fi, char *dst, unsigned int len) {
  int n;
  if (Creader.detected && Creader.type != CMP_GZ)
    return Creader.eof ? 0 : (int)creader_read(dst, len);
  n = gzread(fi, dst, len);
  if (n < 0) {
    int err;
    fprintf(stderr, "FATAL %s:%d: gzread: %s\n", __FILE__, __LINE__, gzerror(fi, &err));
    exit(1);
  }
  /* First read: check for non-gzip compressed formats.  If one is found the
   * raw bytes gzread returned are discarded and the library decoder re-reads
   * from the start. */
  if (!Creader.detected && n > 0) {
    Creader.detected = 1;
    if (creader_detect(dst, n)) {
      gzclose(fi);
      Creader.gz_closed = 1;
      return -1;
    }
  }
  return n;
}

static int rb_eof(gzFile fi) {
  return (Creader.detected && Creader.type != CMP_GZ) ? Creader.eof : gzeof(fi);
}

MDXALIGN static void rb_reader(void *dummy) {
  gzFile fi = RBfile;
  int slot = 0, prev = 0, eof, capped, n;
  unsigned int carry = 0, total, cut;
  char *buf;

  for (;;) {
    possess(RBready[slot]);
    wait_for(RBready[slot], TO_BE, 0);
    release(RBready[slot]);
    possess(ReadBufL[slot]);
    wait_for(ReadBufL[slot], TO_BE, 0);
    release(ReadBufL[slot]);
    buf = Readbuf + (size_t)slot * READBUFSZ;
    if (carry)
      memcpy(buf, Readbuf + (size_t)prev * READBUFSZ + RBlen[prev], carry);
    n = rb_read(fi, buf + carry, READBUFSZ - carry - 1);
    if (n < 0) {                  /* switched to a library decoder */
      carry = 0;
      n = (int)creader_read(buf, READBUFSZ - 1);
    }
    if (n > 0)
      atomic_fetch_add_explicit(&CurfileBytesRead, (unsigned long long)n,
                                memory_order_relaxed);
    total = carry + (unsigned int)n;
    /* The index cap applies at end of file too: a carried tail can hold more
     * lines than one index, so the buffer is the last only when all that
     * remains fits.  Otherwise the rest is carried and read again, at EOF, as
     * the next buffer. */
    eof = rb_eof(fi);
    cut = rb_cut(buf, eof ? total : (total ? total - 1 : 0), RINDEXSIZE - 1, &capped);
    if (eof && !capped) {
      cut = total;          /* includes a last line with no newline */
      carry = 0;
    } else {
      eof = 0;
      carry = total - cut;
      if (!capped && carry >= MAXLINE)
        carry = 0;
    }
    RBlen[slot] = cut;
    RBeof[slot] = eof;
    possess(RBready[slot]);
    twist(RBready[slot], TO, 1);
    if (eof)
      return;
    prev = slot;
    slot = (slot + 1) % NREADBUF;
  }
}

/* Index one line, buf[start..nl).  Returns 1 if it was indexed, 0 if it is
 * MAXLINE bytes or longer and skipped. */
static inline unsigned int rb_line(char *buf, unsigned int start, unsigned int nl,
                                   struct LineInfo *li, int classify) {
  unsigned int len = nl - start, rlen = len;
  char *w = buf + start;
  if (len >= MAXLINE)
    return 0;
  if (len > 0 && buf[nl - 1] == '\r') {
    buf[nl - 1] = '\n';
    rlen--;
  }
  /* The '$' test first: gcc emits this strncmp as repz cmpsb, which was ~20%
   * of cacheline on a Xeon E5-2697 v4, and almost no line starts with '$'. */
  if (len > 5 && *w == '$' && strncmp(w, "$HEX[", 5) == 0)
    rlen = get32(w + 5, (unsigned char *)w, len - 6);
  li->offset = start;
  li->len = rlen;
  /* Classify AFTER the $HEX[] unwrap, never before: the wrapper is pure ASCII,
   * so classifying the raw line would call every wrapped non-ASCII word ASCII
   * and route it to the byte engine (8,819 of 53,976 lines of
   * tools/ruletests/encoding/UTF8test1.txt).  -E classifies the LOCAL PART
   * only, as procrule does at its apply site; classifying the whole address
   * could pick a different engine for an ASCII local part with a non-ASCII
   * domain, and engine choice can change the result for a locale-ambiguous
   * case mapping. */
  li->enc = classify ? (unsigned char)classify_utf8(w, (size_t)enclen_E(w, rlen))
                     : (unsigned char)U8C_NONE;
  w[rlen] = 0;
  return 1;
}

/* Index buf[0..len), which holds whole lines; at end of file the last one may
 * lack its newline.  Every line ending in a 64-byte block is handled in one
 * inner loop over that block's mask. */
static unsigned int rb_index(char *buf, unsigned int len, int eof, struct LineInfo *idx) {
  unsigned int n = 0, start = 0;
  const int classify = Utf32Rules;
#ifdef NLSCAN_SIMD
  const char *blk = (const char *)((uintptr_t)buf & ~(uintptr_t)63);
  const char *end = buf + len;
  unsigned long long m;
  if (len) {
    m = nlmask(blk) & (~0ULL << ((uintptr_t)buf - (uintptr_t)blk));
    for (;;) {
      if (blk + 64 > end)
        m &= ~0ULL >> (64 - (end - blk));
      while (m) {
        unsigned int nl = (unsigned int)(blk + __builtin_ctzll(m) - buf);
        m &= m - 1;
        n += rb_line(buf, start, nl, &idx[n], classify);
        start = nl + 1;
      }
      blk += 64;
      if (blk >= end)
        break;
      m = nlmask(blk);
    }
  }
#else
  char *p;
  while (start < len && (p = memchr(buf + start, '\n', len - start))) {
    unsigned int nl = (unsigned int)(p - buf);
    n += rb_line(buf, start, nl, &idx[n], classify);
    start = nl + 1;
  }
#endif
  if (start < len && eof) {
    /* The last line of a file with no trailing newline. */
    unsigned int rlen = len - start;
    char *w = buf + start;
    buf[len] = '\n';
    if (rlen > 0 && w[rlen - 1] == '\r')
      rlen--;
    if ((len - start) > 5 && *w == '$' && strncmp(w, "$HEX[", 5) == 0)
      rlen = get32(w + 5, (unsigned char *)w, len - start - 6);
    idx[n].offset = start;
    idx[n].len = rlen;
    idx[n].enc = classify ? (unsigned char)classify_utf8(w, (size_t)enclen_E(w, rlen))
                          : (unsigned char)U8C_NONE;
    if (rlen < MAXLINE)
      n++;
  }
  return n;
}

/* Return the next indexed buffer of fi: its lines in *mybuf, its index in
 * *myindex, and the line count, or 0 at end of file.  The buffer stays the
 * caller's until the next call. */
unsigned int cacheline(gzFile fi, char **mybuf, struct LineInfo **myindex) {
  for (;;) {
    unsigned int Linecount, dest;
    unsigned long long curline;
    int slot, eof;
    char *buf;
    struct LineInfo *index;

    if (RBheld >= 0) {
      possess(RBready[RBheld]);
      twist(RBready[RBheld], TO, 0);
      RBheld = -1;
    }
    if (RBdone) {
      join(RBthread);
      RBthread = NULL;
      RBdone = 0;
      return 0;
    }
    if (!RBthread) {
      RBfile = fi;
      RBnext = 0;
      RBthread = launch(rb_reader, NULL);
    }
    slot = RBnext;
    possess(RBready[slot]);
    wait_for(RBready[slot], TO_BE, 1);
    release(RBready[slot]);
    RBheld = slot;
    RBnext = (slot + 1) % NREADBUF;
    eof = RBeof[slot];
    if (eof)
      RBdone = 1;
    buf = Readbuf + (size_t)slot * READBUFSZ;
    index = Readindex + (size_t)slot * RINDEXSIZE;
    Linecount = rb_index(buf, RBlen[slot], eof, index);
    if (Linecount == 0)
      continue;   /* nothing indexable here: next buffer, or end of file */
    if (Dedupe) {
      Localreadbuf = (unsigned char *)buf;
      qsort(index, Linecount, sizeof(struct LineInfo), linecmp);
      dest = 0;
      for (curline = 1; curline < Linecount; curline++) {
        if (linecmp(&index[dest], &index[curline]) == 0)
          continue;
        index[++dest] = index[curline];
      }
      Linecount = dest + 1;
    }
    *mybuf = buf;
    *myindex = index;
    return Linecount;
  }
}


volatile struct JobHash {
  int length;
  void *work;
} ToAdd;

#define MAXADDHASHBUF (4*1024*1024)

MDXALIGN void addhash(void *dummy) {
  int length, len;
  void *work;
  unsigned char *cur;
  unsigned short int *curi;
  unsigned long long insizel;
  unsigned int minhashlenl;

  insizel = 0;
  minhashlenl = Minhashlen;
  while (1) {
    possess(HashWaiting);
    wait_for(HashWaiting, NOT_TO_BE, 0);
    work = ToAdd.work;
    length = ToAdd.length;
    if (length < 0) {
      Insize = insizel;
      Minhashlen = minhashlenl;
      twist(HashWaiting, BY, -1);
      return;
    }
    cur = work;

    /* Pre-size entry arrays for this batch */
    if (HashDataCount + length > HashDataCap) {
      size_t newcap = HashDataCap ? HashDataCap * 2 : 4096;
      while (newcap < HashDataCount + length) newcap *= 2;
      HashDataOff = realloc(HashDataOff, newcap * sizeof(size_t));
      HashDataLen = realloc(HashDataLen, newcap * sizeof(unsigned short));
      HashDataFlags = realloc(HashDataFlags, newcap * sizeof(unsigned short));
      if (!HashDataOff || !HashDataLen || !HashDataFlags) {
        fprintf(stderr, "Failed to allocate hash entry arrays\n");
        exit(1);
      }
      memset(HashDataFlags + HashDataCap, 0, (newcap - HashDataCap) * sizeof(unsigned short));
      HashDataCap = newcap;
    }

    /* Pre-size data buffer for this batch. Worst case: MYSQL3 input is 10 bytes
     * (2-byte len + 8 raw); stored after sub-128-bit zero-pad is 16 bytes = 1.6x.
     * Allocate 2*MAXADDHASHBUF to cover any sub-128-bit expansion. */
    if (HashDataBufUsed + 2 * MAXADDHASHBUF > HashDataBufCap) {
      size_t newcap = HashDataBufCap ? HashDataBufCap * 2 : 65536;
      while (newcap < HashDataBufUsed + 2 * MAXADDHASHBUF) newcap *= 2;
      HashDataBuf = realloc(HashDataBuf, newcap);
      if (!HashDataBuf) {
        fprintf(stderr, "Failed to allocate hash data buffer\n");
        exit(1);
      }
      HashDataBufCap = newcap;
    }

    for (; length > 0; length--) {
#ifdef SPARC
      len = (*cur++) + ((*cur++) << 8);
      if (len > 256) {
        fprintf(stderr,"Error in addhash - aborting (%d)\n",len);
        exit(1);
      }
#else
      curi = (unsigned short int *) cur;
      len = *curi++;
      cur = (unsigned char *) curi;
      if (len > 256 || len < 8) {
        fprintf(stderr, "Error in addhash - aborting (%d)\n", len);
        exit(1);
      }
#endif
      /* Store hash data — truncate to 4-byte boundary for GPU uint* access.
       * The packed stream cursor must advance by the original length.
       * Sub-128-bit outputs (e.g. MYSQL3, 8 bytes) are zero-padded in storage
       * to 16 bytes so the GPU 4xuint32 probe never spills into the next entry.
       * HashDataLen[i] keeps the real length (CPU memcmp is unaffected). */
      { int rawlen = len;
        int padlen;
        if (rawlen < 8) { cur += rawlen; continue; }
        /* Pad the STORED buffer up to a 4-byte boundary (16 minimum) so the GPU
         * 4xuint32 probe never spills into the next entry, but record the REAL
         * length. Rounding the recorded length DOWN to the alignment made every
         * hash whose byte length is not a multiple of 4 compare short: MSSQL2012
         * is 70 bytes and was matched on 68, so any two hashes agreeing in the
         * first 68 bytes were indistinguishable, and the emitted line was 2
         * bytes shorter than the input and validated nowhere else. */
        padlen = (rawlen + 3) & ~3;
        if (padlen < 16) padlen = 16;
        HashDataOff[HashDataCount] = HashDataBufUsed;
        HashDataLen[HashDataCount] = rawlen;
        HashDataFlags[HashDataCount] = 0;
        memcpy(HashDataBuf + HashDataBufUsed, cur, rawlen);
        memset(HashDataBuf + HashDataBufUsed + rawlen, 0, padlen - rawlen);
        HashDataBufUsed += padlen;
        HashDataCount++;

        if (rawlen < minhashlenl)
          minhashlenl = rawlen;
        insizel++;
        cur += rawlen;
      }
    }
    twist(HashWaiting, BY, -1);
  }
}


#if defined(OPENCL_GPU)
/* Per-device set_compact_table worker thread (Memo C parallelization).
 * Reads file-scope CompactFP/CompactIdx/HashData* globals (set by
 * build_compact_table prior to spawning); writes only into gpu_devs[di]
 * which is owned by this thread for the duration. */
static void sct_thread_fn(void *payload) {
  int di = *(int *)payload;
  gpu_opencl_set_compact_table(di, CompactFP, CompactIdx,
      CompactSize, CompactMask,
      HashDataBuf, HashDataBufUsed,
      HashDataOff, HashDataCount, HashDataLen);
}

/* Per-device set_rules worker thread payload + body (Memo C). The rule
 * program data (prog/offs) is shared read-only across threads; only the
 * per-device gpu_devs[di] state is mutated and that's disjoint per slot.
 * rc field gathers the per-device set_rules return value for tally after
 * join. */
struct srl_arg {
  int            di;
  unsigned char *prog;
  uint32_t       prog_len;
  uint32_t      *offs;
  int            n_rules;
  int            rc;
};
static void srl_thread_fn(void *payload) {
  struct srl_arg *a = (struct srl_arg *)payload;
  a->rc = gpu_opencl_set_rules(a->di, a->prog, a->prog_len, a->offs, a->n_rules);
}
#endif

/* hx codegen sub-phase 2a.6 (2026-05-22): shared validation helper. Both
 * the OpenCL and Metal harness branches in build_compact_table() call
 * this function with backend_id = 0 (OpenCL) or 1 (Metal). The function
 * parses the fixture, computes the CPU oracle for every (pass,salt) pair,
 * plants the oracle digests into a synthetic compact-fp table, uploads
 * the table + salts + planted hashes to the GPU via the per-backend
 * set_compact_table/set_salts entry points, packs candidates, JIT-compiles
 * the codegen-emitted kernel B via the per-backend _keep wrapper,
 * dispatches via the per-backend e347_validate_dispatch helper, sorts +
 * diffs hits against the expected pair set, and exits with code 0 on
 * byte-exact PASS or code 1 on FAIL.
 *
 * The function NEVER RETURNS -- it exits() at the end of the diff stage.
 * That mirrors the 2a.5 harness behavior: each invocation of mdxfind with
 * MDXFIND_HX_CODEGEN_VALIDATE=1 runs one validation cell and exits.
 *
 * Refactor rationale: ~270 LOC of fixture-parse + oracle-build + compact-
 * table-plant + candidate-pack + hits-diff was identical between the two
 * backends; only the JIT entry point name + dispatch function name + a
 * handful of release calls differ. Extracting the shared logic eliminates
 * copy-paste drift; the per-backend diff is now ~25 LOC each. */
#if defined(OPENCL_GPU) || (defined(__APPLE__) && defined(METAL_GPU))
static void hx_e347_validate_run_shared(int backend_id, char *hxsrc,
                                        const char *fixture_env,
                                        const char *hxhost)
{
    const char *backend_name = (backend_id == 0) ? "OpenCL" : "Metal";

    if (!fixture_env || !*fixture_env) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen 2a.6 validate (%s): "
            "MDXFIND_HX_CODEGEN_FIXTURE not set on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }
    FILE *ff = fopen(fixture_env, "r");
    if (!ff) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen 2a.6 validate (%s): cannot open "
            "fixture '%s' on %s (errno=%d)\n",
            __FILE__, __LINE__, backend_name, fixture_env, hxhost, errno);
        exit(1);
    }

    /* --- Parse fixture: PASS:<text> / SALT:<text> / # comments --- */
    char **passes = NULL; size_t n_pass = 0, cap_pass = 0;
    char **salts  = NULL; size_t n_salt = 0, cap_salt = 0;
    char line[4096];
    while (fgets(line, sizeof(line), ff)) {
        size_t L = strlen(line);
        while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) {
            line[--L] = '\0';
        }
        if (L == 0 || line[0] == '#') continue;
        if (L > 5 && memcmp(line, "PASS:", 5) == 0) {
            if (n_pass == cap_pass) {
                cap_pass = cap_pass ? cap_pass * 2 : 16;
                passes = realloc(passes, cap_pass * sizeof(char *));
            }
            passes[n_pass++] = strdup(line + 5);
        } else if (L > 5 && memcmp(line, "SALT:", 5) == 0) {
            if (n_salt == cap_salt) {
                cap_salt = cap_salt ? cap_salt * 2 : 16;
                salts = realloc(salts, cap_salt * sizeof(char *));
            }
            salts[n_salt++] = strdup(line + 5);
        }
    }
    fclose(ff);
    if (n_pass == 0 || n_salt == 0) {
        fprintf(stderr,
            "FATAL: %s:%d 2a.6 validate (%s): fixture %s has no PASS "
            "(n=%zu) or SALT (n=%zu) lines on %s\n",
            __FILE__, __LINE__, backend_name, fixture_env,
            n_pass, n_salt, hxhost);
        exit(1);
    }
    size_t n_pairs = n_pass * n_salt;
    fprintf(stderr,
        "hx codegen 2a.6 (%s): fixture %s parsed n_pass=%zu n_salt=%zu "
        "n_pairs=%zu on %s\n",
        backend_name, fixture_env, n_pass, n_salt, n_pairs, hxhost);

    /* --- Compute CPU oracle digests for every (pass, salt) pair.
     * Chain: MD5( hex32(MD5(hex32(MD5(pass)))) || salt ) */
    unsigned char *oracle = (unsigned char *)malloc(n_pairs * 16);
    if (!oracle) {
        fprintf(stderr,
            "FATAL: %s:%d 2a.6 validate (%s): malloc(oracle %zu B) "
            "failed on %s\n",
            __FILE__, __LINE__, backend_name, n_pairs * 16, hxhost);
        exit(1);
    }
    for (size_t pi = 0; pi < n_pass; pi++) {
        unsigned char inner1[16], inner2[16];
        char hex32a[33], hex32b[33];
        mymd5((void *)passes[pi], (int)strlen(passes[pi]), inner1);
        prmd5_32(inner1, hex32a);
        hex32a[32] = '\0';
        mymd5((void *)hex32a, 32, inner2);
        prmd5_32(inner2, hex32b);
        hex32b[32] = '\0';

        for (size_t si = 0; si < n_salt; si++) {
            size_t slen = strlen(salts[si]);
            char *concat = (char *)malloc(32 + slen + 1);
            memcpy(concat, hex32b, 32);
            memcpy(concat + 32, salts[si], slen);
            concat[32 + slen] = '\0';
            unsigned char outd[16];
            mymd5((void *)concat, (int)(32 + slen), outd);
            memcpy(oracle + (pi * n_salt + si) * 16, outd, 16);
            free(concat);
        }
    }
    fprintf(stderr,
        "hx codegen 2a.6 (%s): oracle digests computed for %zu pairs\n",
        backend_name, n_pairs);

    /* --- Build synthetic compact-hash table sized for n_pairs. */
    uint64_t cap_pow2 = 1;
    while (cap_pow2 < n_pairs * 2) cap_pow2 <<= 1;
    if (cap_pow2 < 64) cap_pow2 = 64;
    uint64_t cmask = cap_pow2 - 1;
    uint32_t *cfp  = (uint32_t *)calloc(cap_pow2, sizeof(uint32_t));
    uint32_t *cidx = (uint32_t *)calloc(cap_pow2, sizeof(uint32_t));
    unsigned char *hbuf = (unsigned char *)malloc(n_pairs * 16);
    size_t *hoff = (size_t *)malloc(n_pairs * sizeof(size_t));
    unsigned short *hlen = (unsigned short *)malloc(n_pairs * sizeof(unsigned short));
    if (!cfp || !cidx || !hbuf || !hoff || !hlen) {
        fprintf(stderr,
            "FATAL: %s:%d 2a.6 validate (%s): alloc compact buffers "
            "failed on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }
    for (size_t i = 0; i < n_pairs; i++) {
        memcpy(hbuf + i * 16, oracle + i * 16, 16);
        hoff[i] = i * 16;
        hlen[i] = 16;
        uint32_t hx = ((uint32_t *)(oracle + i * 16))[0];
        uint32_t hy = ((uint32_t *)(oracle + i * 16))[1];
        uint64_t key = ((uint64_t)hy << 32) | hx;
        uint32_t fp = (uint32_t)(key >> 32);
        if (fp == 0) fp = 1;
        uint64_t pos = (key ^ (key >> 32)) & cmask;
        for (int probe = 0; probe < 256; probe++) {
            uint64_t p = (pos + probe) & cmask;
            if (cfp[p] == 0) {
                cfp[p] = fp;
                cidx[p] = (uint32_t)i;
                break;
            }
        }
    }
    fprintf(stderr,
        "hx codegen 2a.6 (%s): compact table built cap=%llu mask=0x%llx\n",
        backend_name,
        (unsigned long long)cap_pow2, (unsigned long long)cmask);

    /* --- Upload compact table + salts to GPU dev[0] via per-backend
     * entry points. */
    if (backend_id == 0) {
#if defined(OPENCL_GPU)
        if (gpu_opencl_set_compact_table(0, cfp, cidx, cap_pow2, cmask,
                hbuf, n_pairs * 16, hoff, n_pairs, hlen) != 0) {
            fprintf(stderr,
                "FATAL: %s:%d 2a.6 validate (OpenCL): "
                "set_compact_table failed on %s dev[0]\n",
                __FILE__, __LINE__, hxhost);
            exit(1);
        }
#endif
    } else {
#if defined(__APPLE__) && defined(METAL_GPU)
        if (gpu_metal_set_compact_table(0, cfp, cidx, cap_pow2, cmask,
                hbuf, n_pairs * 16, hoff, n_pairs, hlen) != 0) {
            fprintf(stderr,
                "FATAL: %s:%d 2a.6 validate (Metal): "
                "set_compact_table failed on %s dev[0]\n",
                __FILE__, __LINE__, hxhost);
            exit(1);
        }
#endif
    }

    /* Pack salts: contiguous buffer + offsets + lens (uint16). */
    {
        size_t sbytes = 0;
        for (size_t si = 0; si < n_salt; si++)
            sbytes += strlen(salts[si]);
        char *sbuf = (char *)malloc(sbytes + 16);
        uint32_t *soff = (uint32_t *)malloc(n_salt * sizeof(uint32_t));
        uint16_t *slens = (uint16_t *)malloc(n_salt * sizeof(uint16_t));
        if (!sbuf || !soff || !slens) {
            fprintf(stderr,
                "FATAL: %s:%d 2a.6 validate (%s): alloc salt buffers "
                "failed on %s\n",
                __FILE__, __LINE__, backend_name, hxhost);
            exit(1);
        }
        size_t pos = 0;
        for (size_t si = 0; si < n_salt; si++) {
            size_t L = strlen(salts[si]);
            memcpy(sbuf + pos, salts[si], L);
            if ((uint64_t)pos + (uint64_t)L > (uint64_t)GPU_SALT_PACK_MAX) {
                fprintf(stderr,
                    "FATAL: %s:%d salt pack: offset %llu + %zu exceeds "
                    "GPU_SALT_PACK_MAX (%u). soff[] is uint32 and truncating "
                    "it here would index the wrong salt data silently.\n",
                    __FILE__, __LINE__, (unsigned long long)pos, L,
                    (unsigned)GPU_SALT_PACK_MAX);
                exit(1);
            }
            soff[si]  = (uint32_t)pos;
            slens[si] = (uint16_t)L;
            pos += L;
        }
        int set_rc = -1;
        if (backend_id == 0) {
#if defined(OPENCL_GPU)
            set_rc = gpu_opencl_set_salts(0, sbuf, soff, slens, (int)n_salt);
#endif
        } else {
#if defined(__APPLE__) && defined(METAL_GPU)
            set_rc = gpu_metal_set_salt(sbuf, soff, slens, (int)n_salt);
#endif
        }
        if (set_rc != 0) {
            fprintf(stderr,
                "FATAL: %s:%d 2a.6 validate (%s): set_salts failed "
                "on %s dev[0] rc=%d\n",
                __FILE__, __LINE__, backend_name, hxhost, set_rc);
            exit(1);
        }
        free(sbuf); free(soff); free(slens);
    }

    /* --- Pack candidates: [len_u8][bytes] per word. */
    size_t pack_bytes = 0;
    for (size_t pi = 0; pi < n_pass; pi++)
        pack_bytes += 1 + strlen(passes[pi]);
    unsigned char *packed = (unsigned char *)malloc(pack_bytes + 16);
    uint32_t *woff = (uint32_t *)malloc(n_pass * sizeof(uint32_t));
    if (!packed || !woff) {
        fprintf(stderr,
            "FATAL: %s:%d 2a.6 validate (%s): alloc packed words "
            "failed on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }
    {
        size_t pos = 0;
        for (size_t pi = 0; pi < n_pass; pi++) {
            size_t L = strlen(passes[pi]);
            if (L > 255) {
                fprintf(stderr,
                    "FATAL: %s:%d 2a.6 validate (%s): pass %zu length "
                    "%zu > 255 on %s\n",
                    __FILE__, __LINE__, backend_name, pi, L, hxhost);
                exit(1);
            }
            woff[pi] = (uint32_t)pos;
            packed[pos++] = (unsigned char)L;
            memcpy(packed + pos, passes[pi], L);
            pos += L;
        }
        pack_bytes = pos;
    }

    /* --- JIT codegen kernel via _keep wrapper + dispatch via per-backend
     * e347_validate_dispatch helper. */
    int max_hits = (int)n_pairs;
    if (max_hits < 16) max_hits = 16;
    uint32_t *vhits = (uint32_t *)calloc((size_t)max_hits * 19, sizeof(uint32_t));
    int vn_hits = 0;
    if (!vhits) {
        fprintf(stderr,
            "FATAL: %s:%d 2a.6 validate (%s): alloc vhits failed on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }

    void *jit_handle_a = NULL;   /* OpenCL: cl_program; Metal: MTLLibrary */
    void *jit_handle_b = NULL;   /* OpenCL: cl_kernel ; Metal: MTLComputePipelineState */

    if (backend_id == 0) {
#if defined(OPENCL_GPU)
        struct _cl_program *vprog = NULL;
        struct _cl_kernel  *vkern = NULL;
        gpu_opencl_jit_compile_source_with_common_keep(
            0, hxsrc, "-cl-std=CL1.2",
            "kernelb_hx_e347_phase0", &vprog, &vkern);
        jit_handle_a = (void *)vprog;
        jit_handle_b = (void *)vkern;
        gpu_opencl_e347_validate_dispatch(
            0, vkern,
            packed, (uint32_t)pack_bytes, woff, (uint32_t)n_pass,
            (uint32_t)n_salt,
            vhits, &vn_hits, max_hits);
#endif
    } else {
#if defined(__APPLE__) && defined(METAL_GPU)
        gpu_metal_jit_compile_source_with_common_keep(
            0, hxsrc, "kernelb_hx_e347_phase0",
            &jit_handle_a, &jit_handle_b);
        gpu_metal_e347_validate_dispatch(
            0, jit_handle_b,
            packed, (uint32_t)pack_bytes, woff, (uint32_t)n_pass,
            (uint32_t)n_salt,
            vhits, &vn_hits, max_hits);
#endif
    }

    fprintf(stderr,
        "hx codegen 2a.6 (%s): dispatch returned vn_hits=%d (expected %zu)\n",
        backend_name, vn_hits, n_pairs);

    /* --- Diff: every (pi, si) pair should appear exactly once. */
    unsigned char *seen = (unsigned char *)calloc(n_pairs, 1);
    int extras = 0, mismatches = 0;
    size_t first_mismatch_print = 0;
    for (int h = 0; h < vn_hits; h++) {
        uint32_t *slot = vhits + (size_t)h * 19;
        uint32_t widx = slot[0];
        uint32_t sidx = slot[1];
        if (widx >= n_pass || sidx >= n_salt) {
            extras++;
            if (first_mismatch_print < 10) {
                fprintf(stderr,
                    "  HIT-EXTRA #%d widx=%u sidx=%u (out of range)\n",
                    h, widx, sidx);
                first_mismatch_print++;
            }
            continue;
        }
        size_t pid = (size_t)widx * n_salt + (size_t)sidx;
        uint32_t exp0 = ((uint32_t *)(oracle + pid * 16))[0];
        uint32_t exp1 = ((uint32_t *)(oracle + pid * 16))[1];
        uint32_t exp2 = ((uint32_t *)(oracle + pid * 16))[2];
        uint32_t exp3 = ((uint32_t *)(oracle + pid * 16))[3];
        if (slot[3] != exp0 || slot[4] != exp1 ||
            slot[5] != exp2 || slot[6] != exp3)
        {
            mismatches++;
            if (first_mismatch_print < 10) {
                fprintf(stderr,
                    "  HIT-DIFF #%d (pi=%u si=%u) "
                    "exp=%08x%08x%08x%08x got=%08x%08x%08x%08x\n",
                    h, widx, sidx,
                    exp0, exp1, exp2, exp3,
                    slot[3], slot[4], slot[5], slot[6]);
                first_mismatch_print++;
            }
        }
        if (!seen[pid]) seen[pid] = 1;
    }
    size_t matched = 0, missing = 0;
    for (size_t i = 0; i < n_pairs; i++)
        if (seen[i]) matched++; else missing++;
    int miss_print = 0;
    if (missing > 0) {
        fprintf(stderr, "  Missing pairs (showing first 10):\n");
        for (size_t i = 0; i < n_pairs && miss_print < 10; i++) {
            if (!seen[i]) {
                size_t pi = i / n_salt;
                size_t si = i % n_salt;
                fprintf(stderr,
                    "    pid=%zu pi=%zu si=%zu pass=\"%s\" salt=\"%s\"\n",
                    i, pi, si, passes[pi], salts[si]);
                miss_print++;
            }
        }
    }

    int pass_ok = (mismatches == 0 && extras == 0 &&
                   matched == n_pairs && vn_hits == (int)n_pairs);
    fprintf(stderr,
        "hx codegen 2a.6 (%s) RESULT on %s: %s  n_pairs=%zu vn_hits=%d "
        "matched=%zu missing=%zu extras=%d digest_mismatches=%d\n",
        backend_name, hxhost,
        pass_ok ? "PASS" : "FAIL",
        n_pairs, vn_hits, matched, missing, extras, mismatches);

    if (pass_ok) {
        fprintf(stderr,
            "hx codegen 2a.6 (%s) sample byte-exact tuples:\n",
            backend_name);
        size_t sample = vn_hits < 5 ? (size_t)vn_hits : 5;
        for (size_t s = 0; s < sample; s++) {
            uint32_t *slot = vhits + s * 19;
            uint32_t widx = slot[0];
            uint32_t sidx = slot[1];
            size_t pid = (size_t)widx * n_salt + (size_t)sidx;
            uint32_t e0 = ((uint32_t *)(oracle + pid * 16))[0];
            uint32_t e1 = ((uint32_t *)(oracle + pid * 16))[1];
            uint32_t e2 = ((uint32_t *)(oracle + pid * 16))[2];
            uint32_t e3 = ((uint32_t *)(oracle + pid * 16))[3];
            fprintf(stderr,
                "  [%zu] pass=\"%s\" salt=\"%s\" "
                "oracle=%08x%08x%08x%08x codegen=%08x%08x%08x%08x\n",
                s, passes[widx], salts[sidx],
                e0, e1, e2, e3,
                slot[3], slot[4], slot[5], slot[6]);
        }
    }

    /* Per-backend cleanup. */
    if (backend_id == 0) {
#if defined(OPENCL_GPU)
        gpu_opencl_jit_release_keep(
            (struct _cl_program *)jit_handle_a,
            (struct _cl_kernel  *)jit_handle_b);
#endif
    } else {
#if defined(__APPLE__) && defined(METAL_GPU)
        gpu_metal_jit_release_keep(jit_handle_a, jit_handle_b);
#endif
    }
    free(vhits); free(seen);
    free(packed); free(woff);
    free(cfp); free(cidx); free(hbuf); free(hoff); free(hlen);
    free(oracle);
    for (size_t i = 0; i < n_pass; i++) free(passes[i]);
    for (size_t i = 0; i < n_salt; i++) free(salts[i]);
    free(passes); free(salts);
    free(hxsrc);

    exit(pass_ok ? 0 : 1);
}
#endif /* OPENCL_GPU || (APPLE && METAL_GPU) */

/* Sub-phase 5a.2 (2026-05-22): MAKE_MD5PASS family validate harness.
 * Sibling to hx_e347_validate_run_shared above. Differences from e347:
 *
 *   - Fixture has PASS lines only (no SALT lines); family is unsalted.
 *     Oracle is computed per pass (not per pass*salt cartesian).
 *   - Oracle dispatch: switches on job_enum so the harness can validate
 *     any 5a-supported family member by changing only the env. For 5a.2
 *     only e161 SHA1MD5PASS is implemented; other GPU-eligible algos
 *     route to the FATAL "5a.4 implements <primitive>" branch.
 *   - Digest width is per-primitive (e161 SHA1 = 20 bytes). The CPU
 *     oracle stores the FULL digest into HashDataBuf so any future
 *     full-width round-trip readback works; the compact_fp probe still
 *     uses only first 8 bytes (key = h0/h1 64-bit) and EMIT_HIT_4_DEDUP
 *     emits h0..h3 (16 bytes) -- diff compares those 16 bytes against
 *     the first 16 bytes of the oracle.
 *   - Dispatch: gpu_opencl_kernelb_family_validate_dispatch (sibling to
 *     gpu_opencl_e347_validate_dispatch). For Metal, 5a.3 ships its
 *     own sibling.
 *
 * Sub-phase 5a.3 (2026-05-22): Metal twin enabled. backend_id=0 selects
 * OpenCL; backend_id=1 selects Metal. Shared front matter (fixture parse,
 * oracle build, compact-table plant, candidate pack, diff) is identical
 * between backends; only the JIT compile entry-point + dispatch helper +
 * release call differ. Function guard widened to include METAL_GPU
 * mirroring hx_e347_validate_run_shared's pattern.
 *
 * Exits 0 on PASS, 1 on FAIL; never returns. */
#if defined(OPENCL_GPU) || (defined(__APPLE__) && defined(METAL_GPU))

/* Compute oracle digest for a single (pass) per job_enum. Returns
 * digest length in bytes (e161=20). FATAL on unsupported job_enum.
 *
 * Sub-phase 5a.5 (2026-05-22): de-staticized so gpu/gpujob_opencl.c and
 * gpu/gpujob_metal.m hit-replay paths can call it to recompute the FULL
 * digest from the matched plaintext (the family kernel emits only the
 * first 4 uints to the hit record; checkhash for digests > 16 bytes
 * needs the full digest to verify against HashDataBuf). */
int
oracle_compute_md5pass_family(int job_enum, const char *pass, int plen,
                              unsigned char *out)
{
    if (!pass || !out || plen < 0) {
        fprintf(stderr,
            "FATAL: %s:%d oracle_compute_md5pass_family: bad args "
            "(job=%d pass=%p plen=%d out=%p)\n",
            __FILE__, __LINE__, job_enum, (void*)pass, plen, (void*)out);
        exit(1);
    }
    /* Common prefix: MD5(pass) -> hex32 -> concat with pass. */
    unsigned char inner[16];
    mymd5((void *)pass, plen, inner);
    /* Build hex32(inner) || pass into a private buffer. */
    int total = 32 + plen;
    unsigned char *buf = (unsigned char *)malloc((size_t)total + 1);
    if (!buf) {
        fprintf(stderr,
            "FATAL: %s:%d oracle_compute_md5pass_family: malloc(%d) "
            "failed\n", __FILE__, __LINE__, total + 1);
        exit(1);
    }
    /* prmd5 writes 32 hex chars (no NUL) into linebuf -- it returns a
     * char* but signature matches what e347 oracle uses. */
    prmd5_32(inner, (char *)buf);
    memcpy(buf + 32, pass, plen);

    switch (job_enum) {
    case JOB_SHA1MD5PASS:        /* e161 */
        mysha1((char *)buf, total, out);
        free(buf);
        return 20;
    case JOB_MD2MD5PASS:         /* e120 -- 5b.1a Tier 1 */
        {
            sph_md2_context cmd2;
            sph_md2_init(&cmd2);
            sph_md2(&cmd2, buf, total);
            sph_md2_close(&cmd2, out);
        }
        free(buf);
        return 16;
    case JOB_MD4MD5PASS:         /* e122 */
        MD4((char *)buf, total, out);
        free(buf);
        return 16;
    case JOB_RMD128MD5PASS:      /* e157 -- 5b.1b Tier 1 */
        RIPEMD128((char *)buf, total, out);
        free(buf);
        return 16;
    case JOB_RMD160MD5PASS:      /* e159 */
        RIPEMD160((char *)buf, total, out);
        free(buf);
        return 20;
    case JOB_SHA224MD5PASS:      /* e163 */
        {
            sph_sha224_context c224;
            sph_sha224_init(&c224);
            sph_sha224(&c224, buf, total);
            sph_sha224_close(&c224, out);
        }
        free(buf);
        return 28;
    case JOB_SHA256MD5PASS:      /* e165 */
        mysha256((char *)buf, total, out);
        free(buf);
        return 32;
    case JOB_SHA384MD5PASS:      /* e167 */
        {
            sph_sha384_context c384;
            sph_sha384_init(&c384);
            sph_sha384(&c384, buf, total);
            sph_sha384_close(&c384, out);
        }
        free(buf);
        return 48;
    case JOB_SHA512MD5PASS:      /* e169 */
        mysha512((char *)buf, total, out);
        free(buf);
        return 64;
    case JOB_TIGERMD5PASS:       /* e171 -- 5b.2b Tier 2 */
        {
            sph_tiger_context ctiger;
            sph_tiger_init(&ctiger);
            sph_tiger(&ctiger, buf, total);
            sph_tiger_close(&ctiger, out);
        }
        free(buf);
        return 24;
    case JOB_WRLMD5PASS:         /* e173 -- 5b.2a Tier 2 */
        WHIRLPOOL((char *)buf, total, out);
        free(buf);
        return 64;
    /* Phase 5b Tier 3 sub-phase 5b.3a (2026-05-27): 5 3-pass HAVAL
     * variants. Each calls the matching sph_haval<W>_3 entry point;
     * sph_haval is the verified-conformant CPU oracle (R15 pre-flight
     * test_haval_paper_vectors.c 120/120 cells PASS). */
    case JOB_HAV128MD5PASS:      /* e127 -- 5b.3a Tier 3 (3-pass, 128-bit) */
        {
            sph_haval128_3_context c;
            sph_haval128_3_init(&c);
            sph_haval128_3(&c, buf, total);
            sph_haval128_3_close(&c, out);
        }
        free(buf);
        return 16;
    case JOB_HAV160_3MD5PASS:    /* e133 -- 5b.3a Tier 3 (3-pass, 160-bit) */
        {
            sph_haval160_3_context c;
            sph_haval160_3_init(&c);
            sph_haval160_3(&c, buf, total);
            sph_haval160_3_close(&c, out);
        }
        free(buf);
        return 20;
    case JOB_HAV192_3MD5PASS:    /* e139 -- 5b.3a Tier 3 (3-pass, 192-bit) */
        {
            sph_haval192_3_context c;
            sph_haval192_3_init(&c);
            sph_haval192_3(&c, buf, total);
            sph_haval192_3_close(&c, out);
        }
        free(buf);
        return 24;
    case JOB_HAV224_3MD5PASS:    /* e145 -- 5b.3a Tier 3 (3-pass, 224-bit) */
        {
            sph_haval224_3_context c;
            sph_haval224_3_init(&c);
            sph_haval224_3(&c, buf, total);
            sph_haval224_3_close(&c, out);
        }
        free(buf);
        return 28;
    case JOB_HAV256MD5PASS:      /* e151 -- 5b.3a Tier 3 (3-pass, 256-bit) */
        {
            sph_haval256_3_context c;
            sph_haval256_3_init(&c);
            sph_haval256_3(&c, buf, total);
            sph_haval256_3_close(&c, out);
        }
        free(buf);
        return 32;
    case JOB_HAV128_4MD5PASS:    /* e129 -- 5b.3b Tier 3 (4-pass, 128-bit) */
        {
            sph_haval128_4_context c;
            sph_haval128_4_init(&c);
            sph_haval128_4(&c, buf, total);
            sph_haval128_4_close(&c, out);
        }
        free(buf);
        return 16;
    case JOB_HAV160_4MD5PASS:    /* e135 -- 5b.3b Tier 3 (4-pass, 160-bit) */
        {
            sph_haval160_4_context c;
            sph_haval160_4_init(&c);
            sph_haval160_4(&c, buf, total);
            sph_haval160_4_close(&c, out);
        }
        free(buf);
        return 20;
    case JOB_HAV192_4MD5PASS:    /* e141 -- 5b.3b Tier 3 (4-pass, 192-bit) */
        {
            sph_haval192_4_context c;
            sph_haval192_4_init(&c);
            sph_haval192_4(&c, buf, total);
            sph_haval192_4_close(&c, out);
        }
        free(buf);
        return 24;
    case JOB_HAV224_4MD5PASS:    /* e147 -- 5b.3b Tier 3 (4-pass, 224-bit) */
        {
            sph_haval224_4_context c;
            sph_haval224_4_init(&c);
            sph_haval224_4(&c, buf, total);
            sph_haval224_4_close(&c, out);
        }
        free(buf);
        return 28;
    case JOB_HAV256_4MD5PASS:    /* e153 -- 5b.3b Tier 3 (4-pass, 256-bit) */
        {
            sph_haval256_4_context c;
            sph_haval256_4_init(&c);
            sph_haval256_4(&c, buf, total);
            sph_haval256_4_close(&c, out);
        }
        free(buf);
        return 32;
    case JOB_HAV128_5MD5PASS:    /* e131 -- 5b.3c Tier 3 (5-pass, 128-bit) */
        {
            sph_haval128_5_context c;
            sph_haval128_5_init(&c);
            sph_haval128_5(&c, buf, total);
            sph_haval128_5_close(&c, out);
        }
        free(buf);
        return 16;
    case JOB_HAV160_5MD5PASS:    /* e137 -- 5b.3c Tier 3 (5-pass, 160-bit) */
        {
            sph_haval160_5_context c;
            sph_haval160_5_init(&c);
            sph_haval160_5(&c, buf, total);
            sph_haval160_5_close(&c, out);
        }
        free(buf);
        return 20;
    case JOB_HAV192_5MD5PASS:    /* e143 -- 5b.3c Tier 3 (5-pass, 192-bit) */
        {
            sph_haval192_5_context c;
            sph_haval192_5_init(&c);
            sph_haval192_5(&c, buf, total);
            sph_haval192_5_close(&c, out);
        }
        free(buf);
        return 24;
    case JOB_HAV224_5MD5PASS:    /* e149 -- 5b.3c Tier 3 (5-pass, 224-bit) */
        {
            sph_haval224_5_context c;
            sph_haval224_5_init(&c);
            sph_haval224_5(&c, buf, total);
            sph_haval224_5_close(&c, out);
        }
        free(buf);
        return 28;
    case JOB_HAV256_5MD5PASS:    /* e155 -- 5b.3c Tier 3 (5-pass, 256-bit) */
        {
            sph_haval256_5_context c;
            sph_haval256_5_init(&c);
            sph_haval256_5(&c, buf, total);
            sph_haval256_5_close(&c, out);
        }
        free(buf);
        return 32;
    /* Phase 5b Tier 4 sub-phase 5b.4a (2026-05-27): the 2 Snefru widths.
     * Use a FRESH LOCAL rhash handle (NOT the global rhash_con, which is
     * configured per-op and is not safe for a standalone recompute helper
     * -- R-Tier4-oracle-rhash-context). librhash is the LIVE CPU oracle
     * for e175/e177 (matches the JOB_SNE128/256MD5PASS case bodies which
     * use rhash_init(RHASH_SNEFRU*)). */
    case JOB_SNE128MD5PASS:      /* e175 -- 5b.4a Tier 4 (Snefru-128) */
        {
            rhash rc128 = rhash_init(RHASH_SNEFRU128);
            if (!rc128) {
                fprintf(stderr,
                    "FATAL: %s:%d oracle_compute_md5pass_family: "
                    "rhash_init(RHASH_SNEFRU128) failed (e175)\n",
                    __FILE__, __LINE__);
                free(buf);
                exit(1);
            }
            rhash_update(rc128, buf, total);
            rhash_final(rc128, out);
            rhash_free(rc128);
        }
        free(buf);
        return 16;
    case JOB_SNE256MD5PASS:      /* e177 -- 5b.4a Tier 4 (Snefru-256) */
        {
            rhash rc256 = rhash_init(RHASH_SNEFRU256);
            if (!rc256) {
                fprintf(stderr,
                    "FATAL: %s:%d oracle_compute_md5pass_family: "
                    "rhash_init(RHASH_SNEFRU256) failed (e177)\n",
                    __FILE__, __LINE__);
                free(buf);
                exit(1);
            }
            rhash_update(rc256, buf, total);
            rhash_final(rc256, out);
            rhash_free(rc256);
        }
        free(buf);
        return 32;
    /* Phase 5b Tier 4 sub-phase 5b.4b (2026-05-27): GOST R 34.11-94 (e125),
     * the FINAL GPU-eligible MAKE_MD5PASS member. Call the in-tree
     * gosthash() directly (the LIVE CPU oracle; matches the JOB_GOSTMD5PASS
     * case body at mdxfind.c:29076). TEST S-box set (gosthash_init computes
     * the derived tables once at startup; NOT CryptoPro -- R-Tier4-gost-sbox
     * HIGH). 32-byte digest. */
    case JOB_GOSTMD5PASS:        /* e125 -- 5b.4b Tier 4 (GOST R 34.11-94) */
        gosthash((char *)buf, total, out);
        free(buf);
        return 32;
    /* Sub-phase 5c.2 (2026-05-27): e123 MD5MD5PASS multi-emit.
     *
     * e123 produces TWO digests per password (variant 0 canonical +
     * variant 1 colon). This 4-arg oracle returns variant 0 (canonical):
     *   md5( hex32(md5(pass)) . pass )
     * which is exactly the `buf` built above (hex32 || pass). The colon
     * variant is built separately by the validation harness (it injects
     * a ':' at offset 32). On the gpujob hit-replay path this arm is NOT
     * reached for e123: the recompute branch only fires for digests > 16
     * bytes, and e123's MD5 digest is 16 bytes (checkhash re-probes the
     * loaded table directly from the kernel-emitted fingerprint, which
     * self-identifies the matched variant -- no recompute needed). So
     * returning the canonical variant here is sufficient + the signature
     * stays unchanged for all callers. */
    case JOB_MD5MD5PASS:         /* e123 -- 5c.2 multi-emit (variant 0) */
        mymd5((char *)buf, total, out);
        free(buf);
        return 16;
    default:
        fprintf(stderr,
            "FATAL: %s:%d oracle_compute_md5pass_family: e%d not in "
            "MAKE_MD5PASS family (or in 5b-deferred set).\n",
            __FILE__, __LINE__, job_enum);
        free(buf);
        exit(1);
    }
}

static void hx_family_md5pass_validate_run_shared(
    int backend_id, char *hxsrc, const char *fixture_env,
    const char *hxhost, int job_enum)
{
    const char *backend_name = (backend_id == 0) ? "OpenCL" : "Metal";

    if (backend_id != 0 && backend_id != 1) {
        fprintf(stderr,
            "FATAL: %s:%d 5a family validate: unknown backend %d "
            "(0=OpenCL, 1=Metal)\n",
            __FILE__, __LINE__, backend_id);
        exit(1);
    }
    if (!fixture_env || !*fixture_env) {
        fprintf(stderr,
            "FATAL: %s:%d 5a.2 family validate (%s): "
            "MDXFIND_HX_CODEGEN_FIXTURE not set on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }
    FILE *ff = fopen(fixture_env, "r");
    if (!ff) {
        fprintf(stderr,
            "FATAL: %s:%d 5a.2 family validate (%s): cannot open fixture "
            "'%s' on %s (errno=%d)\n",
            __FILE__, __LINE__, backend_name, fixture_env, hxhost, errno);
        exit(1);
    }

    /* Parse fixture: PASS:<text> only (family is unsalted; SALT lines
     * are accepted but ignored for diagnostic clarity). */
    char **passes = NULL; size_t n_pass = 0, cap_pass = 0;
    char line[4096];
    while (fgets(line, sizeof(line), ff)) {
        size_t L = strlen(line);
        while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) {
            line[--L] = '\0';
        }
        if (L == 0 || line[0] == '#') continue;
        if (L > 5 && memcmp(line, "PASS:", 5) == 0) {
            if (n_pass == cap_pass) {
                cap_pass = cap_pass ? cap_pass * 2 : 16;
                passes = realloc(passes, cap_pass * sizeof(char *));
            }
            passes[n_pass++] = strdup(line + 5);
        }
        /* SALT lines silently ignored (family is unsalted). */
    }
    fclose(ff);
    if (n_pass == 0) {
        fprintf(stderr,
            "FATAL: %s:%d 5a.2 family validate (%s): fixture %s has no "
            "PASS lines on %s\n",
            __FILE__, __LINE__, backend_name, fixture_env, hxhost);
        exit(1);
    }
    fprintf(stderr,
        "hx codegen 5a.2 (%s) e%d: fixture %s parsed n_pass=%zu on %s\n",
        backend_name, job_enum, fixture_env, n_pass, hxhost);

    /* Compute oracle digests per pass. Width depends on primitive.
     * Store full width into oracle buffer so round-trip readback works;
     * planted compact-table entries use the full width too (HashDataBuf
     * stores full digest; compact_fp is the 64-bit key derived from
     * first 8 bytes). Sub-phase 5a.4 (2026-05-23) widens from e161-only
     * to all 7 wired family members.
     *
     * MD4=16, RMD160=20, SHA1=20, SHA224=28, SHA256=32, SHA384=48, SHA512=64. */
    int dlen = 0;
    switch (job_enum) {
        case JOB_MD5MD5PASS:    dlen = 16; break;  /* 5c.2 multi-emit (e123) */
        case JOB_MD2MD5PASS:    dlen = 16; break;  /* 5b.1a Tier 1 */
        case JOB_MD4MD5PASS:    dlen = 16; break;
        case JOB_RMD128MD5PASS: dlen = 16; break;  /* 5b.1b Tier 1 */
        case JOB_RMD160MD5PASS: dlen = 20; break;
        case JOB_SHA1MD5PASS:   dlen = 20; break;
        case JOB_SHA224MD5PASS: dlen = 28; break;
        case JOB_SHA256MD5PASS: dlen = 32; break;
        case JOB_SHA384MD5PASS: dlen = 48; break;
        case JOB_SHA512MD5PASS: dlen = 64; break;
        case JOB_TIGERMD5PASS:  dlen = 24; break;  /* 5b.2b Tier 2 */
        case JOB_WRLMD5PASS:    dlen = 64; break;  /* 5b.2a Tier 2 */
        case JOB_HAV128MD5PASS: dlen = 16; break;  /* 5b.3a Tier 3 (e127) */
        case JOB_HAV160_3MD5PASS: dlen = 20; break;/* 5b.3a Tier 3 (e133) */
        case JOB_HAV192_3MD5PASS: dlen = 24; break;/* 5b.3a Tier 3 (e139) */
        case JOB_HAV224_3MD5PASS: dlen = 28; break;/* 5b.3a Tier 3 (e145) */
        case JOB_HAV256MD5PASS: dlen = 32; break;  /* 5b.3a Tier 3 (e151) */
        case JOB_HAV128_4MD5PASS: dlen = 16; break;/* 5b.3b Tier 3 (e129) */
        case JOB_HAV160_4MD5PASS: dlen = 20; break;/* 5b.3b Tier 3 (e135) */
        case JOB_HAV192_4MD5PASS: dlen = 24; break;/* 5b.3b Tier 3 (e141) */
        case JOB_HAV224_4MD5PASS: dlen = 28; break;/* 5b.3b Tier 3 (e147) */
        case JOB_HAV256_4MD5PASS: dlen = 32; break;/* 5b.3b Tier 3 (e153) */
        case JOB_HAV128_5MD5PASS: dlen = 16; break;/* 5b.3c Tier 3 (e131) */
        case JOB_HAV160_5MD5PASS: dlen = 20; break;/* 5b.3c Tier 3 (e137) */
        case JOB_HAV192_5MD5PASS: dlen = 24; break;/* 5b.3c Tier 3 (e143) */
        case JOB_HAV224_5MD5PASS: dlen = 28; break;/* 5b.3c Tier 3 (e149) */
        case JOB_HAV256_5MD5PASS: dlen = 32; break;/* 5b.3c Tier 3 (e155) */
        case JOB_SNE128MD5PASS: dlen = 16; break;  /* 5b.4a Tier 4 (e175) */
        case JOB_SNE256MD5PASS: dlen = 32; break;  /* 5b.4a Tier 4 (e177) */
        case JOB_GOSTMD5PASS:   dlen = 32; break;  /* 5b.4b Tier 4 (e125) */
        default: break;
    }
    if (dlen == 0) {
        fprintf(stderr,
            "FATAL: %s:%d 5a family validate: digest width unknown for "
            "e%d on %s (family + 5c.2 multi-emit e123 wired)\n",
            __FILE__, __LINE__, job_enum, hxhost);
        exit(1);
    }

    /* Sub-phase 5c.2 (2026-05-27): multi-emit support.
     *
     * A multi-emit member (e123 MD5MD5PASS) produces N>1 digests per
     * password (variant 0 canonical + variant 1 colon). The harness must
     * plant ALL N variant digests as separate loaded-hash candidates and
     * expect the GPU to emit ALL N cracks per password (the G1b
     * dual-hash canary). We model this with n_variants distinct "rows"
     * per pass: row r = pi*n_variants + v carries pass pi's variant-v
     * digest. The compact table is sized for n_rows = n_pass*n_variants.
     * Single-emit members keep n_variants=1 -> identical to prior
     * behavior (G2 regression safety). */
    int n_variants = (job_enum == JOB_MD5MD5PASS) ? 2 : 1;
    size_t n_rows = n_pass * (size_t)n_variants;

    unsigned char *oracle = (unsigned char *)malloc(n_rows * dlen);
    if (!oracle) {
        fprintf(stderr,
            "FATAL: %s:%d 5a.2 family validate (%s): malloc(oracle %zu B) "
            "failed on %s\n",
            __FILE__, __LINE__, backend_name,
            n_rows * dlen, hxhost);
        exit(1);
    }
    for (size_t pi = 0; pi < n_pass; pi++) {
        int plen = (int)strlen(passes[pi]);
        /* Variant 0 (canonical) via the shared oracle: returns
         * md5(hex32(md5(pass)) . pass). */
        unsigned char *row0 = oracle + (pi * n_variants + 0) * dlen;
        int rdlen = oracle_compute_md5pass_family(
                        job_enum, passes[pi], plen, row0);
        if (rdlen != dlen) {
            fprintf(stderr,
                "FATAL: %s:%d 5a.2 family validate: oracle returned "
                "rdlen=%d (expected %d) for pi=%zu pass=\"%s\"\n",
                __FILE__, __LINE__, rdlen, dlen, pi, passes[pi]);
            exit(1);
        }
        if (n_variants == 2) {
            /* Variant 1 (colon): md5(hex32(md5(pass)) . ':' . pass).
             * Built inline (mirrors mdxfind.c CPU oracle linebuf2 at
             * the JOB_MD5MD5PASS case). 16-byte MD5 digest. */
            unsigned char inner[16];
            char vbuf[32 + 1 + 256];
            mymd5((void *)passes[pi], plen, inner);
            prmd5_32(inner, vbuf);
            vbuf[32] = ':';
            if (plen > (int)sizeof(vbuf) - 33) plen = (int)sizeof(vbuf) - 33;
            memcpy(vbuf + 33, passes[pi], plen);
            unsigned char *row1 = oracle + (pi * n_variants + 1) * dlen;
            mymd5(vbuf, 33 + plen, row1);
        }
    }
    fprintf(stderr,
        "hx codegen 5a.2 (%s) e%d: oracle digests computed (%d bytes each) "
        "for %zu passes x %d variants = %zu rows\n",
        backend_name, job_enum, dlen, n_pass, n_variants, n_rows);

    /* Build synthetic compact-hash table sized for n_rows. */
    uint64_t cap_pow2 = 1;
    while (cap_pow2 < (uint64_t)n_rows * 2) cap_pow2 <<= 1;
    if (cap_pow2 < 64) cap_pow2 = 64;
    uint64_t cmask = cap_pow2 - 1;
    uint32_t *cfp  = (uint32_t *)calloc(cap_pow2, sizeof(uint32_t));
    uint32_t *cidx = (uint32_t *)calloc(cap_pow2, sizeof(uint32_t));
    unsigned char *hbuf = (unsigned char *)malloc(n_rows * dlen);
    size_t *hoff = (size_t *)malloc(n_rows * sizeof(size_t));
    unsigned short *hlen =
        (unsigned short *)malloc(n_rows * sizeof(unsigned short));
    if (!cfp || !cidx || !hbuf || !hoff || !hlen) {
        fprintf(stderr,
            "FATAL: %s:%d 5a.2 family validate (%s): alloc compact "
            "buffers failed on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }
    for (size_t i = 0; i < n_rows; i++) {
        memcpy(hbuf + i * dlen, oracle + i * dlen, dlen);
        hoff[i] = i * dlen;
        hlen[i] = (unsigned short)dlen;
        uint32_t hx = ((uint32_t *)(oracle + i * dlen))[0];
        uint32_t hy = ((uint32_t *)(oracle + i * dlen))[1];
        uint64_t key = ((uint64_t)hy << 32) | hx;
        uint32_t fp = (uint32_t)(key >> 32);
        if (fp == 0) fp = 1;
        uint64_t pos = (key ^ (key >> 32)) & cmask;
        for (int probe = 0; probe < 256; probe++) {
            uint64_t p = (pos + probe) & cmask;
            if (cfp[p] == 0) {
                cfp[p] = fp;
                cidx[p] = (uint32_t)i;
                break;
            }
        }
    }
    fprintf(stderr,
        "hx codegen 5a.2 (%s) e%d: compact table built cap=%llu "
        "mask=0x%llx n_rows=%zu\n",
        backend_name, job_enum,
        (unsigned long long)cap_pow2, (unsigned long long)cmask, n_rows);

    /* Upload compact table + a single 1-byte salt placeholder (kernel
     * ignores salts but the buffer slot must bind; both backends'
     * set_salts/set_salt reject zero-length entries because
     * clEnqueueWriteBuffer / newBufferWithBytes refuse 0 bytes). */
    if (backend_id == 0) {
#if defined(OPENCL_GPU)
        if (gpu_opencl_set_compact_table(0, cfp, cidx, cap_pow2, cmask,
                hbuf, n_rows * dlen, hoff, n_rows, hlen) != 0) {
            fprintf(stderr,
                "FATAL: %s:%d 5a.2 family validate (OpenCL): "
                "set_compact_table failed on %s dev[0]\n",
                __FILE__, __LINE__, hxhost);
            exit(1);
        }
        {
            char zsalt[1] = { 0 };
            uint32_t zoff = 0;
            uint16_t zlen = 1;
            if (gpu_opencl_set_salts(0, zsalt, &zoff, &zlen, 1) != 0) {
                fprintf(stderr,
                    "FATAL: %s:%d 5a.2 family validate (OpenCL): set_salts "
                    "(zero-placeholder) failed on %s dev[0]\n",
                    __FILE__, __LINE__, hxhost);
                exit(1);
            }
        }
#endif
    } else {
#if defined(__APPLE__) && defined(METAL_GPU)
        if (gpu_metal_set_compact_table(0, cfp, cidx, cap_pow2, cmask,
                hbuf, n_rows * dlen, hoff, n_rows, hlen) != 0) {
            fprintf(stderr,
                "FATAL: %s:%d 5a.3 family validate (Metal): "
                "set_compact_table failed on %s dev[0]\n",
                __FILE__, __LINE__, hxhost);
            exit(1);
        }
        {
            char zsalt[1] = { 0 };
            uint32_t zoff = 0;
            uint16_t zlen = 1;
            if (gpu_metal_set_salt(zsalt, &zoff, &zlen, 1) != 0) {
                fprintf(stderr,
                    "FATAL: %s:%d 5a.3 family validate (Metal): set_salt "
                    "(1-byte placeholder) failed on %s dev[0]\n",
                    __FILE__, __LINE__, hxhost);
                exit(1);
            }
        }
#endif
    }

    /* Pack candidates [len_u8][bytes] per word. */
    size_t pack_bytes = 0;
    for (size_t pi = 0; pi < n_pass; pi++)
        pack_bytes += 1 + strlen(passes[pi]);
    unsigned char *packed = (unsigned char *)malloc(pack_bytes + 16);
    uint32_t *woff = (uint32_t *)malloc(n_pass * sizeof(uint32_t));
    if (!packed || !woff) {
        fprintf(stderr,
            "FATAL: %s:%d 5a.2 family validate (%s): alloc packed "
            "words failed on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }
    {
        size_t pos = 0;
        for (size_t pi = 0; pi < n_pass; pi++) {
            size_t L = strlen(passes[pi]);
            if (L > 255) {
                fprintf(stderr,
                    "FATAL: %s:%d 5a.2 family validate (%s): pass %zu "
                    "length %zu > 255 on %s\n",
                    __FILE__, __LINE__, backend_name, pi, L, hxhost);
                exit(1);
            }
            woff[pi] = (uint32_t)pos;
            packed[pos++] = (unsigned char)L;
            memcpy(packed + pos, passes[pi], L);
            pos += L;
        }
        pack_bytes = pos;
    }

    /* JIT family kernel via _keep wrapper + dispatch via family validate
     * sibling. Family kernel entry-point is kernelb_hx_codegen_phase0.
     * Multi-emit can produce up to n_rows hits (n_variants per pass). */
    int max_hits = (int)n_rows;
    if (max_hits < 16) max_hits = 16;
    uint32_t *vhits =
        (uint32_t *)calloc((size_t)max_hits * 19, sizeof(uint32_t));
    int vn_hits = 0;
    if (!vhits) {
        fprintf(stderr,
            "FATAL: %s:%d 5a.2 family validate (%s): alloc vhits failed "
            "on %s\n",
            __FILE__, __LINE__, backend_name, hxhost);
        exit(1);
    }
    /* JIT handles: per-backend opaque pointers held across dispatch +
     * released at function exit. */
    void *fam_jit_a = NULL;   /* OpenCL: cl_program; Metal: MTLLibrary */
    void *fam_jit_b = NULL;   /* OpenCL: cl_kernel ; Metal: MTLComputePipelineState */

    if (backend_id == 0) {
#if defined(OPENCL_GPU)
        struct _cl_program *vprog = NULL;
        struct _cl_kernel  *vkern = NULL;
        gpu_opencl_jit_compile_source_with_common_keep(
            0, hxsrc, "-cl-std=CL1.2",
            "kernelb_hx_codegen_phase0", &vprog, &vkern);
        fam_jit_a = (void *)vprog;
        fam_jit_b = (void *)vkern;
        gpu_opencl_kernelb_family_validate_dispatch(
            0, vkern,
            packed, (uint32_t)pack_bytes, woff, (uint32_t)n_pass,
            vhits, &vn_hits, max_hits);
#endif
    } else {
#if defined(__APPLE__) && defined(METAL_GPU)
        gpu_metal_jit_compile_source_with_common_keep(
            0, hxsrc, "kernelb_hx_codegen_phase0",
            &fam_jit_a, &fam_jit_b);
        gpu_metal_kernelb_family_validate_dispatch(
            0, fam_jit_b,
            packed, (uint32_t)pack_bytes, woff, (uint32_t)n_pass,
            vhits, &vn_hits, max_hits);
#endif
    }

    fprintf(stderr,
        "hx codegen 5a.2 (%s) e%d: dispatch returned vn_hits=%d "
        "(expected %zu)\n",
        backend_name, job_enum, vn_hits, n_pass);

    /* Diff: every ROW (pass x variant) should appear EXACTLY once.
     *
     * Sub-phase 5c.2 (2026-05-27) multi-emit: a hit carries widx (the
     * pass index) + its 16-byte digest (slot[3..6]). For multi-emit each
     * pass emits n_variants hits sharing the same widx but with DISTINCT
     * digests. We resolve each hit to its (pass, variant) ROW by matching
     * the hit digest against that pass's planted variant digests, then
     * mark seen_row[r]. A hit whose digest matches NO variant of its pass
     * is a digest mismatch; a widx out of range is an extra. The total
     * expected hit count is n_rows. This is THE G1b dual-hash canary
     * check: for the canary pass, BOTH variant rows must be seen. */
    unsigned char *seen_row = (unsigned char *)calloc(n_rows, 1);
    int extras = 0, mismatches = 0;
    size_t first_mismatch_print = 0;
    for (int h = 0; h < vn_hits; h++) {
        uint32_t *slot = vhits + (size_t)h * 19;
        uint32_t widx = slot[0];
        /* sidx (slot[1]) is always 0 in family kernel emit. */
        if (widx >= n_pass) {
            extras++;
            if (first_mismatch_print < 10) {
                fprintf(stderr,
                    "  HIT-EXTRA #%d widx=%u (out of range, n_pass=%zu)\n",
                    h, widx, n_pass);
                first_mismatch_print++;
            }
            continue;
        }
        /* Match the hit digest against pass widx's n_variants rows. */
        int matched_variant = -1;
        for (int v = 0; v < n_variants; v++) {
            size_t r = (size_t)widx * n_variants + v;
            uint32_t e0 = ((uint32_t *)(oracle + r * dlen))[0];
            uint32_t e1 = ((uint32_t *)(oracle + r * dlen))[1];
            uint32_t e2 = ((uint32_t *)(oracle + r * dlen))[2];
            uint32_t e3 = ((uint32_t *)(oracle + r * dlen))[3];
            if (slot[3] == e0 && slot[4] == e1 &&
                slot[5] == e2 && slot[6] == e3) {
                matched_variant = v;
                break;
            }
        }
        if (matched_variant < 0) {
            mismatches++;
            if (first_mismatch_print < 10) {
                uint32_t e0 = ((uint32_t *)(oracle + (size_t)widx * n_variants * dlen))[0];
                fprintf(stderr,
                    "  HIT-DIFF #%d (pi=%u) no variant matched; "
                    "got=%08x%08x%08x%08x (variant0[0]=%08x)\n",
                    h, widx,
                    slot[3], slot[4], slot[5], slot[6], e0);
                first_mismatch_print++;
            }
            continue;
        }
        size_t r = (size_t)widx * n_variants + matched_variant;
        if (!seen_row[r]) seen_row[r] = 1;
    }
    size_t matched = 0, missing = 0;
    for (size_t i = 0; i < n_rows; i++)
        if (seen_row[i]) matched++; else missing++;
    int miss_print = 0;
    if (missing > 0) {
        fprintf(stderr, "  Missing rows (pass x variant, first 10):\n");
        for (size_t i = 0; i < n_rows && miss_print < 10; i++) {
            if (!seen_row[i]) {
                size_t pi = i / n_variants;
                size_t v  = i % n_variants;
                fprintf(stderr, "    row=%zu pi=%zu variant=%zu pass=\"%s\"\n",
                        i, pi, v, passes[pi]);
                miss_print++;
            }
        }
    }

    int pass_ok = (mismatches == 0 && extras == 0 &&
                   matched == n_rows && vn_hits == (int)n_rows);
    fprintf(stderr,
        "hx codegen 5a.2 (%s) e%d RESULT on %s: %s  n_pass=%zu n_variants=%d "
        "n_rows=%zu vn_hits=%d matched=%zu missing=%zu extras=%d "
        "digest_mismatches=%d\n",
        backend_name, job_enum, hxhost,
        pass_ok ? "PASS" : "FAIL",
        n_pass, n_variants, n_rows, vn_hits, matched, missing, extras,
        mismatches);

    if (pass_ok) {
        fprintf(stderr,
            "hx codegen 5a.2 (%s) e%d sample byte-exact tuples:\n",
            backend_name, job_enum);
        size_t sample = vn_hits < 5 ? (size_t)vn_hits : 5;
        for (size_t s = 0; s < sample; s++) {
            uint32_t *slot = vhits + s * 19;
            uint32_t widx = slot[0];
            /* Print the hit's own digest; for multi-emit the matching
             * planted variant is whichever row equals slot[3..6]. */
            fprintf(stderr,
                "  [%zu] pass=\"%s\" codegen=%08x%08x%08x%08x\n",
                s, (widx < n_pass) ? passes[widx] : "(oob)",
                slot[3], slot[4], slot[5], slot[6]);
        }
    }

    if (backend_id == 0) {
#if defined(OPENCL_GPU)
        gpu_opencl_jit_release_keep(
            (struct _cl_program *)fam_jit_a,
            (struct _cl_kernel  *)fam_jit_b);
#endif
    } else {
#if defined(__APPLE__) && defined(METAL_GPU)
        gpu_metal_jit_release_keep(fam_jit_a, fam_jit_b);
#endif
    }
    free(vhits); free(seen_row);
    free(packed); free(woff);
    free(cfp); free(cidx); free(hbuf); free(hoff); free(hlen);
    free(oracle);
    for (size_t i = 0; i < n_pass; i++) free(passes[i]);
    free(passes);
    free(hxsrc);

    exit(pass_ok ? 0 : 1);
}
#endif /* OPENCL_GPU || (APPLE && METAL_GPU) */

#ifdef GPU_ENABLED
/* gpu_ops[] -- file-scope single source of truth for the legacy
 * template-path GPU op set. Sentinel-terminated with -1.
 *
 * Iterated in two places, both of which must agree by construction:
 *   1. The GPU init preamble inside build_compact_table() (~line 39185) --
 *      sets need_gpu=1 if any Dohash[] entry matches one of these JOBs
 *      (so gpu_opencl_init or gpu_metal_init fires).
 *   2. gpu_op_advertise_for_h_listing (~line 45515) -- drives the [GPU]
 *      tag on the `mdxfind -h` hash-type listing.
 *
 * Prior to rev 1.511 (2026-05-27) this lived at function scope and the
 * helper carried a parallel mirror; the two could drift silently. New
 * legacy-template-path GPU ops are added here ONCE and both sites pick
 * up the change automatically.
 *
 * Codegen-path ops (JOB_MD5MD5SALT e347; MAKE_MD5PASS family e120/e122/
 * e157/e159/e161/e163/e165/e167/e169/e171/e173) are intentionally
 * ABSENT from this table -- they are admitted via separate predicates
 * (the JOB_MD5MD5SALT check and gpu_codegen_kernelb_family_md5pass_-
 * eligible) so the legacy template path is not advertised as covering
 * them. Both the init preamble and gpu_op_advertise_for_h_listing apply
 * those extra predicates after the gpu_ops[] scan.
 */
static const int gpu_ops[] = {
    JOB_MD5SALT, JOB_MD5UCSALT, JOB_MD5revMD5SALT, JOB_MD5sub8_24SALT,
    JOB_MD5SALTPASS, JOB_MD5PASSSALT,
    JOB_SHA256PASSSALT, JOB_SHA256SALTPASS,
    /* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS joins the GPU-init
     * op list so `mdxfind -G N -M SHA224SALTPASS` (without other GPU
     * types) actually initializes GPU devices. Without this entry the
     * chokepoint admit gate never fires (GPU init is skipped in the
     * type-loader preamble). */
    JOB_SHA224SALTPASS,
    JOB_MD5CRYPT, JOB_MD5_MD5SALTMD5PASS,
    JOB_SHA1SALTPASS, JOB_SHA1PASSSALT,
    JOB_SHA1DRU, JOB_PHPBB3, JOB_MD5, JOB_MD5UC, JOB_MD4, JOB_SHA1,
    JOB_SHA224, JOB_SHA256, JOB_SHA384, JOB_SHA512,
    /* JOB_NTLM (e369): permanently CPU-only by design. The rules engine
     * assumes ASCII input bytes; NTLM emits 2-3 UTF-16LE candidate plains
     * per word via iconv (utf-8/cp1251/cp1252) which the rules engine
     * cannot process correctly. NTLMH (e786) is the GPU-capable variant
     * (single zero-extend, the hashcat-broken-conversion); MD4UTF16
     * (e496) is the iter sibling. */
    JOB_WRL, JOB_NTLMH, JOB_MD4UTF16, JOB_MD6256, JOB_SHA256RAW,
    JOB_KECCAK224, JOB_KECCAK256, JOB_KECCAK384, JOB_KECCAK512,
    JOB_SHA3_224, JOB_SHA3_256, JOB_SHA3_384, JOB_SHA3_512,
    JOB_HMAC_MD5, JOB_HMAC_MD5_KPASS,
    JOB_HMAC_SHA1, JOB_HMAC_SHA1_KPASS,
    JOB_HMAC_SHA224, JOB_HMAC_SHA224_KPASS,
    JOB_HMAC_SHA256, JOB_HMAC_SHA256_KPASS,
    JOB_HMAC_SHA384, JOB_HMAC_SHA384_KPASS,
    JOB_HMAC_SHA512, JOB_HMAC_SHA512_KPASS,
    JOB_HMAC_RMD160, JOB_HMAC_RMD160_KPASS,
    JOB_HMAC_RMD320, JOB_HMAC_RMD320_KPASS,
    JOB_HMAC_BLAKE2S,
    JOB_STREEBOG_32, JOB_STREEBOG_64,
    JOB_HMAC_STREEBOG256_KPASS, JOB_HMAC_STREEBOG256_KSALT,
    JOB_HMAC_STREEBOG512_KPASS, JOB_HMAC_STREEBOG512_KSALT,
    JOB_SHA512PASSSALT, JOB_SHA512SALTPASS,
    /* Phase 2d.5.7-followup (2026-05-13): admit JOB_SHA384SALTPASS to
     * the need_gpu activation set so Metal init runs when -m e812 is in
     * Dohash. Mirrors B6.9/B6.10 SHA512 sibling admission. */
    JOB_SHA384SALTPASS,
    JOB_SHA512CRYPT, JOB_SHA256CRYPT, JOB_SHA512CRYPTMD5,
    JOB_MD5RAW, JOB_SHA1RAW, JOB_SHA384RAW, JOB_SHA512RAW,
    JOB_SQL5, JOB_MYSQL3, JOB_DESCRYPT,
    /* MD5SHA256 (e179): composite sha256-then-two-block-MD5 template core. */
    JOB_MD5SHA256,
    JOB_RMD160, JOB_RMD320, JOB_BLAKE2S256,
    /* B5 sub-batch 3 (2026-05-06): BLAKE2B-256 / BLAKE2B-512 GPU
     * template. BLAKE2B-160 omitted -- no JOB_BLAKE2B160. */
    JOB_BLAKE2B256, JOB_BLAKE2B512,
    JOB_BCRYPT, -1
};
#endif

/* op_can_reach_gpu: can this op reach a GPU kernel at all?
 *
 * Defined in BOTH build configurations on purpose. gpu_ops[] lives inside
 * #ifdef GPU_ENABLED, so a plain CPU build (e.g. the macOS x86_64 target)
 * does not have it -- referencing it directly from the dispatch gate broke
 * that build. When no GPU backend is compiled in, nothing can reach a
 * kernel, so the honest answer is 0 and every expensive type takes the
 * CPU-only fan-out path.
 *
 * Sole consumer is the FRAGILE lineswanted gate; see the header there. */
#ifdef GPU_ENABLED
static int op_can_reach_gpu(int op) {
    int i;
    for (i = 0; gpu_ops[i] >= 0; i++)
        if (gpu_ops[i] == op) return 1;
    return 0;
}
#else
static int op_can_reach_gpu(int op) { (void)op; return 0; }
#endif


/* Capacity helpers for the HashData entry arrays and byte buffer.
 *
 * addhash() sizes HashDataOff/Len/Flags for the hex hashes it loads, but
 * build_compact_table() APPENDS more entries afterwards (bcrypt, sha512crypt,
 * md5crypt digests). Each of those append sites has to grow the arrays itself.
 * The bcrypt block did not: it checked only HashDataBufCap (the byte buffer)
 * and then wrote HashDataOff[HashDataCount] unconditionally, one element past
 * the end once HashDataCount reached HashDataCap. That corrupted the malloc
 * heap; the damage surfaced later and elsewhere, as "free(): invalid pointer"
 * inside compact_resize() when the compact table doubled.
 *
 * Reproduced 2026-08-07 on the CMIYC corpus: any run whose hash file mixes
 * bcrypt with enough hex hashes to fill the entry arrays. Confirmed
 * pre-existing (the RCS-tip baseline crashes identically).
 *
 * These two helpers centralise the growth so a future append site cannot
 * forget it again. Both exit(1) on allocation failure, matching the existing
 * behaviour at the other sites. */
static void hashdata_reserve(size_t need) {
    size_t newcap;
    if (HashDataCount + need <= HashDataCap) return;
    newcap = HashDataCap ? HashDataCap : 4096;
    while (newcap < HashDataCount + need) newcap *= 2;
    HashDataOff   = realloc(HashDataOff,   newcap * sizeof(size_t));
    HashDataLen   = realloc(HashDataLen,   newcap * sizeof(unsigned short));
    HashDataFlags = realloc(HashDataFlags, newcap * sizeof(unsigned short));
    if (!HashDataOff || !HashDataLen || !HashDataFlags) {
        fprintf(stderr, "Failed to grow hash entry arrays to %llu entries\n",
                (unsigned long long)newcap);
        exit(1);
    }
    memset(HashDataFlags + HashDataCap, 0,
           (newcap - HashDataCap) * sizeof(unsigned short));
    HashDataCap = newcap;
}

static void hashdatabuf_reserve(size_t need) {
    size_t newcap;
    if (HashDataBufUsed + need <= HashDataBufCap) return;
    newcap = HashDataBufCap ? HashDataBufCap : 65536;
    while (newcap < HashDataBufUsed + need) newcap *= 2;
    HashDataBuf = realloc(HashDataBuf, newcap);
    if (!HashDataBuf) {
        fprintf(stderr, "Failed to grow hash data buffer to %llu bytes\n",
                (unsigned long long)newcap);
        exit(1);
    }
    HashDataBufCap = newcap;
}

void build_compact_table(void) {
  size_t i;
  uint64_t tsize;
  uint32_t overflow_count = 0;
  struct timespec bstart, bend;

  current_utc_time(&bstart);

  tsize = 1;
  while (tsize <= HashDataCount) tsize <<= 1;
  tsize <<= 1;

  tsfprintf(stderr, "Compact hash table: growing from 0 to %llu slots\n",
         (unsigned long long)tsize);

  CompactSize = tsize;
  CompactMask = tsize - 1;
  CompactUsed = 0;
  CompactFP = calloc(tsize, sizeof(uint32_t));
  CompactIdx = malloc(tsize * sizeof(uint32_t));
  if (!CompactFP || !CompactIdx) {
    fprintf(stderr, "Failed to allocate compact table (%llu slots)\n", (unsigned long long)tsize);
    exit(1);
  }
  OverflowHash = NULL;
  Dupcnt = 0;

  for (i = 0; i < HashDataCount; i++) {
    unsigned char *hash = HashDataBuf + HashDataOff[i];
    int hlen = HashDataLen[i];
    uint64_t key = *(uint64_t *)hash;
    uint32_t fp = (uint32_t)(key >> 32);
    uint64_t pos = compact_mix(key) & CompactMask;
    int p;

    /* Prefetch compact table slot for a future entry */
    if (i + 8 < HashDataCount) {
      uint64_t nkey = *(uint64_t *)(HashDataBuf + HashDataOff[i + 8]);
      uint64_t npos = compact_mix(nkey) & CompactMask;
      __builtin_prefetch(&CompactFP[npos], 1, 1);
      __builtin_prefetch(&CompactIdx[npos], 1, 1);
    }

    if (!fp) fp = 1;

    for (p = 0; p < COMPACT_MAX_PROBE; p++) {
      if (!CompactFP[pos]) {
        CompactFP[pos] = fp;
        CompactIdx[pos] = (uint32_t)i;
        CompactUsed++;
        goto inserted;
      }
      if (CompactFP[pos] == fp) {
        size_t eidx = CompactIdx[pos];
        int elen = HashDataLen[eidx];
        if (elen == hlen && memcmp(hash, HashDataBuf + HashDataOff[eidx], hlen) == 0) {
          Dupcnt++;
          goto inserted;
        }
      }
      pos = (pos + 1) & CompactMask;
    }

    /* Overflow to JudyL + Hashchain */
    {
      Word_t *PV;
      struct Hashchain *chain, *next, *prev;

      JLI(PV, OverflowHash, (Word_t)key);
      next = (struct Hashchain *)(*PV);
      prev = NULL;

      while (next) {
        if (hlen < next->len) break;
        /* Only compare bytes when the lengths match. The chain is ordered
         * by length first, so hlen > next->len simply means keep walking --
         * comparing here read hlen-8 bytes out of an entry that holds only
         * next->len-8 (raw32 vs raw96 in one file overruns by 32 bytes). */
        if (hlen == next->len && memcmp(hash + 8, next->hash, hlen - 8) <= 0) break;
        prev = next;
        next = next->next;
      }
      if (next && hlen == next->len && memcmp(hash + 8, next->hash, hlen - 8) == 0) {
        Dupcnt++;
        goto inserted;
      }

      /* malloc(), NOT mymalloc(): compact_resize() releases overflow chain
       * entries with free(). Entries carved out of the mymalloc arena are not
       * individually free()-able, and handing one to free() aborts the process
       * with "free(): invalid pointer" the next time the compact table doubles.
       * The other two chain-allocation sites (compact_resize, compact_insert)
       * already use malloc; this was the odd one out. Fixed 2026-08-07. */
      chain = malloc(sizeof(struct Hashchain) + hlen - 9);
      if (!chain) {
        fprintf(stderr, "Failed to allocate overflow chain entry (%d bytes)\n",
                (int)(sizeof(struct Hashchain) + hlen - 9));
        exit(1);
      }
      chain->len = hlen;
      chain->flags = 0;
      memmove(chain->hash, hash + 8, hlen - 8);
      chain->next = next;
      if (!prev) *PV = (Word_t)chain;
      else prev->next = chain;
      overflow_count++;
    }
    inserted: ;
  }

  Insize = HashDataCount - Dupcnt;
  Maxdepth = overflow_count ? 2 : 1;

  current_utc_time(&bend);
  {
    double bt = (double)bend.tv_sec + bend.tv_nsec * 1e-9
              - (double)bstart.tv_sec - bstart.tv_nsec * 1e-9;
    tsfprintf(stderr, "Compact table: %llu entries, %llu slots (%.1f%% load), %u overflow, %.2fs build\n",
            (unsigned long long)Insize, (unsigned long long)tsize,
            100.0 * CompactUsed / tsize,
            overflow_count, bt);
  }
  /* Decode bcrypt hash bytes into compact table for GPU probing.
   * Each bcrypt string is $2x$CC$<22-char salt><31-char hash>.
   * Decode the 31-char base64 hash into 23 raw bytes (big-endian from BF_encode),
   * byte-swap to little-endian uint32 words, and insert into compact table. */
  { static const unsigned char bf_atoi64[0x60] = {
      64,64,64,64,64,64,64,64,64,64,64,64,64,64, 0, 1,
      54,55,56,57,58,59,60,61,62,63,64,64,64,64,64,64,
      64, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,
      17,18,19,20,21,22,23,24,25,26,27,64,64,64,64,64,
      64,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
      43,44,45,46,47,48,49,50,51,52,53,64,64,64,64,64
    };
    char bcline[128];
    Word_t *BPV;
    int bc_added = 0;
    bcline[0] = 0;
    JSLF(BPV, JUDYJ(JOB_BCRYPT), (unsigned char *)bcline);
    /* Ensure HashDataBuf is allocated (may be NULL if no hex hashes were loaded) */
    if (BPV && !HashDataBuf) {
      HashDataBufCap = 65536;
      HashDataBuf = malloc(HashDataBufCap);
      HashDataOff = malloc(65536 * sizeof(size_t));
      HashDataLen = malloc(65536 * sizeof(unsigned short));
      HashDataFlags = malloc(65536 * sizeof(unsigned short));
      if (!HashDataBuf || !HashDataOff || !HashDataLen || !HashDataFlags) {
        fprintf(stderr, "Failed to allocate hash data arrays\n");
        exit(1);
      }
    }
    while (BPV) {
      int pfx = (bcline[2] == 'k') ? 28 : 29;
      const unsigned char *src = (unsigned char *)bcline + pfx;
      unsigned char raw[24];
      unsigned char *dp = raw;
      unsigned char *end = raw + 23;
      /* BF_decode: 31 base64 chars -> 23 bytes */
      while (dp < end) {
        unsigned int c1 = bf_atoi64[(*src++) - 0x20];
        unsigned int c2 = bf_atoi64[(*src++) - 0x20];
        *dp++ = (c1 << 2) | ((c2 & 0x30) >> 4);
        if (dp >= end) break;
        unsigned int c3 = bf_atoi64[(*src++) - 0x20];
        *dp++ = ((c2 & 0x0F) << 4) | ((c3 & 0x3C) >> 2);
        if (dp >= end) break;
        unsigned int c4 = bf_atoi64[(*src++) - 0x20];
        *dp++ = ((c3 & 0x03) << 6) | c4;
      }
      /* Raw decoded bytes are BE from BF_encode. On a LE host, reading them
       * as native uint32 gives the same values the GPU kernel produces after
       * its BE-to-LE byte-swap. No additional swap needed here. */
      /* Append to HashDataBuf and insert into compact table.
       * Grow BOTH the entry arrays and the byte buffer first. This used to
       * test only HashDataBufCap, which meant (a) HashDataOff[HashDataCount]
       * could be written one past the end of the entry arrays -- heap
       * corruption -- and (b) once the byte buffer filled, the remaining
       * bcrypt hashes were silently dropped instead of searched. */
      hashdata_reserve(1);
      hashdatabuf_reserve(24);
      {
        HashDataOff[HashDataCount] = HashDataBufUsed;
        HashDataLen[HashDataCount] = 23;
        HashDataFlags[HashDataCount] = 0;
        memcpy(HashDataBuf + HashDataBufUsed, raw, 23);
        raw[23] = 0;  /* pad to 24 bytes for 4-byte alignment */
        HashDataBufUsed += 24;
        compact_insert();
        HashDataCount++;
        bc_added++;
      }
      JSLN(BPV, JUDYJ(JOB_BCRYPT), (unsigned char *)bcline);
    }
    if (bc_added)
      fprintf(stderr, "Compact table: added %d bcrypt hashes for GPU probing\n", bc_added);
  }
  /* Decode SHA512CRYPT ($6$) hashes from JudyJ and insert into compact table
   * for GPU probing.  Only the first 16 bytes of the 64-byte raw hash are
   * stored (enough for compact_fp/idx key + probe_compact's 4-word compare). */
  { static const unsigned char sha512_perm[21][3] = {
        {0,21,42},{22,43,1},{44,2,23},{3,24,45},{25,46,4},
        {47,5,26},{6,27,48},{28,49,7},{50,8,29},{9,30,51},
        {31,52,10},{53,11,32},{12,33,54},{34,55,13},{56,14,35},
        {15,36,57},{37,58,16},{59,17,38},{18,39,60},{40,61,19},
        {62,20,41}
    };
    char sc_line[256];
    Word_t *SC_PV;
    int sc_added = 0;
    init_phpatoi64();
    sc_line[0] = 0;
    JSLF(SC_PV, JUDYJ(JOB_SHA512CRYPT), (unsigned char *)sc_line);
    /* Ensure HashDataBuf is allocated (may be NULL if no hex hashes were loaded) */
    if (SC_PV) {
      /* Allocate/grow through the shared helpers rather than assuming a size.
       *
       * This previously had an "else if (HashDataCap == 0) HashDataCap = 65536"
       * arm commented "arrays exist (e.g. from bcrypt block) but HashDataCap
       * not tracked". That claimed 65536 entries of capacity the arrays might
       * not actually have, so every later grow check compared against a
       * fiction and the appends below could run far past the end. The bcrypt
       * block now maintains HashDataCap correctly, so nothing has to guess. */
      hashdata_reserve(1);
      hashdatabuf_reserve(64);
    }
    while (SC_PV) {
      /* Find hash portion: $6$[rounds=N$]salt$HASH (86 chars) */
      char *hp = sc_line + 3; /* skip "$6$" */
      if (strncmp(hp, "rounds=", 7) == 0) {
        hp = strchr(hp, '$');
        if (hp) hp++;
      }
      if (hp) hp = strchr(hp, '$');
      if (hp) hp++; /* hp now points to 86-char hash */
      if (hp && strlen(hp) >= 86) {
        unsigned char raw[64];
        int decode_ok = 1;
        /* Decode 21 groups of 4 base64 chars -> 3 bytes each */
        for (int g = 0; g < 21 && decode_ok; g++) {
          int c0 = phpatoi64[(unsigned char)hp[g*4]];
          int c1 = phpatoi64[(unsigned char)hp[g*4+1]];
          int c2 = phpatoi64[(unsigned char)hp[g*4+2]];
          int c3 = phpatoi64[(unsigned char)hp[g*4+3]];
          if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) { decode_ok = 0; break; }
          unsigned int v = c0 | (c1 << 6) | (c2 << 12) | (c3 << 18);
          raw[sha512_perm[g][0]] = (v >> 16) & 0xff;
          raw[sha512_perm[g][1]] = (v >> 8) & 0xff;
          raw[sha512_perm[g][2]] = v & 0xff;
        }
        if (decode_ok) {
          /* Final byte[63]: 2 base64 chars */
          int c0 = phpatoi64[(unsigned char)hp[84]];
          int c1 = phpatoi64[(unsigned char)hp[85]];
          if (c0 >= 0 && c1 >= 0) {
            raw[63] = (c0 | (c1 << 6)) & 0xff;
          } else {
            decode_ok = 0;
          }
        }
        if (decode_ok) {
          /* Grow entry arrays + byte buffer via the shared helpers. */
          hashdata_reserve(1);
          hashdatabuf_reserve(16);
          HashDataOff[HashDataCount] = HashDataBufUsed;
          HashDataLen[HashDataCount] = 16;
          HashDataFlags[HashDataCount] = 0;
          memcpy(HashDataBuf + HashDataBufUsed, raw, 16);
          HashDataBufUsed += 16;
          compact_insert();
          HashDataCount++;
          sc_added++;
        }
      }
      JSLN(SC_PV, JUDYJ(JOB_SHA512CRYPT), (unsigned char *)sc_line);
    }
    if (sc_added)
      fprintf(stderr, "Compact table: added %d SHA512CRYPT hashes for GPU probing\n", sc_added);
    /* SHA512CRYPTMD5 shares the same $6$ hashes in JudyJ[JOB_SHA512CRYPT],
     * so the compact table entries above already cover both types. */
  }
  /* Decode SHA256CRYPT ($5$) hashes from JudyJ and insert into compact table
   * for GPU probing.  Mirrors the SHA512CRYPT block above.  The 43-char b64
   * trailer decodes to 32 raw bytes via 10 three-byte permutation tuples plus
   * a final 2-byte tail (encoded as 3 b64 chars).  Only the first 16 bytes of
   * the 32-byte raw hash are stored (enough for compact_fp/idx key +
   * probe_compact's 4-word compare).  Permutation tuples are the inverse of
   * the SHA-256 b64 encoder at mdxfind.c:12784 (cryptlen==32 path). */
  { static const unsigned char sha256_perm[10][3] = {
        { 0,10,20},{21, 1,11},{12,22, 2},{ 3,13,23},{24, 4,14},
        {15,25, 5},{ 6,16,26},{27, 7,17},{18,28, 8},{ 9,19,29}
    };
    char s5_line[256];
    Word_t *S5_PV;
    int s5_added = 0;
    init_phpatoi64();
    s5_line[0] = 0;
    JSLF(S5_PV, JUDYJ(JOB_SHA256CRYPT), (unsigned char *)s5_line);
    /* Ensure HashDataBuf is allocated (may be NULL if no hex hashes were loaded) */
    if (S5_PV && !HashDataBuf) {
      HashDataBufCap = 65536;
      HashDataBuf = malloc(HashDataBufCap);
      HashDataOff = malloc(65536 * sizeof(size_t));
      HashDataLen = malloc(65536 * sizeof(unsigned short));
      HashDataFlags = malloc(65536 * sizeof(unsigned short));
      if (HashDataFlags) memset(HashDataFlags, 0, 65536 * sizeof(unsigned short));
      HashDataCap = 65536;
    } else if (S5_PV && HashDataCap == 0) {
      /* Arrays exist (e.g. from bcrypt/SHA512CRYPT block) but HashDataCap not tracked */
      HashDataCap = 65536;
    }
    while (S5_PV) {
      /* Find hash portion: $5$[rounds=N$]salt$HASH (43 chars) */
      char *hp = s5_line + 3; /* skip "$5$" */
      if (strncmp(hp, "rounds=", 7) == 0) {
        hp = strchr(hp, '$');
        if (hp) hp++;
      }
      if (hp) hp = strchr(hp, '$');
      if (hp) hp++; /* hp now points to 43-char hash */
      if (hp && strlen(hp) >= 43) {
        unsigned char raw[32];
        int decode_ok = 1;
        /* Decode 10 groups of 4 base64 chars -> 3 bytes each (covers raw[0..29]) */
        for (int g = 0; g < 10 && decode_ok; g++) {
          int c0 = phpatoi64[(unsigned char)hp[g*4]];
          int c1 = phpatoi64[(unsigned char)hp[g*4+1]];
          int c2 = phpatoi64[(unsigned char)hp[g*4+2]];
          int c3 = phpatoi64[(unsigned char)hp[g*4+3]];
          if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) { decode_ok = 0; break; }
          unsigned int v = c0 | (c1 << 6) | (c2 << 12) | (c3 << 18);
          raw[sha256_perm[g][0]] = (v >> 16) & 0xff;
          raw[sha256_perm[g][1]] = (v >> 8) & 0xff;
          raw[sha256_perm[g][2]] = v & 0xff;
        }
        if (decode_ok) {
          /* Final 2 bytes raw[30],raw[31]: 3 base64 chars (encoder emits 3,
           * top 4 bits of the 18-bit value are zero; we ignore them). */
          int c0 = phpatoi64[(unsigned char)hp[40]];
          int c1 = phpatoi64[(unsigned char)hp[41]];
          int c2 = phpatoi64[(unsigned char)hp[42]];
          if (c0 >= 0 && c1 >= 0 && c2 >= 0) {
            unsigned int v = c0 | (c1 << 6) | (c2 << 12);
            raw[30] = v & 0xff;
            raw[31] = (v >> 8) & 0xff;
          } else {
            decode_ok = 0;
          }
        }
        if (decode_ok) {
          /* Grow entry arrays + byte buffer via the shared helpers. */
          hashdata_reserve(1);
          hashdatabuf_reserve(16);
          HashDataOff[HashDataCount] = HashDataBufUsed;
          HashDataLen[HashDataCount] = 16;
          HashDataFlags[HashDataCount] = 0;
          memcpy(HashDataBuf + HashDataBufUsed, raw, 16);
          HashDataBufUsed += 16;
          compact_insert();
          HashDataCount++;
          s5_added++;
        }
      }
      JSLN(S5_PV, JUDYJ(JOB_SHA256CRYPT), (unsigned char *)s5_line);
    }
    if (s5_added)
      fprintf(stderr, "Compact table: added %d SHA256CRYPT hashes for GPU probing\n", s5_added);
  }
#ifdef GPU_ENABLED
  /* Only initialize GPU if at least one GPU-capable hash type is selected.
   *
   * gpu_ops[] is declared at file scope as `static const int` (just above
   * gpu_op_advertise_for_h_listing near `void *Stack;`). Both this init
   * preamble loop AND gpu_op_advertise_for_h_listing iterate the same
   * sentinel-terminated table, eliminating the prior drift class where a
   * mirror table in the helper silently fell out of sync.
   *
   * Cleanup 2026-05-27 (rev 1.511): lifted from function-scope mutable
   * to file-scope static const; mirror table in helper deleted; inline
   * 11-JOB family override table replaced with a J1F/J1N walk over
   * Dohash calling gpu_codegen_kernelb_family_md5pass_eligible. */
  { int need_gpu = 0;
    for (int gi = 0; gpu_ops[gi] >= 0; gi++) {
      Word_t grc;
      J1T(grc, Dohash, gpu_ops[gi]);
      if (grc) { need_gpu = 1; break; }
    }
    /* Phase 4 sub-phase 4a.1 (2026-05-21): JOB_MD5MD5SALT need_gpu
     * override is now unconditional on Dohash membership. JOB_MD5MD5SALT
     * remains absent from gpu_ops[] (the existing list of "by default
     * activate GPU init" ops); per feedback_gpu_init_gate_for_non_gpu_-
     * ops_dispatch.md the codegen production path still requires this
     * explicit override to trigger gpu_opencl_init, otherwise the proto
     * dispatch silently no-ops. Prior to 4a.1 this override was gated
     * on MDXFIND_KERNEL_B_PROTO || mdx_kernel_a_proto_enabled; that
     * gate has been removed.
     *
     * Sub-phase 5a.5 (2026-05-22): the seven 5a-eligible MAKE_MD5PASS
     * family JOB enums (e122/e159/e161/e163/e165/e167/e169) join e347
     * in the override. Like e347 they are absent from gpu_ops[] (kept
     * out so their being "advertised" as production GPU support
     * remains scoped to deliberate codegen routing); per
     * feedback_gpu_init_gate_for_non_gpu_ops_dispatch.md without this
     * override the GPU subsystem never initializes and the chokepoint
     * admit silently no-ops. We linear-scan the 7-element family table
     * via the eligibility helper; first hit forces need_gpu=1. */
    if (!need_gpu) {
        Word_t _grc_md5md5salt;
        J1T(_grc_md5md5salt, Dohash, JOB_MD5MD5SALT);
        if (_grc_md5md5salt) need_gpu = 1;
    }
    if (!need_gpu) {
        /* Cleanup 2026-05-27 (rev 1.511): inline 11-JOB literal family
         * table replaced with a J1F/J1N walk over Dohash applying the
         * single-source-of-truth predicate
         * gpu_codegen_kernelb_family_md5pass_eligible. Future Phase 5b
         * Tier 3/4 widens (haval/gost/sne128/sne256) auto-propagate
         * here once the predicate widens in gpu_codegen_eligible.c --
         * no edit needed at this site. */
        Word_t _fi_op = 0;
        Word_t _grc_fam;
        J1F(_grc_fam, Dohash, _fi_op);
        while (_grc_fam) {
            if (gpu_codegen_kernelb_family_md5pass_eligible((int)_fi_op)) {
                need_gpu = 1;
                break;
            }
            J1N(_grc_fam, Dohash, _fi_op);
        }
    }
    /* hx codegen sub-phase 2a.1 (2026-05-21): force need_gpu=1 when the
     * harness mode is explicitly requested via MDXFIND_HX_CODEGEN_JOB or
     * MDXFIND_HX_CODEGEN_VALIDATE so gpu_opencl_init fires and the
     * harness can JIT-compile.
     *
     * Every MDXFIND_HX_CODEGEN* environment variable was removed on
     * 2026-09-16, so this gate is never taken and the harness never fires.
     * It read _JOB or _VALIDATE rather than plain MDXFIND_HX_CODEGEN because
     * the latter's "=0" meant "force the legacy hand-written kernel for
     * e347" in production dispatch, not "fire the harness". Reaching the
     * harness again needs a native CLI flag, not an environment variable. */
    if (!need_gpu && (NULL /* was MDXFIND_HX_CODEGEN_JOB */
                      || NULL /* was MDXFIND_HX_CODEGEN_VALIDATE */)) {
        need_gpu = 1;
    }
    if (!need_gpu) NoMetal = 1;
  }
  if (Printall) {
#ifdef GPU_ENABLED
    if (!NoMetal) fprintf(stderr, "GPU disabled (-z mode)\n");
#endif
    NoMetal = 1;
  }
  if (!NoMetal) {
    int gpu_ok = -1;
#if defined(__APPLE__) && defined(METAL_GPU)
    gpu_ok = gpu_metal_init();
#elif defined(CUDA_GPU)
    gpu_ok = cuda_md5salt_init();
#elif defined(OPENCL_GPU)
    /* Initialize GPU kernel binary cache before any clBuildProgram calls.
     * Idempotent. If MDXFIND_CACHE is unset, warns once and disables —
     * subsequent gpu_kernel_cache_* calls become no-ops, gpu_opencl_init()
     * proceeds to JIT every kernel from source as before. */
    gpu_kernel_cache_init(mdxfind_rev_string());
    /* A device that enumerates and initialises is not thereby known to
     * COMPUTE. NVIDIA's OpenCL 1.2 on Fermi rejects clEnqueueFillBuffer
     * with CL_INVALID_OPERATION although the symbol resolves, and until
     * gpu_opencl.c 1.221 that failure abandoned the dispatch and returned
     * no hits at all: a full run over the wordlist, exit 0, None found.
     * A Tesla M2070 hashed 400,002 words and reported 0 hits where the
     * same binary on a GTX 1080 found both planted targets.
     *
     * So if a GPU path ever reports hashes computed but zero hits where
     * -G none finds them, suspect a silently-failing device call before
     * suspecting the kernel arithmetic. */
    gpu_ok = gpu_opencl_init();
#endif
    if (gpu_ok == 0 && CompactFP && HashDataBuf && HashDataOff && HashDataLen && CompactSize > 0) {
#if defined(__APPLE__) && defined(METAL_GPU)
      /* Phase 1 Metal: single-device only; dev_idx=0 always. The dev_idx
       * arg exists in the gpu_metal_set_compact_table signature for
       * future multi-device support + signature parity with OpenCL. */
      gpu_metal_set_compact_table(0, CompactFP, CompactIdx,
          CompactSize, CompactMask,
          HashDataBuf, HashDataBufUsed,
          HashDataOff, HashDataCount, HashDataLen);
#elif defined(CUDA_GPU)
      cuda_md5salt_set_compact_table(CompactFP, CompactIdx,
          CompactSize, CompactMask,
          HashDataBuf, HashDataBufUsed,
          HashDataOff, HashDataCount, HashDataLen);
#elif defined(OPENCL_GPU)
      { int ndev = gpu_opencl_num_devices();
        /* Parallel per-device compact-table upload (Memo C). Each device has
         * its own cl_context + cl_command_queue; CL_MEM_COPY_HOST_PTR copies
         * host data into device-resident memory and concurrent reads of the
         * same host pages from N driver threads are safe (memcpy reads of
         * read-only data). The per-thread function lives at file scope as
         * sct_thread_fn() — see just above main. */
        thread *sct_threads[64];
        int sct_args[64];
        for (int di = 0; di < ndev && di < 64; di++) {
          sct_args[di] = di;
          sct_threads[di] = launch(sct_thread_fn, &sct_args[di]);
        }
        for (int di = 0; di < ndev && di < 64; di++)
          join(sct_threads[di]);
        /* Aggregate result: how many devices survived set_compact_table?
         * If zero, this flips gpu_opencl_available() back to false so
         * the rest of mdxfind treats it like -G none. Per-device disable
         * details (the "DEVICE DISABLED" lines) were already emitted by
         * set_compact_table itself. */
        gpu_opencl_finalize_active_count();
      }
#endif
    } else if (gpu_ok == 0) {
      /* Say it AND do it. This arm printed "GPU disabled" and set nothing, so
       * gpu_opencl_available() stayed true and the dispatch ran against a
       * compact table that was never built -- the kernel faulted and NVIDIA
       * surfaced it as CL_OUT_OF_RESOURCES on a later clEnqueueReadBuffer.
       * Reached whenever no hashes load for the selected types, which an empty
       * -f file or a wholly unparseable one both produce. */
      fprintf(stderr, "GPU: compact table not ready, GPU disabled\n");
      #if defined(OPENCL_GPU)
      gpu_opencl_disable("compact table not built (no hashes loaded for the selected types)");
      #endif
    }
#if defined(OPENCL_GPU)
    /* hx codegen sub-phase 2a.1 (2026-05-21): harness mode. When the
     * env flag is set, drive the in-process P4 state-machine walker
     * with a TRIVIAL HARDCODED SPEC (2-op pass-through:
     * OP_PUSH_PASS + OP_EMIT_DIGEST). Optionally dump emitted source
     * via MDXFIND_HX_CODEGEN_DUMP=/tmp/...  Then JIT-compile the
     * result against dev[0] (FATAL on any failure per
     * feedback_external_failures_are_fatal.md) and exit(0).
     *
     * 2a.1 is harness-mode only -- codegen is NOT wired to the runtime
     * dispatch path. Full e347 (MD5MD5SALT) tp0 emission arrives with
     * sub-phase 2a.3; Metal twin with 2a.4.
     *
     * The harness env vars (_JOB / _VALIDATE) were removed 2026-09-16, so
     * this block is unreachable. They were preferred over plain
     * MDXFIND_HX_CODEGEN because its "=0" was a production opt-out. */
    if (gpu_ok == 0
        && (NULL /* was MDXFIND_HX_CODEGEN_JOB */
            || NULL /* was MDXFIND_HX_CODEGEN_VALIDATE */)) {
        char hxhost[256];
        mdx_gethostname(hxhost, sizeof(hxhost));

        /* Sub-phase 2a.2: look up e1 MD5 in the auto-generated
         * hx_specs_data[] (replaces 2a.1's hardcoded trivial spec).
         * MDXFIND_HX_CODEGEN_JOB env var overrides the default eN. */
        int target_job = 1; /* e1 MD5 default */
        const char *job_env = NULL /* was MDXFIND_HX_CODEGEN_JOB */;
        if (job_env && *job_env) target_job = atoi(job_env);

        const struct hx_spec_entry *entry = hx_specs_lookup(target_job);
        if (!entry) {
            fprintf(stderr,
                "FATAL: %s:%d hx_specs_lookup(%d) returned NULL on %s "
                "(check codegen/hx_specs_data.c)\n",
                __FILE__, __LINE__, target_job, hxhost);
            exit(1);
        }
        if (entry->is_outlier || entry->compile_failed || !entry->program) {
            fprintf(stderr,
                "FATAL: %s:%d hx codegen: e%d %s is %s on %s -- routes to "
                "override path, not codegen\n",
                __FILE__, __LINE__, entry->job_enum, entry->name,
                entry->is_outlier ? "outlier" :
                entry->compile_failed ? "compile_failed" : "no-program",
                hxhost);
            exit(1);
        }
        fprintf(stderr,
            "hx codegen: 2a.2 harness using e%d %s `%s` (ncode=%d nvars=%d)\n",
            entry->job_enum, entry->name, entry->expression,
            entry->program->ncode, entry->program->nvars);

        struct hx_specialization zone;
        memset(&zone, 0, sizeof(zone));
        zone.iter_count_if_fixed = 1;
        zone.has_rules           = 0;
        zone.has_masks           = 0;
        zone.has_bf              = 0;
        zone.salt_minlen         = 0;
        zone.salt_maxlen         = 0;
        zone.salt_count_regime   = HX_SALT_SINGLE;
        zone.iter_shape          = HX_ITER_NONE;
        zone.digest_endianness   = HX_DIGEST_LE;
        zone.emit_width          = 16;

        /* Sub-phase 2a.3 (2026-05-21): for e347 the harness pre-fills
         * specialization fields that drive the tp0 emitter:
         *   has_rules=1, salt_count_regime=BATCH_64
         *   salt_minlen/maxlen left at 0 (the emitted code reads salt
         *   lengths at runtime from payload->salt_lens[]; baking them
         *   constant is a later micro-opt).
         * 2a.5 will exercise edge cases. */
        if (target_job == 347) {
            zone.has_rules         = 1;
            zone.salt_count_regime = HX_SALT_BATCH_64;
            zone.salt_minlen       = 1;
            zone.salt_maxlen       = 64;
        }

        char *hxsrc = NULL;
        size_t hxcap = 0;
        /* Sub-phase 5a.2 (2026-05-22): walker signature takes entry so
         * family emitters can reach the per-program callnames sidecar. */
        int hxrc = hx_emit_kernel(entry->program, &zone,
                                  HX_BACKEND_OPENCL, entry,
                                  &hxsrc, &hxcap);
        if (hxrc != 0 || !hxsrc) {
            fprintf(stderr,
                "FATAL: %s:%d hx codegen walker failed (rc=%d) on %s for e%d %s\n",
                __FILE__, __LINE__, hxrc, hxhost,
                entry->job_enum, entry->name);
            exit(1);
        }
        if (hx_dump_source(hxsrc, "MDXFIND_HX_CODEGEN_DUMP") < 0) {
            fprintf(stderr,
                "FATAL: %s:%d hx codegen source dump failed on %s\n",
                __FILE__, __LINE__, hxhost);
            exit(1);
        }
        /* Sub-phase 2a.5 (2026-05-21): byte-exact validation harness.
         *
         * When MDXFIND_HX_CODEGEN_VALIDATE=1 AND target_job==347, drive
         * the full dispatch-and-diff path: parse fixture, compute CPU
         * oracle digests, plant them into a synthetic compact table,
         * JIT codegen kernel via _keep wrapper, dispatch, read hits,
         * sort + diff against expected (pass,salt) pair set.
         *
         * Exits 0 on PASS (zero diff), exits 1 on FAIL.
         *
         * Otherwise (validate flag unset OR job != 347): preserve the
         * 2a.3 JIT-only harness behavior.
         *
         * CPU oracle: mdxfind's mymd5 + prmd5 (declared in mdxfind.c
         * head) implement the canonical chain
         *     MD5( hex32(MD5(hex32(MD5(pass)))) || salt )
         * which matches case JOB_MD5MD5SALT: at line 23174 above. */
        const char *validate_env = NULL /* was MDXFIND_HX_CODEGEN_VALIDATE */;
        const char *fixture_env  = NULL /* was MDXFIND_HX_CODEGEN_FIXTURE */;
        int validate_on = (validate_env && *validate_env &&
                           *validate_env != '0');

        if (target_job == 347 && validate_on) {
            /* Sub-phase 2a.6 (2026-05-22) refactor: parser + oracle +
             * diff body extracted into hx_e347_validate_run_shared so the
             * Metal branch below reuses 100% of the host-side logic. The
             * helper exits with the validation result code (0 PASS, 1
             * FAIL) and never returns. */
            hx_e347_validate_run_shared(0, hxsrc, fixture_env, hxhost);
            /* NOTREACHED */
        }
        /* Sub-phase 5b.3a.0.3 (2026-05-27) D17.4.b refactor: 11-entry
         * OR-chain replaced with single-source-of-truth predicate call
         * via gpu_codegen_kernelb_family_md5pass_eligible. Future Tier
         * 3 sub-phases (5b.3b 4-pass HAVAL, 5b.3c 5-pass HAVAL) + Tier
         * 4 (gost/sne128/sne256) flip prim_table.supported_5a and this
         * OR-chain auto-propagates. */
        if (gpu_codegen_kernelb_family_md5pass_eligible(target_job)
            && validate_on) {
            hx_family_md5pass_validate_run_shared(
                0, hxsrc, fixture_env, hxhost, target_job);
            /* NOTREACHED */
        }
#if 0  /* === 2a.5 inline harness body removed in 2a.6; logic moved to ===
        * === hx_e347_validate_run_shared above (refactor). Kept inside ===
        * === a #if 0 block to preserve the original LOC anchors during ===
        * === git/RCS history review; will be deleted in 2a.7 cleanup.  === */
        if (target_job == 347 && validate_on) {
            FILE *ff = fopen(fixture_env, "r");
            if (!ff) {
                fprintf(stderr,
                    "FATAL: %s:%d hx codegen 2a.5 validate: cannot open "
                    "fixture '%s' on %s (errno=%d)\n",
                    __FILE__, __LINE__, fixture_env, hxhost, errno);
                exit(1);
            }
            /* --- Parse fixture: PASS:<text> / SALT:<text> / # comments --- */
            char **passes = NULL; size_t n_pass = 0, cap_pass = 0;
            char **salts  = NULL; size_t n_salt = 0, cap_salt = 0;
            char line[4096];
            while (fgets(line, sizeof(line), ff)) {
                size_t L = strlen(line);
                while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) {
                    line[--L] = '\0';
                }
                if (L == 0 || line[0] == '#') continue;
                if (L > 5 && memcmp(line, "PASS:", 5) == 0) {
                    if (n_pass == cap_pass) {
                        cap_pass = cap_pass ? cap_pass * 2 : 16;
                        passes = realloc(passes, cap_pass * sizeof(char *));
                    }
                    passes[n_pass++] = strdup(line + 5);
                } else if (L > 5 && memcmp(line, "SALT:", 5) == 0) {
                    if (n_salt == cap_salt) {
                        cap_salt = cap_salt ? cap_salt * 2 : 16;
                        salts = realloc(salts, cap_salt * sizeof(char *));
                    }
                    salts[n_salt++] = strdup(line + 5);
                }
            }
            fclose(ff);
            if (n_pass == 0 || n_salt == 0) {
                fprintf(stderr,
                    "FATAL: %s:%d 2a.5 validate: fixture %s has no PASS "
                    "(n=%zu) or SALT (n=%zu) lines on %s\n",
                    __FILE__, __LINE__, fixture_env,
                    n_pass, n_salt, hxhost);
                exit(1);
            }
            size_t n_pairs = n_pass * n_salt;
            fprintf(stderr,
                "hx codegen 2a.5: fixture %s parsed n_pass=%zu n_salt=%zu "
                "n_pairs=%zu on %s\n",
                fixture_env, n_pass, n_salt, n_pairs, hxhost);

            /* --- Compute CPU oracle digests for every (pass, salt) pair.
             * Chain: MD5( hex32(MD5(hex32(MD5(pass)))) || salt )
             * Order: outer-major iter over passes; inner iter over salts.
             * Per-pair plant index == pi * n_salt + si. */
            unsigned char *oracle =
                (unsigned char *)malloc(n_pairs * 16);
            if (!oracle) {
                fprintf(stderr,
                    "FATAL: %s:%d 2a.5 validate: malloc(oracle %zu B) "
                    "failed on %s\n",
                    __FILE__, __LINE__, n_pairs * 16, hxhost);
                exit(1);
            }
            for (size_t pi = 0; pi < n_pass; pi++) {
                unsigned char inner1[16], inner2[16];
                char hex32a[33], hex32b[33];
                /* MD5(pass) -> inner1 */
                mymd5((void *)passes[pi], (int)strlen(passes[pi]),
                      inner1);
                /* hex32(inner1) -> hex32a */
                prmd5_32(inner1, hex32a);
                hex32a[32] = '\0';
                /* MD5(hex32a) -> inner2 */
                mymd5((void *)hex32a, 32, inner2);
                /* hex32(inner2) -> hex32b */
                prmd5_32(inner2, hex32b);
                hex32b[32] = '\0';

                for (size_t si = 0; si < n_salt; si++) {
                    size_t slen = strlen(salts[si]);
                    /* concat hex32b (32) + salt (slen) into temp */
                    char *concat = (char *)malloc(32 + slen + 1);
                    memcpy(concat, hex32b, 32);
                    memcpy(concat + 32, salts[si], slen);
                    concat[32 + slen] = '\0';
                    unsigned char outd[16];
                    mymd5((void *)concat, (int)(32 + slen), outd);
                    memcpy(oracle + (pi * n_salt + si) * 16, outd, 16);
                    free(concat);
                }
            }
            fprintf(stderr,
                "hx codegen 2a.5: oracle digests computed for %zu pairs\n",
                n_pairs);

            /* --- Build synthetic compact-hash table sized for n_pairs. */
            uint64_t cap_pow2 = 1;
            while (cap_pow2 < n_pairs * 2) cap_pow2 <<= 1;
            if (cap_pow2 < 64) cap_pow2 = 64;
            uint64_t cmask = cap_pow2 - 1;
            uint32_t *cfp  = (uint32_t *)calloc(cap_pow2, sizeof(uint32_t));
            uint32_t *cidx = (uint32_t *)calloc(cap_pow2, sizeof(uint32_t));
            unsigned char *hbuf =
                (unsigned char *)malloc(n_pairs * 16);
            size_t *hoff = (size_t *)malloc(n_pairs * sizeof(size_t));
            unsigned short *hlen =
                (unsigned short *)malloc(n_pairs * sizeof(unsigned short));
            if (!cfp || !cidx || !hbuf || !hoff || !hlen) {
                fprintf(stderr,
                    "FATAL: %s:%d 2a.5 validate: alloc compact buffers "
                    "failed on %s\n",
                    __FILE__, __LINE__, hxhost);
                exit(1);
            }
            for (size_t i = 0; i < n_pairs; i++) {
                memcpy(hbuf + i * 16, oracle + i * 16, 16);
                hoff[i] = i * 16;
                hlen[i] = 16;
                uint32_t hx = ((uint32_t *)(oracle + i * 16))[0];
                uint32_t hy = ((uint32_t *)(oracle + i * 16))[1];
                uint64_t key = ((uint64_t)hy << 32) | hx;
                uint32_t fp = (uint32_t)(key >> 32);
                if (fp == 0) fp = 1;
                /* compact_mix == k ^ (k >> 32) */
                uint64_t pos = (key ^ (key >> 32)) & cmask;
                for (int probe = 0; probe < 256; probe++) {
                    uint64_t p = (pos + probe) & cmask;
                    if (cfp[p] == 0) {
                        cfp[p] = fp;
                        cidx[p] = (uint32_t)i;
                        break;
                    }
                }
            }
            fprintf(stderr,
                "hx codegen 2a.5: compact table built cap=%llu mask=0x%llx\n",
                (unsigned long long)cap_pow2,
                (unsigned long long)cmask);

            /* --- Upload compact table + salts to GPU dev[0]. */
            if (gpu_opencl_set_compact_table(0, cfp, cidx, cap_pow2, cmask,
                    hbuf, n_pairs * 16, hoff, n_pairs, hlen) != 0) {
                fprintf(stderr,
                    "FATAL: %s:%d 2a.5 validate: set_compact_table "
                    "failed on %s dev[0]\n",
                    __FILE__, __LINE__, hxhost);
                exit(1);
            }
            /* Pack salts: contiguous buffer + offsets + lens (uint16). */
            {
                size_t sbytes = 0;
                for (size_t si = 0; si < n_salt; si++)
                    sbytes += strlen(salts[si]);
                char *sbuf = (char *)malloc(sbytes + 16);
                uint32_t *soff =
                    (uint32_t *)malloc(n_salt * sizeof(uint32_t));
                uint16_t *slens =
                    (uint16_t *)malloc(n_salt * sizeof(uint16_t));
                if (!sbuf || !soff || !slens) {
                    fprintf(stderr,
                        "FATAL: %s:%d 2a.5 validate: alloc salt buffers "
                        "failed on %s\n",
                        __FILE__, __LINE__, hxhost);
                    exit(1);
                }
                size_t pos = 0;
                for (size_t si = 0; si < n_salt; si++) {
                    size_t L = strlen(salts[si]);
                    memcpy(sbuf + pos, salts[si], L);
                    if ((uint64_t)pos + (uint64_t)L > (uint64_t)GPU_SALT_PACK_MAX) {
                        fprintf(stderr,
                            "FATAL: %s:%d salt pack: offset %llu + %zu exceeds "
                            "GPU_SALT_PACK_MAX (%u). soff[] is uint32 and truncating "
                            "it here would index the wrong salt data silently.\n",
                            __FILE__, __LINE__, (unsigned long long)pos, L,
                            (unsigned)GPU_SALT_PACK_MAX);
                        exit(1);
                    }
                    soff[si]  = (uint32_t)pos;
                    slens[si] = (uint16_t)L;
                    pos += L;
                }
                if (gpu_opencl_set_salts(0, sbuf, soff, slens,
                                          (int)n_salt) != 0) {
                    fprintf(stderr,
                        "FATAL: %s:%d 2a.5 validate: set_salts failed "
                        "on %s dev[0]\n",
                        __FILE__, __LINE__, hxhost);
                    exit(1);
                }
                free(sbuf); free(soff); free(slens);
            }

            /* --- Pack candidates: [len_u8][bytes] per word.
             * word_offset[pi] = byte index of word pi in the packed buf. */
            size_t pack_bytes = 0;
            for (size_t pi = 0; pi < n_pass; pi++)
                pack_bytes += 1 + strlen(passes[pi]);
            unsigned char *packed = (unsigned char *)malloc(pack_bytes + 16);
            uint32_t *woff = (uint32_t *)malloc(n_pass * sizeof(uint32_t));
            if (!packed || !woff) {
                fprintf(stderr,
                    "FATAL: %s:%d 2a.5 validate: alloc packed words "
                    "failed on %s\n",
                    __FILE__, __LINE__, hxhost);
                exit(1);
            }
            {
                size_t pos = 0;
                for (size_t pi = 0; pi < n_pass; pi++) {
                    size_t L = strlen(passes[pi]);
                    if (L > 255) {
                        fprintf(stderr,
                            "FATAL: %s:%d 2a.5 validate: pass %zu length "
                            "%zu > 255 on %s\n",
                            __FILE__, __LINE__, pi, L, hxhost);
                        exit(1);
                    }
                    woff[pi] = (uint32_t)pos;
                    packed[pos++] = (unsigned char)L;
                    memcpy(packed + pos, passes[pi], L);
                    pos += L;
                }
                pack_bytes = pos;
            }

            /* --- JIT codegen kernel via _keep wrapper. --- */
            struct _cl_program *vprog = NULL;
            struct _cl_kernel  *vkern = NULL;
            gpu_opencl_jit_compile_source_with_common_keep(
                0, hxsrc, "-cl-std=CL1.2",
                "kernelb_hx_e347_phase0", &vprog, &vkern);

            /* --- Allocate hit buffer (cap = n_pairs slots). --- */
            int max_hits = (int)n_pairs;
            if (max_hits < 16) max_hits = 16;
            uint32_t *vhits =
                (uint32_t *)calloc((size_t)max_hits * 19,
                                    sizeof(uint32_t));
            int vn_hits = 0;
            if (!vhits) {
                fprintf(stderr,
                    "FATAL: %s:%d 2a.5 validate: alloc vhits failed on %s\n",
                    __FILE__, __LINE__, hxhost);
                exit(1);
            }

            /* --- Dispatch. --- */
            gpu_opencl_e347_validate_dispatch(
                0, vkern,
                packed, (uint32_t)pack_bytes, woff, (uint32_t)n_pass,
                (uint32_t)n_salt,
                vhits, &vn_hits, max_hits);

            fprintf(stderr,
                "hx codegen 2a.5: dispatch returned vn_hits=%d (expected %zu)\n",
                vn_hits, n_pairs);

            /* --- Diff: every (pi, si) pair should appear exactly once. */
            unsigned char *seen =
                (unsigned char *)calloc(n_pairs, 1);
            int extras = 0, mismatches = 0;
            size_t first_mismatch_print = 0;
            for (int h = 0; h < vn_hits; h++) {
                uint32_t *slot = vhits + (size_t)h * 19;
                uint32_t widx = slot[0];
                uint32_t sidx = slot[1];
                /* iter at slot[2]; digest at slot[3..6] */
                if (widx >= n_pass || sidx >= n_salt) {
                    extras++;
                    if (first_mismatch_print < 10) {
                        fprintf(stderr,
                            "  HIT-EXTRA #%d widx=%u sidx=%u (out of range)\n",
                            h, widx, sidx);
                        first_mismatch_print++;
                    }
                    continue;
                }
                size_t pid = (size_t)widx * n_salt + (size_t)sidx;
                /* Validate digest matches oracle exactly. */
                uint32_t exp0 = ((uint32_t *)(oracle + pid * 16))[0];
                uint32_t exp1 = ((uint32_t *)(oracle + pid * 16))[1];
                uint32_t exp2 = ((uint32_t *)(oracle + pid * 16))[2];
                uint32_t exp3 = ((uint32_t *)(oracle + pid * 16))[3];
                if (slot[3] != exp0 || slot[4] != exp1 ||
                    slot[5] != exp2 || slot[6] != exp3)
                {
                    mismatches++;
                    if (first_mismatch_print < 10) {
                        fprintf(stderr,
                            "  HIT-DIFF #%d (pi=%u si=%u) "
                            "exp=%08x%08x%08x%08x got=%08x%08x%08x%08x\n",
                            h, widx, sidx,
                            exp0, exp1, exp2, exp3,
                            slot[3], slot[4], slot[5], slot[6]);
                        first_mismatch_print++;
                    }
                }
                if (!seen[pid]) seen[pid] = 1;
            }
            size_t matched = 0, missing = 0;
            for (size_t i = 0; i < n_pairs; i++)
                if (seen[i]) matched++; else missing++;
            int miss_print = 0;
            if (missing > 0) {
                fprintf(stderr,
                    "  Missing pairs (showing first 10):\n");
                for (size_t i = 0; i < n_pairs && miss_print < 10; i++) {
                    if (!seen[i]) {
                        size_t pi = i / n_salt;
                        size_t si = i % n_salt;
                        fprintf(stderr,
                            "    pid=%zu pi=%zu si=%zu pass=\"%s\" salt=\"%s\"\n",
                            i, pi, si, passes[pi], salts[si]);
                        miss_print++;
                    }
                }
            }

            int pass = (mismatches == 0 && extras == 0 &&
                        matched == n_pairs && vn_hits == (int)n_pairs);
            fprintf(stderr,
                "hx codegen 2a.5 validate RESULT on %s: "
                "%s  n_pairs=%zu vn_hits=%d matched=%zu missing=%zu "
                "extras=%d digest_mismatches=%d\n",
                hxhost,
                pass ? "PASS" : "FAIL",
                n_pairs, vn_hits, matched, missing, extras, mismatches);

            /* --- Print 3-5 byte-exact tuples for the report. */
            if (pass) {
                fprintf(stderr,
                    "hx codegen 2a.5 sample byte-exact tuples:\n");
                size_t sample = vn_hits < 5 ? (size_t)vn_hits : 5;
                for (size_t s = 0; s < sample; s++) {
                    uint32_t *slot = vhits + s * 19;
                    uint32_t widx = slot[0];
                    uint32_t sidx = slot[1];
                    size_t pid = (size_t)widx * n_salt + (size_t)sidx;
                    uint32_t e0 = ((uint32_t *)(oracle + pid * 16))[0];
                    uint32_t e1 = ((uint32_t *)(oracle + pid * 16))[1];
                    uint32_t e2 = ((uint32_t *)(oracle + pid * 16))[2];
                    uint32_t e3 = ((uint32_t *)(oracle + pid * 16))[3];
                    fprintf(stderr,
                        "  [%zu] pass=\"%s\" salt=\"%s\" "
                        "oracle=%08x%08x%08x%08x codegen=%08x%08x%08x%08x\n",
                        s, passes[widx], salts[sidx],
                        e0, e1, e2, e3,
                        slot[3], slot[4], slot[5], slot[6]);
                }
            }

            /* Cleanup. */
            gpu_opencl_jit_release_keep(vprog, vkern);
            free(vhits); free(seen);
            free(packed); free(woff);
            free(cfp); free(cidx); free(hbuf); free(hoff); free(hlen);
            free(oracle);
            for (size_t i = 0; i < n_pass; i++) free(passes[i]);
            for (size_t i = 0; i < n_salt; i++) free(salts[i]);
            free(passes); free(salts);
            free(hxsrc);

            exit(pass ? 0 : 1);
        }
#endif  /* end 2a.5 inline-harness #if 0 block; logic now in hx_e347_validate_run_shared */

        /* JIT against dev[0]. FATAL on failure (function exits internally).
         *
         * Sub-phase 2a.3 (2026-05-21): pattern-recognized fast-paths emit
         * source that depends on gpu_common.cl symbols (OCLParams,
         * md5_block, EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx).
         * Route those through the with-common JIT helper which prepends
         * gpu_common_str at clCreateProgramWithSource. Other (generic-
         * dispatch placeholder) emissions remain on the original helper. */
        /* Sub-phase 5b.3a (2026-05-27): the MAKE_MD5PASS family emitter
         * (5a.2+) also produces source that references gpu_common.cl
         * symbols (OCLParams, md5_block, state_to_hex32_bytes, the
         * primitive *_block functions, HAVAL_IV, etc.), so the JIT-only
         * dump harness path MUST route family members through the
         * _with_common helper too -- not just e347. Prior to this fix the
         * non-VALIDATE dump path JIT-failed on family members with
         * "undeclared identifier" errors (gpu_common never prepended);
         * latent since 5a.2 but only surfaced now because Tier 1/2/3
         * always validated via MDXFIND_HX_CODEGEN_VALIDATE=1 (which uses
         * the _with_common_keep dispatch path). */
        if (target_job == 347
            || gpu_codegen_kernelb_family_md5pass_eligible(target_job)) {
            gpu_opencl_jit_compile_source_with_common(0, hxsrc, "-cl-std=CL1.2");
        } else {
            gpu_opencl_jit_compile_source(0, hxsrc, "-cl-std=CL1.2");
        }
        free(hxsrc);
        fprintf(stderr,
            "hx codegen: sub-phase 2a.3 harness complete on %s (e%d %s JIT ok) -- exiting\n",
            hxhost, entry->job_enum, entry->name);
        exit(0);
    }
#endif
#if defined(__APPLE__) && defined(METAL_GPU)
    /* hx codegen sub-phase 2a.4 (2026-05-21): Metal harness branch.
     * Mirrors the OpenCL harness block above exactly; the only differences
     * are the backend enum, the JIT helper called, and the stderr log
     * markers. Both branches are independently gated on MDXFIND_HX_CODEGEN
     * so a single build can exercise both backends (mutually exclusive at
     * runtime since each backend's GPU init is platform-conditional).
     *
     * The harness env vars (_JOB / _VALIDATE) were removed 2026-09-16, so
     * this block is unreachable. They were preferred over plain
     * MDXFIND_HX_CODEGEN because its "=0" was a production opt-out. */
    if (gpu_ok == 0
        && (NULL /* was MDXFIND_HX_CODEGEN_JOB */
            || NULL /* was MDXFIND_HX_CODEGEN_VALIDATE */)) {
        char hxhost[256];
        mdx_gethostname(hxhost, sizeof(hxhost));

        int target_job = 1;
        const char *job_env = NULL /* was MDXFIND_HX_CODEGEN_JOB */;
        if (job_env && *job_env) target_job = atoi(job_env);

        const struct hx_spec_entry *entry = hx_specs_lookup(target_job);
        if (!entry) {
            fprintf(stderr,
                "FATAL: %s:%d hx_specs_lookup(%d) returned NULL on %s "
                "(check codegen/hx_specs_data.c)\n",
                __FILE__, __LINE__, target_job, hxhost);
            exit(1);
        }
        if (entry->is_outlier || entry->compile_failed || !entry->program) {
            fprintf(stderr,
                "FATAL: %s:%d hx codegen: e%d %s is %s on %s -- routes to "
                "override path, not codegen\n",
                __FILE__, __LINE__, entry->job_enum, entry->name,
                entry->is_outlier ? "outlier" :
                entry->compile_failed ? "compile_failed" : "no-program",
                hxhost);
            exit(1);
        }
        fprintf(stderr,
            "hx codegen: 2a.4 Metal harness using e%d %s `%s` "
            "(ncode=%d nvars=%d)\n",
            entry->job_enum, entry->name, entry->expression,
            entry->program->ncode, entry->program->nvars);

        struct hx_specialization zone;
        memset(&zone, 0, sizeof(zone));
        zone.iter_count_if_fixed = 1;
        zone.has_rules           = 0;
        zone.has_masks           = 0;
        zone.has_bf              = 0;
        zone.salt_minlen         = 0;
        zone.salt_maxlen         = 0;
        zone.salt_count_regime   = HX_SALT_SINGLE;
        zone.iter_shape          = HX_ITER_NONE;
        zone.digest_endianness   = HX_DIGEST_LE;
        zone.emit_width          = 16;

        if (target_job == 347) {
            zone.has_rules         = 1;
            zone.salt_count_regime = HX_SALT_BATCH_64;
            zone.salt_minlen       = 1;
            zone.salt_maxlen       = 64;
        }
        /* Sub-phase 5a.3 (2026-05-22): MAKE_MD5PASS family Metal
         * specialization. Same shape as the OpenCL e161 path: has_rules
         * is unset (family kernel is per-thread, no SALT_BATCH outer
         * loop), salts are placeholder-only. The emit-stage uses
         * salt_minlen/maxlen=1/64 to mirror e347 for byte-identical
         * banner output (the family kernel body ignores both fields). */
        if (target_job == 161) {
            zone.has_rules         = 0;
            zone.salt_count_regime = HX_SALT_BATCH_64;
            zone.salt_minlen       = 1;
            zone.salt_maxlen       = 64;
        }

        char *hxsrc = NULL;
        size_t hxcap = 0;
        /* Sub-phase 5a.2 (2026-05-22): walker signature takes entry. */
        int hxrc = hx_emit_kernel(entry->program, &zone,
                                  HX_BACKEND_METAL, entry,
                                  &hxsrc, &hxcap);
        if (hxrc != 0 || !hxsrc) {
            fprintf(stderr,
                "FATAL: %s:%d hx codegen Metal walker failed (rc=%d) on %s "
                "for e%d %s\n",
                __FILE__, __LINE__, hxrc, hxhost,
                entry->job_enum, entry->name);
            exit(1);
        }
        if (hx_dump_source(hxsrc, "MDXFIND_HX_CODEGEN_DUMP") < 0) {
            fprintf(stderr,
                "FATAL: %s:%d hx codegen Metal source dump failed on %s\n",
                __FILE__, __LINE__, hxhost);
            exit(1);
        }

        /* Sub-phase 2a.6 (2026-05-22): Metal byte-exact validation harness.
         * Mirrors the OpenCL branch above. When MDXFIND_HX_CODEGEN_VALIDATE
         * =1 AND target_job==347, drive the full dispatch-and-diff path via
         * the shared helper hx_e347_validate_run_shared (defined above the
         * function). backend_id=1 selects the Metal dispatch arm. The helper
         * exits with code 0 (PASS) or 1 (FAIL); never returns. */
        const char *m_validate_env = NULL /* was MDXFIND_HX_CODEGEN_VALIDATE */;
        const char *m_fixture_env  = NULL /* was MDXFIND_HX_CODEGEN_FIXTURE */;
        int m_validate_on = (m_validate_env && *m_validate_env &&
                             *m_validate_env != '0');
        if (target_job == 347 && m_validate_on) {
            hx_e347_validate_run_shared(1, hxsrc, m_fixture_env, hxhost);
            /* NOTREACHED */
        }
        /* Sub-phase 5b.3a.0.3 (2026-05-27) D17.4.b refactor: 11-entry
         * Metal OR-chain replaced with single-source-of-truth predicate
         * call. Mirror of the OpenCL twin's refactor at line 39527-era.
         */
        if (gpu_codegen_kernelb_family_md5pass_eligible(target_job)
            && m_validate_on) {
            hx_family_md5pass_validate_run_shared(
                1, hxsrc, m_fixture_env, hxhost, target_job);
            /* NOTREACHED */
        }

        /* JIT against dev[0]. FATAL on failure (function exits internally).
         * The Metal codegen-emitted source for the e347 pattern references
         * metal_common.metal symbols (md5_block, OCLParams via typedef,
         * EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx); route through
         * the with-common JIT helper which prepends metal_common_str at
         * newLibraryWithSource. */
        gpu_metal_jit_compile_source_with_common(0, hxsrc);
        free(hxsrc);
        fprintf(stderr,
            "hx codegen: sub-phase 2a.4 Metal harness complete on %s "
            "(e%d %s JIT ok) -- exiting\n",
            hxhost, entry->job_enum, entry->name);
        exit(0);
    }
#endif
  }
#endif
}


/* ---- Background line counter for -W auto ---- */

/* Resolve cache database path: CLI override > env var > ./mdxfind.db */
static void linecount_cache_resolve(void) {
    if (LineCountCachePath[0]) return;  /* already set by -W auto:/path */
    const char *env = getenv("MDXFIND_CACHE");
    if (env && env[0]) {
        strncpy(LineCountCachePath, env, sizeof(LineCountCachePath) - 1);
    } else {
        strncpy(LineCountCachePath, "mdxfind.db", sizeof(LineCountCachePath) - 1);
    }
}

/* Open cache DB, create table if needed.
 * Fatal error if DB doesn't exist and can't be created.
 * Warning if DB exists but can't be opened (wrong format, etc). */
static sqlite3 *linecount_cache_open(void) {
    linecount_cache_resolve();
    int existed = (access(LineCountCachePath, F_OK) == 0);
    sqlite3 *db = NULL;
    if (sqlite3_open(LineCountCachePath, &db) != SQLITE_OK) {
        if (existed)
            fprintf(stderr, "Warning: cache %s exists but cannot be opened: %s\n",
                    LineCountCachePath, db ? sqlite3_errmsg(db) : "unknown error");
        else {
            fprintf(stderr, "Fatal: cannot create cache %s: %s\n",
                    LineCountCachePath, db ? sqlite3_errmsg(db) : "unknown error");
            if (db) sqlite3_close(db);
            exit(1);
        }
        if (db) sqlite3_close(db);
        return NULL;
    }
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA busy_timeout=5000;", NULL, NULL, NULL);
    char *errmsg = NULL;
    int rc = sqlite3_exec(db,
        "CREATE TABLE IF NOT EXISTS linecount ("
        "  filepath TEXT NOT NULL,"
        "  filesize INTEGER NOT NULL,"
        "  mtime INTEGER NOT NULL,"
        "  lines INTEGER NOT NULL,"
        "  cached_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),"
        "  PRIMARY KEY (filepath, filesize, mtime)"
        ");", NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        if (existed)
            fprintf(stderr, "Warning: cache %s opened but schema invalid: %s\n",
                    LineCountCachePath, errmsg ? errmsg : "unknown");
        else {
            fprintf(stderr, "Fatal: cannot initialize cache %s: %s\n",
                    LineCountCachePath, errmsg ? errmsg : "unknown");
            sqlite3_free(errmsg);
            sqlite3_close(db);
            exit(1);
        }
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return NULL;
    }
    /* Prune /tmp/ entries older than 7 days */
    sqlite3_exec(db,
        "DELETE FROM linecount WHERE filepath LIKE '/tmp/%'"
        " AND cached_at < strftime('%s','now') - 604800;",
        NULL, NULL, NULL);
    return db;
}

static sqlite3 *LineCountDB;
static sqlite3_stmt *LC_stmt_lookup;
static sqlite3_stmt *LC_stmt_store;

/* Lookup cached line count. Returns -1 if not found. */
static long long linecount_cache_lookup(sqlite3 *db, const char *path,
                                         long long size, long long mtime) {
    if (!LC_stmt_lookup) return -1;
    sqlite3_reset(LC_stmt_lookup);
    sqlite3_bind_text(LC_stmt_lookup, 1, path, -1, SQLITE_STATIC);
    sqlite3_bind_int64(LC_stmt_lookup, 2, size);
    sqlite3_bind_int64(LC_stmt_lookup, 3, mtime);
    long long result = -1;
    if (sqlite3_step(LC_stmt_lookup) == SQLITE_ROW)
        result = sqlite3_column_int64(LC_stmt_lookup, 0);
    return result;
}

/* Store line count in cache. */
static void linecount_cache_store(sqlite3 *db, const char *path,
                                   long long size, long long mtime,
                                   long long lines) {
    if (!LC_stmt_store) return;
    sqlite3_reset(LC_stmt_store);
    sqlite3_bind_text(LC_stmt_store, 1, path, -1, SQLITE_STATIC);
    sqlite3_bind_int64(LC_stmt_store, 2, size);
    sqlite3_bind_int64(LC_stmt_store, 3, mtime);
    sqlite3_bind_int64(LC_stmt_store, 4, lines);
    sqlite3_step(LC_stmt_store);
}

/* Count newlines in a buffer */
/* Line counting buffer size — must be a multiple of 4096.
 * Larger buffers improve throughput on systems with fast I/O. */
#if defined(ARM) && ARM < 8
#define LINECOUNT_BUFSZ  (64 * 1024)
#else
#define LINECOUNT_BUFSZ  (256 * 1024)
#endif

#ifndef NOTINTEL
__attribute__((target("avx2")))
static unsigned long long count_newlines_avx2(const unsigned char *buf, size_t len) {
    unsigned long long count = 0;
    const unsigned char *p = buf, *end = buf + len;
    __m256i nl32 = _mm256_set1_epi8('\n');
    while (p + 32 <= end) {
        __m256i v = _mm256_loadu_si256((const __m256i *)p);
        __m256i cmp = _mm256_cmpeq_epi8(v, nl32);
        count += __builtin_popcount(_mm256_movemask_epi8(cmp));
        p += 32;
    }
    if (p + 16 <= end) {
        __m128i nl = _mm_set1_epi8('\n');
        __m128i v = _mm_loadu_si128((const __m128i *)p);
        count += __builtin_popcount(_mm_movemask_epi8(_mm_cmpeq_epi8(v, nl)));
        p += 16;
    }
    while (p < end) { if (*p++ == '\n') count++; }
    return count;
}
#endif

static unsigned long long count_newlines(const unsigned char *buf, size_t len) {
    unsigned long long count = 0;
    const unsigned char *p = buf, *end = buf + len;
#ifndef NOTINTEL
    /* AVX2 fast path (runtime-detected) */
    static int has_avx2 = -1;
    if (has_avx2 < 0) has_avx2 = __builtin_cpu_supports("avx2");
    if (has_avx2) return count_newlines_avx2(buf, len);
    /* SSE2 fallback */
    { __m128i nl = _mm_set1_epi8('\n');
      while (p + 16 <= end) {
        __m128i v = _mm_loadu_si128((const __m128i *)p);
        __m128i cmp = _mm_cmpeq_epi8(v, nl);
        count += __builtin_popcount(_mm_movemask_epi8(cmp));
        p += 16;
      }
    }
#elif defined(ARM) && ARM >= 8
    /* NEON: scan 16 bytes at a time */
    { uint8x16_t nl = vdupq_n_u8('\n');
      uint8x16_t one = vdupq_n_u8(1);
      while (p + 16 <= end) {
        uint8x16_t v = vld1q_u8(p);
        uint8x16_t cmp = vandq_u8(vceqq_u8(v, nl), one);
        count += vaddlvq_u8(cmp);
        p += 16;
      }
    }
#endif
    /* Scalar tail */
    while (p < end) {
        p = memchr(p, '\n', end - p);
        if (!p) break;
        count++;
        p++;
    }
    return count;
}

/* Count lines in a single file, handling gzip/xz/bz2/zstd/zip compression.
 * Returns line count, or 0 on error.
 * `complete` (if non-NULL) is set to 1 only if counting ran to EOF without
 * bailing on HashWaiting. The caller MUST check this before persisting the
 * count to the SQLite cache — otherwise an early-bail partial count poisons
 * future sessions (the cause of the wrong-ETA bug seen on Shooter
 * 2026-05-03 with hashmob.net.found.v7: 12M cached vs 3.16B actual). */
static unsigned long long linecount_file(const char *filename, int *complete) {
    if (complete) *complete = 0;   /* default: assume incomplete until proven otherwise */
    struct stat sb;
    if (stat(filename, &sb) != 0) return 0;
    /* Skip pipes, sockets, devices */
    if (S_ISFIFO(sb.st_mode) || !S_ISREG(sb.st_mode)) return 0;

    unsigned long long lines = 0;
    int completed = 0;             /* reaches 1 only via natural fall-through after read loop */
    unsigned char hdrbuf[256];

    /* Read first bytes to detect format */
    int fd = open(filename, O_RDONLY | O_BINARY);
    if (fd < 0) return 0;
    ssize_t hdrlen = read(fd, hdrbuf, sizeof(hdrbuf));
    close(fd);
    if (hdrlen < 4) return 0;

    /* Check for early exit */
    { int done = 0;
      possess(HashWaiting);
      done = peek_lock(HashWaiting);
      release(HashWaiting);
      if (done) return 0;
    }

    unsigned char *dbuf = (unsigned char *)malloc_lock(LINECOUNT_BUFSZ, "linecount");
    unsigned char *inbuf = (unsigned char *)malloc_lock(LINECOUNT_BUFSZ, "linecount");
    if (!dbuf || !inbuf) { free(dbuf); free(inbuf); return 0; }
#ifdef __linux__
#define LC_FADVISE(fd) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)
#else
#define LC_FADVISE(fd) ((void)0)
#endif

    if (hdrbuf[0] == 0xfd && hdrbuf[1] == 0x37 && hdrbuf[2] == 0x7a &&
        hdrbuf[3] == 0x58 && hdrbuf[4] == 0x5a && hdrbuf[5] == 0x00) {
        /* XZ */
        fd = open(filename, O_RDONLY | O_BINARY);
        if (fd < 0) goto linecount_done;
        LC_FADVISE(fd);
        lzma_stream xz = LZMA_STREAM_INIT;
        {
            lzma_ret r = lzma_stream_decoder(&xz, UINT64_MAX, LZMA_CONCATENATED);
            if (r != LZMA_OK) {
                fprintf(stderr, "FATAL: %s:%d lzma_stream_decoder failed (ret=%d) for %s\n",
                        __FILE__, __LINE__, (int)r, filename);
                exit(1);
            }
        }
        xz.avail_in = 0;
        int eof = 0;
        while (!eof) {
            if (xz.avail_in == 0) {
                ssize_t n = read(fd, inbuf, LINECOUNT_BUFSZ);
                if (n <= 0) eof = 1;
                else { xz.next_in = inbuf; xz.avail_in = n; }
            }
            xz.next_out = dbuf; xz.avail_out = LINECOUNT_BUFSZ;
            lzma_ret r = lzma_code(&xz, eof ? LZMA_FINISH : LZMA_RUN);
            size_t got = LINECOUNT_BUFSZ - xz.avail_out;
            if (got > 0) lines += count_newlines(dbuf, got);
            if (r == LZMA_STREAM_END || (r != LZMA_OK)) eof = 1;
            if ((lines & 0xFFFFF) == 0) {
                possess(HashWaiting); int done = peek_lock(HashWaiting); release(HashWaiting);
                if (done) { lzma_end(&xz); close(fd); goto linecount_done; }
            }
        }
        lzma_end(&xz);
        close(fd);
    } else if (hdrbuf[0] == 'B' && hdrbuf[1] == 'Z' && hdrbuf[2] == 'h') {
        /* bzip2 */
        fd = open(filename, O_RDONLY | O_BINARY);
        if (fd < 0) goto linecount_done;
        LC_FADVISE(fd);
        bz_stream bz;
        memset(&bz, 0, sizeof(bz));
        BZ2_bzDecompressInit(&bz, 0, 0);
        bz.avail_in = 0;
        int eof = 0;
        while (!eof) {
            if (bz.avail_in == 0) {
                ssize_t n = read(fd, inbuf, LINECOUNT_BUFSZ);
                if (n <= 0) eof = 1;
                else { bz.next_in = (char *)inbuf; bz.avail_in = n; }
            }
            bz.next_out = (char *)dbuf; bz.avail_out = LINECOUNT_BUFSZ;
            int r = BZ2_bzDecompress(&bz);
            size_t got = LINECOUNT_BUFSZ - bz.avail_out;
            if (got > 0) lines += count_newlines(dbuf, got);
            if (r == BZ_STREAM_END || r != BZ_OK) eof = 1;
            if ((lines & 0xFFFFF) == 0) {
                possess(HashWaiting); int done = peek_lock(HashWaiting); release(HashWaiting);
                if (done) { BZ2_bzDecompressEnd(&bz); close(fd); goto linecount_done; }
            }
        }
        BZ2_bzDecompressEnd(&bz);
        close(fd);
    } else if (hdrbuf[0] == 0x28 && hdrbuf[1] == 0xb5 && hdrbuf[2] == 0x2f && hdrbuf[3] == 0xfd) {
        /* Zstandard */
        fd = open(filename, O_RDONLY | O_BINARY);
        if (fd < 0) goto linecount_done;
        LC_FADVISE(fd);
        ZSTD_DStream *ds = ZSTD_createDStream();
        ZSTD_initDStream(ds);
        int eof = 0;
        while (!eof) {
            ssize_t n = read(fd, inbuf, LINECOUNT_BUFSZ);
            if (n <= 0) break;
            ZSTD_inBuffer zin = { inbuf, (size_t)n, 0 };
            while (zin.pos < zin.size) {
                ZSTD_outBuffer zout = { dbuf, LINECOUNT_BUFSZ, 0 };
                size_t r = ZSTD_decompressStream(ds, &zout, &zin);
                if (ZSTD_isError(r)) { eof = 1; break; }
                if (zout.pos > 0) lines += count_newlines(dbuf, zout.pos);
            }
            if ((lines & 0xFFFFF) == 0) {
                possess(HashWaiting); int done = peek_lock(HashWaiting); release(HashWaiting);
                if (done) { ZSTD_freeDStream(ds); close(fd); goto linecount_done; }
            }
        }
        ZSTD_freeDStream(ds);
        close(fd);
    } else if (hdrbuf[0] == 'P' && hdrbuf[1] == 'K' && hdrbuf[2] == 3 && hdrbuf[3] == 4 && hdrlen >= 30) {
        /* ZIP — parse local file header, decompress first file */
        unsigned short method = hdrbuf[8] | (hdrbuf[9] << 8);
        uint64_t comp_size;

        fd = open(filename, O_RDONLY | O_BINARY);
        if (fd < 0) goto linecount_done;
        LC_FADVISE(fd);
        comp_size = zip_entry_comp_size(fd, hdrbuf);

        if (method == 0) {
            /* Stored */
            uint64_t remaining = comp_size;
            while (remaining > 0) {
                size_t want = LINECOUNT_BUFSZ;
                if (want > remaining) want = remaining;
                ssize_t n = read(fd, dbuf, want);
                if (n <= 0) break;
                lines += count_newlines(dbuf, n);
                remaining -= n;
            }
        } else if (method == 8) {
            /* Deflate */
            z_stream zs;
            memset(&zs, 0, sizeof(zs));
            inflateInit2(&zs, -15);
            uint64_t remaining = comp_size;
            int eof = 0;
            while (!eof) {
                if (zs.avail_in == 0 && remaining > 0) {
                    size_t want = LINECOUNT_BUFSZ;
                    if (want > remaining) want = remaining;
                    ssize_t n = read(fd, inbuf, want);
                    if (n <= 0) { eof = 1; break; }
                    remaining -= n;
                    zs.next_in = inbuf; zs.avail_in = n;
                }
                zs.next_out = dbuf; zs.avail_out = LINECOUNT_BUFSZ;
                int r = inflate(&zs, Z_NO_FLUSH);
                size_t got = LINECOUNT_BUFSZ - zs.avail_out;
                if (got > 0) lines += count_newlines(dbuf, got);
                if (r == Z_STREAM_END || (r != Z_OK && r != Z_BUF_ERROR)) eof = 1;
            }
            inflateEnd(&zs);
        } else if (method == 14) {
            /* LZMA in ZIP */
            lzma_stream xz = LZMA_STREAM_INIT;
            {
                lzma_ret r = lzma_alone_decoder(&xz, UINT64_MAX);
                if (r != LZMA_OK) {
                    fprintf(stderr, "FATAL: %s:%d lzma_alone_decoder failed (ret=%d) for %s\n",
                            __FILE__, __LINE__, (int)r, filename);
                    exit(1);
                }
            }
            uint64_t remaining = comp_size;
            int eof = 0;
            lseek(fd, 4, SEEK_CUR);
            if (remaining > 4) remaining -= 4; else remaining = 0;
            while (!eof) {
                if (xz.avail_in == 0 && remaining > 0) {
                    size_t want = LINECOUNT_BUFSZ;
                    if (want > remaining) want = remaining;
                    ssize_t n = read(fd, inbuf, want);
                    if (n <= 0) { eof = 1; break; }
                    remaining -= n;
                    xz.next_in = inbuf; xz.avail_in = n;
                }
                xz.next_out = dbuf; xz.avail_out = LINECOUNT_BUFSZ;
                lzma_ret r = lzma_code(&xz, (remaining == 0) ? LZMA_FINISH : LZMA_RUN);
                size_t got = LINECOUNT_BUFSZ - xz.avail_out;
                if (got > 0) lines += count_newlines(dbuf, got);
                if (r == LZMA_STREAM_END || r != LZMA_OK) eof = 1;
            }
            lzma_end(&xz);
        }
        /* bzip2 and zstd in ZIP — less common, skip for now */
        close(fd);
    } else if (hdrbuf[0] == 0x1f && hdrbuf[1] == 0x8b) {
        /* gzip — use gzread */
        gzFile gz = gzopen(filename, "rb");
        if (!gz) goto linecount_done;
        gzbuffer(gz, LINECOUNT_BUFSZ);
        int n;
        while ((n = gzread(gz, dbuf, LINECOUNT_BUFSZ)) > 0) {
            lines += count_newlines(dbuf, n);
            if ((lines & 0xFFFFF) == 0) {
                possess(HashWaiting); int done = peek_lock(HashWaiting); release(HashWaiting);
                if (done) { gzclose(gz); goto linecount_done; }
            }
        }
        gzclose(gz);
    } else {
        /* Plain text — direct read with SIMD newline counting */
        fd = open(filename, O_RDONLY | O_BINARY);
        if (fd < 0) goto linecount_done;
        LC_FADVISE(fd);
        ssize_t n;
        while ((n = read(fd, dbuf, LINECOUNT_BUFSZ)) > 0) {
            lines += count_newlines(dbuf, n);
            if ((lines & 0xFFFFF) == 0) {
                possess(HashWaiting); int done = peek_lock(HashWaiting); release(HashWaiting);
                if (done) { close(fd); goto linecount_done; }
            }
        }
        close(fd);
    }
    completed = 1;     /* reached only via natural fall-through; all early bails goto linecount_done */
linecount_done:
    free(dbuf);
    free(inbuf);
    if (complete) *complete = completed;
    return lines;
}

/* LineCountDB, LC_stmt_lookup, LC_stmt_store declared before linecount_cache_lookup */

MDXALIGN void linecount_thread(void *dummy) {
    sqlite3 *db = LineCountDB;
    int cache_hits = 0, cache_misses = 0;
    struct timespec lc_start, lc_end;
    clock_gettime(CLOCK_MONOTONIC, &lc_start);

    for (int i = 0; i < LineCountArgc; i++) {
        /* Check for shutdown */
        possess(HashWaiting);
        int done = peek_lock(HashWaiting);
        release(HashWaiting);
        if (done) break;

        /* Skip stdin */
        if (strcmp(LineCountArgv[i], "stdin") == 0) continue;

        /* Check cache first */
        struct stat sb;
        if (stat(LineCountArgv[i], &sb) != 0) continue;
        if (S_ISFIFO(sb.st_mode) || !S_ISREG(sb.st_mode)) continue;

        char resolved[4096];
#ifdef _WIN32
        char *rp = _fullpath(resolved, LineCountArgv[i], sizeof(resolved));
#else
        char *rp = realpath(LineCountArgv[i], resolved);
#endif
        if (!rp) strncpy(resolved, LineCountArgv[i], sizeof(resolved) - 1);

        long long cached = linecount_cache_lookup(db, resolved,
                            (long long)sb.st_size, (long long)sb.st_mtime);
        if (cached >= 0) {
            __sync_fetch_and_add(&AutoCountTotalLines, (unsigned long long)cached);
            cache_hits++;
            continue;
        }

        int complete = 0;
        /* Per-file linecount perf instrumentation: time the actual file
         * scan (cache-miss path) and emit MB/s based on on-disk byte
         * size from stat above. For compressed inputs this is the
         * compressed read rate — what end-users can measure externally. */
        struct timespec lcf_start, lcf_end;
        clock_gettime(CLOCK_MONOTONIC, &lcf_start);
        unsigned long long count = linecount_file(LineCountArgv[i], &complete);
        clock_gettime(CLOCK_MONOTONIC, &lcf_end);
        if (count > 0) {
            __sync_fetch_and_add(&AutoCountTotalLines, count);
            double lcf_ms = (lcf_end.tv_sec - lcf_start.tv_sec) * 1000.0
                          + (lcf_end.tv_nsec - lcf_start.tv_nsec) / 1e6;
            double mb = (double)sb.st_size / (1024.0 * 1024.0);
            double mbps = (lcf_ms > 0.0) ? mb * 1000.0 / lcf_ms : 0.0;
            tsfprintf(stderr,
                "linecount: %s (%.2f MB): scanned in %.0fms = %.1f MB/s\n",
                LineCountArgv[i], mb, lcf_ms, mbps);
            /* Only persist to cache if we counted to EOF — partial counts
             * from HashWaiting early-bail would poison future sessions. */
            if (complete) {
                linecount_cache_store(db, resolved,
                    (long long)sb.st_size, (long long)sb.st_mtime, (long long)count);
                cache_misses++;
            } else {
                fprintf(stderr,
                    "Wordlist ETA: line count for %s bailed early at %llu lines (incomplete) — not cached\n",
                    LineCountArgv[i], count);
            }
        }
    }
    /* Finalize prepared statements before close so sqlite3_close()
     * can succeed and run its WAL checkpoint + cleanup. Without this
     * the close returns SQLITE_BUSY (silently — rc was historically
     * dropped) and the .db-wal/.db-shm files persist after exit. */
    if (LC_stmt_lookup) { sqlite3_finalize(LC_stmt_lookup); LC_stmt_lookup = NULL; }
    if (LC_stmt_store)  { sqlite3_finalize(LC_stmt_store);  LC_stmt_store  = NULL; }
    if (db) {
        int crc = sqlite3_close(db);
        if (crc != SQLITE_OK)
            fprintf(stderr, "Warning: sqlite3_close(%s) returned %d (%s) — WAL/-shm may persist\n",
                    LineCountCachePath, crc, sqlite3_errstr(crc));
        LineCountDB = NULL;
    }
    clock_gettime(CLOCK_MONOTONIC, &lc_end);
    double lc_secs = (lc_end.tv_sec - lc_start.tv_sec)
                   + (lc_end.tv_nsec - lc_start.tv_nsec) / 1e9;
    if (cache_hits > 0 || cache_misses > 0)
        tsfprintf(stderr, "Line count: %d cached, %d counted, %llu total lines in %.2fs\n",
                cache_hits, cache_misses, (unsigned long long)AutoCountTotalLines, lc_secs);
    __sync_synchronize();
    AutoCountDone = 1;
}

/* Persist the AUTHORITATIVE line count for `filepath` to the cache,
 * overwriting any prior partial-count entry. Called from the main read
 * loop after a wordlist has been fully consumed (Fileline = true total).
 *
 * Opens its own short-lived SQLite connection — WAL mode + busy_timeout
 * provide concurrency vs the background linecount_thread. Atomic via
 * single INSERT OR REPLACE inside an implicit transaction. WAL-checkpoint-
 * truncate at close so the .db-wal file doesn't linger.
 *
 * No-op if MDXFIND_CACHE / -W auto wasn't used in this session
 * (LineCountCachePath is empty). */
static void linecount_cache_finalize(const char *filepath,
                                     long long size, long long mtime,
                                     unsigned long long actual_lines) {
    if (!*LineCountCachePath) return;        /* cache not enabled this session */
    sqlite3 *db = NULL;
    if (sqlite3_open(LineCountCachePath, &db) != SQLITE_OK) {
        if (db) sqlite3_close(db);
        return;
    }
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA busy_timeout=5000;", NULL, NULL, NULL);
    /* Schema may not exist if linecount_thread didn't run — create defensively */
    sqlite3_exec(db,
        "CREATE TABLE IF NOT EXISTS linecount ("
        "  filepath TEXT NOT NULL,"
        "  filesize INTEGER NOT NULL,"
        "  mtime INTEGER NOT NULL,"
        "  lines INTEGER NOT NULL,"
        "  cached_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),"
        "  PRIMARY KEY (filepath, filesize, mtime)"
        ");", NULL, NULL, NULL);
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db,
            "INSERT OR REPLACE INTO linecount (filepath, filesize, mtime, lines) "
            "VALUES (?, ?, ?, ?)", -1, &st, NULL) == SQLITE_OK) {
        sqlite3_bind_text (st, 1, filepath, -1, SQLITE_STATIC);
        sqlite3_bind_int64(st, 2, size);
        sqlite3_bind_int64(st, 3, mtime);
        sqlite3_bind_int64(st, 4, (long long)actual_lines);
        sqlite3_step(st);
        sqlite3_finalize(st);
    }
    /* Truncate the WAL into the main DB so we leave a single clean file. */
    sqlite3_exec(db, "PRAGMA wal_checkpoint(TRUNCATE);", NULL, NULL, NULL);
    int crc = sqlite3_close(db);
    if (crc != SQLITE_OK) {
        fprintf(stderr,
            "Warning: linecount_cache_finalize: sqlite3_close(%s) returned %d (%s)\n",
            LineCountCachePath, crc, sqlite3_errstr(crc));
    }
}

/* Expunge a cached row when the runtime detection in the main read loop
 * discovers that the cached line count is stale (Fileline > AutoCountTotalLines
 * — the cached value was a partial count, almost certainly poisoned by a
 * pre-1.375 mdxfind that crashed before EOF). After this DELETE the next
 * session re-counts from scratch via linecount_thread; the natural
 * linecount_cache_finalize at gzclose still writes the authoritative count
 * if THIS session runs to completion. Idempotent — DELETE on missing row is
 * a no-op. Same connection-lifecycle pattern as linecount_cache_finalize. */
static void linecount_cache_delete(const char *filepath) {
    if (!*LineCountCachePath || !filepath) return;
    sqlite3 *db = NULL;
    if (sqlite3_open(LineCountCachePath, &db) != SQLITE_OK) {
        if (db) sqlite3_close(db);
        return;
    }
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;",  NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA busy_timeout=5000;", NULL, NULL, NULL);
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db,
            "DELETE FROM linecount WHERE filepath = ?", -1, &st, NULL) == SQLITE_OK) {
        sqlite3_bind_text(st, 1, filepath, -1, SQLITE_STATIC);
        sqlite3_step(st);
        sqlite3_finalize(st);
    }
    sqlite3_exec(db, "PRAGMA wal_checkpoint(TRUNCATE);", NULL, NULL, NULL);
    int crc = sqlite3_close(db);
    if (crc != SQLITE_OK) {
        fprintf(stderr,
            "Warning: linecount_cache_delete: sqlite3_close(%s) returned %d (%s)\n",
            LineCountCachePath, crc, sqlite3_errstr(crc));
    }
}

static void format_eta(double seconds, char *buf, size_t bufsz) {
    if (seconds <= 0) { snprintf(buf, bufsz, "done"); return; }
    if (seconds < 60) { snprintf(buf, bufsz, "%.0fs", seconds); return; }
    if (seconds < 3600) { snprintf(buf, bufsz, "%dm%02ds", (int)(seconds/60), (int)((int)seconds%60)); return; }
    if (seconds < 86400) { snprintf(buf, bufsz, "%dh%02dm", (int)(seconds/3600), ((int)seconds%3600)/60); return; }
    if (seconds < 86400*30) { snprintf(buf, bufsz, "%dd%dh", (int)(seconds/86400), ((int)seconds%86400)/3600); return; }
    if (seconds < 86400*365) { snprintf(buf, bufsz, "~%dmo%dd", (int)(seconds/(86400*30)), ((int)seconds%(86400*30))/86400); return; }
    snprintf(buf, bufsz, "~%dy%dmo", (int)(seconds/(86400*365)), ((int)seconds%(86400*365))/(86400*30));
}

/* Fixed internal round count per candidate-salt pair, for the brute-force
 * progress divisor only. PAIRED WITH gpu_compute_iter_sum() in
 * gpu/gpujob_opencl.c -- these constants are the same protocol values
 * (25 for Unix DES crypt(3), 1000 for md5crypt, 1000000 for Drupal SHA1)
 * and must stay in step with that table. Compiled unconditionally: the
 * progress display exists in non-GPU builds too, whereas
 * gpu_compute_iter_sum is static inside a GPU-only translation unit.
 *
 * Types whose round count is encoded per salt (bcrypt cost, phpBB3 iter
 * char, the rounds= prefix on $5$/$6$) return 1 and clear *exact, because
 * reconstructing their true divisor needs the salt strings of the salts
 * that are still live, and a salt that retired has left LIVESALTS while
 * its operations remain in Tothash. Reporting 1 under-divides, which is
 * why the caller caps the percentage and refuses to print an ETA rather
 * than showing a confident wrong one. */
static unsigned long long bf_progress_iter_per_salt(int op, int *exact) {
  switch (op) {
    case JOB_DESCRYPT:       return 25ULL;
    case JOB_MD5CRYPT:
    case JOB_MD5CRYPTMD5:    return 1000ULL;
    case JOB_SHA1DRU:        return 1000000ULL;
    case JOB_BCRYPT:
    case JOB_PHPBB3:
    case JOB_SHA256CRYPT:
    case JOB_SHA512CRYPT:
    case JOB_SHA512CRYPTMD5:
      if (exact) *exact = 0;
      return 1ULL;
    default:                 return 1ULL;
  }
}

static void format_rate(double val, double *out, char **suffix) {
  *suffix = "";
  if (val < 0.0) val = 0.0;
  if (val > 1000.0) { *suffix = "K"; val /= 1000.0; }
  if (val > 1000.0) { *suffix = "M"; val /= 1000.0; }
  if (val > 1000.0) { *suffix = "G"; val /= 1000.0; }
  if (val > 1000.0) { *suffix = "T"; val /= 1000.0; }
  *out = val;
}

/* ReportStats ticks once a second, and shutdown has to wait for it to leave
 * before the run's end time is taken.  A plain sleep(1) made that wait up to
 * a second, so "seconds hashing" was quantised to ReportStats's tick.  It
 * waits here instead, and stats_wake() ends the wait at once. */
#include <pthread.h>   /* not reached through the other headers on every target (mingw) */
static pthread_mutex_t Stats_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t Stats_cv = PTHREAD_COND_INITIALIZER;
static int Stats_stop;

static void stats_nap(void) {
  struct timespec ts;
  int rc = 0;
  current_utc_time(&ts);
  ts.tv_sec += 1;
  pthread_mutex_lock(&Stats_mx);
  while (!Stats_stop && rc != ETIMEDOUT) {
    rc = pthread_cond_timedwait(&Stats_cv, &Stats_mx, &ts);
    if (rc != 0 && rc != ETIMEDOUT) {
      fprintf(stderr, "FATAL %s:%d: pthread_cond_timedwait: %s\n",
              __FILE__, __LINE__, strerror(rc));
      exit(1);
    }
  }
  pthread_mutex_unlock(&Stats_mx);
}

static void stats_wake(void) {
  pthread_mutex_lock(&Stats_mx);
  Stats_stop = 1;
  pthread_cond_broadcast(&Stats_cv);
  pthread_mutex_unlock(&Stats_mx);
}

MDXALIGN void ReportStats(void *dummy) {
  int x;
  double stime, wtime, lps, hps;
  double RetiredLines_rate = 0.0;  /* words retired/sec, updated each 15s tick */
  unsigned long long RetiredLines_now_snap = 0;  /* snapshot of lowest retired_line */
  char *mult, *mult1;
  unsigned long long lastline;
  int pause_state = 0;        /* 0=running, 1=just paused, 2=minutes, 3=hours */
  time_t pause_start = 0;     /* when pause began */
  int pause_ticks = 0;        /* 15-second ticks since pause started */

  stime = (double) starthash.tv_sec + (double) (starthash.tv_nsec) / 1000000000.0;
  lastline = 0;
  while (1) {
    if (!MDXpause) {
      atomic_store_explicit(&MDXlowest_line, ULLONG_MAX, memory_order_relaxed);
      MDXlowest_file = NULL;
    }
    for (x = 0; x < 15; x++) {
      possess(HashWaiting);
      if (peek_lock(HashWaiting) != 0) {
        release(HashWaiting);
        return;
      }
      release(HashWaiting);
      stats_nap();
      current_utc_time(&current);
    }

    if (MDXpause) {
      /* --- Paused states --- */
      if (pause_state == 0) {
        /* Transition: draining */
        pause_state = 1;
        pause_start = time(NULL);
        pause_ticks = 0;
      }

      if (pause_state == 1) {
        /* Draining: waiting for all threads to reach pause loop */
        int paused = MDXpaused_count;
        int total = ThreadCount;
        if (paused >= total) {
          /* All threads paused — transition to fully paused */
          pause_state = 2;
          pause_start = time(NULL);
          pause_ticks = 0;
          { char tbuf[64];
            struct tm *tm = localtime(&pause_start);
            strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", tm);
            { unsigned long long restart = atomic_load_explicit(&MDXlowest_line, memory_order_relaxed);
              const char *rfile = (const char *)MDXlowest_file;
              if (restart == ULLONG_MAX) restart = 0;
              fprintf(stderr, "MDXfind process (%d) paused at %s, restart with -w %llu on %s\n",
                    (int)getpid(), tbuf, restart,
                    rfile ? rfile : "(unknown)");
            }
          }
        } else {
          fprintf(stderr, "MDXfind process (%d) pausing... %d/%d threads paused\n",
                  (int)getpid(), paused, total);
        }
        continue;
      }

      pause_ticks++;
      time_t elapsed = time(NULL) - pause_start;
      int minutes = (int)(elapsed / 60);
      int hours = minutes / 60;

      if (pause_state == 2 && pause_ticks >= 4) {
        /* After ~1 minute, transition to minutes display */
        pause_state = 3;
      }
      if (pause_state == 3 && minutes >= 60) {
        /* After 60 minutes, transition to hours display */
        pause_state = 4;
      }

      if (pause_state == 3 && (pause_ticks % 4) == 0) {
        fprintf(stderr, "MDXfind process (%d) paused for %d minute%s\n",
                (int)getpid(), minutes, minutes == 1 ? "" : "s");
      } else if (pause_state == 4 && (pause_ticks % (15 * 60)) == 0) {
        fprintf(stderr, "MDXfind process (%d) paused for %d hour%s\n",
                (int)getpid(), hours, hours == 1 ? "" : "s");
      }
      continue;
    }

    /* --- Running state --- */
    if (pause_state > 0) {
      /* Transition back from paused: print total pause duration */
      time_t elapsed = time(NULL) - pause_start;
      int days = (int)(elapsed / 86400);
      int hrs  = (int)((elapsed % 86400) / 3600);
      int mins = (int)((elapsed % 3600) / 60);
      int secs = (int)(elapsed % 60);
      fprintf(stderr, "MDXfind process (%d) resumed (paused %d:%02d:%02d:%02d)\n",
              (int)getpid(), days, hrs, mins, secs);
      pause_state = 0;
      pause_ticks = 0;
    }

    /* EMA feedback: update per-algorithm CPU rates from accumulated hashes.
     *
     * GPU-side correction: the rules-engine path (gpu/gpujob_opencl.c) bumps
     * the global Tothash atomic but does NOT feed linehints[op].hashes_accum.
     * Without this snapshot+delta, the EMA decays based only on the tiny
     * (~0.007% on rockyou) FastRule CPU contribution from words > 55 chars
     * that bypass the C2.2 GPU pack and fall through to the SIMD walker.
     * Snapshot Tothash per tick and attribute the delta to JOB_MD5 (the only
     * op the rules engine processes; gpu_rule_count > 0 implies op==JOB_MD5
     * gates at mdxfind.c:9765 and 9979). Multi-op rules-engine workloads are
     * not supported today; if added, this attribution becomes approximate.
     */
    if (linehints && gpu_rule_count > 0 && JOB_MD5 < linehints_count) {
      static unsigned long long tothash_snapshot = 0;
      unsigned long long tothash_now = atomic_load_explicit(&Tothash, memory_order_relaxed);
      unsigned long long gpu_delta = tothash_now - tothash_snapshot;
      tothash_snapshot = tothash_now;
      if (gpu_delta > 0)
        __sync_fetch_and_add(&LINEHINTS(JOB_MD5).hashes_accum, (long long)gpu_delta);
    }
    if (linehints) {
      unsigned long long lowest_line = ULLONG_MAX;
      for (x = 0; x < linehints_count; x++) {
        long long accum = __sync_lock_test_and_set(&LINEHINTS(x).hashes_accum, 0);
        if (accum > 0) {
          long long measured = accum / 15;  /* hashes/sec over 15s window */
          long long old = LINEHINTS(x).rate;
          if (old <= 0)
            LINEHINTS(x).rate = measured;
          else
            LINEHINTS(x).rate = (long long)(0.3 * measured + 0.7 * old);
        }
        /* Track lowest curline across active algorithms for progress */
        if (LINEHINTS(x).curline > 0 && LINEHINTS(x).curline < lowest_line)
          lowest_line = LINEHINTS(x).curline;
      }
      if (lowest_line < ULLONG_MAX)
        Lowline = lowest_line;
    }

    /* word-retirement ETA: find lowest retired_line across active ops.
     * Filter by curline > 0 (mirrors the existing Lowline filter at
     * 38828) — `linehints` is sized to MAX_OP+1 so inactive slots have
     * curline = retired_line = 0; without the filter the min returns 0
     * from inactive slots and RetiredLines_rate stays 0 forever. */
    { unsigned long long lowest_retired = ULLONG_MAX;
      int any_retired_op = 0;
      for (x = 0; x < linehints_count; x++) {
        if (LINEHINTS(x).curline == 0) continue;   /* skip inactive slots */
        if (TYPEDONE(x)) continue;
        unsigned long long r = LINEHINTS(x).retired_line;
        if (r < lowest_retired) lowest_retired = r;
        any_retired_op = 1;
      }
      static unsigned long long RetiredLines_lasttick = 0;
      /* Only update lasttick when we observed a valid retirement frontier
       * this tick. Otherwise we'd zero lasttick during brief any_retired_op==0
       * windows (e.g. chunk reset with no dispatch yet), making the NEXT
       * tick compute rate as cumulative-since-start instead of delta. */
      if (any_retired_op && lowest_retired != ULLONG_MAX) {
        unsigned long long RetiredLines_now = lowest_retired;
        RetiredLines_rate = (RetiredLines_now > RetiredLines_lasttick)
                            ? (double)(RetiredLines_now - RetiredLines_lasttick) / 15.0 : 0;
        RetiredLines_now_snap = RetiredLines_now;
        RetiredLines_lasttick = RetiredLines_now;
      } else {
        RetiredLines_rate = 0;
        /* leave RetiredLines_now_snap and lasttick at prior values */
      }
    }

    wtime = (double) current.tv_sec + (double) (current.tv_nsec) / 1000000000.0;
    wtime -= stime;
    if (wtime <= 0.0) wtime = 1;
    {
      unsigned long long total_rules = Totrules + atomic_load(&Totrules_gpu);
      if (total_rules) {
        lps = (double)(total_rules - lastline)/15.0;
        lastline = total_rules;
      } else {
        lps = (double)((TotLines - lastline))/15.0;
        lastline = TotLines;
      }
    }
    format_rate((double)Tothash / wtime, &hps, &mult1);
    format_rate(lps, &lps, &mult);

    if (BruteForceMode && BruteForceTotal > 0) {
      /* Brute-force progress display:
       * Show keyspace progress, rate, sample candidate, and ETA */
      /* Tothash tracks completed hash computations across all active types.
       * For brute-force with N types, each candidate is hashed N times.
       * For salted types, each candidate is also multiplied by the salt count.
       * Divide by the total multiplier to recover the candidate progress. */
      unsigned long long progress = Tothash;
      /* Recover candidate progress from Tothash. Tothash counts hash
       * OPERATIONS, so for an iterated type it already carries that type's
       * internal round count per candidate-salt pair: the GPU accounting
       * multiplies by gpu_compute_iter_sum() (gpu/gpujob_opencl.c), which is
       * iter x nsalts_packed. Dividing by the live salt count alone therefore
       * leaves the iteration factor in, and the displayed position runs past
       * the keyspace -- DESCRYPT showed 1730% of a 100M 8-digit mask at 69%
       * complete, which also clamped the sample candidate to the last mask
       * value and forced ETA to report done. Divide by iter x salts instead.
       *
       * iter_exact stays 1 only while every selected type has a FIXED round
       * count. The variable-iter types derive theirs per salt from the salt
       * string, and a retired salt is gone from LIVESALTS but its operations
       * are still in Tothash, so their divisor is not reconstructible here.
       * For those the percentage is capped and the ETA says unknown rather
       * than naming a wrong time -- see the iter_exact uses below. */
      int iter_exact = 1;
      { unsigned long long multiplier = 0;
        /* -i N is the THIRD factor in Tothash and the one 1.593 left out.  A
         * chain N deep costs N hash operations per candidate-salt pair and
         * Tothash counts every one, so the divisor needs the depth as well as
         * the per-type round count and the salt count.  Measured on .205, MD5
         * over a 1000-candidate ?d?d?d mask: 1,000 / 2,000 / 3,000 / 10,000
         * operations at -i 1 / 2 / 3 / 10 -- exactly linear.  Maxiter is global
         * rather than per-type, so it scales every term, not one of them.
         * Without it a -i 2 run reported 13.3T of a 6.7T keyspace: the
         * percentage clamped to 100, the sample candidate clamped to the last
         * mask value, and the ETA said done -- the same three symptoms, from
         * the same cause, as the DESCRYPT case 1.593 fixed. */
        unsigned long long depth = (Maxiter > 0) ? (unsigned long long)Maxiter : 1ULL;
        Word_t ti = 0; int trc;
        J1F(trc, Dohash, ti);
        while (trc) {
          unsigned int ns = LIVESALTS(ti);
          unsigned long long iter = bf_progress_iter_per_salt((int)ti, &iter_exact);
          multiplier += ((ns > 0) ? (unsigned long long)ns : 1ULL) * iter * depth;
          J1N(trc, Dohash, ti);
        }
        if (multiplier > 1) progress /= multiplier;
      }
      double pct = (BruteForceTotal > 0) ? (100.0 * progress / BruteForceTotal) : 0;
      if (pct > 100.0) pct = 100.0;
      /* Generate sample candidate from completed progress position */
      char sample[MAX_MASK_POS + 1];
      unsigned long long sample_idx = progress < BruteForceTotal ? progress : BruteForceTotal - 1;
      int slen = mask_expand_into(sample_idx, MaskAppendPattern, MaskAppendLen, sample);
      sample[slen] = 0;
      /* ETA based on keyspace progress rate */
      double rate = (double)progress / wtime;
      double remaining = (rate > 0 && progress < BruteForceTotal) ?
                          (BruteForceTotal - progress) / rate : 0;
      char eta[64];
      if (!iter_exact && progress >= BruteForceTotal)
        snprintf(eta, sizeof(eta), "unknown");
      else
        format_eta(remaining, eta, sizeof(eta));
      /* Format progress with SI prefix */
      double dprog = (double)progress, dtotal = (double)BruteForceTotal;
      char *pmult = "", *tmult = "";
      format_rate(dprog, &dprog, &pmult);
      format_rate(dtotal, &dtotal, &tmult);
      fprintf(stderr, "Brute %s, %.1f%s/%.1f%s (%.1f%%), Found=%llu, %.2f%sh/s, '%s', ETA %s\n",
              Curfile, dprog, pmult, dtotal, tmult, pct, Totfound, hps, mult1, sample, eta);
    } else {
      { unsigned long long total_est = EstimatedTotalLines;
        /* Authoritative count first, projection only until it lands. The '~'
         * prefix below already marks the figure provisional while
         * AutoCountRequested && !AutoCountDone. */
        if (!total_est && AutoCountDone && AutoCountTotalLines > 0)
          total_est = AutoCountTotalLines;     /* settled: authoritative  */
        else if (!total_est && ProjectedTotalLines > 0)
          total_est = ProjectedTotalLines;     /* provisional: marked '~' */
        /* Deliberately NO fallback to AutoCountTotalLines while the count is
         * still running: mid-accumulation it is a partial multi-file sum, and
         * showing that as a total is the same category of error as the
         * doubling this change removes. */
        if (total_est > 0 && TotLines > 0) {
          long wq = peek_lock(WorkWaiting);
          char eta[64];
          double progress_frac;
          double remaining = 0;
          char prefix;
          /* Retirement-based ETA: each word is retired once all op-specific work
           * (rules x masks x salts x iters) is complete. Immune to type count
           * and iter shape. Bootstrap fallback uses hash-rate until first tick fires. */
          if (RetiredLines_rate > 0 && total_est > 0) {
            /* Bug 1 (2026-05-20): include LowSkip in numerator so progress/ETA
             * remain consistent with no-ETA branches (which display Fileline,
             * skip-inclusive) and so percent reaches 100% at completion under -w. */
            unsigned long long retired_inclusive = RetiredLines_now_snap + LowSkip;
            unsigned long long remaining_lines = (total_est > retired_inclusive)
                                                   ? total_est - retired_inclusive : 0;
            remaining = (RetiredLines_rate > 0) ? (double)remaining_lines / RetiredLines_rate : 0;
            progress_frac = (total_est > 0) ? (double)retired_inclusive / (double)total_est : 0;
            if (progress_frac > 1.0) progress_frac = 1.0;
            if (remaining <= 0 && wq > 0)
              snprintf(eta, sizeof(eta), "finishing");
            else
              format_eta(remaining, eta, sizeof(eta));
            prefix = ' ';
          } else {
            /* Bootstrap: hash-rate fallback until retirement events fire.
             * Compute effective work multiplier: salts x rules x masks x iters. */
            unsigned long long salt_mult = 0;
            { Word_t ti = 0; int trc;
              J1F(trc, Dohash, ti);
              while (trc) {
                unsigned int ns = LIVESALTS(ti);
                salt_mult += (ns > 0) ? ns : 1;
                J1N(trc, Dohash, ti);
              }
            }
            if (salt_mult < 1) salt_mult = 1;
            unsigned long long expected_total = total_est * salt_mult;
            if (Numrules > 1) expected_total *= Numrules;
            if (Maxiter > 1) expected_total *= Maxiter;
            if (MaskTotal > 1) expected_total *= MaskTotal;
            /* -1 = NO ESTIMATE YET, which is NOT the same as zero remaining.
             * format_eta(0) prints "done", so leaving remaining at 0 when the
             * estimate could not be made asserted that a run had finished at
             * exactly the moment the code had least information about it. An
             * -i 9999 run over 25,000 rules reported "ETA~done" for a job with
             * 16 days left, because expected_total was 6.3e16 and the 0.001
             * progress floor below was never cleared (A1131, 2026-10-30). The
             * floor itself is right -- an estimate off the first fraction of a
             * percent is worthless -- so the fix is to say so instead of
             * printing the one answer that is certainly wrong. */
            remaining = -1;
            if (expected_total > 0 && Tothash > 0) {
              progress_frac = (double)Tothash / (double)expected_total;
              if (progress_frac > 1.0) progress_frac = 1.0;
              double hps_raw = (double)Tothash / wtime;
              /* Three states, not two. remaining was overloaded: 0 meant BOTH
               * "no estimate" and "nothing left", and the wq>0 test below
               * turned the second into "finishing". Introducing -1 for the
               * first without pinning the second to 0 reported a COMPLETE run
               * as "unknown" -- caught by eta_check.c before it shipped. */
              if (progress_frac >= 1.0)
                remaining = 0;                    /* complete: nothing left */
              else if (progress_frac > 0.001 && hps_raw > 0)
                remaining = (double)(expected_total - Tothash) / hps_raw;
            } else {
              progress_frac = 0;
            }
            if (remaining < 0)
              snprintf(eta, sizeof(eta), "unknown");
            else if (remaining <= 0 && wq > 0)
              snprintf(eta, sizeof(eta), "finishing");
            else
              format_eta(remaining, eta, sizeof(eta));
            prefix = (AutoCountRequested && !AutoCountDone) ? '~' : '~';  /* bootstrap provisional */
          }
          /* First comfort-line: emit the consolidated end-of-init pin summary. */
          {
            static int _pin_summary_emitted = 0;
            if (!_pin_summary_emitted) {
              _pin_summary_emitted = 1;
              tsfprintf_pin_summary();
            }
          }
#ifdef GPU_ENABLED
          if (gpujob_available())
            tsfprintf(stderr, "Working on %s, w=%ld, gq=%d/%d, line %llu, %.2f lines/s (%.1f%%), %.2f%sh/s, Found=%llu, ETA%c%s\n",
                  Curfile, wq, gpujob_queue_depth(), gpujob_free_count(),
                  (RetiredLines_rate > 0 ? RetiredLines_now_snap : TotLines) + LowSkip,
                  RetiredLines_rate > 0 ? RetiredLines_rate : lps,
                  100.0*progress_frac, hps, mult1, Totfound, prefix, eta);
          else
#endif
            tsfprintf(stderr, "Working on %s, w=%ld, line %llu, %.2f lines/s (%.1f%%), %.2f%sh/s, Found=%llu, ETA%c%s\n",
                  Curfile, wq,
                  (RetiredLines_rate > 0 ? RetiredLines_now_snap : TotLines) + LowSkip,
                  RetiredLines_rate > 0 ? RetiredLines_rate : lps,
                  100.0*progress_frac, hps, mult1, Totfound, prefix, eta);
        } else {
#ifdef GPU_ENABLED
          if (gpujob_available())
            tsfprintf(stderr, "Working on %s, w=%ld, gq=%d/%d, line %llu, Found=%llu, %.2f%sh/s, %.2f%sc/s\n",
                  Curfile, peek_lock(WorkWaiting), gpujob_queue_depth(), gpujob_free_count(), Fileline, Totfound, hps, mult1, lps, mult);
          else
#endif
            tsfprintf(stderr, "Working on %s, w=%ld, line %llu, Found=%llu, %.2f%sh/s, %.2f%sc/s\n",
                  Curfile, peek_lock(WorkWaiting), Fileline, Totfound, hps, mult1, lps, mult);
        }
      }
    }
    fflush(stdout);
    fflush(stderr);
  }
}


#ifdef _SC_NPROCESSORS_ONLN
#ifdef MACOSX
#include <sys/sysctl.h>
int get_nprocs() {
    int numCPUs;
    size_t len = sizeof(numCPUs);
    int mib[2] = { CTL_HW, HW_NCPU };
    if (sysctl(mib, 2, &numCPUs, &len, NULL, 0))
      return 1;
    return numCPUs;
}
#else
int get_nprocs() {
    int numCPUs;
    numCPUs = sysconf(_SC_NPROCESSORS_ONLN);
    if (numCPUs <=0)
      numCPUs = 1;
    return(numCPUs);
}
#endif
#endif

#ifdef SPARC
#ifndef AIX
int get_nprocs() { return(1); }
#endif
#endif
#ifdef _WIN32
#include <windows.h>

int get_nprocs() {
    SYSTEM_INFO SysInfo;
    ZeroMemory(&SysInfo,sizeof(SYSTEM_INFO));
    GetSystemInfo(&SysInfo);
    return SysInfo.dwNumberOfProcessors;
}
#endif


int _dowildcard = -1; /* enable wildcard expansion for Windows */
#define MDXMAXPATHLEN 5000

Pvoid_t Files = NULL;
Word_t Filecount = 0;

void addfiles(char *path) {
  struct stat bstat;
  DIR *dir;
  struct dirent *result;
  char cur[MDXMAXPATHLEN * 3], *
  new;
  int bufpathlen, len;
  PWord_t PV;

  bufpathlen = strlen(path);
  if (bufpathlen > MDXMAXPATHLEN) return;

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
    /* Both strncpy calls in this block intentionally omit a trailing NUL —
     * the explicit cur[N] = 0 / cur[len+bufpathlen] = 0 below provides it.
     * Suppress -Wstringop-truncation for this directory-walk path. */
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
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
      if ((len + bufpathlen + 1) > MDXMAXPATHLEN) {
        fprintf(stderr, "New path too long at %s, ignored\n", result->d_name);
      } else {
        strncpy(cur + bufpathlen, result->d_name, len);
        cur[len + bufpathlen] = 0;
        addfiles(cur);
      }
    }
    closedir(dir);
#pragma GCC diagnostic pop
  } else {
    new = strdup(path);
    JLI(PV, Files, Filecount);
    *PV = (Word_t)
    new;
    Filecount++;
  }
}


char **dirprocess(int *argcp, char **argvp) {
int myc;
char **ret;
  Word_t filecount = 0;
  PWord_t PV;

  Filecount = 0;

  for (myc = 0; myc < *argcp; myc++)
    addfiles(argvp[myc]);
  fprintf(stderr, "Found %d files to process\n", (int) Filecount);
  ret = NULL;
  if (Filecount > 0) {
    ret = malloc_lock(sizeof(char **) * Filecount + 16, "dirprocess");
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




int strlccmp(char *line, char *match) {
  char c;
  while ((c = *line)) {
    if (isupper(c))
      c = tolower(c);
    if (c != *match)
      return(1);
    line++;
    match++;
    if (*match == 0)
      return(0);
  }
  return(1);
}

/* Write length header for hash already at inhashbuf+2, advance pointer,
 * and flush to worker thread if buffer is near full.
 * Caller must write hash data to inhashbuf+2 before calling. */
/* Hashes committed to the input set by the load in progress. Reset per file
 * by load_hash_file(), which reports it: a file that contributed nothing
 * has to say so out loud. */
static unsigned long HashesThisFile = 0;
static void commit_compact(int hlen) {
    HashesThisFile++;
#ifdef SPARC
    *inhashbuf++ = (hlen & 0xff); *inhashbuf++ = (hlen >> 8);
    inhashbuf += hlen;
#else
    *(unsigned short int *)inhashbuf = hlen;
    inhashbuf += hlen + 2;
#endif
    inhashcnt++;
    if ((inhashbuf - inhashstart) > (MAXADDHASHBUF - 258)) {
        possess(HashWaiting);
        wait_for(HashWaiting, TO_BE, 0);
        ToAdd.length = inhashcnt;
        ToAdd.work = (inhash) ? inhash2 : inhash1;
        twist(HashWaiting, BY, +1);
        inhash ^= 1;
        if (inhash)
            inhashstart = inhashbuf = inhash2;
        else
            inhashstart = inhashbuf = inhash1;
        inhashcnt = 0;
    }
}

/* Store a salt string (or substring) into Typesalt[job] Judy array */
/* Store a salt in the per-type Typesalt Judy.
 *
 * This silently truncated any salt of 256 bytes or more to 255. The JudyJ hash
 * key built by the loaders keeps the FULL salt, so for a long salt the two
 * disagreed: the compute path recovered a truncated salt from the snapshot,
 * hashed the wrong message, and produced a value that could never match the
 * stored key. IPMI2-SHA1 and IPMI2-MD5 carry RAKP blobs well over 255 bytes
 * (hashcat's 7300 vector is 316), so both were unfindable -- the hash loaded,
 * the salt loaded, and every candidate simply failed to match.
 *
 * Truncation is never the desired behaviour here, so the buffer now follows
 * the salt. The consumers are sized for it already: tsalt is MAXLINE and
 * saltpool is allocated from the measured Judy key lengths. */
/* Upper bound on a stored salt. The old limit was 256, which silently cut
 * IPMI2's 316-byte RAKP blob and the KRB5TGS23 blob. No known type needs
 * anywhere near this, but the ceiling is deliberately generous because an
 * as-yet-unsupported type could carry a longer salt, and a bound that is too
 * tight fails the same silent way the 256 did. Salts beyond this are refused
 * with a one-time diagnostic rather than truncated: a truncated salt hashes
 * the wrong message and reports "not found" on a hash that should have
 * cracked, which is indistinguishable from a genuine miss. */
#define MAXSALTLEN 4096

static int store_typesalt(int job, const char *src, int slen) {
    static int oversize_warned = 0;
    char stackts[256];
    char *ts = stackts;
    Word_t *PV;
    int ret = 0;
    if (slen < 0) slen = 0;
    if (slen > MAXSALTLEN) {
        if (!oversize_warned) {
            oversize_warned = 1;
            fprintf(stderr,
                "Warning: salt of %d bytes for type %d exceeds MAXSALTLEN %d; "
                "salt not loaded. Hashes using it cannot be found. "
                "Raise MAXSALTLEN in %s if this type is legitimate.\n",
                slen, job, MAXSALTLEN, __FILE__);
        }
        return 0;
    }
    if (slen >= (int)sizeof(stackts)) {
        ts = malloc(slen + 1);
        if (!ts) {
            fprintf(stderr, "%s:%d: store_typesalt: malloc of %d bytes failed: %s\n",
                    __FILE__, __LINE__, slen + 1, strerror(errno));
            exit(1);
        }
    }
    memcpy(ts, src, slen);
    ts[slen] = 0;
    JSLI(PV, TYPESALT(job), (unsigned char *)ts);
    if (PV) { if ((*PV)++ == 0) ret = 1; }
    if (ts != stackts) free(ts);
    return ret;
}

/* Number of distinct salts in a type's Typesalt. */
static long typesalt_entries(int job) {
    unsigned char k[MAXSALTLEN + 1];
    Word_t *PV;
    long n = 0;
    k[0] = 0;
    JSLF(PV, TYPESALT(job), k);
    while (PV) { n++; JSLN(PV, TYPESALT(job), k); }
    return n;
}

/* $5$/$6$ file a full hash in Typesalt as the whole line, and a bare
 * setting as the setting alone, so the same salt supplied both ways would
 * be generated twice under -z.  Given a full line, drop the bare setting it
 * already carries. */
static void drop_bare_setting(int job, const char *full) {
    char set[256];
    const char *e = strrchr(full, '$');
    int n, rc;
    if (!e) return;
    n = (int)(e - full) + 1;
    if (n >= (int)sizeof(set) || full[n] == 0) return;
    memcpy(set, full, (size_t)n);
    set[n] = 0;
    JSLD(rc, TYPESALT(job), (unsigned char *)set);
    (void)rc;
}

/* Load structured hash formats from a file into Judy arrays.
 * When pDoload is NULL, operates as standard -J loading.
 * When pDoload is non-NULL, also handles MD5SALT hashes gated by *pDoload. */
/* Types that consume the phpBB3 / phpass wrapper "$H$Csalt8chr" + 22-char hash.
 *
 * Each member keeps its OWN Typesalt entries, filed by the loader for whichever
 * members the -M load set names. Adding a third -- a PHPBB3SHA1, say -- is one
 * entry here plus its compute arm; the loader, the salt accounting and the -z
 * default-salt path all follow from this table.
 *
 * The previous shape filed every hash under JOB_PHPBB3 and had the bootstrap
 * COPY the keys into Typesalt[JOB_PHPBB3MD5] with a hardcoded count of 1. The
 * loader keys on 12 characters, which is the salt and not the whole line, so
 * one entry serves every hash sharing that salt; a count of 1 retired the salt
 * after the first match and the rest were silently never tried. PHPBB3MD5
 * returned 11 of the fixture's 121 lines, one per salt. store_typesalt()
 * counts as it inserts and is the only place that knows the real number, which
 * is why the count is no longer produced anywhere else.
 */
static const int Phpbb3Family[] = { JOB_PHPBB3, JOB_PHPBB3MD5 };
#define PHPBB3FAMILY_N ((int)(sizeof(Phpbb3Family) / sizeof(Phpbb3Family[0])))

/* $1$ md5crypt family.  e511 MD5CRYPT and e1061 MD5CRYPTMD5 consume the
 * identical stored format, and the e1061 compute arm falls through to the
 * e511 arm, exactly as e537 does into e455.  Filing the salt under one
 * member and copying it later is what invented the wrong count for the
 * phpBB3 pair, so this list is walked at load time and each SELECTED
 * member is counted on its own. */
static const int Md5cryptFamily[] = { JOB_MD5CRYPT, JOB_MD5CRYPTMD5 };
#define MD5CRYPTFAMILY_N ((int)(sizeof(Md5cryptFamily) / sizeof(Md5cryptFamily[0])))

/* Stored forms whose own grammar cannot contain a ':'.
 *
 * A -F file is very often mdxfind's OWN output fed back in -- the
 * "<stored form>:<plaintext>" of a founds file -- and for most types that
 * cannot be read back, because the -F form of a salted hash is
 * "<hash>:<salt>" and NOTHING in the line distinguishes a salt from a
 * recovered plaintext.  In "bc53abf8154e5967bef8da502fc93025:~}::password"
 * the salt is "~}:" and only the operator knows it.  That ambiguity is why
 * the -F parser takes the whole line, and it is not solvable here; those
 * lines are still handed to the recognizers untouched.
 *
 * It IS solvable for a hash that is self-delimiting -- phpBB3, the crypt(3)
 * families, the "{...}" and "$name$..." wrappers -- because their alphabets
 * and separators exclude the colon outright.  For those the first colon
 * cannot be part of the hash whatever follows it, so the line is cut there
 * and the recognizers see the shape they measure.  Requested by Waffle
 * 2026-10-17 after "$H$9Pno.ZQTcecz9lrMxdq6X9cLAWz4SX1:password" was refused
 * by -F while hashpipe read the identical line without difficulty.
 *
 * Membership is a property of the FORMAT, and the test for it is exact: run
 * the type through -z and look at what it emits.  If the stored form it
 * prints contains no colon, the prefix belongs here.  Two kinds must stay
 * OUT, and both are silent failures if admitted:
 *
 *   - a form that legitimately carries a trailing field.  "$BLAKE2$<hex>"
 *     takes ":<salt>" for the six BLAKE2 PASSSALT/SALTPASS types, and
 *     e1059 BCRYPTMD5PASSSALT stores "<60-char MCF>:<site salt>".  Cutting
 *     would throw the salt away before its recognizer ever saw it, which is
 *     why bcrypt is admitted only when e1059 is not in the selection.
 *   - a form whose grammar USES the colon: "sha256:1000:<salt>:<digest>",
 *     "SCRYPT:...", "$7z$...:KEY=...", NETNTLM, MURMUR3, IPMI2, CISCOASA,
 *     the QNX "@m,N@hash@salt" shapes' siblings, SAP, and the plain
 *     "<hex>:<salt>" channel itself.
 *
 * Known limit: a stored form that embeds a USERNAME ($krb5pa$, $as400$,
 * $mongodb$, $racf$ ...) is cut at the first colon like any other, so a
 * username containing a colon loses its tail.  Such a line is already
 * ambiguous to every reader of a founds file, including hashpipe.
 *
 * Adding a format is one line in this table.
 */
static const char * const Colonfree[] = {
  "$1$",              /* md5crypt, MD5CRYPTMD5                            */
  "$3$", "$4$",       /* DragonFly BSD 3 and 4, 32- and 64-bit            */
  "$B$",              /* MediaWiki                                        */
  "$H$", "$P$",       /* phpass: phpBB3, PHPBB3MD5, WordPress             */
  "$S$",              /* Drupal 7                                         */
  "$h$",              /* H3C                                              */
  "$y$", "$gy$",      /* yescrypt, gost-yescrypt                          */
  "$md5$",            /* SUNMD5                                           */
  "$sha1$",           /* SHA1CRYPT                                        */
  "$apr1$",           /* Apache apr1                                      */
  "$wp$",             /* WPBCRYPT                                         */
  "$ASN$",            /* Apple Secure Notes                               */
  "$DCC2$",           /* MS Domain Cached Credentials 2                   */
  "$vbk$", "$vmx$",   /* Veeam backup, VMware VMX                         */
  "$as400$",          /* AS/400 DES and SSHA1                             */
  "$cmiyc$",          /* CMIYC                                            */
  "$fvde$",           /* FileVault / APFS                                 */
  "$iwork$",          /* Apple iWork                                      */
  "$keychain$",       /* Apple keychain                                   */
  "$krb5pa$", "$krb5tgs$", "$krb5db$",
  "$mysql$", "$mysqlna$", "$postgres$",
  "$mongodb$", "$mongodb-scram$",
  "$racf$", "$racf-kdfaes$",
  "$sspr$", "$sntp-ms$",
  "$ansible$", "$argon2",
  "$axcrypt$", "$axcrypt_sha1$",
  "$bitwarden$", "$ecryptfs$", "$episerver$",
  "$gost94hash$", "$gost12256hash$", "$gost12512hash$",
  "$solarwinds$", "$tacacs-plus$", "$telegram$",
  "$AWS-Sig-v4$", "$MSONLINEACCOUNT$",
  "{CRAM-MD5}", "{PBKDF2_SHA256}", "{PKCS5S2}",
  "{SSHA}", "{SSHA256}", "{SSHA512}",
  "{smd5}", "{ssha1}", "{ssha256}", "{ssha512}",
  "(",                /* Lotus Domino 6 and 8, "(Gxxx)" / "(Hxxx)"        */
  "@m,", "@s,", "@S,",/* QNX md5, sha256, sha512                          */
  "SQLCIPHER*",       /* SQLCipher                                        */
  "pbkdf2(",          /* web2py                                           */
  "v1;PPH1_",         /* Azure AD Connect sync                            */
  NULL
};

/* Where does the stored form at the start of this line end?  0 means "not a
 * form whose end can be found", which is the normal answer and leaves the
 * line alone.  See Colonfree[] above for what qualifies and why. */
static int stored_form_len(const char *s, int len, const char *lf)
{
  const char * const *pp;
  const char *c;

  /* One test throws out every plain-hex line, which is the bulk case and the
   * one that must not be touched: "<hex>:<salt>" is the ordinary salted -F
   * form.  It also throws out the hex-bodied wrappers -- MSSQL and Sybase
   * "0x...", Oracle, NetScaler, CiscoISE -- which are self-delimiting but
   * indistinguishable from a digest here, so they belong in their own
   * recognizers where lf[] already says the type was asked for.  The same
   * goes for the bare base64 and itoa64 tokens: DESCRYPT, BSDICRYPT,
   * CISCOPIX, PROGRESSENCODE, UMBRACO, PEOPLESOFT, JWT and the rest. */
  if (isxdigit((unsigned char)s[0])) return 0;

  if (s[0] == '$' && s[1] == '2') {
    /* bcrypt and the types that share its MCF.  e1059 BCRYPTMD5PASSSALT
     * reads a site salt out of the field AFTER the MCF, so when it is in
     * the selection this line's tail is data, not a plaintext. */
    if (lf[JOB_BCRYPTMD5PASSSALT]) return 0;
  } else {
    for (pp = Colonfree; *pp; pp++)
      if (strncmp(s, *pp, strlen(*pp)) == 0) break;
    if (*pp == NULL) return 0;
  }

  c = memchr(s, ':', (size_t)len);
  return c ? (int)(c - s) : 0;
}

/* How much has load_hash_file filed so far?  Compared across one line to
 * tell "nothing loaded it" from "something did", for the trailing-field
 * retry.  HashesThisFile counts the plain-hex commits. */
static unsigned long claim_mark(const int *found, const int *sel, int nsel)
{
  unsigned long t = HashesThisFile;
  int i;
  for (i = 0; i < nsel; i++) t += (unsigned long)found[sel[i]];
  return t;
}

/* Count one salt for this file's load receipt, once per distinct value.
 * For the crypt types whose stored line is filed whole (DESCRYPT, $5$, $6$),
 * the salt reaches Typesalt only at start-up, so the loader has no
 * store_typesalt() return to count; this set stands in for it.  It never
 * touches Typesalt, so a salt's reference count is exactly what it was.
 * The key is the type in two non-zero bytes, then the salt. */
static Pvoid_t RcptSalts;
static void rcpt_salt(unsigned long *saltloaded, int job, const char *s, int n)
{
  char k[264];
  Word_t *PV;
  if (n <= 0 || n > 256) return;
  k[0] = (char)(1 + (job & 0x7f));
  k[1] = (char)(1 + ((job >> 7) & 0x7f));
  memcpy(k + 2, s, (size_t)n);
  k[n + 2] = 0;
  JSLI(PV, RcptSalts, (unsigned char *)k);
  if (PV && (*PV)++ == 0) saltloaded[job]++;
}

/* A crypt(3) setting with no hash after it: $5$ or $6$, an optional
 * rounds=N$ field, a salt of 1 to 64 characters, and an optional closing
 * $.  It is how a salt is supplied for these types, since their -F form is
 * the whole MCF.  Returns the length of the normalised setting written to
 * out, which always ends in $, or 0 when s is not a bare setting. */
static int crypt_setting_only(const char *s, int len, char *out, int outsz)
{
  int p = 3, salt0, n;
  if (len < 4 || s[0] != '$' || (s[1] != '5' && s[1] != '6') || s[2] != '$')
    return 0;
  if (len > 10 && strncmp(s + 3, "rounds=", 7) == 0) {
    p = 10;
    while (p < len && isdigit((unsigned char)s[p])) p++;
    if (p == 10 || p >= len || s[p] != '$') return 0;
    p++;
  }
  salt0 = p;
  while (p < len && s[p] != '$') {
    if ((signed char)s[p] <= ' ' || s[p] == ':' || (signed char)s[p] > 'z')
      return 0;
    p++;
  }
  if (p == salt0 || p - salt0 > 64) return 0;  /* crypt() uses the first 16 */
  if (p < len && p + 1 != len) return 0;   /* a hash follows: not bare */
  n = p + 1;
  if (n >= outsz) return 0;
  memcpy(out, s, (size_t)p);
  out[p] = '$';
  out[n] = 0;
  return n;
}

/* A fixed-width, all-hex stored form followed by ":<anything>".
 *
 * Oracle 11 and 12, Android FDE and the NetScaler forms are one hex blob
 * that carries its own salt.  Followed by a colon, the plain-hex loader
 * claims the line as "<hex>:<salt>" before the trailing-field retry ever
 * sees it, so "<blob>:bad-password" loads a digest that cannot match.  The
 * blob's width is the type's own, so when that type is selected and the
 * line is exactly n hex digits and then a colon, the colon ends the hash
 * and the rest is ignored.  first, when not 0, is the required lead char. */
static void hexform_cut(char *line, int *len, int n, char first)
{
  int i;
  if (*len <= n || line[n] != ':') return;
  if (first && line[0] != first) return;
  for (i = 0; i < n; i++)
    if (!isxdigit((unsigned char)line[i])) return;
  line[n] = 0;
  *len = n;
}

static void load_hash_file(gzFile gi, const char *filename, Pvoid_t *pDoload,
                           int hex_only) {
  char line[MAXLINE + 16], *cur, *s;
  int len, x, y, cryptlen, any;
  Word_t *PV, RC;
  static const int jsigpos[] = {0, 6, 12, 17, 23, 29};
  static const char jsig[] = "nrcstn";
  int Foundcnt[JOB_DONE];
  static char rt_orig[MAXLINE + 16];  /* trailing-field retry, see loop end */
  int rt_cut = 0, rt_tries = 0, rt_nsel = 0, rt_sel[JOB_DONE];
  unsigned long rt_mark = 0;
  char lf[JOB_DONE],LoadStructured = 0, salttmp[MAXLINE+16];
  int Dosalt[JOB_DONE],Douser[JOB_DONE];
  int Dosaltcnt = 0, Dousercnt = 0;
  char LoadHex = 0, LoadUser = 0, LoadSalt = 0;
  unsigned long Saltloaded[JOB_DONE], Userloaded[JOB_DONE];
  Word_t ti;

  HashesThisFile = 0;
  memset(Foundcnt, 0, sizeof(Foundcnt));
  memset(lf, 0, sizeof(lf));
  memset(Saltloaded, 0, sizeof(Saltloaded));
  memset(Userloaded, 0, sizeof(Userloaded));
  any = 0;

  /* Build lf[] (Load Format) array: which hash types are we looking for?
   * If -F was used with -h (not -M), Doload is empty; fall back to Dohash. */
  if (pDoload && *pDoload == NULL && Dohash != NULL)
    pDoload = &Dohash;
  if (pDoload) {
    ti = 0;
    J1F(RC, *pDoload, ti);
    while (RC) {
      if (ti < JOB_DONE) {
	any = 1;
        lf[ti] = 1;
        if (TYPEOPTS(ti) & TYPEOPT_NEEDSF)
          LoadHex = 1;
        if (TYPEOPTS(ti) & TYPEOPT_NEEDUSER) {
          LoadUser = 1;
	  Douser[Dousercnt++] = ti;
	}
        if ((TYPEOPTS(ti) & TYPEOPT_NEEDSALT)) {
          LoadSalt = 1;
	  Dosalt[Dosaltcnt++] = ti;
	}
	if (lf[ti] && !(TYPEOPTS(ti) & TYPEOPT_NEEDSF))
	  LoadStructured = 1;
      }
      J1N(RC, *pDoload, ti);
    }
  } else {
    ti = 0;
    J1F(RC, Dohash, ti);
    while (RC) {
      if (ti < JOB_DONE) {
	any = 1;
        lf[ti] = 1;
        if (TYPEOPTS(ti) & TYPEOPT_NEEDSF)
          LoadHex = 1;
        if (TYPEOPTS(ti) & TYPEOPT_NEEDUSER) {
          LoadUser = 1;
	  Douser[Dousercnt++] = ti;
	}
        if ((TYPEOPTS(ti) & TYPEOPT_NEEDSALT)) {
          LoadSalt = 1;
	  Dosalt[Dosaltcnt++] = ti;
	}
	if (lf[ti] && !(TYPEOPTS(ti) & TYPEOPT_NEEDSF))
	  LoadStructured = 1;
      }
      J1N(RC, Dohash, ti);
    }
  }

  /* Four TYPEOPT_NEEDSF types also have a stored form that is NOT plain
   * hex: PROGRESSENCODE (16 letters) and the unsalted $BLAKE2$<hex> forms.
   * Their recognizers live in the slow path. Selected alone, NEEDSF left
   * LoadStructured at 0, the hex-only fast path ran, and every one of
   * their own -z lines loaded nothing. hex_only (-f) never runs the
   * recognizers, so the fast path stays correct there. */
  if (!hex_only && (lf[JOB_PROGRESSENCODE] || lf[JOB_BLAKE2B512] ||
                    lf[JOB_BLAKE2S256] || lf[JOB_BLAKE2B256]))
    LoadStructured = 1;

  if (!any) {
    fprintf(stderr, "No hash types selected for %s\n", filename);
    return;
  }


  /* Determine if any structured (non-hex) types are selected */
  { 

    /* Fast path: hex-only types with compact table available.
     * Skip fast path when LoadUser + pDoload (-F/-J): slow path
     * extracts :user suffixes from hex hashes. */
    if (LoadHex && !LoadStructured && !(LoadSalt && pDoload) && !(LoadUser && pDoload) && inhashbuf) {
      char *readbuf;
      struct LineInfo *readindex;
      unsigned int Linecount, curline;
      while ((Linecount = cacheline(gi, &readbuf, &readindex))) {
        for (curline = 0; curline < Linecount; curline++) {
          char *_fl = &readbuf[readindex[curline].offset];
          int hlen = get32(_fl, inhashbuf + 2, 256);
          if (hlen >= 8 && hlen <= 255) {
            commit_compact(hlen);
            /* Same DCC2 filing as the slow path below. This is the branch a
             * plain hex list actually takes under -f, so omitting it here left
             * -f -u loading the digests, printing a receipt, and matching
             * nothing -- while -F worked. See the comment at the slow-path
             * copy for why DCC2 needs its own JudyJ at all. */
            if (hlen == 16 && lf[JOB_DCC2]) {
              char _dh[33];
              int _dy;
              for (_dy = 0; _dy < 32; _dy++)
                _dh[_dy] = tolower((unsigned char)_fl[_dy]);
              _dh[32] = 0;
              JSLI(PV, JUDYJ(JOB_DCC2), (unsigned char *)_dh);
            }
          }
        }
      }
      /* Not `return`: the receipt below has to fire on THIS path above all --
       * it is the plain-hex fast path, the one a salted hash fed to -f takes,
       * and it used to leave without a word. */
      goto load_done;
    }
  }

  /* Slow path: structured format handlers + hex fallthrough */
  /* Only a selected type can be filed, so only those counters move.  A few
   * recognizers (YAF-SHA1) gate on Dohash rather than lf[], so take both. */
  for (x = 0; x < JOB_DONE; x++) {
    if (lf[x]) { rt_sel[rt_nsel++] = x; continue; }
    J1T(RC, Dohash, x);
    if (RC) rt_sel[rt_nsel++] = x;
  }
  while (gzgets(gi, line, MAXLINE)) {
    cur = strchr(line, 13);
    if (cur) *cur = 0;
    cur = strchr(line, 10);
    if (cur) *cur = 0;
    len = mystrlen(line);
    /* -f is the PLAIN HEX channel: parse hex from the start of the line and
     * DISCARD the remainder. Any non-hex character terminates the hash, and
     * what follows it is not a salt, not a userid and not structure.
     *
     * Without this the slow path below sees the whole line. The plain-hex
     * fast path further up is gated on !LoadStructured, so selecting ANY
     * structured type diverts a -f file into the slow path and every format
     * recognizer gets a look at it. Reported by Waffle 2026-09-16 and
     * replicated with
     *     echo 5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8:password >r.txt
     *     mdxfind -h "^SHA1|^MD4|^NTLM|^SMF" -h '!salt,!user,!md5x,!sha1x' \
     *             -f r.txt
     * which loaded "password" as a SALT for SHA1-8TRACK, MD4DESCRYPT,
     * SHA1DESCRYPT, MD4UTF16DESCRYPT, SHA1-S1PS2 and SHA1CRYPT, and as a
     * USERID for SMF and SHA1-HMAC-MD5: "1 hashes, 6 salts, 2 users loaded"
     * where it must read "1 hashes, 0 salts, 0 users loaded".
     *
     * -F and -J pass hex_only = 0 and are untouched; a type whose Options
     * column says F still does not load from -f, which is the documented
     * contract. */
    if (hex_only) {
      int _hx = 0;
      while (_hx < len &&
             ((line[_hx] >= '0' && line[_hx] <= '9') ||
              (line[_hx] >= 'a' && line[_hx] <= 'f') ||
              (line[_hx] >= 'A' && line[_hx] <= 'F')))
        _hx++;
      if (_hx < len) { line[_hx] = 0; len = _hx; }
      if (len == 0) continue;
    }
    /* Arm the trailing-field retry at the bottom of this loop.  See there. */
    rt_cut = 0;
    if (!hex_only && memchr(line, ':', (size_t)len)) {
      memcpy(rt_orig, line, (size_t)len + 1);
      rt_cut = len;
      rt_tries = 0;
      rt_mark = claim_mark(Foundcnt, rt_sel, rt_nsel);
    }
  reparse:
    /* mdxfind's own output read back: "<stored form>:<plaintext>".  Where the
     * format's grammar excludes the colon, cut at the first one so the
     * recognizers below see the shape they measure.  -f is exempt: it has
     * already truncated at the first non-hex character above.
     *
     * Done HERE rather than inside each recognizer because the two ways this
     * fails are different and both are silent.  phpBB3, md5crypt and WBB4
     * MEASURE the hash and reject the longer line, so nothing loads at all,
     * while APR1, Drupal7 and the "$name$..." wrappers accept it and file a
     * key with ":password" welded on -- a hash that can never match, loaded
     * with a clean count and no complaint.  One cut above all of them covers
     * both shapes, and a line whose format is not in the table is passed on
     * untouched.  See Colonfree[] and stored_form_len().  */
    if (!hex_only) {
      int sf = stored_form_len(line, len, lf);
      if (sf > 0) { line[sf] = 0; len = sf; }
    }
    /* AIX {smd5}salt$hash (6-char prefix + salt + $ + 22 encoded) */
    if (lf[JOB_AIXMD5] && len > 6 && strncmp(line, "{smd5}", 6) == 0) {
      char *p = line + 6; /* points to salt$hash */
      char *dollar = strchr(p, '$');
      if (dollar && (dollar - p) >= 1 && (dollar - p) <= 8 &&
          mystrlen(dollar + 1) == 22) {
        JSLI(PV, JUDYJ(JOB_AIXMD5), (unsigned char *)p);
        Foundcnt[JOB_AIXMD5]++;
        continue;
      }
    }
    /* AIX {ssha1}NN$salt$hash */
    if (lf[JOB_AIXSHA1] && len > 7 && strncmp(line, "{ssha1}", 7) == 0) {
      char *p = line + 7; /* points to NN$salt$hash */
      JSLI(PV, JUDYJ(JOB_AIXSHA1), (unsigned char *)p);
      char *last = strrchr(p, '$');
      if (last && last > p) {
        int slen = last - p;
        if (slen < 79) {
          memcpy(salttmp, p, slen);
          salttmp[slen] = 0;
          JSLI(PV, TYPESALT(JOB_AIXSHA1), (unsigned char *)salttmp);
	  if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_AIXSHA1]++;} }
        }
      }
      Foundcnt[JOB_AIXSHA1]++;
      continue;
    }
    /* AIX {ssha256}NN$salt$hash */
    if (lf[JOB_AIXSHA256] && len > 9 && strncmp(line, "{ssha256}", 9) == 0) {
      char *p = line + 9;
      JSLI(PV, JUDYJ(JOB_AIXSHA256), (unsigned char *)p);
      char *last = strrchr(p, '$');
      if (last && last > p) {
        int slen = last - p;
        if (slen < 79) {
          memcpy(salttmp, p, slen);
          salttmp[slen] = 0;
          JSLI(PV, TYPESALT(JOB_AIXSHA256), (unsigned char *)salttmp);
	  if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_AIXSHA256]++;} }
        }
      }
      Foundcnt[JOB_AIXSHA256]++;
      continue;
    }
    /* AIX {ssha512}NN$salt$hash */
    if (lf[JOB_AIXSHA512] && len > 9 && strncmp(line, "{ssha512}", 9) == 0) {
      char *p = line + 9;
      JSLI(PV, JUDYJ(JOB_AIXSHA512), (unsigned char *)p);
      char *last = strrchr(p, '$');
      if (last && last > p) {
        int slen = last - p;
        if (slen < 79) {
          memcpy(salttmp, p, slen);
          salttmp[slen] = 0;
          JSLI(PV, TYPESALT(JOB_AIXSHA512), (unsigned char *)salttmp);
	  if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_AIXSHA512]++;} }
        }
      }
      Foundcnt[JOB_AIXSHA512]++;
      continue;
    }
    /* Juniper SSG: 30-char hash + colon + username */
    s = strchr(line, ':');
    if (lf[JOB_JUNIPERSSG] && s && (s - line) == 30) {
      int ok = 1, si;
      for (si = 0; si < 6; si++) {
        if (line[jsigpos[si]] != jsig[si]) { ok = 0; break; }
      }
      if (ok) {
        *s = 0;
        JSLI(PV, JUDYJ(JOB_JUNIPERSSG), (unsigned char *)line);
        *s = ':';
        if (s[1] && user_canon(s + 1, (int)strlen(s + 1), Ucanon) > 0)
          JSLI(PV, UserArray, (unsigned char *)Ucanon);
        Foundcnt[JOB_JUNIPERSSG]++;
        continue;
      }
    }
    /* JudyJ[JOB_PEOPLESOFT]: 28-char base64 ending with '=' */
    if (lf[JOB_PEOPLESOFT] && len == 28 && line[27] == '=') {
      int ok = 1;
      for (y = 0; y < 27; y++) {
        if (!((line[y] >= 'A' && line[y] <= 'Z') ||
              (line[y] >= 'a' && line[y] <= 'z') ||
              (line[y] >= '0' && line[y] <= '9') ||
              line[y] == '+' || line[y] == '/')) {
          ok = 0; break;
        }
      }
      if (ok) {
        JSLI(PV, JUDYJ(JOB_PEOPLESOFT), (unsigned char *)line);
        Foundcnt[JOB_PEOPLESOFT]++;
        continue;
      }
    }
    /* Juniper IVE: 104-char base64 ending with == -> AES decrypt -> md5crypt */
    if (lf[JOB_JUNIPERIVE] && len == 104 && line[102] == '=' && line[103] == '=') {
      unsigned char jdecoded[80], jiv[16], jplain[80];
      int jdlen, jcryptlen;
      static const unsigned char jive_key[16] = {
        0xa6, 0x70, 0x7a, 0x7e, 0x8d, 0xf9, 0x10, 0x59,
        0xde, 0xa7, 0x0a, 0xe5, 0x2f, 0x9c, 0x24, 0x42
      };
      AES_KEY jaeskey;
      jdlen = b64_decode(line, (char *)jdecoded, &jcryptlen);
      if (jdlen >= 76) {
        memcpy(jiv, jdecoded, 12);
        memset(jiv + 12, 0, 4);
        AES_set_decrypt_key(jive_key, 128, &jaeskey);
        AES_cbc_encrypt(jdecoded + 12, jplain, 64, &jaeskey, jiv, AES_DECRYPT);
        /* find $1$ md5crypt hash in decrypted data */
        for (x = 0; x < 61; x++) {
          if (jplain[x] == '$' && jplain[x+1] == '1' && jplain[x+2] == '$') {
            char *md5h = (char *)&jplain[x];
            for (y = 0; md5h[y] && (signed char)md5h[y] > ' ' && y < 34; y++)
              ;
            if (y >= 26) {
              md5h[y] = 0;
              JSLI(PV, JUDYJ(JOB_JUNIPERIVE), (unsigned char *)md5h);
              /* store original base64 line keyed by md5crypt hash */
              JSLI(PV, JuniperIVE, (unsigned char *)md5h);
              if (*PV == 0) {
                char *orig = malloc(105);
                if (!orig) { fprintf(stderr, "Out of memory\n"); exit(1); }
                memcpy(orig, line, 104);
                orig[104] = 0;
                *PV = (Word_t)orig;
              }
              Foundcnt[JOB_JUNIPERIVE]++;
            }
            break;
          }
        }
      }
      continue;
    }
    /* JudyJ[JOB_CISCO4] (IOS type 4): 43 phpitoa64 chars, SHA256 */
    if (lf[JOB_CISCO4] && len == 43) {
      int c4ok = 1;
      for (y = 0; y < 43; y++) {
        if (!strchr(phpitoa64, line[y])) { c4ok = 0; break; }
      }
      if (c4ok) {
        JSLI(PV, JUDYJ(JOB_CISCO4), (unsigned char *)line);
        Foundcnt[JOB_CISCO4]++;
        continue;
      }
    }
    /* Cisco ASA: 16 phpitoa64 chars + ":" + 1-4 char salt */
    if (lf[JOB_CISCOASA] && len >= 18 && len <= 21 && line[16] == ':') {
      int pix_ok = 1;
      for (y = 0; y < 16; y++) {
        if (!strchr(phpitoa64, line[y])) { pix_ok = 0; break; }
      }
      if (pix_ok) {
        int slen = len - 17;
        memcpy(salttmp, line + 17, slen);
        salttmp[slen] = 0;
	JSLI(PV, TYPESALT(JOB_CISCOASA), (unsigned char *)salttmp);
	if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_CISCOASA]++;} }
        JSLI(PV, JUDYJ(JOB_CISCOASA), (unsigned char *)line);
        Foundcnt[JOB_CISCOASA]++;
        continue;
      }
    }
    /* Cisco PIX: exactly 16 phpitoa64 chars, no colon */
    if (lf[JOB_CISCOPIX] && len == 16 && !strchr(line, ':')) {
      int pix_ok = 1;
      for (y = 0; y < 16; y++) {
        if (!strchr(phpitoa64, line[y])) { pix_ok = 0; break; }
      }
      if (pix_ok) {
        JSLI(PV, JUDYJ(JOB_CISCOPIX), (unsigned char *)line);
        Foundcnt[JOB_CISCOPIX]++;
        continue;
      }
    }
    /* JudyJ[JOB_WBB3] (8400) in -J: 40hex:40hex = 81 chars — must be before IPMI2_SHA1 */
    /*
     * User-defined types. The format parser offers each SELECTED user type the
     * line in turn and the first whose layout matches claims it. A type that was
     * NOT selected never sees the line: an unselected type must not quietly absorb
     * input meant for something else.
     *
     * Layout comes from the type's "form =" declaration when it has one. Without
     * one the default shapes apply, chosen by which slots the expression uses:
     *   no slots      digest
     *   salt          digest:salt
     *   user          digest:user
     *   salt + user   digest:salt:user
     */
    {
      int _uc = userdef_count(), _ui, _claimed = 0;
      for (_ui = 0; _ui < _uc && !_claimed; _ui++) {
        int _uop = (int)user_id_op(_ui);
        struct userdef_type *_ut;
        const char *_dg = NULL, *_sl = NULL, *_us = NULL;
        int _dgl = 0, _sll = 0, _usl = 0;

        if (_uop >= JOB_DONE || !lf[_uop]) continue;   /* not selected: ignore */
        _ut = userdef_get(_uop);
        if (!_ut) continue;

        if (_ut->form.npieces) {
          if (!userdef_form_split(&_ut->form, line, len,
                                  &_dg, &_dgl, &_sl, &_sll, &_us, &_usl))
            continue;                                  /* line is not this form */
        } else {
          char *_c1 = strchr(line, ':');
          char *_c2 = _c1 ? strchr(_c1 + 1, ':') : NULL;
          int _wants_salt = (_ut->slot_mask & USERDEF_SLOT_SALT) != 0;
          int _wants_user = (_ut->slot_mask & USERDEF_SLOT_USER) != 0;
          _dg = line;
          if (!_wants_salt && !_wants_user) {
            if (_c1) continue;
            _dgl = len;
          } else if (_wants_salt && _wants_user) {
            if (!_c1 || !_c2) continue;
            _dgl = (int)(_c1 - line);
            _sl = _c1 + 1; _sll = (int)(_c2 - _c1 - 1);
            _us = _c2 + 1; _usl = len - (int)(_c2 + 1 - line);
          } else {
            if (!_c1 || _c2) continue;
            _dgl = (int)(_c1 - line);
            if (_wants_salt) { _sl = _c1 + 1; _sll = len - _dgl - 1; }
            else             { _us = _c1 + 1; _usl = len - _dgl - 1; }
          }
        }
        if (_dgl != _ut->diglen_hex) continue;

        JSLI(PV, JUDYJ(_uop), (unsigned char *)line);
        if (_sl && _sll > 0)
          Saltloaded[_uop] += store_typesalt(_uop, (char *)_sl, _sll);
        if (_us && _usl > 0 && user_canon((char *)_us, _usl, Ucanon) > 0) {
          Word_t *_UV;
          JSLI(_UV, TYPEUSER(_uop), (unsigned char *)Ucanon);
          if (_UV) { if ((*_UV)++ == 0) Userloaded[_uop]++; }
        }
        Foundcnt[_uop]++;
        _claimed = 1;
      }
      if (_claimed) continue;
    }

    /* ORACLE11: Oracle 11g stored form -- a 40-hex SHA1 digest followed by a
     * 20-hex salt, concatenated into ONE 60-character field. No other type in
     * the catalogue has a 30-byte digest, so the width identifies the format on
     * its own and no wrapper is needed to disambiguate it.
     *
     * The construction is the same digest e834 SHA1PASSHEXSALT computes, but
     * e834 takes hash and salt as separate fields. This type owns Oracle's
     * stored form: it parses the concatenation on input and reproduces it on
     * output, so a found line can be fed straight back to Oracle or to us.
     * Oracle stores uppercase, and so does the JudyJ key built here, matching
     * what procjob assembles with prmd5UC. */
    if (lf[JOB_ORACLE11]) hexform_cut(line, &len, 60, 0);
    if (lf[JOB_ORACLE11] && len == 60 && !strchr(line, ':')) {
      int ook = 1;
      for (y = 0; y < 60; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { ook = 0; break; }
      }
      if (ook) {
        for (y = 0; y < 60; y++)
          salttmp[y] = toupper((unsigned char)line[y]);
        salttmp[60] = 0;
        JSLI(PV, JUDYJ(JOB_ORACLE11), (unsigned char *)salttmp);
        Saltloaded[JOB_ORACLE11] += store_typesalt(JOB_ORACLE11, salttmp + 40, 20);
        Foundcnt[JOB_ORACLE11]++;
        continue;
      }
    }
    if (lf[JOB_WBB3] && len == 81 && line[40] == ':') {
      int wok = 1;
      for (y = 0; y < 40; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { wok = 0; break; }
      }
      if (wok) {
        for (y = 41; y < 81; y++) {
          if (trhex[(unsigned char)line[y]] > 15) { wok = 0; break; }
        }
      }
      if (wok) {
        JSLI(PV, JUDYJ(JOB_WBB3), (unsigned char *)line);
        Saltloaded[JOB_WBB3] += store_typesalt(JOB_WBB3, line + 41, 40);
        Foundcnt[JOB_WBB3]++;
        continue;
      }
    }
    /* IPMI2_SHA1 (7300): SALT_HEX:HASH_HEX — last 40 hex = hash, rest = salt */
    if (lf[JOB_IPMI_SHA1]) {
      char *lastcolon = strrchr(line, ':');
      if (lastcolon && mystrlen(lastcolon + 1) == 40) {
        int hexok = 1;
        for (y = 0; y < 40; y++) {
          if (trhex[(unsigned char)lastcolon[1+y]] > 15) { hexok = 0; break; }
        }
        if (hexok) {
          char ipmi_key[512];
          /* key = hash_lc:salt_hex */
          for (y = 0; y < 40; y++)
            ipmi_key[y] = lastcolon[1+y] | 0x20;
          ipmi_key[40] = ':';
          memcpy(ipmi_key + 41, line, lastcolon - line);
          ipmi_key[41 + (lastcolon - line)] = 0;
          JSLI(PV, JUDYJ(JOB_IPMI_SHA1), (unsigned char *)ipmi_key);
          Saltloaded[JOB_IPMI_SHA1] += store_typesalt(JOB_IPMI_SHA1, ipmi_key + 41, lastcolon - line);
          Foundcnt[JOB_IPMI_SHA1]++;
          continue;
        }
      }
    }
    /* IPMI2_MD5 (7350): HASH_HEX:SALT_HEX — first 32 hex = hash */
    if (lf[JOB_IPMI_MD5] && len > 33 && line[32] == ':') {
      int hexok = 1;
      for (y = 0; y < 32; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
      }
      if (hexok) {
        char ipmi_key[512];
        /* key = hash_lc:salt_hex */
        for (y = 0; y < 32; y++)
          ipmi_key[y] = line[y] | 0x20;
        ipmi_key[32] = ':';
        strcpy(ipmi_key + 33, line + 33);
        JSLI(PV, JUDYJ(JOB_IPMI_MD5), (unsigned char *)ipmi_key);
        Saltloaded[JOB_IPMI_MD5] += store_typesalt(JOB_IPMI_MD5, ipmi_key + 33, mystrlen(ipmi_key + 33));
        Foundcnt[JOB_IPMI_MD5]++;
        continue;
      }
    }
    /* ISCSI-CHAP (4800): HASH(32hex):CHALLENGE(hex):ID(2hex) → MD5(id_byte || pass || challenge_bytes) */
    if (lf[JOB_ISCSI_CHAP] && len > 35 && line[32] == ':') {
      int hexok = 1;
      for (y = 0; y < 32; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
      }
      if (hexok) {
        /* find second colon (end of challenge) */
        char *c2 = strchr(line + 33, ':');
        if (c2 && (c2 - (line + 33)) >= 2 && mystrlen(c2 + 1) == 2) {
          /* verify challenge and id are hex */
          int challen = c2 - (line + 33);
          int challok = 1;
          for (y = 0; y < challen; y++) {
            if (trhex[(unsigned char)line[33+y]] > 15) { challok = 0; break; }
          }
          if (challok && trhex[(unsigned char)c2[1]] <= 15 && trhex[(unsigned char)c2[2]] <= 15) {
            char chap_key[512];
            /* key = hash_lc:challenge:id */
            for (y = 0; y < 32; y++)
              chap_key[y] = line[y] | 0x20;
            chap_key[32] = ':';
            strcpy(chap_key + 33, line + 33);
            JSLI(PV, JUDYJ(JOB_ISCSI_CHAP), (unsigned char *)chap_key);
            /* salt = challenge:id */
            Saltloaded[JOB_ISCSI_CHAP] += store_typesalt(JOB_ISCSI_CHAP, chap_key + 33, mystrlen(chap_key + 33));
            Foundcnt[JOB_ISCSI_CHAP]++;
            continue;
          }
        }
      }
    }
    /* MD5_SALT_SHA1PEPPASS (21310): hash(32hex):salt1:salt2
     * md5(salt1 . sha1(salt2 . pass)) — store "salt1:salt2" in Typesalt */
    if (lf[JOB_MD5_SALT_SHA1PEPPASS] && len > 35 && line[32] == ':' && inhashbuf) {
      int hexok = 1;
      for (y = 0; y < 32; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
      }
      if (hexok) {
        char *rest = line + 33;  /* everything after "hash:" = "salt1:salt2" */
        int restlen = mystrlen(rest);
        if (restlen > 0 && strchr(rest, ':')) {
          int hlen = get32(line, inhashbuf + 2, 32);
          if (hlen == 16) {
            commit_compact(hlen);
            Saltloaded[JOB_MD5_SALT_SHA1PEPPASS] += store_typesalt(JOB_MD5_SALT_SHA1PEPPASS, rest, restlen);
            Foundcnt[JOB_MD5_SALT_SHA1PEPPASS]++;
            continue;
          }
        }
      }
    }
    if (lf[JOB_MD5_MD5MD5PASSSALT_PEP2] && len > 35 && line[32] == ':' && inhashbuf) {
      int hexok = 1;
      for (y = 0; y < 32; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
      }
      if (hexok) {
        char *rest = line + 33;  /* everything after "hash:" = "salt1:salt2" */
        int restlen = mystrlen(rest);
        if (restlen > 0 && strchr(rest, ':')) {
          int hlen = get32(line, inhashbuf + 2, 32);
          if (hlen == 16) {
            commit_compact(hlen);
            Saltloaded[JOB_MD5_MD5MD5PASSSALT_PEP2] += store_typesalt(JOB_MD5_MD5MD5PASSSALT_PEP2, rest, restlen);
            Foundcnt[JOB_MD5_MD5MD5PASSSALT_PEP2]++;
            continue;
          }
        }
      }
    }
    if (lf[JOB_MD5_MD5MD5PASSSALT_PEP] && len > 35 && line[32] == ':' && inhashbuf) {
      int hexok = 1;
      for (y = 0; y < 32; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
      }
      if (hexok) {
        char *rest = line + 33;
        int restlen = mystrlen(rest);
        if (restlen > 0 && strchr(rest, ':')) {
          int hlen = get32(line, inhashbuf + 2, 32);
          if (hlen == 16) {
            commit_compact(hlen);
            Saltloaded[JOB_MD5_MD5MD5PASSSALT_PEP] += store_typesalt(JOB_MD5_MD5MD5PASSSALT_PEP, rest, restlen);
            Foundcnt[JOB_MD5_MD5MD5PASSSALT_PEP]++;
            continue;
          }
        }
      }
    }
    /* KRB5PA23 (7500): $krb5pa$23$user$realm$salt$enc_hex */
    if (lf[JOB_KRB5PA23] && strncmp(line, "$krb5pa$23$", 11) == 0) {
      /* find enc_hex after 3rd $ past $krb5pa$23$ */
      char *p = line + 11;
      int dc = 0;
      while (*p && dc < 3) { if (*p == '$') dc++; p++; }
      if (dc == 3 && mystrlen(p) >= 72) {
        /* verify enc_hex is all hex */
        int hexok = 1;
        for (y = 0; p[y]; y++) {
          if (trhex[(unsigned char)p[y]] > 15) { hexok = 0; break; }
        }
        if (hexok && y >= 72) {
          JSLI(PV, JUDYJ(JOB_KRB5PA23), (unsigned char *)line);
          /* store full hash as Typesalt key for dispatch iteration */
          Saltloaded[JOB_KRB5PA23] += store_typesalt(JOB_KRB5PA23, line, mystrlen(line));
          Foundcnt[JOB_KRB5PA23]++;
          continue;
        }
      }
    }
    /* MySQL $A$ (7401): $mysql$A$NNN*hex_salt*hex_hash */
    if (lf[JOB_MYSQL_SHA256CRYPT] && strncmp(line, "$mysql$A$", 9) == 0) {
      char *p = line + 9;
      /* NNN digits */
      while (*p >= '0' && *p <= '9') p++;
      if (*p == '*') {
        char *salt_start = p + 1;
        char *salt_end = strchr(salt_start, '*');
        if (salt_end && salt_end > salt_start) {
          char *hash_start = salt_end + 1;
          int hlen = mystrlen(hash_start);
          if (hlen >= 40) {
            /* trim trailing whitespace */
            while (hlen > 0 && (hash_start[hlen-1] == '\r' || hash_start[hlen-1] == '\n'))
              hash_start[--hlen] = 0;
            JSLI(PV, JUDYJ(JOB_MYSQL_SHA256CRYPT), (unsigned char *)line);
            /* extract $mysql$A$NNN*hex_salt into Typesalt */
            { char savec = *salt_end;
              *salt_end = 0;
              Saltloaded[JOB_MYSQL_SHA256CRYPT] += store_typesalt(JOB_MYSQL_SHA256CRYPT, line, mystrlen(line));
              *salt_end = savec;
            }
            Foundcnt[JOB_MYSQL_SHA256CRYPT]++;
            continue;
          }
        }
      }
    }
    /* JudyJ[JOB_DRUPAL7] (7900): $S$Xsalt...hash... */
    if (lf[JOB_DRUPAL7] && strncmp(line, "$S$", 3) == 0 && mystrlen(line) >= 55) {
      char *p = line + 3;
      /* p[0] = iteration char (phpitoa64), then 8 chars salt, then 43 chars hash */
      if (strchr(phpitoa64, p[0]) && mystrlen(p) >= 52) {
        int sl = mystrlen(line);
        while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
          line[--sl] = 0;
        JSLI(PV, JUDYJ(JOB_DRUPAL7), (unsigned char *)line);
        Saltloaded[JOB_DRUPAL7] += store_typesalt(JOB_DRUPAL7, line + 3, 9);
        Foundcnt[JOB_DRUPAL7]++;
        continue;
      }
    }
    /* Sybase ASE (8000): 0xc007 + 80 hex = 86 chars */
    if (lf[JOB_SYBASE_ASE] && (strncmp(line, "0xc007", 6) == 0 || strncmp(line, "0xC007", 6) == 0)) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      if (sl == 86) {
        int hexok = 1;
        for (y = 6; y < 86; y++) {
          if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
        }
        if (hexok) {
          JSLI(PV, JUDYJ(JOB_SYBASE_ASE), (unsigned char *)line);
          Saltloaded[JOB_SYBASE_ASE] += store_typesalt(JOB_SYBASE_ASE, line + 6, 16);
          Foundcnt[JOB_SYBASE_ASE]++;
          continue;
        }
      }
    }
    /* NetScaler (8100): 1 + 8hex + 40hex = 49 chars */
    if (lf[JOB_NETSCALER] && line[0] == '1') {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      if (sl == 49) {
        int hexok = 1;
        for (y = 1; y < 49; y++) {
          if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
        }
        if (hexok) {
          JSLI(PV, JUDYJ(JOB_NETSCALER), (unsigned char *)line);
          Saltloaded[JOB_NETSCALER] += store_typesalt(JOB_NETSCALER, line + 1, 8);
          Foundcnt[JOB_NETSCALER]++;
          continue;
        }
      }
    }
    /* NSEC3 (8300): base32hex_hash:domain:salt_hex:iterations */
    if (lf[JOB_NSEC3]) {
      char *colon1 = strchr(line, ':');
      if (colon1 && (colon1 - line) == 32) {
        /* verify 32 base32hex chars */
        int b32ok = 1;
        for (y = 0; y < 32; y++) {
          char c = line[y] | 0x20;
          if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'v'))) { b32ok = 0; break; }
        }
        if (b32ok) {
          char *colon2 = strchr(colon1 + 1, ':');
          if (colon2) {
            char *colon3 = strchr(colon2 + 1, ':');
            if (colon3 && atoi(colon3 + 1) >= 0) {
              int sl = mystrlen(line);
              while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
                line[--sl] = 0;
              JSLI(PV, JUDYJ(JOB_NSEC3), (unsigned char *)line);
              /* extract domain:salt_hex:iterations into Typesalt */
              Saltloaded[JOB_NSEC3] += store_typesalt(JOB_NSEC3, colon1 + 1, mystrlen(colon1 + 1));
              Foundcnt[JOB_NSEC3]++;
              continue;
            }
          }
        }
      }
    }
    /* RACF (8500): $racf$*USER*16hex */
    if (lf[JOB_RACF] && strncmp(line, "$racf$*", 7) == 0) {
      char *star2 = strchr(line + 7, '*');
      if (star2 && (star2 - (line + 7)) >= 1 && (star2 - (line + 7)) <= 8) {
        int hlen = mystrlen(star2 + 1);
        while (hlen > 0 && (star2[1+hlen-1] == '\r' || star2[1+hlen-1] == '\n')) {
          star2[1+hlen-1] = 0; hlen--;
        }
        if (hlen == 16) {
          int hexok = 1;
          for (y = 0; y < 16; y++) {
            if (trhex[(unsigned char)star2[1+y]] > 15) { hexok = 0; break; }
          }
          if (hexok) {
            JSLI(PV, JUDYJ(JOB_RACF), (unsigned char *)line);
            /* extract username into Typesalt */
            Saltloaded[JOB_RACF] += store_typesalt(JOB_RACF, line + 7, star2 - (line + 7));
            Foundcnt[JOB_RACF]++;
            continue;
          }
        }
      }
    }
    /* PS_TOKEN (13500): 40hex_hash:hex_salt */
    if (lf[JOB_PS_TOKEN] && len >= 42) {
      /* Check for 40-hex hash followed by colon */
      int hexok = 1;
      for (y = 0; y < 40; y++)
        if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
      if (hexok && line[40] == ':') {
        int slen = len - 41;
        if (slen >= 2 && slen <= 1024 && (slen & 1) == 0) {
          /* Verify salt is hex */
          int saltok = 1;
          for (y = 41; y < len; y++)
            if (trhex[(unsigned char)line[y]] > 15) { saltok = 0; break; }
          if (saltok) {
            /* JudyJ key = 40-hex lc hash, Typesalt = hex salt */
            char ps_hashkey[41];
            for (y = 0; y < 40; y++)
              ps_hashkey[y] = tolower(line[y]);
            ps_hashkey[40] = 0;
            JSLI(PV, JUDYJ(JOB_PS_TOKEN), (unsigned char *)ps_hashkey);
            /* Salt can exceed 255 chars — use JSLI directly */
            { char *ps_salt = salttmp;
              if (slen < (int)sizeof(salttmp)) {
                memcpy(ps_salt, line + 41, slen);
                ps_salt[slen] = 0;
                JSLI(PV, TYPESALT(JOB_PS_TOKEN), (unsigned char *)ps_salt);
                if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_PS_TOKEN]++; }
              }
            }
            Foundcnt[JOB_PS_TOKEN]++;
            continue;
          }
        }
      }
    }
    /* WINPHONE (13800): 64hex_hash:256hex_salt */
    if (lf[JOB_WINPHONE] && len == 64 + 1 + 256) {
      int hexok = 1;
      for (y = 0; y < 64; y++)
        if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
      if (hexok && line[64] == ':') {
        int saltok = 1;
        for (y = 65; y < len; y++)
          if (trhex[(unsigned char)line[y]] > 15) { saltok = 0; break; }
        if (saltok) {
          char wp_hashkey[65];
          for (y = 0; y < 64; y++)
            wp_hashkey[y] = tolower(line[y]);
          wp_hashkey[64] = 0;
          JSLI(PV, JUDYJ(JOB_WINPHONE), (unsigned char *)wp_hashkey);
          /* Salt is 256 hex — too long for store_typesalt (255 limit), use JSLI directly */
          { char wp_salt[257];
            memcpy(wp_salt, line + 65, 256);
            wp_salt[256] = 0;
            JSLI(PV, TYPESALT(JOB_WINPHONE), (unsigned char *)wp_salt);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_WINPHONE]++; }
          }
          Foundcnt[JOB_WINPHONE]++;
          continue;
        }
      }
    }
    /* RACF-KDFAES (14200): $racf-kdfaes$*USER*HEADER32HEX*SALT32HEX*HASH32HEX */
    if (lf[JOB_RACF_KDFAES] && strncmp(line, "$racf-kdfaes$*", 14) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      /* Parse: *USER*HEADER*SALT*HASH — need 4 star-separated fields */
      char *f1 = line + 14; /* username start */
      char *s2 = strchr(f1, '*');
      if (s2 && (s2 - f1) >= 1 && (s2 - f1) <= 8) {
        char *s3 = strchr(s2 + 1, '*');
        if (s3 && (s3 - s2 - 1) == 32) {
          char *s4 = strchr(s3 + 1, '*');
          if (s4 && (s4 - s3 - 1) == 32 && (int)strlen(s4 + 1) == 32) {
            /* Verify all hex fields */
            int hexok = 1;
            char *hp;
            for (hp = s2 + 1; hp < s4 + 33; hp++) {
              if (hp == s3 || hp == s4) continue; /* skip star separators */
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            }
            if (hexok) {
              /* JudyJ key = full line, Typesalt = USER*HEADER*SALT */
              JSLI(PV, JUDYJ(JOB_RACF_KDFAES), (unsigned char *)line);
              /* Typesalt: USER*HEADER*SALT (from f1 to s4-1) */
              Saltloaded[JOB_RACF_KDFAES] += store_typesalt(JOB_RACF_KDFAES, f1, s4 - f1);
              Foundcnt[JOB_RACF_KDFAES]++;
              continue;
            }
          }
        }
      }
    }
    /* TACACS (16100): $tacacs-plus$0$SESSION(8hex)$CT(12-256hex)$SEQ(4hex) */
    if (lf[JOB_TACACS] && strncmp(line, "$tacacs-plus$0$", 15) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      char *f1 = line + 15; /* session hex start */
      char *d1 = strchr(f1, '$');
      if (d1 && (d1 - f1) == 8) {
        char *f2 = d1 + 1; /* ct_data hex start */
        char *d2 = strchr(f2, '$');
        if (d2 && (d2 - f2) >= 12 && (d2 - f2) <= 256 && ((d2 - f2) % 2) == 0) {
          char *f3 = d2 + 1; /* sequence hex */
          int seqlen = strlen(f3);
          if (seqlen == 4) {
            /* Verify all hex */
            int hexok = 1;
            char *hp;
            for (hp = f1; hp < f1 + 8; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = f2; hp < d2; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = f3; hp < f3 + 4; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) {
              /* Typesalt = "session:ct:seq", JudyJ key = same */
              int ct_hexlen = d2 - f2;
              snprintf(salttmp, MAXLINE, "%.*s:%.*s:%s",
                       8, f1, ct_hexlen, f2, f3);
              /* Lowercase the salt */
              for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
              JSLI(PV, JUDYJ(JOB_TACACS), (unsigned char *)salttmp);
              JSLI(PV, TYPESALT(JOB_TACACS), (unsigned char *)salttmp);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_TACACS]++; }
              Foundcnt[JOB_TACACS]++;
              continue;
            }
          }
        }
      }
    }
    /* FileVault 2 (16700): $fvde$ver$saltlen$salt32hex$iter$wrappedkey48hex
     * Convert to ASN internal format: $ASN$*ver*iter*salt*wkey */
    if (lf[JOB_APPLE_SECURE_NOTES] && strncmp(line, "$fvde$", 6) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      /* Parse: $fvde$ver$saltlen$salt$iter$wkey */
      char *f1 = line + 6; /* ver */
      char *s1 = strchr(f1, '$');
      if (s1 && (s1 - f1) >= 1 && (s1 - f1) <= 4) {
        int ver = atoi(f1);
        char *f2 = s1 + 1; /* saltlen */
        char *s2 = strchr(f2, '$');
        if (s2 && (s2 - f2) >= 1 && (s2 - f2) <= 4) {
          int saltlen_val = atoi(f2);
          char *f3 = s2 + 1; /* salt hex */
          char *s3 = strchr(f3, '$');
          if (s3 && (s3 - f3) == saltlen_val * 2 && saltlen_val == 16) {
            char *f4 = s3 + 1; /* iter */
            char *s4 = strchr(f4, '$');
            if (s4 && (s4 - f4) >= 1 && (s4 - f4) <= 8) {
              int iter = atoi(f4);
              char *f5 = s4 + 1; /* wrapped key hex */
              if ((int)strlen(f5) == 48 && iter > 0) {
                int hexok = 1;
                char *hp;
                for (hp = f3; hp < s3; hp++)
                  if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
                if (hexok) for (hp = f5; hp < f5 + 48; hp++)
                  if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
                if (hexok) {
                  /* Typesalt = "zpk:iter:salt32hex:wrappedkey48hex" (lowercased) */
                  snprintf(salttmp, MAXLINE, "%d:%d:%.*s:%s",
                           ver, iter, 32, f3, f5);
                  for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
                  JSLI(PV, TYPESALT(JOB_APPLE_SECURE_NOTES), (unsigned char *)salttmp);
                  if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_APPLE_SECURE_NOTES]++; }
                  /* JudyJ key = $ASN$*... format (hex parts from lowered Typesalt) */
                  { const char *tc1 = strchr(salttmp, ':');
                    const char *tc2 = tc1 ? strchr(tc1 + 1, ':') : NULL;
                    const char *tc3 = tc2 ? strchr(tc2 + 1, ':') : NULL;
                    if (tc2 && tc3) {
                      char judykey[256];
                      snprintf(judykey, sizeof(judykey), "$ASN$*%d*%d*%.*s*%s",
                               ver, iter, (int)(tc3 - tc2 - 1), tc2 + 1, tc3 + 1);
                      JSLI(PV, JUDYJ(JOB_APPLE_SECURE_NOTES), (unsigned char *)judykey);
                    }
                  }
                  Foundcnt[JOB_APPLE_SECURE_NOTES]++;
                  continue;
                }
              }
            }
          }
        }
      }
    }
    /* APPLE-SECURE-NOTES (16200): $ASN$*zpk*iter*salt32hex*wrappedkey48hex */
    if (lf[JOB_APPLE_SECURE_NOTES] && strncmp(line, "$ASN$*", 6) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      char *f1 = line + 6; /* zpk start */
      char *s1 = strchr(f1, '*');
      if (s1 && (s1 - f1) >= 1 && (s1 - f1) <= 4) {
        int zpk = atoi(f1);
        char *f2 = s1 + 1; /* iter start */
        char *s2 = strchr(f2, '*');
        if (s2 && (s2 - f2) >= 1 && (s2 - f2) <= 8) {
          int iter = atoi(f2);
          char *f3 = s2 + 1; /* salt hex */
          char *s3 = strchr(f3, '*');
          if (s3 && (s3 - f3) == 32 && iter > 0) {
            char *f4 = s3 + 1; /* wrapped key hex */
            if ((int)strlen(f4) == 48) {
              int hexok = 1;
              char *hp;
              for (hp = f3; hp < s3; hp++)
                if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
              if (hexok) for (hp = f4; hp < f4 + 48; hp++)
                if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
              if (hexok) {
                /* JudyJ key = full format line */
                JSLI(PV, JUDYJ(JOB_APPLE_SECURE_NOTES), (unsigned char *)line);
                /* Typesalt = "zpk:iter:salt32hex:wrappedkey48hex" */
                snprintf(salttmp, MAXLINE, "%d:%d:%.*s:%s",
                         zpk, iter, 32, f3, f4);
                /* Lowercase */
                for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
                JSLI(PV, TYPESALT(JOB_APPLE_SECURE_NOTES), (unsigned char *)salttmp);
                if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_APPLE_SECURE_NOTES]++; }
                Foundcnt[JOB_APPLE_SECURE_NOTES]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* CRAMMD5-DOVECOT (16400): {CRAM-MD5}32hex32zeros */
    if (lf[JOB_CRAMMD5_DOVECOT] && strncmp(line, "{CRAM-MD5}", 10) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      if (sl == 10 + 64) {
        int hexok = 1;
        char *hp;
        for (hp = line + 10; hp < line + 10 + 64; hp++)
          if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
        if (hexok) {
          /* JudyJ key = first 32 hex chars, lowercase */
          char dc_key[33];
          for (y = 0; y < 32; y++)
            dc_key[y] = tolower(line[10 + y]);
          dc_key[32] = 0;
          JSLI(PV, JUDYJ(JOB_CRAMMD5_DOVECOT), (unsigned char *)dc_key);
          Foundcnt[JOB_CRAMMD5_DOVECOT]++;
          continue;
        }
      }
    }
    /* JWT (16500): base64url.base64url.base64url (header.payload.signature) */
    if (lf[JOB_JWT] && len > 10) {
      /* Must have exactly 2 dots, no colon/dollar/star prefix */
      char *dot1 = memchr(line, '.', len);
      if (dot1 && dot1 > line && line[0] != '$' && line[0] != '{') {
        char *dot2 = memchr(dot1 + 1, '.', len - (dot1 + 1 - line));
        if (dot2 && dot2 > dot1 + 1) {
          char *sig_b64 = dot2 + 1;
          int sig_b64_len = len - (sig_b64 - line);
          /* Strip trailing whitespace */
          while (sig_b64_len > 0 && (sig_b64[sig_b64_len-1] == '\r' || sig_b64[sig_b64_len-1] == '\n'))
            sig_b64_len--;
          /* Check sig length for HS256(43), HS384(64), HS512(86) */
          if (sig_b64_len == 43 || sig_b64_len == 64 || sig_b64_len == 86) {
            /* No third dot */
            if (!memchr(sig_b64, '.', sig_b64_len)) {
              /* Convert base64url to base64 */
              char jwt_b64[128];
              int ji;
              for (ji = 0; ji < sig_b64_len && ji < 120; ji++) {
                if (sig_b64[ji] == '-') jwt_b64[ji] = '+';
                else if (sig_b64[ji] == '_') jwt_b64[ji] = '/';
                else jwt_b64[ji] = sig_b64[ji];
              }
              while (ji % 4) jwt_b64[ji++] = '=';
              jwt_b64[ji] = 0;
              /* Decode signature */
              int incount;
              int siglen = b64_decode(jwt_b64, salttmp, &incount);
              if (siglen == 32 || siglen == 48 || siglen == 64) {
                /* JudyJ key = lowercase hex of decoded signature */
                { char *jwt_hexkey = salttmp + 68;
                  prmd5((unsigned char *)salttmp, jwt_hexkey, siglen * 2);
                  jwt_hexkey[siglen * 2] = 0;
                  JSLI(PV, JUDYJ(JOB_JWT), (unsigned char *)jwt_hexkey);
                }
                /* Typesalt = full JWT string (JSLI directly — can be very long) */
                { char *jwt_copy = salttmp + 128;
                  int jwt_len = (dot2 + 1 + sig_b64_len) - line;
                  if (jwt_len < MAXLINE - 128) {
                    memcpy(jwt_copy, line, jwt_len);
                    jwt_copy[jwt_len] = 0;
                    JSLI(PV, TYPESALT(JOB_JWT), (unsigned char *)jwt_copy);
                    if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_JWT]++; }
                  }
                }
                Foundcnt[JOB_JWT]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* QNX /etc/shadow (19000/19100/19200/19210): @X@HASH@SALT or @X,ITER@HASH@SALT */
    if ((lf[JOB_QNX_MD5] || lf[JOB_QNX_SHA256] || lf[JOB_QNX_SHA512] || lf[JOB_QNX7_SHA512])
        && len > 6 && line[0] == '@' && (line[1] == 'm' || line[1] == 's' || line[1] == 'S')) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      char type_ch = line[1];
      /* Parse optional iteration count: @X@ or @X,ITER@ */
      int iter = 1000;
      char *at2; /* second @ */
      if (line[2] == ',') {
        iter = atoi(line + 3);
        at2 = strchr(line + 3, '@');
      } else if (line[2] == '@') {
        at2 = line + 2;
      } else {
        at2 = NULL;
      }
      if (at2 && iter > 0) {
        char *hash_start = at2 + 1;
        char *at3 = strchr(hash_start, '@');
        if (at3 && at3 > hash_start) {
          int hash_hexlen = at3 - hash_start;
          char *salt_start = at3 + 1;
          int salt_len_s = strlen(salt_start);
          int target_job = 0;
          /* Determine type by type_ch and hash encoding */
          if (type_ch == 'm' && hash_hexlen == 32 && lf[JOB_QNX_MD5])
            target_job = JOB_QNX_MD5;
          else if (type_ch == 's' && hash_hexlen == 64 && lf[JOB_QNX_SHA256])
            target_job = JOB_QNX_SHA256;
          else if (type_ch == 'S' && hash_hexlen == 128 && lf[JOB_QNX_SHA512])
            target_job = JOB_QNX_SHA512;
          else if (type_ch == 'S' && lf[JOB_QNX7_SHA512]) {
            /* 19210: base64 hash, distinguish from 19200 hex by checking non-hex chars */
            int is_b64 = 0;
            { char *hp; for (hp = hash_start; hp < at3; hp++)
                if (trhex[(unsigned char)*hp] > 15) { is_b64 = 1; break; } }
            if (is_b64) target_job = JOB_QNX7_SHA512;
          }
          if (target_job && salt_len_s > 0 && salt_len_s < 256) {
            int hexok = 1;
            if (target_job != JOB_QNX7_SHA512) {
              /* Validate hex hash for 19000/19100/19200 */
              char *hp;
              for (hp = hash_start; hp < at3; hp++)
                if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            }
            if (hexok) {
              if (target_job == JOB_QNX7_SHA512) {
                /* 19210: iter defaults to 4096 if not specified */
                if (line[2] == '@') iter = 4096;
              }
              /* JudyJ key = lowercase hex hash (19000/19100/19200) or normalized format (19210) */
              if (target_job == JOB_QNX7_SHA512) {
                /* Normalize: always @S,iter@hash@salt */
                char save = *at3; *at3 = 0;
                snprintf(salttmp + MAXLINE/2, MAXLINE/2, "@S,%d@%s@%s", iter, hash_start, salt_start);
                *at3 = save;
                JSLI(PV, JUDYJ(target_job), (unsigned char *)(salttmp + MAXLINE/2));
              } else {
                /* Lowercase the hex hash for JudyJ key */
                { char *hp; for (hp = hash_start; hp < at3; hp++) *hp = tolower(*hp); }
                char save = *at3; *at3 = 0;
                JSLI(PV, JUDYJ(target_job), (unsigned char *)hash_start);
                *at3 = save;
              }
              /* Typesalt = "iter:salt" */
              snprintf(salttmp, MAXLINE, "%d:%s", iter, salt_start);
              JSLI(PV, TYPESALT(target_job), (unsigned char *)salttmp);
              if (PV) { if ((*PV)++ == 0) Saltloaded[target_job]++; }
              Foundcnt[target_job]++;
              continue;
            }
          }
        }
      }
    }
    /* SHA1-S1PS2 (19300) and RAILS-RESTFUL (19500):
     * Format is 40hex:salt1:salt2 — hex fallthrough at bottom handles
     * compact table insertion + Typesalt population via Dosalt[] */
    /* KRB5PA-17 (19800): $krb5pa$17$user$realm$hexdata */
    if (lf[JOB_KRB5PA_17] && strncmp(line, "$krb5pa$17$", 11) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      char *user = line + 11;
      char *d2 = strchr(user, '$');
      if (d2 && d2 > user) {
        char *realm = d2 + 1;
        char *d3 = strchr(realm, '$');
        if (d3 && d3 > realm) {
          char *hexdata = d3 + 1;
          int hexlen = strlen(hexdata);
          if (hexlen >= 104 && hexlen <= 112 && (hexlen % 2) == 0) {
            int hexok = 1;
            { char *hp; for (hp = hexdata; *hp; hp++)
                if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; } }
            if (hexok) {
              JSLI(PV, JUDYJ(JOB_KRB5PA_17), (unsigned char *)line);
              /* Typesalt = "REALM:user:hexdata" (realm uppercased for PBKDF2 salt) */
              { char *hp;
                int realmlen = d3 - realm;
                snprintf(salttmp, MAXLINE, "%.*s:%.*s:%s",
                         realmlen, realm, (int)(d2 - user), user, hexdata);
                /* Uppercase realm portion */
                for (hp = salttmp; hp < salttmp + realmlen; hp++) *hp = toupper(*hp);
              }
              JSLI(PV, TYPESALT(JOB_KRB5PA_17), (unsigned char *)salttmp);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_KRB5PA_17]++; }
              Foundcnt[JOB_KRB5PA_17]++;
              continue;
            }
          }
        }
      }
    }
    /* KRB5PA-18 (19900): $krb5pa$18$user$realm$hexdata */
    if (lf[JOB_KRB5PA_18] && strncmp(line, "$krb5pa$18$", 11) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      char *user = line + 11;
      char *d2 = strchr(user, '$');
      if (d2 && d2 > user) {
        char *realm = d2 + 1;
        char *d3 = strchr(realm, '$');
        if (d3 && d3 > realm) {
          char *hexdata = d3 + 1;
          int hexlen = strlen(hexdata);
          if (hexlen >= 104 && hexlen <= 112 && (hexlen % 2) == 0) {
            int hexok = 1;
            { char *hp; for (hp = hexdata; *hp; hp++)
                if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; } }
            if (hexok) {
              JSLI(PV, JUDYJ(JOB_KRB5PA_18), (unsigned char *)line);
              /* Typesalt = "REALM:user:hexdata" */
              { char *hp;
                int realmlen = d3 - realm;
                snprintf(salttmp, MAXLINE, "%.*s:%.*s:%s",
                         realmlen, realm, (int)(d2 - user), user, hexdata);
                for (hp = salttmp; hp < salttmp + realmlen; hp++) *hp = toupper(*hp);
              }
              JSLI(PV, TYPESALT(JOB_KRB5PA_18), (unsigned char *)salttmp);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_KRB5PA_18]++; }
              Foundcnt[JOB_KRB5PA_18]++;
              continue;
            }
          }
        }
      }
    }
    /* WPA-PMKID (16800/22000 type=01): WPA*01*PMKID32*MACAP12*MACSTA12*ESSID_hex*** */
    if ((lf[JOB_WPA_PMKID] || lf[JOB_WPA_EAPOL] || lf[JOB_WPA_PMK]) && strncmp(line, "WPA*", 4) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      /* Parse: WPA*TYPE*HASH32*MACAP12*MACSTA12*ESSID_hex*[ANONCE64*EAPOL_hex*MP] */
      char *f[9] = {NULL}; /* up to 9 fields separated by * — init silences flow-insensitive false positive */
      int nf = 0;
      char *p = line;
      while (nf < 9 && p < line + sl) {
        f[nf++] = p;
        char *np = strchr(p, '*');
        if (np) { *np = 0; p = np + 1; }
        else { p = line + sl; break; }
      }
      if (nf >= 6) {
        int wtype = (f[1][0] == '0' && f[1][1] == '1') ? 1 :
                    (f[1][0] == '0' && f[1][1] == '2') ? 2 : 0;
        int hash_hexlen = mystrlen(f[2]);
        int macap_len = mystrlen(f[3]);
        int macsta_len = mystrlen(f[4]);
        int essid_hexlen = mystrlen(f[5]);
        if (wtype == 1 && (lf[JOB_WPA_PMKID] || lf[JOB_WPA_PMK]) && hash_hexlen == 32 &&
            macap_len == 12 && macsta_len == 12 &&
            essid_hexlen >= 2 && essid_hexlen <= 64 && (essid_hexlen & 1) == 0) {
          int hexok = 1;
          char *hp;
          for (hp = f[2]; *hp; hp++)
            if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
          if (hexok) for (hp = f[3]; *hp; hp++)
            if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
          if (hexok) for (hp = f[4]; *hp; hp++)
            if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
          if (hexok) for (hp = f[5]; *hp; hp++)
            if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
          if (hexok) {
            /* JudyJ key = lowercase PMKID hex */
            char judykey[64];
            for (x = 0; x < 32; x++) judykey[x] = tolower(f[2][x]);
            judykey[32] = 0;
            if (lf[JOB_WPA_PMKID]) {
              JSLI(PV, JUDYJ(JOB_WPA_PMKID), (unsigned char *)judykey);
              /* Typesalt = "essid_hex:macap:macsta" (lowercased) */
              snprintf(salttmp, MAXLINE, "%s:%s:%s", f[5], f[3], f[4]);
              for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
              Saltloaded[JOB_WPA_PMKID] += store_typesalt(JOB_WPA_PMKID, salttmp, mystrlen(salttmp));
              Foundcnt[JOB_WPA_PMKID]++;
            }
            /* Also load into WPA_PMK if enabled */
            if (lf[JOB_WPA_PMK]) {
              JSLI(PV, JUDYJ(JOB_WPA_PMK), (unsigned char *)judykey);
              /* Typesalt = "1:essid_hex:macap:macsta" */
              snprintf(salttmp, MAXLINE, "1:%s:%s:%s", f[5], f[3], f[4]);
              for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
              JSLI(PV, TYPESALT(JOB_WPA_PMK), (unsigned char *)salttmp);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_WPA_PMK]++; }
              Foundcnt[JOB_WPA_PMK]++;
            }
            /* Restore * separators for next parse attempt */
            continue;
          }
        }
        if (wtype == 2 && (lf[JOB_WPA_EAPOL] || lf[JOB_WPA_PMKID] || lf[JOB_WPA_PMK]) && nf >= 9 &&
            hash_hexlen == 32 && macap_len == 12 && macsta_len == 12 &&
            essid_hexlen >= 2 && essid_hexlen <= 64 && (essid_hexlen & 1) == 0) {
          int anonce_len = mystrlen(f[6]);
          int eapol_hexlen = mystrlen(f[7]);
          if (anonce_len == 64 && eapol_hexlen >= 2 && (eapol_hexlen & 1) == 0) {
            int hexok = 1;
            char *hp;
            for (hp = f[2]; *hp; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = f[3]; *hp; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = f[4]; *hp; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = f[5]; *hp; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = f[6]; *hp; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) {
              /* Decode EAPOL to get keyver from key_information */
              unsigned char eapol_bin[320];
              int eapol_binlen = get32(f[7], eapol_bin, 256);
              if (eapol_binlen >= 99) {
                /* keyver from key_information (bytes 5-6, big-endian) & 0x07 */
                int keyver = ((eapol_bin[5] << 8) | eapol_bin[6]) & 7;
                if (keyver == 1 || keyver == 2) {
                  /* JudyJ key = lowercase MIC hex */
                  char judykey[64];
                  for (x = 0; x < 32; x++) judykey[x] = tolower(f[2][x]);
                  judykey[32] = 0;
                  /* Enable WPA_EAPOL if PMKID was requested (auto-enable for type 02) */
                  if (!lf[JOB_WPA_EAPOL] && lf[JOB_WPA_PMKID]) {
                    Word_t RC2;
                    lf[JOB_WPA_EAPOL] = 1;
                    J1S(RC2, Dohash, JOB_WPA_EAPOL);
                  }
                  if (lf[JOB_WPA_EAPOL]) {
                    JSLI(PV, JUDYJ(JOB_WPA_EAPOL), (unsigned char *)judykey);
                  }
                  /* Typesalt for WPA_EAPOL */
                  if (lf[JOB_WPA_EAPOL]) {
                    snprintf(salttmp, MAXLINE, "%d:%s:%s:%s:%s:%s",
                             keyver, f[5], f[3], f[4], f[6], f[7]);
                    for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
                    JSLI(PV, TYPESALT(JOB_WPA_EAPOL), (unsigned char *)salttmp);
                    if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_WPA_EAPOL]++; }
                    Foundcnt[JOB_WPA_EAPOL]++;
                  }
                  /* Also load into WPA_PMK if enabled */
                  if (lf[JOB_WPA_PMK]) {
                    JSLI(PV, JUDYJ(JOB_WPA_PMK), (unsigned char *)judykey);
                    /* Typesalt = "2:keyver:essid_hex:macap:macsta:anonce:eapol" */
                    snprintf(salttmp, MAXLINE, "2:%d:%s:%s:%s:%s:%s",
                             keyver, f[5], f[3], f[4], f[6], f[7]);
                    for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
                    JSLI(PV, TYPESALT(JOB_WPA_PMK), (unsigned char *)salttmp);
                    if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_WPA_PMK]++; }
                    Foundcnt[JOB_WPA_PMK]++;
                  }
                  continue;
                }
              }
            }
          }
        }
      }
    }
    /* WPA-PMKID old format (16800): PMKID32:MACAP12:MACSTA12:ESSID_hex */
    if (lf[JOB_WPA_PMKID] && len >= 32 + 3 + 12 + 12 + 2) {
      char *c1 = line + 32;
      if (*c1 == ':' || *c1 == '*') {
        char sep = *c1;
        char *c2 = (len > 45) ? line + 45 : NULL;
        char *c3 = (len > 58) ? line + 58 : NULL;
        if (c2 && *c2 == sep && c3 && *c3 == sep) {
          int essid_hexlen = len - 59;
          if (essid_hexlen >= 2 && essid_hexlen <= 64 && (essid_hexlen & 1) == 0) {
            int hexok = 1;
            char *hp;
            for (hp = line; hp < c1; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = c1 + 1; hp < c2; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = c2 + 1; hp < c3; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) for (hp = c3 + 1; hp < line + len; hp++)
              if (trhex[(unsigned char)*hp] > 15) { hexok = 0; break; }
            if (hexok) {
              char judykey[64];
              for (x = 0; x < 32; x++) judykey[x] = tolower(line[x]);
              judykey[32] = 0;
              JSLI(PV, JUDYJ(JOB_WPA_PMKID), (unsigned char *)judykey);
              /* Typesalt = "essid_hex:macap:macsta" */
              snprintf(salttmp, MAXLINE, "%.*s:%.*s:%.*s",
                       essid_hexlen, c3 + 1, 12, c1 + 1, 12, c2 + 1);
              for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
              Saltloaded[JOB_WPA_PMKID] += store_typesalt(JOB_WPA_PMKID, salttmp, mystrlen(salttmp));
              Foundcnt[JOB_WPA_PMKID]++;
              continue;
            }
          }
        }
      }
    }
    /* ANSIBLE-VAULT (16900): $ansible$C*V*SALT64hex*CT_hex*HMAC64hex */
    if (lf[JOB_ANSIBLE_VAULT] && len > 80 && strncmp(line, "$ansible$", 9) == 0) {
      char *p = line + 9;
      /* Parse: cipher*version*salt*ct*hmac */
      char *s1 = strchr(p, '*');
      if (s1) {
        char *f2 = s1 + 1; /* version */
        char *s2 = strchr(f2, '*');
        if (s2) {
          char *f3 = s2 + 1; /* salt hex */
          char *s3 = strchr(f3, '*');
          if (s3 && (s3 - f3) == 64) {
            char *f4 = s3 + 1; /* ct hex */
            char *s4 = strchr(f4, '*');
            if (s4) {
              int ct_hexlen = s4 - f4;
              char *f5 = s4 + 1; /* hmac hex */
              int hmac_hexlen = len - (f5 - line);
              if (ct_hexlen >= 2 && (ct_hexlen & 1) == 0 && hmac_hexlen == 64) {
                int hexok = 1;
                char *hp;
                for (hp = f3; hp < s3 && hexok; hp++)
                  if (trhex[(unsigned char)*hp] > 15) hexok = 0;
                for (hp = f4; hp < s4 && hexok; hp++)
                  if (trhex[(unsigned char)*hp] > 15) hexok = 0;
                for (hp = f5; hp < line + len && hexok; hp++)
                  if (trhex[(unsigned char)*hp] > 15) hexok = 0;
                if (hexok) {
                  /* JudyJ key = lowercase HMAC hex (64 chars) */
                  char judykey[80];
                  for (x = 0; x < 64; x++) judykey[x] = tolower(f5[x]);
                  judykey[64] = 0;
                  JSLI(PV, JUDYJ(JOB_ANSIBLE_VAULT), (unsigned char *)judykey);
                  /* Typesalt = "salt64hex:ct_hex" (lowercased, uses JSLI for long ct) */
                  snprintf(salttmp, MAXLINE, "%.*s:%.*s", 64, f3, ct_hexlen, f4);
                  for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
                  JSLI(PV, TYPESALT(JOB_ANSIBLE_VAULT), (unsigned char *)salttmp);
                  if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_ANSIBLE_VAULT]++; }
                  Foundcnt[JOB_ANSIBLE_VAULT]++;
                  continue;
                }
              }
            }
          }
        }
      }
    }
    /* APFS (18300): $fvde$2$16$SALT32hex$ITER$WKEY80hex */
    if (lf[JOB_APFS] && len > 40 && strncmp(line, "$fvde$2$", 8) == 0) {
      char *f1 = line + 8; /* saltlen */
      char *s1 = strchr(f1, '$');
      if (s1 && (s1 - f1) >= 1 && (s1 - f1) <= 4) {
        int saltlen_val = atoi(f1);
        if (saltlen_val == 16) {
          char *f2 = s1 + 1; /* salt hex */
          char *s2 = strchr(f2, '$');
          if (s2 && (s2 - f2) == 32) {
            char *f3 = s2 + 1; /* iter */
            char *s3 = strchr(f3, '$');
            if (s3 && (s3 - f3) >= 1 && (s3 - f3) <= 8) {
              int iter = atoi(f3);
              char *f4 = s3 + 1; /* wrapped key hex */
              if ((int)strlen(f4) == 80 && iter > 0) {
                int hexok = 1;
                char *hp;
                for (hp = f2; hp < s2 && hexok; hp++)
                  if (trhex[(unsigned char)*hp] > 15) hexok = 0;
                for (hp = f4; hp < f4 + 80 && hexok; hp++)
                  if (trhex[(unsigned char)*hp] > 15) hexok = 0;
                if (hexok) {
                  /* JudyJ key = full $fvde$2$... format, lowercased hex parts */
                  char judykey[256];
                  snprintf(judykey, sizeof(judykey), "$fvde$2$16$%.*s$%d$%s",
                           32, f2, iter, f4);
                  for (hp = judykey; *hp; hp++) *hp = tolower(*hp);
                  JSLI(PV, JUDYJ(JOB_APFS), (unsigned char *)judykey);
                  /* Typesalt = "iter:salt32hex:wkey80hex" */
                  snprintf(salttmp, MAXLINE, "%d:%.*s:%s", iter, 32, f2, f4);
                  for (hp = salttmp; *hp; hp++) *hp = tolower(*hp);
                  JSLI(PV, TYPESALT(JOB_APFS), (unsigned char *)salttmp);
                  if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_APFS]++; }
                  Foundcnt[JOB_APFS]++;
                  continue;
                }
              }
            }
          }
        }
      }
    }
    /* OTM-SHA256 (20600): otm_sha256:ITER:SALT:HASH_B64 */
    if (lf[JOB_OTM_SHA256] && len > 20 && strncmp(line, "otm_sha256:", 11) == 0) {
      char *iter_s = line + 11;
      char *d1 = strchr(iter_s, ':');
      if (d1) {
        int iter = atoi(iter_s);
        if (iter > 0) {
          char *salt = d1 + 1;
          char *d2 = strchr(salt, ':');
          if (d2 && d2 > salt && (d2 - salt) <= 16) {
            char *hash_b64 = d2 + 1;
            int hash_b64_len = mystrlen(hash_b64);
            if (hash_b64_len == 44) {
              /* Decode hash to verify it's valid base64 */
              int b64incount;
              int hashbinlen = b64_decode(hash_b64, salttmp, &b64incount);
              if (hashbinlen >= 32) {
                /* JudyJ key = lowercase hex of 32-byte hash */
                char hexkey[65];
                prmd5_64((unsigned char *)salttmp, hexkey); /* 32 bytes → 64 hex chars */
                hexkey[64] = 0;
                JSLI(PV, JUDYJ(JOB_OTM_SHA256), (unsigned char *)hexkey);
                /* Typesalt = "iter:salt" */
                snprintf(salttmp, MAXLINE, "%d:%.*s", iter, (int)(d2 - salt), salt);
                JSLI(PV, TYPESALT(JOB_OTM_SHA256), (unsigned char *)salttmp);
                if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_OTM_SHA256]++; }
                Foundcnt[JOB_OTM_SHA256]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* TELEGRAM-SHA256 (22301): $telegram$0*HASH64hex*SALT32hex */
    if (lf[JOB_TELEGRAM_SHA256] && len > 80 && strncmp(line, "$telegram$0*", 12) == 0) {
      char *hash_hex = line + 12;
      char *sep1 = strchr(hash_hex, '*');
      if (sep1 && (sep1 - hash_hex) == 64) {
        char *salt_hex = sep1 + 1;
        int salt_hex_len = mystrlen(salt_hex);
        if (salt_hex_len == 32) {
          /* JudyJ key = lowercase hash hex */
          char hexkey[65];
          memcpy(hexkey, hash_hex, 64);
          hexkey[64] = 0;
          { char *hp; for (hp = hexkey; *hp; hp++) *hp = tolower(*hp); }
          JSLI(PV, JUDYJ(JOB_TELEGRAM_SHA256), (unsigned char *)hexkey);
          /* Typesalt = lowercase salt hex (32 chars) */
          char saltkey[33];
          memcpy(saltkey, salt_hex, 32);
          saltkey[32] = 0;
          { char *hp; for (hp = saltkey; *hp; hp++) *hp = tolower(*hp); }
          JSLI(PV, TYPESALT(JOB_TELEGRAM_SHA256), (unsigned char *)saltkey);
          if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_TELEGRAM_SHA256]++; }
          Foundcnt[JOB_TELEGRAM_SHA256]++;
          continue;
        }
      }
    }
    /* WEB2PY-SHA512 (21600): pbkdf2(ITER,20,sha512)$SALT$HASH40hex */
    if (lf[JOB_WEB2PY_SHA512] && len > 30 && strncmp(line, "pbkdf2(", 7) == 0) {
      char *iter_s = line + 7;
      char *comma1 = strchr(iter_s, ',');
      if (comma1) {
        int iter = atoi(iter_s);
        if (iter > 0 && strncmp(comma1, ",20,sha512)$", 12) == 0) {
          char *salt = comma1 + 12;
          char *sep = strchr(salt, '$');
          if (sep && sep > salt) {
            char *hash_hex = sep + 1;
            int hash_hex_len = mystrlen(hash_hex);
            if (hash_hex_len == 40) {
              /* JudyJ key = lowercase hash hex (40 chars) */
              char hexkey[41];
              memcpy(hexkey, hash_hex, 40);
              hexkey[40] = 0;
              { char *hp; for (hp = hexkey; *hp; hp++) *hp = tolower(*hp); }
              JSLI(PV, JUDYJ(JOB_WEB2PY_SHA512), (unsigned char *)hexkey);
              /* Typesalt = "iter:salt" */
              int slen = sep - salt;
              snprintf(salttmp, MAXLINE, "%d:%.*s", iter, slen, salt);
              JSLI(PV, TYPESALT(JOB_WEB2PY_SHA512), (unsigned char *)salttmp);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_WEB2PY_SHA512]++; }
              Foundcnt[JOB_WEB2PY_SHA512]++;
              continue;
            }
          }
        }
      }
    }
    /* SOLARWINDS (21500): $solarwinds$0$USERNAME$BASE64HASH */
    if (lf[JOB_SOLARWINDS] && len > 20 && strncmp(line, "$solarwinds$0$", 14) == 0) {
      char *user = line + 14;
      char *sep = strchr(user, '$');
      if (sep && sep > user) {
        char *hash_b64 = sep + 1;
        int hash_b64_len = mystrlen(hash_b64);
        if (hash_b64_len == 88) {
          /* Decode hash to get 64 bytes of SHA512 */
          int b64ic;
          int hashbinlen = b64_decode(hash_b64, salttmp, &b64ic);
          if (hashbinlen >= 64) {
            /* JudyJ key = lowercase hex of 64-byte SHA512 hash */
            char hexkey[129];
            prmd5_128((unsigned char *)salttmp, hexkey);
            hexkey[128] = 0;
            JSLI(PV, JUDYJ(JOB_SOLARWINDS), (unsigned char *)hexkey);
            /* Typesalt = username (will be processed: lc, pad/trunc to 8) */
            int ulen = sep - user;
            snprintf(salttmp, MAXLINE, "%.*s", ulen, user);
            JSLI(PV, TYPESALT(JOB_SOLARWINDS), (unsigned char *)salttmp);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_SOLARWINDS]++; }
            Foundcnt[JOB_SOLARWINDS]++;
            continue;
          }
        }
      }
    }
    /* SOLARWINDS2 (21501): $solarwinds$1$SALT_B64$HASH_B64 */
    if (lf[JOB_SOLARWINDS2] && len > 20 && strncmp(line, "$solarwinds$1$", 14) == 0) {
      char *salt_b64 = line + 14;
      char *sep = strchr(salt_b64, '$');
      if (sep && sep > salt_b64) {
        char *hash_b64 = sep + 1;
        int hash_b64_len = mystrlen(hash_b64);
        if (hash_b64_len == 88) {
          /* Decode hash */
          int b64ic;
          int hashbinlen = b64_decode(hash_b64, salttmp, &b64ic);
          if (hashbinlen >= 64) {
            char hexkey[129];
            prmd5_128((unsigned char *)salttmp, hexkey);
            hexkey[128] = 0;
            JSLI(PV, JUDYJ(JOB_SOLARWINDS2), (unsigned char *)hexkey);
            /* Typesalt = base64 salt (decoded to hex for storage) */
            int sb64len = sep - salt_b64;
            char salt_b64_copy[64];
            memcpy(salt_b64_copy, salt_b64, sb64len);
            salt_b64_copy[sb64len] = 0;
            int saltbinlen = b64_decode(salt_b64_copy, salttmp + 256, &b64ic);
            if (saltbinlen == 16) {
              /* Store as 32-char hex */
              char salthex[33];
              prmd5_32((unsigned char *)salttmp + 256, salthex);
              salthex[32] = 0;
              JSLI(PV, TYPESALT(JOB_SOLARWINDS2), (unsigned char *)salthex);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_SOLARWINDS2]++; }
              Foundcnt[JOB_SOLARWINDS2]++;
              continue;
            }
          }
        }
      }
    }
    /* SIMPLACMS (22800): md5($salt.$pass.md5($pass)) -- stored form 32hex:salt.
     * No special case here, deliberately: the stored form IS <hex>:<salt>, which
     * the generic hex loader already consumes -- digest into the compact table,
     * remainder into Typesalt. The branch that used to sit here claimed the line,
     * put the digest in the per-type Judy array and continued, so the type was
     * loadable ONLY as a combined record through -F. Reformatting types such as
     * Werkzeug MD5 reach that same generic loader by rewriting themselves into
     * <hex>:<salt> and falling through, which is the pattern this type needs. */
    /* APPLE-KEYCHAIN (23100): $keychain$*SALT40hex*IV16hex*DATA96hex */
    if (lf[JOB_APPLE_KEYCHAIN] && len > 60 && strncmp(line, "$keychain$*", 11) == 0) {
      char *f1 = line + 11; /* salt */
      char *s1 = strchr(f1, '*');
      if (s1 && (s1 - f1) == 40) {
        char *iv = s1 + 1;
        char *s2 = strchr(iv, '*');
        if (s2 && (s2 - iv) == 16) {
          char *data = s2 + 1;
          int datalen = mystrlen(data);
          if (datalen == 96) {
            /* JudyJ key = lowercase data hex (what we verify after decrypt) */
            char judykey[97];
            memcpy(judykey, data, 96);
            judykey[96] = 0;
            { char *hp; for (hp = judykey; *hp; hp++) *hp = tolower(*hp); }
            JSLI(PV, JUDYJ(JOB_APPLE_KEYCHAIN), (unsigned char *)judykey);
            /* Typesalt = "salt40hex:iv16hex" */
            snprintf(salttmp, MAXLINE, "%.*s:%.*s", 40, f1, 16, iv);
            { char *hp; for (hp = salttmp; *hp; hp++) *hp = tolower(*hp); }
            JSLI(PV, TYPESALT(JOB_APPLE_KEYCHAIN), (unsigned char *)salttmp);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_APPLE_KEYCHAIN]++; }
            Foundcnt[JOB_APPLE_KEYCHAIN]++;
            continue;
          }
        }
      }
    }
    /* APPLE-IWORK (23300): $iwork$hash_ver$file_ver$1$iter$SALThex$IV32hex$DATA128hex */
    if (lf[JOB_APPLE_IWORK] && len > 60 && strncmp(line, "$iwork$", 7) == 0) {
      char *p = line + 7;
      /* Parse: hash_ver$file_ver$1$iter$salt$iv$data */
      int hash_ver = 0, fmt_ver = 0, iter = 0;
      char *d1, *d2, *d3, *d4, *d5, *d6;
      d1 = strchr(p, '$'); /* after hash_ver */
      if (d1) { hash_ver = atoi(p); d2 = strchr(d1 + 1, '$'); }
      if (d1 && d2) { d3 = strchr(d2 + 1, '$'); }
      if (d1 && d2 && d3) { fmt_ver = atoi(d2 + 1); d4 = strchr(d3 + 1, '$'); }
      if (d1 && d2 && d3 && d4 && fmt_ver == 1) {
        iter = atoi(d3 + 1);
        d5 = strchr(d4 + 1, '$'); /* after salt */
        if (d5 && iter > 0) {
          d6 = strchr(d5 + 1, '$'); /* after IV */
          if (d6 && (d6 - d5 - 1) == 32) {
            char *data = d6 + 1;
            int datalen = mystrlen(data);
            int saltlen = d5 - d4 - 1;
            if (datalen == 128 && saltlen >= 16 && saltlen <= 32) {
              /* JudyJ key = lowercase data hex */
              char judykey[129];
              memcpy(judykey, data, 128);
              judykey[128] = 0;
              { char *hp; for (hp = judykey; *hp; hp++) *hp = tolower(*hp); }
              JSLI(PV, JUDYJ(JOB_APPLE_IWORK), (unsigned char *)judykey);
              /* Typesalt = "hash_ver:iter:salthex:iv32hex" */
              snprintf(salttmp, MAXLINE, "%d:%d:%.*s:%.*s",
                       hash_ver, iter, saltlen, d4 + 1, 32, d5 + 1);
              { char *hp; for (hp = salttmp; *hp; hp++) *hp = tolower(*hp); }
              JSLI(PV, TYPESALT(JOB_APPLE_IWORK), (unsigned char *)salttmp);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_APPLE_IWORK]++; }
              Foundcnt[JOB_APPLE_IWORK]++;
              continue;
            }
          }
        }
      }
    }
    /* BITWARDEN (23400): $bitwarden$2*iter1*iter2*B64_SALT*B64_HASH */
    if (lf[JOB_BITWARDEN] && len > 30 && strncmp(line, "$bitwarden$2*", 13) == 0) {
      char *p = line + 13;
      char *s1 = strchr(p, '*');
      if (s1) {
        int iter1 = atoi(p);
        char *s2 = strchr(s1 + 1, '*');
        if (s2 && iter1 > 0) {
          int iter2 = atoi(s1 + 1);
          char *salt_b64 = s2 + 1;
          char *s3 = strchr(salt_b64, '*');
          if (s3 && iter2 > 0) {
            char *hash_b64 = s3 + 1;
            int hash_b64_len = mystrlen(hash_b64);
            if (hash_b64_len == 44) {
              /* Decode hash */
              int b64ic;
              int hashbinlen = b64_decode(hash_b64, salttmp + 512, &b64ic);
              if (hashbinlen >= 32) {
                /* JudyJ key = lowercase hex of 32-byte hash */
                char hexkey[65];
                prmd5_64((unsigned char *)salttmp + 512, hexkey);
                hexkey[64] = 0;
                JSLI(PV, JUDYJ(JOB_BITWARDEN), (unsigned char *)hexkey);
                /* Typesalt = "iter1:iter2:salt_b64" */
                int sb64len = s3 - salt_b64;
                snprintf(salttmp, MAXLINE, "%d:%d:%.*s", iter1, iter2, sb64len, salt_b64);
                JSLI(PV, TYPESALT(JOB_BITWARDEN), (unsigned char *)salttmp);
                if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_BITWARDEN]++; }
                Foundcnt[JOB_BITWARDEN]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* MONGODB-SHA1 (24100): $mongodb-scram$*0*B64_USER*ITER*B64_SALT*B64_HASH */
    if (lf[JOB_MONGODB_SHA1] && len > 40 && strncmp(line, "$mongodb-scram$*0*", 18) == 0) {
      char *user_b64 = line + 18;
      char *s1 = strchr(user_b64, '*');
      if (s1) {
        int iter = atoi(s1 + 1);
        char *s2 = strchr(s1 + 1, '*');
        if (s2 && iter > 0) {
          char *salt_b64 = s2 + 1;
          char *s3 = strchr(salt_b64, '*');
          if (s3) {
            char *hash_b64 = s3 + 1;
            int hash_b64_len = mystrlen(hash_b64);
            if (hash_b64_len >= 24 && hash_b64_len <= 32) {
              /* Decode hash (20 bytes) */
              int b64ic;
              int hashbinlen = b64_decode(hash_b64, salttmp + 512, &b64ic);
              if (hashbinlen >= 20) {
                /* JudyJ key = lowercase hex of 20-byte hash */
                char hexkey[41];
                prmd5_40((unsigned char *)salttmp + 512, hexkey);
                hexkey[40] = 0;
                JSLI(PV, JUDYJ(JOB_MONGODB_SHA1), (unsigned char *)hexkey);
                /* Decode user and salt for Typesalt */
                char userbuf[128];
                int userlen = b64_decode(user_b64, userbuf, &b64ic);
                if (userlen < 0) userlen = 0;
                if (userlen > 57) userlen = 57;
                userbuf[userlen] = 0;
                /* Decode salt */
                int sb64len = s3 - salt_b64;
                char saltbuf_b64[64];
                memcpy(saltbuf_b64, salt_b64, sb64len);
                saltbuf_b64[sb64len] = 0;
                unsigned char saltbin[32];
                int saltbinlen = b64_decode(saltbuf_b64, (char *)saltbin, &b64ic);
                if (saltbinlen == 16) {
                  /* Typesalt = "user:iter:salthex" */
                  char salthex[33];
                  prmd5_32(saltbin, salthex);
                  salthex[32] = 0;
                  snprintf(salttmp, MAXLINE, "%s:%d:%s", userbuf, iter, salthex);
                  JSLI(PV, TYPESALT(JOB_MONGODB_SHA1), (unsigned char *)salttmp);
                  if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_MONGODB_SHA1]++; }
                  Foundcnt[JOB_MONGODB_SHA1]++;
                  continue;
                }
              }
            }
          }
        }
      }
    }
    /* MONGODB-SHA256 (24200): $mongodb-scram$*1*B64_USER*ITER*B64_SALT*B64_HASH */
    if (lf[JOB_MONGODB_SHA256] && len > 40 && strncmp(line, "$mongodb-scram$*1*", 18) == 0) {
      char *user_b64 = line + 18;
      char *s1 = strchr(user_b64, '*');
      if (s1) {
        int iter = atoi(s1 + 1);
        char *s2 = strchr(s1 + 1, '*');
        if (s2 && iter > 0) {
          char *salt_b64 = s2 + 1;
          char *s3 = strchr(salt_b64, '*');
          if (s3) {
            char *hash_b64 = s3 + 1;
            int hash_b64_len = mystrlen(hash_b64);
            if (hash_b64_len >= 40 && hash_b64_len <= 48) {
              /* Decode hash (32 bytes) */
              int b64ic;
              int hashbinlen = b64_decode(hash_b64, salttmp + 512, &b64ic);
              if (hashbinlen >= 32) {
                /* JudyJ key = lowercase hex of 32-byte hash */
                char hexkey[65];
                prmd5_64((unsigned char *)salttmp + 512, hexkey);
                hexkey[64] = 0;
                JSLI(PV, JUDYJ(JOB_MONGODB_SHA256), (unsigned char *)hexkey);
                /* Decode salt */
                int sb64len = s3 - salt_b64;
                char saltbuf_b64[64];
                memcpy(saltbuf_b64, salt_b64, sb64len);
                saltbuf_b64[sb64len] = 0;
                unsigned char saltbin[32];
                int saltbinlen = b64_decode(saltbuf_b64, (char *)saltbin, &b64ic);
                if (saltbinlen == 28) {
                  /* Typesalt = "iter:salthex" (user not needed for computation) */
                  char salthex[57];
                  prmd5_56(saltbin, salthex);
                  salthex[56] = 0;
                  snprintf(salttmp, MAXLINE, "%d:%s", iter, salthex);
                  JSLI(PV, TYPESALT(JOB_MONGODB_SHA256), (unsigned char *)salttmp);
                  if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_MONGODB_SHA256]++; }
                  Foundcnt[JOB_MONGODB_SHA256]++;
                  continue;
                }
              }
            }
          }
        }
      }
    }
    /* SAP-PASSCODE (7800): SALT$40HEX_HASH — salt 1-12 chars */
    if (lf[JOB_SAP_PASSCODE] && len >= 3) {
      char *dollar = memchr(line, '$', len);
      if (dollar) {
        int sp_slen = dollar - line;
        int sp_hlen = len - sp_slen - 1;
        if (sp_slen >= 1 && sp_slen <= 12 && sp_hlen == 40) {
          int hexok = 1;
          for (y = sp_slen + 1; y < len; y++)
            if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
          if (hexok) {
            /* JudyJ key = lowercase 40-hex hash */
            char sp_hashkey[41];
            for (y = 0; y < 40; y++)
              sp_hashkey[y] = tolower(line[sp_slen + 1 + y]);
            sp_hashkey[40] = 0;
            JSLI(PV, JUDYJ(JOB_SAP_PASSCODE), (unsigned char *)sp_hashkey);
            /* Typesalt = salt string (as-is) */
            Saltloaded[JOB_SAP_PASSCODE] += store_typesalt(JOB_SAP_PASSCODE, line, sp_slen);
            Foundcnt[JOB_SAP_PASSCODE]++;
            continue;
          }
        }
      }
    }
    /* SAP-PASSCODE5 (7801): SALT$10HEX_HASH — salt 1-12 chars, truncated hash */
    if (lf[JOB_SAP_PASSCODE5] && len >= 3) {
      char *dollar = memchr(line, '$', len);
      if (dollar) {
        int sp_slen = dollar - line;
        int sp_hlen = len - sp_slen - 1;
        if (sp_slen >= 1 && sp_slen <= 12 && sp_hlen == 40) {
          int hexok = 1;
          for (y = sp_slen + 1; y < len; y++)
            if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
          if (hexok) {
            /* JudyJ key = lowercase 40-hex hash (full length stored) */
            char sp_hashkey[41];
            for (y = 0; y < 40; y++)
              sp_hashkey[y] = tolower(line[sp_slen + 1 + y]);
            sp_hashkey[40] = 0;
            JSLI(PV, JUDYJ(JOB_SAP_PASSCODE5), (unsigned char *)sp_hashkey);
            /* Typesalt = salt string (as-is) */
            Saltloaded[JOB_SAP_PASSCODE5] += store_typesalt(JOB_SAP_PASSCODE5, line, sp_slen);
            Foundcnt[JOB_SAP_PASSCODE5]++;
            continue;
          }
        }
      }
    }
    /* SAP-BCODE (7700): SALT$16HEX_HASH — salt 1-12 chars, 8-byte hash */
    if (lf[JOB_SAP_BCODE] && len >= 3) {
      char *dollar = memchr(line, '$', len);
      if (dollar) {
        int sb_slen = dollar - line;
        int sb_hlen = len - sb_slen - 1;
        if (sb_slen >= 1 && sb_slen <= 12 && sb_hlen == 16) {
          int hexok = 1;
          for (y = sb_slen + 1; y < len; y++)
            if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
          if (hexok) {
            /* JudyJ key = uppercase 16-hex hash */
            char sb_hashkey[17];
            for (y = 0; y < 16; y++)
              sb_hashkey[y] = toupper(line[sb_slen + 1 + y]);
            sb_hashkey[16] = 0;
            JSLI(PV, JUDYJ(JOB_SAP_BCODE), (unsigned char *)sb_hashkey);
            Saltloaded[JOB_SAP_BCODE] += store_typesalt(JOB_SAP_BCODE, line, sb_slen);
            Foundcnt[JOB_SAP_BCODE]++;
            continue;
          }
        }
      }
    }
    /* SAP-BCODE4 (7701): SALT$16HEX_HASH — salt 1-12 chars, 4-byte hash (zero-padded to 16 hex) */
    if (lf[JOB_SAP_BCODE4] && len >= 3) {
      char *dollar = memchr(line, '$', len);
      if (dollar) {
        int sb_slen = dollar - line;
        int sb_hlen = len - sb_slen - 1;
        if (sb_slen >= 1 && sb_slen <= 12 && sb_hlen == 16) {
          int hexok = 1;
          for (y = sb_slen + 1; y < len; y++)
            if (trhex[(unsigned char)line[y]] > 15) { hexok = 0; break; }
          if (hexok) {
            char sb_hashkey[17];
            for (y = 0; y < 16; y++)
              sb_hashkey[y] = toupper(line[sb_slen + 1 + y]);
            sb_hashkey[16] = 0;
            JSLI(PV, JUDYJ(JOB_SAP_BCODE4), (unsigned char *)sb_hashkey);
            Saltloaded[JOB_SAP_BCODE4] += store_typesalt(JOB_SAP_BCODE4, line, sb_slen);
            Foundcnt[JOB_SAP_BCODE4]++;
            continue;
          }
        }
      }
    }
    /* AS400-DES (8501): $as400$des$*USER*16HEXHASH */
    if (lf[JOB_AS400_DES] && strncmp(line, "$as400$des$*", 12) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      char *u_start = line + 12;
      char *u_end = strchr(u_start, '*');
      if (u_end && (u_end - u_start) >= 1 && (u_end - u_start) <= 10) {
        int hashlen_a = sl - (int)(u_end + 1 - line);
        if (hashlen_a == 16) {
          int hexok = 1;
          for (y = 0; y < 16; y++)
            if (trhex[(unsigned char)u_end[1+y]] > 15) { hexok = 0; break; }
          if (hexok) {
            /* JudyJ key = full format string */
            JSLI(PV, JUDYJ(JOB_AS400_DES), (unsigned char *)line);
            /* Typesalt = username */
            int ulen = u_end - u_start;
            Saltloaded[JOB_AS400_DES] += store_typesalt(JOB_AS400_DES, u_start, ulen);
            Foundcnt[JOB_AS400_DES]++;
            continue;
          }
        }
      }
    }
    /* Domino 6 (8700): (Gxxxxxxxxxxxxxxxxxx) — 22 chars total */
    if (lf[JOB_DOMINO6] && line[0] == '(' && line[1] == 'G') {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      if (sl == 22 && line[21] == ')') {
        /* decode, normalize padding bits, then store */
        { unsigned char d6_dec[16];
          int dret = lotus64_decode(line + 2, d6_dec, 19);
          if (dret >= 14) {
            /* re-encode to normalize padding bits in last char */
            char d6_norm[24];
            d6_norm[0] = '('; d6_norm[1] = 'G';
            lotus64_encode(d6_dec, 14, d6_norm + 2, 20);
            d6_norm[21] = ')'; d6_norm[22] = 0;
            JSLI(PV, JUDYJ(JOB_DOMINO6), (unsigned char *)d6_norm);
            /* extract 5-byte salt (with quirk removed) as hex into Typesalt */
            d6_dec[3] -= 4;
            char salthex[11];
            for (y = 0; y < 5; y++) {
              salthex[y*2]   = hextab[(d6_dec[y]>>4)&0xf];
              salthex[y*2+1] = hextab[d6_dec[y]&0xf];
            }
            salthex[10] = 0;
            JSLI(PV, TYPESALT(JOB_DOMINO6), (unsigned char *)salthex);
            if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_DOMINO6]++;} }
          }
        }
        Foundcnt[JOB_DOMINO6]++;
        continue;
      }
    }
    /* scrypt (8900): SCRYPT:N:r:p:base64_salt:base64_hash */
    if (lf[JOB_SCRYPT] && strncmp(line, "SCRYPT:", 7) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      /* verify at least 6 colon-separated fields */
      { int nc = 0;
        char *p = line;
        while (*p) { if (*p == ':') nc++; p++; }
        if (nc >= 5) {
          JSLI(PV, JUDYJ(JOB_SCRYPT), (unsigned char *)line);
          /* extract SCRYPT:N:r:p:base64_salt into Typesalt */
          { char *lastsep = strrchr(line, ':');
            if (lastsep && lastsep > line) {
              char savec = *lastsep;
              *lastsep = 0;
              JSLI(PV, TYPESALT(JOB_SCRYPT), (unsigned char *)line);
              if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_SCRYPT]++;} }
              *lastsep = savec;
            }
          }
          Foundcnt[JOB_SCRYPT]++;
          continue;
        }
      }
    }
    /* SCRYPT (8900) alternate input format: the $7$ crypt encoding used
     * by libxcrypt / yescrypt.
     *
     *   $7$<log2N:1><r:5><p:5><salt>$<hash:43>
     *
     * Params are crypt64 (./0-9A-Za-z, little-endian 6-bit groups), the
     * salt is RAW ASCII -- not base64, unlike every other field here --
     * and the hash is 43 crypt64 chars decoding to 32 bytes. See
     * yescrypt/yescrypt-common.c:335 for the reference parse.
     *
     * Normalized into the canonical SCRYPT:N:r:p:b64salt:b64hash key so
     * that procjob, -z, the GPU path and the rest of e884 are untouched.
     * Same scrypt parameters, different spelling on the wire. */
    if (lf[JOB_SCRYPT] && strncmp(line, "$7$", 3) == 0) {
      int sl = mystrlen(line);
      while (sl > 0 && (line[sl-1] == '\r' || line[sl-1] == '\n'))
        line[--sl] = 0;
      /* 3 prefix + 11 param chars + >=1 salt + '$' + 43 hash */
      if (sl >= 3 + 11 + 1 + 1 + 43) {
        char *lastsep = strrchr(line, '$');
        const char *pfields = line + 3;
        const char *saltp = line + 3 + 11;
        if (lastsep && lastsep > saltp && (int)(sl - (lastsep + 1 - line)) == 43) {
          unsigned int n_log2v = 0, sc_r = 0, sc_p = 0;
          int saltlen7 = (int)(lastsep - saltp);
          int nlv = crypt64_val((unsigned char)pfields[0]);
          if (nlv >= 1 && nlv <= 63 && saltlen7 > 0 &&
              crypt64_uint32(pfields + 1, 5, &sc_r) == 0 &&
              crypt64_uint32(pfields + 6, 5, &sc_p) == 0 &&
              sc_r > 0 && sc_p > 0) {
            unsigned char hashbin7[40];
            char b64salt7[512], b64hash7[128], key7[MAXLINE];
            n_log2v = (unsigned int)nlv;
            if (crypt64_bytes(lastsep + 1, 43, hashbin7, sizeof(hashbin7)) == 32 &&
                saltlen7 < (int)(sizeof(b64salt7) / 2)) {
              /* b64_encode() takes a length, not a NUL-terminated string,
               * so the raw salt bytes go in directly. */
              b64_encode((char *)saltp, b64salt7, saltlen7);
              b64_encode((char *)hashbin7, b64hash7, 32);
              snprintf(key7, MAXLINE, "SCRYPT:%llu:%u:%u:%s:%s",
                       (unsigned long long)1ULL << n_log2v,
                       sc_r, sc_p, b64salt7, b64hash7);
              JSLI(PV, JUDYJ(JOB_SCRYPT), (unsigned char *)key7);
              { char *ls2 = strrchr(key7, ':');
                if (ls2 && ls2 > key7) {
                  *ls2 = 0;
                  JSLI(PV, TYPESALT(JOB_SCRYPT), (unsigned char *)key7);
                  if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_SCRYPT]++;} }
                }
              }
              Foundcnt[JOB_SCRYPT]++;
              continue;
            }
          }
        }
      }
    }
    /* A bare bcrypt setting, $2a$NN$ and the 22-character salt ($2k$: 21),
     * supplies a SALT and no hash.  It goes into Typesalt exactly as the
     * full-MCF arm below would file its prefix, so -z generates with it,
     * and into nothing else: JudyJ holds the MCFs that can be matched. */
    if (Arg_have_z && (lf[JOB_BCRYPT] || lf[JOB_BCRYPT256] || lf[JOB_BCRYPTMD5] || lf[JOB_BCRYPTSHA1] || lf[JOB_BCRYPTSHA512]) &&
        line[0] == '$' && line[1] == '2' && line[3] == '$' &&
        (line[2] == 'a' || line[2] == 'b' || line[2] == 'y' ||
         line[2] == 'x' || line[2] == 'k') &&
        len == ((line[2] == 'k') ? 28 : 29) &&
        isdigit((unsigned char)line[4]) && isdigit((unsigned char)line[5]) &&
        line[6] == '$') {
      static const int bs_types[] = { JOB_BCRYPT, JOB_BCRYPT256,
          JOB_BCRYPTMD5, JOB_BCRYPTSHA1, JOB_BCRYPTSHA512, -1 };
      int bs_ok = 1, bs_i, filed = 0;
      for (bs_i = 7; bs_i < len; bs_i++)
        if (!strchr(phpitoa64, line[bs_i])) { bs_ok = 0; break; }
      if (bs_ok) {
        for (bs_i = 0; bs_types[bs_i] >= 0; bs_i++) {
          int bj = bs_types[bs_i];
          if (!lf[bj]) continue;
          Saltloaded[bj] += store_typesalt(bj, line, len);
          filed = 1;
        }
        if (filed) continue;
      }
    }
    /* bcrypt: $2a$, $2b$, $2y$, $2x$, $2k$ */
    if ((lf[JOB_BCRYPT] || lf[JOB_BCRYPT256] || lf[JOB_BCRYPTMD5] || lf[JOB_BCRYPTSHA1] || lf[JOB_BCRYPTSHA512] ||
         lf[JOB_BCRYPTMD5PASSSALT]) &&
        len >= 58 && line[0] == '$' && line[1] == '2' &&
        (line[2] == 'a' || line[2] == 'b' || line[2] == 'y' ||
         line[2] == 'x' || line[2] == 'k') && line[3] == '$') {
      /* e1059 BCRYPTMD5PASSSALT stores a SITE salt after the MCF:
       *   <60-char MCF>:<site salt>
       * It has to be captured HERE, before the line[60] = 0 below throws
       * the remainder away.  The bcrypt salt and the cost still come out of
       * the MCF -- crypt_rn is handed the stored string as its setting -- so
       * what this salt adds is the separate per-record site value folded
       * into the md5 preimage.  It gets its OWN Judy array rather than the
       * family's shared JudyJ[JOB_BCRYPT]: a found MCF here is only
       * meaningful paired with a site salt, and the shared array has nowhere
       * to put one.  The salt text is stored VERBATIM including any $HEX[]
       * wrapper, because build_salt_snapshot already decodes that form; a
       * line this type emitted for a salt with high or control bytes thus
       * reloads without a second decoder here.
       *
       * A bare MCF with no trailing salt still loads -- that is the -s / -S
       * channel, where the pairing is unknown and the site salt comes from
       * the salt pool.  Typesalt stays empty in that case, which is exactly
       * what lets the SaltArray copy in main() fill it. */
      if (lf[JOB_BCRYPTMD5PASSSALT]) {
        int mcflen = (line[2] == 'k') ? 59 : 60;
        char bc_mcf[64];
        if (len >= mcflen) {
          if (len > mcflen + 1 && line[mcflen] == ':') {
            char *sp = line + mcflen + 1;
            int splen = len - mcflen - 1;
            if (splen > 0 && splen < MAXLINE) {
              JSLI(PV, TYPESALT(JOB_BCRYPTMD5PASSSALT), (unsigned char *)sp);
              if (PV) { if ((*PV)++ == 0) {
                Saltloaded[JOB_BCRYPTMD5PASSSALT]++;
                Typesaltcnt[JOB_BCRYPTMD5PASSSALT]++;
                Typesaltbytes[JOB_BCRYPTMD5PASSSALT] += splen + 1;
              } }
            }
          }
          memcpy(bc_mcf, line, mcflen);
          bc_mcf[mcflen] = 0;
          JSLI(PV, JUDYJ(JOB_BCRYPTMD5PASSSALT), (unsigned char *)bc_mcf);
          Foundcnt[JOB_BCRYPTMD5PASSSALT]++;
        }
      }
      /* Scoped to the five types that share JudyJ[JOB_BCRYPT].  Without this
       * guard, adding e1059 to the branch condition above made an e1059-only
       * run file every record under BCRYPT as well -- which reported the hashes
       * twice and announced a type the invocation had not selected. */
      if (lf[JOB_BCRYPT] || lf[JOB_BCRYPT256] || lf[JOB_BCRYPTMD5] ||
          lf[JOB_BCRYPTSHA1] || lf[JOB_BCRYPTSHA512]) {
        /* $2k$ is 59 characters, every other variant 60 -- the same split
         * the e1059 arm above already makes with mcflen.  A flat 60 left a
         * trailing ':' welded to the key of any $2k$ record that carried a
         * tail, so e577 BCRYPT256 loaded a hash that could never match. */
        { int bclen = (line[2] == 'k') ? 59 : 60;
          if (len > bclen) line[bclen] = 0; }
        JSLI(PV, JUDYJ(JOB_BCRYPT),(unsigned char *)line);
        /* extract salt prefix into Typesalt[job->op] */
        /* $2k$ has 21-char salt (28 prefix), others have 22 (29 prefix) */
        { char bc_salt[30];
          int pfx = (line[2] == 'k') ? 28 : 29;
          memcpy(bc_salt, line, pfx);
          bc_salt[pfx] = 0;
          /* Populate Typesalt for each selected BCRYPT type that uses $2a$ salts */
          static const int bc_types[] = { JOB_BCRYPT, JOB_BCRYPT256,
              JOB_BCRYPTMD5, JOB_BCRYPTSHA1, JOB_BCRYPTSHA512, -1 };
          for (int ci = 0; bc_types[ci] >= 0; ci++) {
            int bj = bc_types[ci];
            if (!lf[bj]) continue;
            JSLI(PV, TYPESALT(bj), (unsigned char *)bc_salt);
            if (PV) { if ((*PV)++ == 0) {
              Saltloaded[bj]++;
              Typesaltcnt[bj]++;
              Typesaltbytes[bj] += pfx + 1;
            } }
          }
        }
        Foundcnt[JOB_BCRYPT]++;
      }
      continue;
    }
    /* BCRYPTHMACSHA256 (30601): $bcrypt-sha256$v=2,t=2b,r=NN$SALT22$HASH31 */
    if (lf[JOB_BCRYPTHMACSHA256] && len >= 60 &&
        strncmp(line, "$bcrypt-sha256$v=2,t=2b,r=", 26) == 0) {
      const char *rp = line + 26;
      const char *d1 = strchr(rp, '$');
      if (d1 && (d1 - rp) <= 2) {
        int rounds = atoi(rp);
        if (rounds >= 1 && rounds <= 31) {
          const char *salt22 = d1 + 1;
          if (mystrlen(salt22) >= 22 + 1 + 31 && salt22[22] == '$') {
            /* JudyJ key = bcrypt output: $2b$NN$salt22hash31 */
            char bcout[64];
            snprintf(bcout, sizeof(bcout), "$2b$%02d$%.22s%.31s", rounds, salt22, salt22 + 23);
            JSLI(PV, JUDYJ(JOB_BCRYPTHMACSHA256), (unsigned char *)bcout);
            /* Typesalt = bcrypt setting: $2b$NN$salt22 */
            char bcsetting[32];
            snprintf(bcsetting, sizeof(bcsetting), "$2b$%02d$%.22s", rounds, salt22);
            JSLI(PV, TYPESALT(JOB_BCRYPTHMACSHA256), (unsigned char *)bcsetting);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_BCRYPTHMACSHA256]++; }
            Foundcnt[JOB_BCRYPTHMACSHA256]++;
            continue;
          }
        }
      }
    }
    /* A bare 2-character DES salt supplies a SALT and no hash.  It goes into
     * Typesalt for each selected DES-family type, so -z generates with the
     * supplied salts instead of all 4096. */
    if (Arg_have_z && len == 2 && line[0] && line[1] &&
        strchr(phpitoa64, line[0]) && strchr(phpitoa64, line[1])) {
      static const int ds_types[] = { JOB_DESCRYPT, JOB_MD5DESCRYPT,
          JOB_MD4DESCRYPT, JOB_SHA1DESCRYPT, JOB_MD4UTF16DESCRYPT,
          JOB_MD5USERDESCRYPT, -1 };
      int ds_i, filed = 0;
      for (ds_i = 0; ds_types[ds_i] >= 0; ds_i++) {
        int dj = ds_types[ds_i];
        if (!lf[dj]) continue;
        store_typesalt(dj, line, 2);
        rcpt_salt(Saltloaded, dj, line, 2);
        filed = 1;
      }
      if (filed) continue;
    }
    /* DES crypt: 13 phpitoa64 chars (standard 25-round DES) */
    if (lf[JOB_DESCRYPT] && (len == 13 && strchr(phpitoa64, line[0]))) {
      int desok = 1;
      for (y = 1; y < len; y++) {
        if (!strchr(phpitoa64, line[y])) { desok = 0; break; }
      }
      if (desok) {
        rcpt_salt(Saltloaded, JOB_DESCRYPT, line, 2);
        JSLI(PV, JUDYJ(JOB_DESCRYPT),(unsigned char *)line);
        /* Standard DES: decode base64 hash body to (r0,r1), apply IP for
         * pre-FP form, store in compact table for GPU (kernel skips FP) */
        if (inhashbuf && len == 13) {
          static const unsigned char DES_IP[64] = {
            58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,
            62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,
            57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,
            61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7
          };
          init_phpatoi64();
          unsigned char *dp = (unsigned char *)line + 2;
          unsigned int v0 = ((unsigned int)phpatoi64[dp[0]]<<18)|((unsigned int)phpatoi64[dp[1]]<<12)|
                            ((unsigned int)phpatoi64[dp[2]]<<6)|(unsigned int)phpatoi64[dp[3]];
          unsigned int v1 = ((unsigned int)phpatoi64[dp[4]]<<18)|((unsigned int)phpatoi64[dp[5]]<<12)|
                            ((unsigned int)phpatoi64[dp[6]]<<6)|(unsigned int)phpatoi64[dp[7]];
          unsigned int v2 = ((unsigned int)phpatoi64[dp[8]]<<12)|((unsigned int)phpatoi64[dp[9]]<<6)|
                            (unsigned int)phpatoi64[dp[10]];
          unsigned int dr0 = (v0 << 8) | (v1 >> 16);
          unsigned int dr1 = ((v1 & 0xFFFF) << 16) | (v2 >> 2);
          /* Apply DES IP to get pre-FP form (matches GPU output) */
          unsigned int il = 0, ir = 0;
          for (int di = 0; di < 32; di++) {
            int db = DES_IP[di] - 1;
            unsigned int ds = (db < 32) ? dr0 : dr1;
            if (ds & (1u << (31 - (db % 32)))) il |= (1u << (31 - di));
          }
          for (int di = 0; di < 32; di++) {
            int db = DES_IP[32+di] - 1;
            unsigned int ds = (db < 32) ? dr0 : dr1;
            if (ds & (1u << (31 - (db % 32)))) ir |= (1u << (31 - di));
          }
          memcpy(inhashbuf + 2, &il, 4);
          memcpy(inhashbuf + 6, &ir, 4);
          memset(inhashbuf + 10, 0, 8);
          commit_compact(16);
        }
        Foundcnt[JOB_DESCRYPT]++;
        continue;
      }
    }
    /* BSDi extended DES: 20 chars (_CCCCSSSSHHHHHHHHHHH), 9-char salt */
    if (lf[JOB_BSDICRYPT] && (len == 20 && line[0] == '_')) {
      int desok = 1;
      for (y = 1; y < len; y++) {
        if (!strchr(phpitoa64, line[y])) { desok = 0; break; }
      }
      if (desok) {
        JSLI(PV, JUDYJ(JOB_BSDICRYPT),(unsigned char *)line);
        /* Extract 9-char salt (_CCCCSSSS) for Typesalt */
        strncpy(salttmp, line, 9);
        salttmp[9] = 0;
        Saltloaded[JOB_BSDICRYPT] += store_typesalt(JOB_BSDICRYPT, salttmp, 9);
        Foundcnt[JOB_BSDICRYPT]++;
        continue;
      }
    }
    /* {SHA}base64: decode to 20-byte SHA1, write to compact table */
    if ((lf[JOB_APACHE_SHA] || lf[JOB_APACHE_SHA_TRUNC16]) &&
        *line == '{' && line[1] == 'S' && line[2] == 'H' &&
        line[3] == 'A' && line[4] == '}') {
      if (inhashbuf) {  /* only when compact table available (-F, not -J) */
        len = b64_decode(&line[5], (char *)(inhashbuf + 2), &cryptlen);
        /* Both types carry the same {SHA} tag, so the PAYLOAD LENGTH is the only
         * discriminator: a conforming RFC 2307 payload is 20 bytes (28 base64
         * characters), the truncated variant is 16 (24 characters).  Before
         * e1040 existed a 16-byte payload decoded here, failed the >= 20 test
         * and was silently dropped. */
        if (len >= 20 && lf[JOB_APACHE_SHA]) {
          len = 20;
          commit_compact(len);
          Foundcnt[JOB_APACHE_SHA]++;
          continue;
        }
        if (len == 16 && lf[JOB_APACHE_SHA_TRUNC16]) {
          commit_compact(len);
          Foundcnt[JOB_APACHE_SHA_TRUNC16]++;
          continue;
        }
      }
      continue;  /* skip {SHA} in -J mode (no compact table) */
    }
    /* {SSHA}base64: variable length, >20 decoded bytes */
    if (lf[JOB_SSHA1BASE64] && strncmp(line, "{SSHA}", 6) == 0 && len > 10) {
      JSLI(PV, JUDYJ(JOB_SSHA1BASE64), (unsigned char *)line);
      { char ssha_tmp[256];
        int ssha_dlen, ssha_ic;
        ssha_dlen = b64_decode(line + 6, ssha_tmp, &ssha_ic);
        if (ssha_dlen > 20) {
          char salthex[512];
          int si;
          for (si = 0; si < ssha_dlen - 20; si++) {
            salthex[si*2]   = hextab[((unsigned char)ssha_tmp[20+si]>>4)&0xf];
            salthex[si*2+1] = hextab[(unsigned char)ssha_tmp[20+si]&0xf];
          }
          salthex[(ssha_dlen-20)*2] = 0;
          JSLI(PV, TYPESALT(JOB_SSHA1BASE64), (unsigned char *)salthex);
          if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_SSHA1BASE64]++;} }
        }
      }
      Foundcnt[JOB_SSHA1BASE64]++;
      continue;
    }
    /* {SSHA256}base64 */
    if (lf[JOB_SSHA256BASE64] && strncmp(line, "{SSHA256}", 9) == 0 && len > 12) {
      JSLI(PV, JUDYJ(JOB_SSHA256BASE64), (unsigned char *)line);
      { char ssha_tmp[256];
        int ssha_dlen, ssha_ic;
        ssha_dlen = b64_decode(line + 9, ssha_tmp, &ssha_ic);
        if (ssha_dlen > 32) {
          char salthex[512];
          int si;
          for (si = 0; si < ssha_dlen - 32; si++) {
            salthex[si*2]   = hextab[((unsigned char)ssha_tmp[32+si]>>4)&0xf];
            salthex[si*2+1] = hextab[(unsigned char)ssha_tmp[32+si]&0xf];
          }
          salthex[(ssha_dlen-32)*2] = 0;
          JSLI(PV, TYPESALT(JOB_SSHA256BASE64), (unsigned char *)salthex);
          if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_SSHA256BASE64]++;} }
        }
      }
      Foundcnt[JOB_SSHA256BASE64]++;
      continue;
    }
    /* {SSHA512}base64 */
    if (lf[JOB_SSHA512BASE64] && strncmp(line, "{SSHA512}", 9) == 0 && len > 12) {
      JSLI(PV, JUDYJ(JOB_SSHA512BASE64), (unsigned char *)line);
      { char ssha_tmp[256];
        int ssha_dlen, ssha_ic;
        ssha_dlen = b64_decode(line + 9, ssha_tmp, &ssha_ic);
        if (ssha_dlen > 64) {
          char salthex[512];
          int si;
          for (si = 0; si < ssha_dlen - 64; si++) {
            salthex[si*2]   = hextab[((unsigned char)ssha_tmp[64+si]>>4)&0xf];
            salthex[si*2+1] = hextab[(unsigned char)ssha_tmp[64+si]&0xf];
          }
          salthex[(ssha_dlen-64)*2] = 0;
          JSLI(PV, TYPESALT(JOB_SSHA512BASE64), (unsigned char *)salthex);
          if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_SSHA512BASE64]++;} }
        }
      }
      Foundcnt[JOB_SSHA512BASE64]++;
      continue;
    }
    /* JudyJ[JOB_EPISERVER]: $episerver$*0*SALT_B64*HASH_B64 or *1* */
    if (lf[JOB_EPISERVER] && strncmp(line, "$episerver$*", 12) == 0 &&
        (line[12] == '0' || line[12] == '1') && line[13] == '*') {
      char *star2 = strchr(line + 14, '*');
      if (star2 && mystrlen(star2 + 1) >= 27) {
        JSLI(PV, JUDYJ(JOB_EPISERVER), (unsigned char *)line);
        /* extract "V*SALT_B64" into Typesalt */
        Saltloaded[JOB_EPISERVER] += store_typesalt(JOB_EPISERVER, line + 12, star2 - (line + 12));
        Foundcnt[JOB_EPISERVER]++;
        continue;
      }
    }
    /* hMailServer: 70 hex chars = 6 salt + 64 SHA256 */
    if (lf[JOB_HMAILSERVER] && len == 70) {
      int hms_ok = 1;
      for (y = 0; y < 70; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { hms_ok = 0; break; }
      }
      if (hms_ok) {
        JSLI(PV, JUDYJ(JOB_HMAILSERVER), (unsigned char *)line);
        Saltloaded[JOB_HMAILSERVER] += store_typesalt(JOB_HMAILSERVER, line, 6);
        Foundcnt[JOB_HMAILSERVER]++;
        continue;
      }
    }
    /* CiscoISE: 128 hex chars (or 130 with 0x prefix) */
    if (lf[JOB_CISCOISE] && (len == 128 || (len == 130 && line[0] == '0' && (line[1] == 'x' || line[1] == 'X')))) {
      char *hp = line;
      int hlen = 128;
      if (len == 130) { hp += 2; }
      { int cise_ok = 1;
        for (y = 0; y < hlen; y++) {
          if (trhex[(unsigned char)hp[y]] > 15) { cise_ok = 0; break; }
        }
        if (cise_ok) {
          JSLI(PV, JUDYJ(JOB_CISCOISE), (unsigned char *)hp);
          Saltloaded[JOB_CISCOISE] += store_typesalt(JOB_CISCOISE, hp + 64, 64);
          Foundcnt[JOB_CISCOISE]++;
          continue;
        }
      }
    }
    /* MediaWiki: $B$salt$hash */
    if (lf[JOB_MEDIAWIKI] && strncmp(line, "$B$", 3) == 0) {
      char *mw_hash = strchr(line + 3, '$');
      if (mw_hash && mystrlen(mw_hash + 1) >= 32) {
        int slen = mw_hash - (line + 3);
        if (slen > 0 && slen < 32) {
          JSLI(PV, JUDYJ(JOB_MEDIAWIKI), (unsigned char *)line);
          Saltloaded[JOB_MEDIAWIKI] += store_typesalt(JOB_MEDIAWIKI, line + 3, slen);
          Foundcnt[JOB_MEDIAWIKI]++;
          continue;
        }
      }
    }
    /* PHPS: $PHPS$hexsalt$hexhash — md5(md5(pass).salt) */
    if (lf[JOB_PHPS] && strncmp(line, "$PHPS$", 6) == 0) {
      char *phps_shex = line + 6;
      char *phps_hash = strchr(phps_shex, '$');
      if (phps_hash && mystrlen(phps_hash + 1) >= 32) {
        char phps_salt[32];
        int slen = 0;
        char *p;
        for (p = phps_shex; p < phps_hash && p[0] && p[1]; p += 2) {
          int hi = trhex[(unsigned char)p[0]];
          int lo = trhex[(unsigned char)p[1]];
          if (hi > 15 || lo > 15) break;
          phps_salt[slen++] = (hi << 4) | lo;
        }
        if (slen > 0) {
          char *hexhash = phps_hash + 1;
          int hlen = mystrlen(hexhash);
          if (hlen >= 32) {
            /* Store hash keyed by lowercase hexhash in JudyJ[JOB_PHPS] */
            char phps_key[33];
            for (y = 0; y < 32; y++)
              phps_key[y] = hexhash[y] | 0x20;
            phps_key[32] = 0;
            JSLI(PV, JUDYJ(JOB_PHPS), (unsigned char *)phps_key);
            if (PV && *PV == 0) {
              /* Store original $PHPS$... line as value for output */
              char *orig = malloc(mystrlen(line) + 1);
              if (!orig) { fprintf(stderr, "Out of memory\n"); exit(1); }
              strcpy(orig, line);
              *PV = (Word_t)orig;
            }
            /* Store decoded salt in Typesalt[JOB_PHPS] Judy */
            phps_salt[slen] = 0;
            JSLI(PV, TYPESALT(JOB_PHPS), (unsigned char *)phps_salt);
            if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_PHPS]++;} }
            Foundcnt[JOB_PHPS]++;
          }
        }
      }
      continue;
    }
    /* ProgressEncode: 16 alpha chars */
    if (lf[JOB_PROGRESSENCODE] && len == 16) {
      int prok = 1;
      for (y = 0; y < 16; y++) {
        if (!((line[y] >= 'A' && line[y] <= 'Z') || (line[y] >= 'a' && line[y] <= 'z'))) {
          prok = 0; break;
        }
      }
      if (prok) {
        JSLI(PV, JUDYJ(JOB_PROGRESSENCODE), (unsigned char *)line);
        Foundcnt[JOB_PROGRESSENCODE]++;
        continue;
      }
    }
    /* SHA1_CUSTOMUSERSALT: base64:username */
    if (lf[JOB_SHA1_CUSTOMUSERSALT]) {
      char *colon = strchr(line, ':');
      if (colon && (colon - line) >= 27 && (colon - line) <= 30) {
        int cusok = 1;
        for (y = 0; y < (colon - line); y++) {
          if (!strchr(b64, line[y]) && line[y] != '=') { cusok = 0; break; }
        }
        if (cusok && colon[1]) {
          JSLI(PV, TYPESALT(JOB_SHA1_CUSTOMUSERSALT), (unsigned char *)line);
          if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_SHA1_CUSTOMUSERSALT]++;} }
          Foundcnt[JOB_SHA1_CUSTOMUSERSALT]++;
          continue;
        }
      }
    }
    /* Python passlib $pbkdf2-sha512$, $pbkdf2-sha256$, $pbkdf2$ → convert to internal format */
    if ((lf[JOB_PKCS5S2] || lf[JOB_PBKDF2_SHA1] || lf[JOB_PBKDF2_SHA256] || lf[JOB_PBKDF2_SHA512]) &&
        (strncmp(line, "$pbkdf2-sha512$", 15) == 0 ||
         strncmp(line, "$pbkdf2-sha256$", 15) == 0 ||
         strncmp(line, "$pbkdf2$", 8) == 0)) {
      /* Parse: $pbkdf2[-algo]$iter$salt_ab64$hash_ab64
       * Convert to: algo:iter:salt_b64:hash_b64 (standard b64 with + and =) */
      const char *algo_str;
      int algo_len, job_id;
      char *iter_start;
      if (strncmp(line, "$pbkdf2-sha512$", 15) == 0) {
        algo_str = "sha512"; algo_len = 15; job_id = JOB_PBKDF2_SHA512;
      } else if (strncmp(line, "$pbkdf2-sha256$", 15) == 0) {
        algo_str = "sha256"; algo_len = 15; job_id = JOB_PBKDF2_SHA256;
      } else {
        algo_str = "sha1"; algo_len = 8; job_id = JOB_PBKDF2_SHA1;
      }
      iter_start = line + algo_len;
      char *d1 = strchr(iter_start, '$');
      if (d1) {
        int iter = atoi(iter_start);
        if (iter > 0) {
          char *salt_ab64 = d1 + 1;
          char *d2 = strchr(salt_ab64, '$');
          if (d2 && d2 > salt_ab64) {
            char *hash_ab64 = d2 + 1;
            int hash_ab64_len = mystrlen(hash_ab64);
            if (hash_ab64_len > 0) {
              /* Convert ab64→b64: replace . with +, add = padding */
              char convbuf[512];
              int salt_len = d2 - salt_ab64;
              int offset = sprintf(convbuf, "%s:%d:", algo_str, iter);
              int ci, pad;
              /* Copy salt, converting . → + */
              for (ci = 0; ci < salt_len && offset < 480; ci++)
                convbuf[offset++] = (salt_ab64[ci] == '.') ? '+' : salt_ab64[ci];
              /* Add = padding: (4 - len%4) % 4 */
              pad = (4 - (salt_len % 4)) % 4;
              while (pad--) convbuf[offset++] = '=';
              convbuf[offset++] = ':';
              /* Copy hash, converting . → + */
              for (ci = 0; ci < hash_ab64_len && offset < 500; ci++)
                convbuf[offset++] = (hash_ab64[ci] == '.') ? '+' : hash_ab64[ci];
              pad = (4 - (hash_ab64_len % 4)) % 4;
              while (pad--) convbuf[offset++] = '=';
              convbuf[offset] = 0;
              JSLI(PV, JUDYJ(JOB_PKCS5S2), (unsigned char *)convbuf);
              /* Extract prefix (everything before last :) for Typesalt */
              { char *lastsep = strrchr(convbuf, ':');
                if (lastsep && lastsep > convbuf) {
                  char savec = *lastsep;
                  *lastsep = 0;
                  JSLI(PV, TYPESALT(job_id), (unsigned char *)convbuf);
                  if (PV) { if ((*PV)++ == 0) {Saltloaded[job_id]++;} }
                  *lastsep = savec;
                }
              }
              Foundcnt[JOB_PKCS5S2]++;
              continue;
            }
          }
        }
      }
    }
    /* JudyJ[JOB_PKCS5S2] variants: pbkdf2_sha256$, sha256:, {PKCS5S2}, $ml$ */
    if ((lf[JOB_PKCS5S2] || lf[JOB_PBKDF2_MD5] || lf[JOB_PBKDF2_SHA1] || lf[JOB_PBKDF2_SHA256] || lf[JOB_PBKDF2_SHA512]) &&
        (strncmp(line, "pbkdf2_sha256$", 14) == 0 ||
        strncmp(line, "pbkdf2_sha512$", 14) == 0 ||
        strncmp(line, "pbkdf2_sha1$", 12) == 0 ||
        strncmp(line, "pbkdf2_md5$", 11) == 0 ||
        strncmp(line, "sha256:", 7) == 0 ||
        strncmp(line, "sha1:", 5) == 0 ||
        strncmp(line, "sha512:", 7) == 0 ||
        strncmp(line, "{PKCS5S2}", 9) == 0 ||
        strncmp(line, "md5:", 4) == 0)) {
      x = 4;
      if (line[0] == '{') x = 10;
      for (; x < 120; x++)
        if ((signed char)line[x] <= ' ' || (signed char)line[x] > 'z') break;
      if (x < 120) {
        line[x] = 0;
        JSLI(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)line);
        /* PKCS5S2: extract 16-byte salt into Typesalt */
        if (strncmp(line, "{PKCS5S2}", 9) == 0) {
          char pk_raw[64], pk_sb64[32];
          int pk_len;
          pk_len = b64_decode(line + 9, pk_raw, &cryptlen);
          if (pk_len >= 48) {
            b64_encode(pk_raw, pk_sb64, 16);
            JSLI(PV, TYPESALT(JOB_PKCS5S2), (unsigned char *)pk_sb64);
            if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_PKCS5S2]++;} }
          }
        }
        /* JudyJ[JOB_PKCS5S2] md5/sha1/sha256/sha512: extract prefix into Typesalt */
        if (strncmp(line, "md5:", 4) == 0 ||
            strncmp(line, "sha1:", 5) == 0 ||
            strncmp(line, "sha256:", 7) == 0 ||
            strncmp(line, "sha512:", 7) == 0 ||
            strncmp(line, "pbkdf2_sha256$", 14) == 0 ||
            strncmp(line, "pbkdf2_sha512$", 14) == 0 ||
            strncmp(line, "pbkdf2_sha1$", 12) == 0 ||
            strncmp(line, "pbkdf2_md5$", 11) == 0) {
          /* find last separator (: or $) to split prefix from hash */
          char *lastsep = strrchr(line, ':');
          if (!lastsep) lastsep = strrchr(line, '$');
          if (lastsep && lastsep > line) {
            int job_id = 0;
            if (strncmp(line, "md5:", 4) == 0 || strncmp(line, "pbkdf2_md5$", 11) == 0)
              job_id = JOB_PBKDF2_MD5;
            else if (strncmp(line, "sha1:", 5) == 0 || strncmp(line, "pbkdf2_sha1$", 12) == 0)
              job_id = JOB_PBKDF2_SHA1;
            else if (strncmp(line, "sha256:", 7) == 0 || strncmp(line, "pbkdf2_sha256$", 14) == 0)
              job_id = JOB_PBKDF2_SHA256;
            else if (strncmp(line, "sha512:", 7) == 0 || strncmp(line, "pbkdf2_sha512$", 14) == 0)
              job_id = JOB_PBKDF2_SHA512;
            if (job_id) {
              char savec = *lastsep;
              *lastsep = 0;
              JSLI(PV, TYPESALT(job_id), (unsigned char *)line);
              if (PV) { if ((*PV)++ == 0) {Saltloaded[job_id]++;} }
              *lastsep = savec;
            }
          }
        }
        Foundcnt[JOB_PKCS5S2]++;
        continue;
      }
    }
    /* GRUB 2: grub.pbkdf2.sha512.ITER.SALT_HEX.HASH_HEX → convert to $ml$ format */
    if ((lf[JOB_PKCS5S2] || lf[JOB_PBKDF2_SHA512]) &&
        strncmp(line, "grub.pbkdf2.sha512.", 19) == 0) {
      /* convert dots to dollars and replace prefix */
      char mlbuf[MAXLINE + 16];
      int mi = sprintf(mlbuf, "$ml$");
      for (x = 19; line[x] && mi < MAXLINE; x++)
        mlbuf[mi++] = (line[x] == '.') ? '$' : line[x];
      mlbuf[mi] = 0;
      memcpy(line, mlbuf, mi + 1);
      len = mi;
      /* fall through to $ml$ loading below */
    }
    /* $ml$ (GRUB2 / PBKDF2_SHA512): $ml$ITER$SALT(hex)$HASH(128hex) */
    if ((lf[JOB_PKCS5S2] || lf[JOB_PBKDF2_SHA512]) && strncmp(line, "$ml$", 4) == 0) {
      for (x = 4; x < MAXLINE; x++)
        if ((signed char)line[x] <= ' ' || (signed char)line[x] > 'z') break;
      /* Validate: $ml$ + iter + $ + salt_hex + $ + hash_128hex
       * Minimum: 4 + 1 + 1 + 2 + 1 + 128 = 137; hash must end with 128 hex */
      if (x >= 137 && x <= MAXLINE - 1) {
        char *lastsep_v = strrchr(line, '$');
        if (lastsep_v && (int)strlen(lastsep_v + 1) == 128) {
        line[x] = 0;
        JSLI(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)line);
        /* extract prefix into Typesalt for PBKDF2_SHA512 */
        { char *lastsep = strrchr(line, '$');
          if (lastsep && lastsep > line) {
            char savec = *lastsep;
            *lastsep = 0;
            JSLI(PV, TYPESALT(JOB_PBKDF2_SHA512), (unsigned char *)line);
            if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_PBKDF2_SHA512]++;} }
            *lastsep = savec;
          }
        }
        Foundcnt[JOB_PKCS5S2]++;
        continue;
        }
      }
    }
    /* $mysql$A$ (MySQL SHA256CRYPT) */
    if (lf[JOB_MYSQL_SHA256CRYPT] && strncmp(line, "$mysql$A$", 9) == 0) {
      char *p = line + 9;
      while (*p >= '0' && *p <= '9') p++;
      if (*p == '*') {
        char *salt_start = p + 1;
        char *salt_end = strchr(salt_start, '*');
        if (salt_end && salt_end > salt_start) {
          JSLI(PV, JUDYJ(JOB_MYSQL_SHA256CRYPT), (unsigned char *)line);
          { char savec = *salt_end;
            *salt_end = 0;
            Saltloaded[JOB_MYSQL_SHA256CRYPT] += store_typesalt(JOB_MYSQL_SHA256CRYPT, line, mystrlen(line));
            *salt_end = savec;
          }
          Foundcnt[JOB_MYSQL_SHA256CRYPT]++;
          continue;
        }
      }
    }
    /* phpBB3 / phpass: $P$ or $H$.  Admit either JOB_PHPBB3 (e455) or
       JOB_PHPBB3MD5 (e537): both consume this identical hash format, and the
       e537 compute arm falls through to the e455 arm, so the salt stored here
       under JOB_PHPBB3 is copied to TYPESALT(JOB_PHPBB3MD5) in the post-load
       pass.  Gating on JOB_PHPBB3 alone silently dropped e537-only loads. */
    if ((lf[JOB_PHPBB3] || lf[JOB_PHPBB3MD5]) && (line[0] == '$' && (line[1] == 'P' || line[1] == 'H') && line[2] == '$') && len >= 34) {
      /* gate mirrors Phpbb3Family; a new member joins both */
      { int pbbx = i64hex[line[3] & 0xff];
        if (pbbx >= 7 && pbbx <= 30) {
          for (x = 4; x < 12; x++) {
            if (i64hex[line[x] & 0xff] > 63) break;
          }
          if (x == 12 && strlen(line + 12) == 22) {
            /* Decode phpass base64 hash → binary → compact table */
            unsigned char hashbin[16];
            if (phpass_b64decode(line + 12, hashbin) == 0 && inhashbuf) {
              int fi, filed = 0;
              memcpy(inhashbuf + 2, hashbin, 16);
              /* ONCE per line: this commits the binary digest to the input set,
               * not to any one type. */
              commit_compact(16);
              /* Store the "$H$Csalt8chr" prefix (12 chars) in the Typesalt of
               * each SELECTED family member, counting per member. Filing it
               * under JOB_PHPBB3 alone and copying later is what invented the
               * wrong count, and it also populated a type the user had not
               * selected: -M e537 alone used to fill Typesalt[JOB_PHPBB3]. */
              for (fi = 0; fi < PHPBB3FAMILY_N; fi++) {
                int fop = Phpbb3Family[fi];
                if (!lf[fop]) continue;
                Saltloaded[fop] += store_typesalt(fop, line, 12);
                Foundcnt[fop]++;
                filed = 1;
              }
              if (filed) continue;
            }
          }
        }
      }
    }
    /* $1$ md5crypt — decode base64 hash into compact table, salt into Typesalt.
       Admits either JOB_MD5CRYPT (e511) or JOB_MD5CRYPTMD5 (e1061); gating on
       e511 alone silently dropped an e1061-only load, the same defect the
       phpBB3 pair had above. */
    if ((lf[JOB_MD5CRYPT] || lf[JOB_MD5CRYPTMD5]) &&
        line[0] == '$' && line[1] == '1' && line[2] == '$') {
      /* Find salt: between $1$ and next $ */
      char *salt_start = line + 3;
      char *hash_start = NULL;
      for (x = 0; x <= 8; x++) {
        if (salt_start[x] == '$') { hash_start = salt_start + x + 1; break; }
        if (salt_start[x] == 0) break;
      }
      /* A bare setting, $1$salt$ or $1$salt, supplies a SALT and no hash:
       * Typesalt only, in the same $1$salt$ form the full-hash path stores,
       * so -z generates with it.  Nothing goes in the compact table. */
      if (Arg_have_z && x > 0 && x <= 8 && ((hash_start && *hash_start == 0) ||
                              (!hash_start && salt_start[x] == 0))) {
        char mc_set[16];
        int fi, filed = 0;
        memcpy(mc_set, line, (size_t)(3 + x));
        mc_set[3 + x] = '$';
        mc_set[4 + x] = 0;
        for (fi = 0; fi < MD5CRYPTFAMILY_N; fi++) {
          int fop = Md5cryptFamily[fi];
          if (!lf[fop]) continue;
          Saltloaded[fop] += store_typesalt(fop, mc_set, 4 + x);
          filed = 1;
        }
        if (filed) continue;
      }
      if (hash_start && strlen(hash_start) == 22) {
        unsigned char hashbin[16];
        if (md5crypt_b64decode(hash_start, hashbin) == 0 && inhashbuf) {
          /* Store binary hash in compact table */
          memcpy(inhashbuf + 2, hashbin, 16);
          commit_compact(16);
          /* Store salt prefix "$1$salt$" in the Typesalt of each SELECTED
           * family member, counting per member (see Md5cryptFamily). */
          int prefixlen = (int)(hash_start - line);
          int fi, filed = 0;
          for (fi = 0; fi < MD5CRYPTFAMILY_N; fi++) {
            int fop = Md5cryptFamily[fi];
            if (!lf[fop]) continue;
            Saltloaded[fop] += store_typesalt(fop, line, prefixlen);
            Foundcnt[fop]++;
            filed = 1;
          }
          if (filed) continue;
        }
      }
    }
    /* $apr1$ */
    if (lf[JOB_APR1] && strncmp(line, "$apr1$", 6) == 0) {
      for (x = 4; x < 38; x++) {
        if ((signed char)line[x] <= ' ' || (signed char)line[x] > 'z') break;
      }
      if (x >= 29) {
        line[x] = 0;
        JSLI(PV, JUDYJ(JOB_APR1), (unsigned char *)line);
        Foundcnt[JOB_APR1]++;
        continue;
      }
    }
    /* $5$ sha256crypt, $6$ sha512crypt, $8$ cisco8, $9$ cisco9 */
    if ((lf[JOB_SHA256CRYPT] || lf[JOB_SHA512CRYPT] || lf[JOB_SHA512CRYPTMD5] || lf[JOB_CISCO8] || lf[JOB_CISCO9]) &&
        line[0] == '$' && (line[1] == '5' || line[1] == '6' || line[1] == '8' || line[1] == '9') && line[2] == '$') {
      for (x = 3; x < 254; x++) {
        if (line[x] == ':' || (signed char)line[x] <= ' ' || (signed char)line[x] > 'z') break;
      }
      /* A bare setting -- $6$salt$, $6$rounds=N$salt$ -- supplies a SALT,
       * not a hash.  It goes straight into Typesalt, where the compute arm
       * reads its crypt() settings, so -z generates with it and the receipt
       * counts it as a salt.  It is not filed in JudyJ, which holds the
       * lines to be matched; a setting alone can never match anything. */
      { char cs_set[96];
        int cs_n = Arg_have_z ? crypt_setting_only(line, x, cs_set, (int)sizeof(cs_set)) : 0;
        if (cs_n) {
          int cs_jobs[3], cs_k, cs_nj = 0, cs_filed = 0;
          if (line[1] == '5') cs_jobs[cs_nj++] = JOB_SHA256CRYPT;
          else { cs_jobs[cs_nj++] = JOB_SHA512CRYPT; cs_jobs[cs_nj++] = JOB_SHA512CRYPTMD5; }
          for (cs_k = 0; cs_k < cs_nj; cs_k++) {
            if (!lf[cs_jobs[cs_k]]) continue;
            store_typesalt(cs_jobs[cs_k], cs_set, cs_n);
            rcpt_salt(Saltloaded, cs_jobs[cs_k], cs_set, cs_n);
            cs_filed = 1;
          }
          if (cs_filed) continue;
        }
      }
      if (x > 10) {
        line[x] = 0;
        /* The receipt's salt is the setting: everything up to the last $. */
        if (line[1] == '5' || line[1] == '6') {
          char *cs_end = strrchr(line, '$');
          int cs_len = cs_end ? (int)(cs_end - line) + 1 : 0;
          if (cs_len > 3) {
            if (line[1] == '5' && lf[JOB_SHA256CRYPT])
              rcpt_salt(Saltloaded, JOB_SHA256CRYPT, line, cs_len);
            if (line[1] == '6' && lf[JOB_SHA512CRYPT])
              rcpt_salt(Saltloaded, JOB_SHA512CRYPT, line, cs_len);
            if (line[1] == '6' && lf[JOB_SHA512CRYPTMD5])
              rcpt_salt(Saltloaded, JOB_SHA512CRYPTMD5, line, cs_len);
          }
        }
        if (line[1] == '5') { JSLI(PV, JUDYJ(JOB_SHA256CRYPT), (unsigned char *)line); Foundcnt[JOB_SHA256CRYPT]++; }
        if (line[1] == '6') { JSLI(PV, JUDYJ(JOB_SHA512CRYPT), (unsigned char *)line); Foundcnt[JOB_SHA512CRYPT]++; }
        if (line[1] == '8' && x == 61) {
          JSLI(PV, JUDYJ(JOB_CISCO8),(unsigned char *)line);
          Saltloaded[JOB_CISCO8] += store_typesalt(JOB_CISCO8, line + 3, 14);
          Foundcnt[JOB_CISCO8]++;
        }
        if (line[1] == '9' && x == 61) {
          JSLI(PV, JUDYJ(JOB_CISCO9),(unsigned char *)line);
          Saltloaded[JOB_CISCO9] += store_typesalt(JOB_CISCO9, line + 3, 14);
          Foundcnt[JOB_CISCO9]++;
        }
        continue;
      }
    }
    /* SM3CRYPT (35100): $sm3$[rounds=N$]salt$hash43 */
    if (lf[JOB_SM3CRYPT] && strncmp(line, "$sm3$", 5) == 0 && len > 10) {
      for (x = 5; x < 254; x++) {
        if (line[x] == ':' || (signed char)line[x] <= ' ' || (signed char)line[x] > 'z') break;
      }
      if (x > 10) {
        line[x] = 0;
        JSLI(PV, JUDYJ(JOB_SM3CRYPT), (unsigned char *)line);
        Foundcnt[JOB_SM3CRYPT]++;
        continue;
      }
    }
    /* $S$ JudyJ[JOB_DRUPAL7] */
    if (lf[JOB_DRUPAL7] && strncmp(line, "$S$", 3) == 0 && len >= 55) {
      char *p = line + 3;
      if (strchr(phpitoa64, p[0]) && mystrlen(p) >= 52) {
        JSLI(PV, JUDYJ(JOB_DRUPAL7), (unsigned char *)line);
        Saltloaded[JOB_DRUPAL7] += store_typesalt(JOB_DRUPAL7, line + 3, 9);
        Foundcnt[JOB_DRUPAL7]++;
        continue;
      }
    }
    /* $BLAKE2$hex: strip prefix, write hex to compact table */
    if (lf[JOB_BLAKE2S256] && strncmp(line, "$BLAKE2$", 8) == 0 && inhashbuf) {
      int hlen = get32(line + 8, inhashbuf + 2, 256);
      if (hlen >= 16 && hlen <= 255) {
        commit_compact(hlen);
        Foundcnt[JOB_BLAKE2S256]++;
        continue;
      }
    }
    /* MSSQL 0x0100/0x0200: extract salt, write hex to compact table */
    if ((lf[JOB_MSSQL2000] || lf[JOB_MSSQL2005] || lf[JOB_MSSQL2012]) &&
        line[0] == '0' && (line[1] == 'x' || line[1] == 'X') &&
        line[2] == '0' && (line[3] == '1' || line[3] == '2') &&
        line[4] == '0' && line[5] == '0') {
      x = mystrlen(line);
      if (x == 94 || x == 54 || x == 142) {
        char msalt_hex[9];
        memcpy(msalt_hex, line + 6, 8);
        msalt_hex[8] = 0;
        JSLI(PV, JUDYJ(JOB_MSSQL2000), (unsigned char *)msalt_hex);
        if (PV) (*PV)++;
        if (inhashbuf) {
          int hlen;
          if (x == 94) {
            /* MSSQL2000: skip hash_upper, load magic+salt+hash_mixed only */
            char mstmp[64];
            memcpy(mstmp, line + 2, 12);     /* "0100" + salt(8hex) */
            memcpy(mstmp + 12, line + 54, 40); /* hash_mixed (40hex) */
            mstmp[52] = 0;
            hlen = get32(mstmp, inhashbuf + 2, 256);
            Foundcnt[JOB_MSSQL2000]++;
          } else {
            hlen = get32(line + 2, inhashbuf + 2, 256);
            if (x == 54) Foundcnt[JOB_MSSQL2005]++;
            else Foundcnt[JOB_MSSQL2012]++;
          }
          if (hlen >= 8 && hlen <= 255) {
            commit_compact(hlen);
          }
        }
        continue;
      }
    }
    /* macOS v10.4-10.6 (48 hex), v10.7 (136 hex), ArubaOS (50 hex): extract salt, write to compact table */
    if (lf[JOB_MACOSX] || lf[JOB_MACOSX7] || lf[JOB_ARUBAOS]) {
      int hlen2 = 0;
      char *p = line;
      while ((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f') ||
             (*p >= 'A' && *p <= 'F')) { hlen2++; p++; }
      if ((*p == 0 || *p == '\r' || *p == '\n' || *p == ':') &&
          (hlen2 == 48 || hlen2 == 136 || hlen2 == 50)) {
        if (inhashbuf) {
          int hlen3 = get32(line, inhashbuf + 2, 256);
          if (hlen3 >= 8 && hlen3 <= 255) {
            commit_compact(hlen3);
          }
        }
        if (hlen2 == 48) {
          char msalt_hex[9];
          memcpy(msalt_hex, line, 8);
          msalt_hex[8] = 0;
          JSLI(PV, TYPESALT(JOB_MACOSX), (unsigned char *)msalt_hex);
          if (PV) (*PV)++;
          Foundcnt[JOB_MACOSX]++;
        } else if (hlen2 == 136) {
          char msalt_hex[9];
          memcpy(msalt_hex, line, 8);
          msalt_hex[8] = 0;
          JSLI(PV, TYPESALT(JOB_MACOSX7), (unsigned char *)msalt_hex);
          if (PV) (*PV)++;
          Foundcnt[JOB_MACOSX7]++;
        } else if (hlen2 == 50) {
          char msalt_hex[11];
          memcpy(msalt_hex, line, 10);
          msalt_hex[10] = 0;
          JSLI(PV, TYPESALT(JOB_ARUBAOS), (unsigned char *)msalt_hex);
          if (PV) (*PV)++;
          Foundcnt[JOB_ARUBAOS]++;
        }
        continue;
      }
    }
    /* YAF-SHA1: base64hash[:/ ]base64salt → JudyJ[JOB_YAF_SHA1] + Typesalt */
    { J1T(RC, Dohash, JOB_YAF_SHA1);
      if (RC) {
        char *sep = NULL;
        for (x = 0; line[x] && line[x] != ':' && line[x] != ' ' &&
                     line[x] != '\r' && line[x] != '\n'; x++)
          ;
        if (x > 0 && (line[x] == ':' || line[x] == ' '))
          sep = &line[x];
        if (sep && sep[1]) {
          char yaf_hash[64], yaf_salt[64];
          int hlen2, slen2;
          hlen2 = sep - line;
          if (hlen2 > 0 && hlen2 < 64) {
            memcpy(yaf_hash, line, hlen2);
            yaf_hash[hlen2] = 0;
            for (slen2 = 0; sep[1 + slen2] && sep[1 + slen2] != '\r' &&
                 sep[1 + slen2] != '\n' && sep[1 + slen2] != ':' &&
                 sep[1 + slen2] != ' ' && slen2 < 63; slen2++)
              yaf_salt[slen2] = sep[1 + slen2];
            yaf_salt[slen2] = 0;
            if (slen2 > 0) {
              JSLI(PV, JUDYJ(JOB_YAF_SHA1), (unsigned char *)yaf_hash);
              JSLI(PV, TYPESALT(JOB_YAF_SHA1), (unsigned char *)yaf_salt);
              if (PV) { if ((*PV)++ == 0) {Saltloaded[JOB_YAF_SHA1]++;} }
              Foundcnt[JOB_YAF_SHA1]++;
              continue;
            }
          }
        }
      }
    }
    /* NetNTLMv1 (5500): user::domain:lm(0-48hex):ntlm(48hex):challenge(16hex) */
    if (lf[JOB_NETNTLMV1]) {
      char *dc = strstr(line, "::");
      if (dc && dc - line <= 60) {
        /* count remaining colons after :: to find 6-field format */
        char *p = dc + 2;
        int nc = 0;
        char *col[4];  /* pointers to colons 3,4,5 */
        while (*p && nc < 3) { if (*p == ':') { col[nc] = p; nc++; } p++; }
        if (nc == 3) {
          int f3len = col[1] - col[0] - 1;  /* lm_response */
          int f4len = col[2] - col[1] - 1;  /* ntlm_response */
          int f5len = strlen(col[2] + 1);    /* challenge */
          if (f4len == 48 && f5len == 16 && f3len <= 48 && (f3len == 0 || (f3len % 2) == 0)) {
            JSLI(PV, JUDYJ(JOB_NETNTLMV1), (unsigned char *)line);
            Saltloaded[JOB_NETNTLMV1] += store_typesalt(JOB_NETNTLMV1, line, len);
            Foundcnt[JOB_NETNTLMV1]++;
            continue;
          }
        }
      }
    }
    /* NetNTLMv2 (5600): user::domain:srvchall(16hex):response(32hex):blob(2+hex) */
    if (lf[JOB_NETNTLMV2]) {
      char *dc = strstr(line, "::");
      if (dc && dc - line <= 60) {
        char *p = dc + 2;
        int nc = 0;
        char *col[4];
        while (*p && nc < 3) { if (*p == ':') { col[nc] = p; nc++; } p++; }
        if (nc == 3) {
          int f3len = col[0] + 1 < col[1] ? col[1] - col[0] - 1 : 0;  /* srvchall */
          int f4len = col[1] + 1 < col[2] ? col[2] - col[1] - 1 : 0;  /* response */
          int f5len = strlen(col[2] + 1);                               /* blob */
          if (f3len == 16 && f4len == 32 && f5len >= 2 && (f5len % 2) == 0) {
            JSLI(PV, JUDYJ(JOB_NETNTLMV2), (unsigned char *)line);
            Saltloaded[JOB_NETNTLMV2] += store_typesalt(JOB_NETNTLMV2, line, len);
            Foundcnt[JOB_NETNTLMV2]++;
            continue;
          }
        }
      }
    }
    /* Oracle7 (3100): 16-hex-UC:username (1-30 char salt) → Judy + Typesalt */
    if (lf[JOB_ORACLE7] && len > 17 && len <= 47 && line[16] == ':') {
      int o7_ok = 1;
      for (y = 0; y < 16; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { o7_ok = 0; break; }
      }
      if (o7_ok) {
        /* uppercase the hash portion for consistent lookup */
        for (y = 0; y < 16; y++) line[y] = toupper(line[y]);
        JSLI(PV, JUDYJ(JOB_ORACLE7), (unsigned char *)line);
        Saltloaded[JOB_ORACLE7] += store_typesalt(JOB_ORACLE7, line + 17, len - 17);
        Foundcnt[JOB_ORACLE7]++;
        continue;
      }
    }
    /* LastPass (6800): hash(32hex):iterations:email:iv(32hex) → Judy + Typesalt */
    if (lf[JOB_LASTPASS] && len >= 70 && line[32] == ':') {
      int lp_ok = 1;
      for (y = 0; y < 32; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { lp_ok = 0; break; }
      }
      if (lp_ok) {
        /* Find 3 more colons: iter:email:iv */
        char *c1 = line + 32;  /* first colon */
        char *c2 = strchr(c1 + 1, ':');
        char *c3 = c2 ? strchr(c2 + 1, ':') : NULL;
        if (c2 && c3 && (c3 - c2 - 1) <= 32) {
          int ivlen = strlen(c3 + 1);
          int iterlen = c2 - c1 - 1;
          if (ivlen == 32 && iterlen >= 1 && iterlen <= 6) {
            int iv_ok = 1;
            for (y = 0; y < 32; y++) {
              if (trhex[(unsigned char)c3[1+y]] > 15) { iv_ok = 0; break; }
            }
            if (iv_ok) {
              /* lowercase hash for JudyJ key */
              for (y = 0; y < 32; y++) line[y] = tolower(line[y]);
              char jkey[33];
              memcpy(jkey, line, 32);
              jkey[32] = 0;
              JSLI(PV, JUDYJ(JOB_LASTPASS), (unsigned char *)jkey);
              /* salt = "iterations:email:iv" */
              Saltloaded[JOB_LASTPASS] += store_typesalt(JOB_LASTPASS, c1 + 1, len - 33);
              Foundcnt[JOB_LASTPASS]++;
              continue;
            }
          }
        }
      }
    }
    /* FortiGate (7000): AK1 + 44 base64 (47 chars total) → Judy + Typesalt */
    if (lf[JOB_FORTIGATE] && len == 47 && strncmp(line, "AK1", 3) == 0) {
      unsigned char fg_decoded[64];
      int fg_incount;
      int fg_dlen = b64_decode(line + 3, (char *)fg_decoded, &fg_incount);
      if (fg_dlen >= 32) {
        /* JudyJ key = 40-hex lowercase SHA1 hash */
        char fg_hashkey[41];
        prmd5_40(fg_decoded + 12, fg_hashkey);
        fg_hashkey[40] = 0;
        JSLI(PV, JUDYJ(JOB_FORTIGATE), (unsigned char *)fg_hashkey);
        /* Typesalt = 24-hex salt */
        char fg_salthex[25];
        prmd5(fg_decoded, fg_salthex, 24);
        fg_salthex[24] = 0;
        Saltloaded[JOB_FORTIGATE] += store_typesalt(JOB_FORTIGATE, fg_salthex, 24);
        Foundcnt[JOB_FORTIGATE]++;
        continue;
      }
    }
    /* FortiGate256 (26300): SH2 + 60 base64 (63 chars total)
     * base64 decodes to 44 bytes: salt[12] + sha256[32]
     * JudyJ key = 64-hex lowercase SHA256, Typesalt = 24-hex salt */
    if (lf[JOB_FORTIGATE256] && len == 63 && strncmp(line, "SH2", 3) == 0) {
      unsigned char fg_decoded[64];
      int fg_incount;
      int fg_dlen = b64_decode(line + 3, (char *)fg_decoded, &fg_incount);
      if (fg_dlen >= 44) {
        char fg_hashkey[65];
        prmd5_64(fg_decoded + 12, fg_hashkey);
        fg_hashkey[64] = 0;
        JSLI(PV, JUDYJ(JOB_FORTIGATE256), (unsigned char *)fg_hashkey);
        char fg_salthex[25];
        prmd5(fg_decoded, fg_salthex, 24);
        fg_salthex[24] = 0;
        Saltloaded[JOB_FORTIGATE256] += store_typesalt(JOB_FORTIGATE256, fg_salthex, 24);
        Foundcnt[JOB_FORTIGATE256]++;
        continue;
      }
    }
    /* Umbraco HMAC-SHA1 (24800): 28-char base64 (20 bytes = SHA1-sized), unsalted
     * JudyJ key = 40-hex lowercase */
    if (lf[JOB_UMBRACO] && len == 28 && line[27] == '=') {
      unsigned char umb_decoded[32];
      int umb_incount;
      int umb_dlen = b64_decode(line, (char *)umb_decoded, &umb_incount);
      if (umb_dlen == 20) {
        char umb_hashkey[41];
        prmd5_40(umb_decoded, umb_hashkey);
        umb_hashkey[40] = 0;
        JSLI(PV, JUDYJ(JOB_UMBRACO), (unsigned char *)umb_hashkey);
        Foundcnt[JOB_UMBRACO]++;
        continue;
      }
    }
    /* Dahua Auth MD5 (24900): exactly 8 alphanumeric chars, verify-only
     * JudyJ key = the 8-char string itself */
    if ((lf[JOB_DAHUA_AUTH] || lf[JOB_BESDER_AUTH]) && len == 8) {
      int dahua_ok = 1;
      for (x = 0; x < 8; x++) {
        if (!((line[x] >= '0' && line[x] <= '9') ||
              (line[x] >= 'A' && line[x] <= 'Z') ||
              (line[x] >= 'a' && line[x] <= 'z'))) { dahua_ok = 0; break; }
      }
      if (dahua_ok) {
        salttmp[0] = 0; memcpy(salttmp, line, 8); salttmp[8] = 0;
        if (lf[JOB_DAHUA_AUTH]) {
          JSLI(PV, JUDYJ(JOB_DAHUA_AUTH), (unsigned char *)salttmp);
          Foundcnt[JOB_DAHUA_AUTH]++;
        }
        if (lf[JOB_BESDER_AUTH]) {
          JSLI(PV, JUDYJ(JOB_BESDER_AUTH), (unsigned char *)salttmp);
          Foundcnt[JOB_BESDER_AUTH]++;
        }
        continue;
      }
    }
    /* SQLCipher (24600): SQLCIPHER*type*iter*salt32hex*iv32hex*data32hex
     * JudyJ key = lowercase data32hex, Typesalt = "type:iter:salt32hex:iv32hex" */
    if (lf[JOB_SQLCIPHER] && len > 50 && strncmp(line, "SQLCIPHER*", 10) == 0) {
      /* Parse: SQLCIPHER*T*ITER*SALT32*IV32*DATA32 — use salttmp as copy */
      memcpy(salttmp, line, len); salttmp[len] = 0;
      char *f1 = salttmp + 10;  /* type */
      char *s1 = strchr(f1, '*');
      if (s1 && (s1 - f1) == 1) {
        *s1 = 0;
        int sc_type = atoi(f1);
        char *f2 = s1 + 1;  /* iter */
        char *s2 = strchr(f2, '*');
        if (s2 && sc_type >= 1 && sc_type <= 3) {
          *s2 = 0;
          int sc_iter = atoi(f2);
          char *f3 = s2 + 1;  /* salt32hex */
          char *s3 = strchr(f3, '*');
          if (s3 && (s3 - f3) == 32 && sc_iter >= 1) {
            *s3 = 0;
            char *f4 = s3 + 1;  /* iv32hex */
            char *s4 = strchr(f4, '*');
            if (s4 && (s4 - f4) == 32) {
              *s4 = 0;
              char *f5 = s4 + 1;  /* data32hex */
              int f5len = len - (f5 - salttmp);
              if (f5len == 32) {
                /* JudyJ key = lowercase data hex */
                char jkey[33];
                for (x = 0; x < 32; x++) jkey[x] = tolower(f5[x]);
                jkey[32] = 0;
                JSLI(PV, JUDYJ(JOB_SQLCIPHER), (unsigned char *)jkey);
                /* Typesalt = "type:iter:salt32hex_lc:iv32hex_lc" */
                char tskey[128];
                snprintf(tskey, sizeof(tskey), "%d:%d:", sc_type, sc_iter);
                int toff = strlen(tskey);
                for (x = 0; x < 32; x++) tskey[toff + x] = tolower(f3[x]);
                tskey[toff + 32] = ':';
                for (x = 0; x < 32; x++) tskey[toff + 33 + x] = tolower(f4[x]);
                tskey[toff + 65] = 0;
                JSLI(PV, TYPESALT(JOB_SQLCIPHER), (unsigned char *)tskey);
                if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_SQLCIPHER]++; }
                Foundcnt[JOB_SQLCIPHER]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* Ruby on Rails Restful Auth (27200): 40hex_hash:40hex_salt
     * SHA1("--" + salt_hex_string + "--" + pass + "--")
     * JudyJ key = 40-hex lowercase hash, Typesalt = 40-hex salt string */
    if (lf[JOB_RORAILS_SHA1] && len == 81 && line[40] == ':') {
      int rr_ok = 1;
      for (x = 0; x < 40; x++)
        if (!isxdigit(line[x]) || !isxdigit(line[41+x])) { rr_ok = 0; break; }
      if (rr_ok) {
        char rr_hash[41], rr_salt[41];
        for (x = 0; x < 40; x++) rr_hash[x] = tolower(line[x]);
        rr_hash[40] = 0;
        JSLI(PV, JUDYJ(JOB_RORAILS_SHA1), (unsigned char *)rr_hash);
        for (x = 0; x < 40; x++) rr_salt[x] = tolower(line[41+x]);
        rr_salt[40] = 0;
        Saltloaded[JOB_RORAILS_SHA1] += store_typesalt(JOB_RORAILS_SHA1, rr_salt, 40);
        Foundcnt[JOB_RORAILS_SHA1]++;
        continue;
      }
    }
    /* AES-128/192/256-ECB NOKDF (26401/26402/26403): 32hex_ct:32hex_pt
     * JudyJ key = 32-hex lowercase ciphertext, Typesalt = "keysize:32hex_plaintext" */
    if ((lf[JOB_AES128_NOKDF] || lf[JOB_AES192_NOKDF] || lf[JOB_AES256_NOKDF]) &&
        len == 65 && line[32] == ':') {
      int ae_ok = 1;
      for (x = 0; x < 32; x++)
        if (!isxdigit(line[x]) || !isxdigit(line[33+x])) { ae_ok = 0; break; }
      if (ae_ok) {
        char ae_ct[33], ae_pt[33];
        for (x = 0; x < 32; x++) ae_ct[x] = tolower(line[x]);
        ae_ct[32] = 0;
        for (x = 0; x < 32; x++) ae_pt[x] = tolower(line[33+x]);
        ae_pt[32] = 0;
        if (lf[JOB_AES128_NOKDF]) {
          JSLI(PV, JUDYJ(JOB_AES128_NOKDF), (unsigned char *)ae_ct);
          snprintf(salttmp, MAXLINE, "128:%s", ae_pt);
          Saltloaded[JOB_AES128_NOKDF] += store_typesalt(JOB_AES128_NOKDF, salttmp, strlen(salttmp));
          Foundcnt[JOB_AES128_NOKDF]++;
        }
        if (lf[JOB_AES192_NOKDF]) {
          JSLI(PV, JUDYJ(JOB_AES192_NOKDF), (unsigned char *)ae_ct);
          snprintf(salttmp, MAXLINE, "192:%s", ae_pt);
          Saltloaded[JOB_AES192_NOKDF] += store_typesalt(JOB_AES192_NOKDF, salttmp, strlen(salttmp));
          Foundcnt[JOB_AES192_NOKDF]++;
        }
        if (lf[JOB_AES256_NOKDF]) {
          JSLI(PV, JUDYJ(JOB_AES256_NOKDF), (unsigned char *)ae_ct);
          snprintf(salttmp, MAXLINE, "256:%s", ae_pt);
          Saltloaded[JOB_AES256_NOKDF] += store_typesalt(JOB_AES256_NOKDF, salttmp, strlen(salttmp));
          Foundcnt[JOB_AES256_NOKDF]++;
        }
        continue;
      }
    }
    /* VMware VMX (27400): $vmx$0$iter$32hex_salt$64hex_iv_ct
     * PBKDF2-SHA1 + AES-256-CBC, known plaintext "type=key:cipher="
     * JudyJ key = lowercase 32hex_ct, Typesalt = "iter:32hex_salt:32hex_iv" */
    if (lf[JOB_VMWARE_VMX] && len > 50 && strncmp(line, "$vmx$0$", 7) == 0) {
      memcpy(salttmp, line, len); salttmp[len] = 0;
      char *p = salttmp + 7;
      char *s1 = strchr(p, '$');
      if (s1) {
        *s1 = 0;
        int vmx_iter = atoi(p);
        char *f_salt = s1 + 1;
        char *s2 = strchr(f_salt, '$');
        if (s2 && (s2 - f_salt) == 32 && vmx_iter >= 1) {
          *s2 = 0;
          char *f_ivct = s2 + 1;
          int ivct_len = len - (f_ivct - salttmp);
          if (ivct_len == 64) {
            /* IV = first 32 hex, CT = last 32 hex */
            char vmx_ct[33], vmx_tskey[128];
            for (x = 0; x < 32; x++) vmx_ct[x] = tolower(f_ivct[32+x]);
            vmx_ct[32] = 0;
            JSLI(PV, JUDYJ(JOB_VMWARE_VMX), (unsigned char *)vmx_ct);
            /* Typesalt = "iter:salt32hex_lc:iv32hex_lc" */
            snprintf(vmx_tskey, sizeof(vmx_tskey), "%d:", vmx_iter);
            int toff = strlen(vmx_tskey);
            for (x = 0; x < 32; x++) vmx_tskey[toff+x] = tolower(f_salt[x]);
            vmx_tskey[toff+32] = ':';
            for (x = 0; x < 32; x++) vmx_tskey[toff+33+x] = tolower(f_ivct[x]);
            vmx_tskey[toff+65] = 0;
            JSLI(PV, TYPESALT(JOB_VMWARE_VMX), (unsigned char *)vmx_tskey);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_VMWARE_VMX]++; }
            Foundcnt[JOB_VMWARE_VMX]++;
            continue;
          }
        }
      }
    }
    /* PostgreSQL SCRAM-SHA-256 (28600): SCRAM-SHA-256$iter:b64salt$b64StoredKey:b64ServerKey */
    if (lf[JOB_POSTGRESSCRAM256] && len > 30 && strncmp(line, "SCRAM-SHA-256$", 14) == 0) {
      memcpy(salttmp, line, len); salttmp[len] = 0;
      char *p = salttmp + 14;
      /* Parse iter:b64salt$b64StoredKey:b64ServerKey */
      char *colon1 = strchr(p, ':');
      if (colon1) {
        *colon1 = 0;
        int sc_iter = atoi(p);
        char *b64salt = colon1 + 1;
        char *dollar = strchr(b64salt, '$');
        if (dollar && sc_iter >= 1) {
          *dollar = 0;
          char *b64stored = dollar + 1;
          char *colon2 = strchr(b64stored, ':');
          if (colon2) {
            *colon2 = 0;
            /* Decode salt and StoredKey from base64 */
            unsigned char salt_bin[64], stored_bin[32];
            int slen = b64_decode(b64salt, (char *)salt_bin, &x);
            int hlen = b64_decode(b64stored, (char *)stored_bin, &x);
            if (slen > 0 && hlen == 32) {
              /* JudyJ key = lowercase hex of StoredKey */
              char judykey[65];
              prmd5_64(stored_bin, judykey);
              judykey[64] = 0;
              JSLI(PV, JUDYJ(JOB_POSTGRESSCRAM256), (unsigned char *)judykey);
              /* Typesalt = "iter:hex_salt" */
              char tskey[256];
              snprintf(tskey, sizeof(tskey), "%d:", sc_iter);
              int toff = strlen(tskey);
              prmd5(salt_bin, tskey + toff, slen * 2);
              tskey[toff + slen * 2] = 0;
              JSLI(PV, TYPESALT(JOB_POSTGRESSCRAM256), (unsigned char *)tskey);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_POSTGRESSCRAM256]++; }
              Foundcnt[JOB_POSTGRESSCRAM256]++;
              continue;
            }
          }
        }
      }
    }
    /* AWS Signature v4 (28700): $AWS-Sig-v4$0$longdate$region$service$canonical$digest */
    if (lf[JOB_AWSSIGV4] && len > 50 && strncmp(line, "$AWS-Sig-v4$0$", 14) == 0) {
      memcpy(salttmp, line, len); salttmp[len] = 0;
      char *p = salttmp + 14;
      /* Parse longdate$region$service$canonical$digest */
      char *d1 = strchr(p, '$');
      if (d1) {
        *d1 = 0;
        char *aws_longdate = p;
        char *aws_region = d1 + 1;
        char *d2 = strchr(aws_region, '$');
        if (d2) {
          *d2 = 0;
          char *aws_service = d2 + 1;
          char *d3 = strchr(aws_service, '$');
          if (d3) {
            *d3 = 0;
            char *aws_canonical = d3 + 1;
            char *d4 = strchr(aws_canonical, '$');
            if (d4 && (d4 - aws_canonical) == 64) {
              *d4 = 0;
              char *aws_digest = d4 + 1;
              int digest_len = strlen(aws_digest);
              if (digest_len == 64) {
                /* JudyJ key = lowercase digest (signature to match) */
                char judykey[65];
                for (x = 0; x < 64; x++) judykey[x] = tolower(aws_digest[x]);
                judykey[64] = 0;
                JSLI(PV, JUDYJ(JOB_AWSSIGV4), (unsigned char *)judykey);
                /* Typesalt = "longdate:region:service:canonical".
                 * Reuse the thread-local salttmp (MAXLINE+16) so the canonical
                 * request payload (which can be ~40 KB for AWS V4) is not
                 * truncated into a 512-byte stack buffer.
                 * aws_canonical itself comes from upstream parsing bounded by
                 * MAXLINE, so the joined string is realistically <= MAXLINE in
                 * the AWS V4 input shapes mdxfind handles. snprintf bounds
                 * regardless; the upper-bound informational warning is OK. */
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
                snprintf(salttmp, MAXLINE, "%s:%s:%s:%s", aws_longdate, aws_region, aws_service, aws_canonical);
#pragma GCC diagnostic pop
                JSLI(PV, TYPESALT(JOB_AWSSIGV4), (unsigned char *)salttmp);
                if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_AWSSIGV4]++; }
                Foundcnt[JOB_AWSSIGV4]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* Kerberos 5 DB etype 17/18 (28800/28900): $krb5db$17$user$realm$hash or $krb5db$18$user$realm$hash */
    if ((lf[JOB_KRB5DB17] || lf[JOB_KRB5DB18]) && len > 20 && strncmp(line, "$krb5db$", 8) == 0) {
      memcpy(salttmp, line, len); salttmp[len] = 0;
      char *p = salttmp + 8;
      int etype = atoi(p);
      int target_job = (etype == 17) ? JOB_KRB5DB17 : (etype == 18) ? JOB_KRB5DB18 : 0;
      if (target_job && lf[target_job]) {
        char *d1 = strchr(p, '$');
        if (d1) {
          char *user = d1 + 1;
          char *d2 = strchr(user, '$');
          if (d2) {
            *d2 = 0;
            char *realm = d2 + 1;
            char *d3 = strchr(realm, '$');
            if (d3) {
              *d3 = 0;
              char *hash = d3 + 1;
              /* Skip optional *SPN* field */
              if (*hash == '*') {
                char *d4 = strchr(hash + 1, '*');
                if (d4 && *(d4+1) == '$') hash = d4 + 2;
                else continue;
              }
              int hashlen = strlen(hash);
              int expected_len = (etype == 17) ? 32 : 64;
              if (hashlen == expected_len) {
                /* JudyJ key = lowercase hex hash */
                char judykey[65];
                for (x = 0; x < hashlen; x++) judykey[x] = tolower(hash[x]);
                judykey[hashlen] = 0;
                JSLI(PV, JUDYJ(target_job), (unsigned char *)judykey);
                /* Typesalt = "REALM:user" (salt for PBKDF2 = realm + user) */
                char tskey[256];
                snprintf(tskey, sizeof(tskey), "%s:%s", realm, user);
                JSLI(PV, TYPESALT(target_job), (unsigned char *)tskey);
                if (PV) { if ((*PV)++ == 0) Saltloaded[target_job]++; }
                Foundcnt[target_job]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* NetIQ SSPR MD5 (32000): $sspr$0$ITER$NONE$32hex — iterated MD5, unsalted
     * JudyJ key = lowercase 32-hex hash, Typesalt = "ITER:" */
    if (lf[JOB_SSPR_MD5] && strncmp(line, "$sspr$0$", 8) == 0 && inhashbuf) {
      char *iter_start = line + 8;
      char *d1 = strchr(iter_start, '$');
      if (d1 && strncmp(d1, "$NONE$", 6) == 0) {
        int iter = atoi(iter_start);
        char *hash = d1 + 6;
        int hashlen2 = len - (hash - line);
        if (iter > 0 && hashlen2 == 32) {
          int hok = 1; for (x = 0; x < 32; x++) if (!isxdigit((unsigned char)hash[x])) { hok = 0; break; }
          if (hok) {
            char judykey[33];
            for (x = 0; x < 32; x++) judykey[x] = tolower(hash[x]);
            judykey[32] = 0;
            JSLI(PV, JUDYJ(JOB_SSPR_MD5), (unsigned char *)judykey);
            char tskey[32];
            snprintf(tskey, sizeof(tskey), "%d:", iter);
            JSLI(PV, TYPESALT(JOB_SSPR_MD5), (unsigned char *)tskey);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_SSPR_MD5]++; }
            Foundcnt[JOB_SSPR_MD5]++;
            continue;
          }
        }
      }
    }
    /* NetIQ SSPR SHA1/SHA256/SHA512 (32020/32030/32031/32040/32041):
     * $sspr$N$ITER$SALT$HEXHASH — iterated hash
     * Typesalt = "ITER:SALT" (salt is base64 or hex depending on variant) */
    if (strncmp(line, "$sspr$", 6) == 0 && inhashbuf) {
      char *p = line + 6;
      int stype = *p - '0';  /* 1=SHA1(unsalted), 2=SHA1(salted), 3=SHA256, 4=SHA512 */
      int job_id = 0;
      int expect_hexlen = 0;
      if (stype == 1 && lf[JOB_SSPR_SHA1])        { job_id = JOB_SSPR_SHA1; expect_hexlen = 40; }
      else if (stype == 2 && lf[JOB_SSPR_SHA1S])  { job_id = JOB_SSPR_SHA1S; expect_hexlen = 40; }
      else if (stype == 3 && lf[JOB_SSPR_SHA256]) { job_id = JOB_SSPR_SHA256; expect_hexlen = 64; }
      else if (stype == 4 && lf[JOB_SSPR_SHA512]) { job_id = JOB_SSPR_SHA512; expect_hexlen = 128; }
      if (job_id && p[1] == '$') {
        char *iter_start = p + 2;
        char *d1 = strchr(iter_start, '$');
        if (d1) {
          int iter = atoi(iter_start);
          char *salt = d1 + 1;
          char *d2 = strchr(salt, '$');
          if (d2 && iter > 0) {
            char *hash = d2 + 1;
            int hashlen2 = len - (hash - line);
            int saltlen2 = d2 - salt;
            if (hashlen2 == expect_hexlen && saltlen2 > 0) {
              int hok = 1; for (x = 0; x < hashlen2; x++) if (!isxdigit((unsigned char)hash[x])) { hok = 0; break; }
              if (hok) {
                /* JudyJ key = lowercase hex hash */
                char judykey[130];
                for (x = 0; x < hashlen2; x++) judykey[x] = tolower(hash[x]);
                judykey[hashlen2] = 0;
                JSLI(PV, JUDYJ(job_id), (unsigned char *)judykey);
                /* Typesalt = "iter:salt" (NONE → empty salt) */
                char tskey[256];
                if (saltlen2 == 4 && memcmp(salt, "NONE", 4) == 0)
                  snprintf(tskey, sizeof(tskey), "%d:", iter);
                else
                  snprintf(tskey, sizeof(tskey), "%d:%.*s", iter, saltlen2, salt);
                JSLI(PV, TYPESALT(job_id), (unsigned char *)tskey);
                if (PV) { if ((*PV)++ == 0) Saltloaded[job_id]++; }
                Foundcnt[job_id]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* NetIQ SSPR PBKDF2-HMAC-SHA1 (32050): $pbkdf2-hmac-sha1$ITER$64HEXSALT$40HEXHASH
     * Convert to internal sha1:ITER:SALT_B64:HASH_B64 format for JOB_PBKDF2_SHA1 */
    if ((lf[JOB_PBKDF2_SHA1] || lf[JOB_PKCS5S2]) && strncmp(line, "$pbkdf2-hmac-sha1$", 18) == 0 && inhashbuf) {
      char *iter_start = line + 18;
      char *d1 = strchr(iter_start, '$');
      if (d1) {
        int iter = atoi(iter_start);
        char *salthex = d1 + 1;
        char *d2 = strchr(salthex, '$');
        if (d2 && iter > 0) {
          int salthexlen = d2 - salthex;
          char *hashhex = d2 + 1;
          int hashhexlen = len - (hashhex - line);
          if (salthexlen > 0 && salthexlen <= 128 && hashhexlen == 40) {
            /* Decode hex salt and hash to binary, then b64 encode */
            unsigned char saltbin[64], hashbin[20];
            int saltbinlen = get32(salthex, saltbin, salthexlen/2);
            int hashbinlen = get32(hashhex, hashbin, 20);
            if (saltbinlen > 0 && hashbinlen == 20) {
              char convbuf[512];
              int offset = sprintf(convbuf, "sha1:%d:", iter);
              b64_encode((char *)saltbin, convbuf + offset, saltbinlen);
              offset = mystrlen(convbuf);
              convbuf[offset++] = ':';
              b64_encode((char *)hashbin, convbuf + offset, hashbinlen);
              JSLI(PV, JUDYJ(JOB_PKCS5S2), (unsigned char *)convbuf);
              /* Typesalt = everything before last : */
              { char *lastsep = strrchr(convbuf, ':');
                if (lastsep) {
                  *lastsep = 0;
                  JSLI(PV, TYPESALT(JOB_PBKDF2_SHA1), (unsigned char *)convbuf);
                  if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_PBKDF2_SHA1]++; }
                  *lastsep = ':';
                }
              }
              Foundcnt[JOB_PKCS5S2]++;
              continue;
            }
          }
        }
      }
    }
    /* Empire CMS (32300): 32hex:salt1:salt2 — md5(salt2+"E!m^p-i(r#e.C:M?S"+md5hex(md5(md5hex(pass)+salt1))+"d)i.g^o-d"+salt1)
     * Uses checkhashsalt2 with hex-fallthrough: JudyJ key = lowercase 32-hex,
     * Typesalt = "salt1:salt2" */
    if (lf[JOB_EMPIRECMS] && len > 35 && inhashbuf) {
      char *c1 = (char *)memchr(line, ':', len);
      if (c1 && (c1 - line) == 32) {
        int hok = 1; for (x = 0; x < 32; x++) if (!isxdigit((unsigned char)line[x])) { hok = 0; break; }
        if (hok) {
          char *salt1 = c1 + 1;
          char *c2 = (char *)memchr(salt1, ':', len - 33);
          if (c2) {
            int salt1len = c2 - salt1;
            char *salt2 = c2 + 1;
            int salt2len = len - (salt2 - line);
            if (salt1len > 0 && salt2len > 0) {
              char judykey[33];
              for (x = 0; x < 32; x++) judykey[x] = tolower(line[x]);
              judykey[32] = 0;
              JSLI(PV, JUDYJ(JOB_EMPIRECMS), (unsigned char *)judykey);
              char tskey[512];
              snprintf(tskey, sizeof(tskey), "%.*s:%.*s", salt1len, salt1, salt2len, salt2);
              JSLI(PV, TYPESALT(JOB_EMPIRECMS), (unsigned char *)tskey);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_EMPIRECMS]++; }
              Foundcnt[JOB_EMPIRECMS]++;
              continue;
            }
          }
        }
      }
    }
    /* PBKDF1-SHA1 (32900): PBKDF1:sha1:ITER:B64SALT:B64HASH
     * Algorithm: SHA1(pass + salt_decoded), then iterate SHA1(digest) N-1 times
     * JudyJ key = b64hash, Typesalt = "iter:hexsalt" */
    if (lf[JOB_PBKDF1_SHA1] && strncmp(line, "PBKDF1:sha1:", 12) == 0 && inhashbuf) {
      char *iter_s = line + 12;
      char *d1 = strchr(iter_s, ':');
      if (d1) {
        int iter = atoi(iter_s);
        char *saltb64 = d1 + 1;
        char *d2 = strchr(saltb64, ':');
        if (d2 && iter > 0) {
          int saltb64len = d2 - saltb64;
          char *hashb64 = d2 + 1;
          int hashb64len = len - (hashb64 - line);
          if (saltb64len > 0 && hashb64len > 0) {
            /* Decode salt from b64 to hex for Typesalt */
            unsigned char saltdec[256];
            int incount;
            int saltdeclen = b64_decode(saltb64, (char *)saltdec, &incount);
            if (saltdeclen > 0) {
              char judykey[256];
              snprintf(judykey, sizeof(judykey), "%.*s", hashb64len, hashb64);
              JSLI(PV, JUDYJ(JOB_PBKDF1_SHA1), (unsigned char *)judykey);
              /* Typesalt = "iter:hexsalt" */
              char tskey[600];
              int tsoff = snprintf(tskey, 20, "%d:", iter);
              for (y = 0; y < saltdeclen && tsoff + 2 < (int)sizeof(tskey); y++) {
                tskey[tsoff++] = hextab[saltdec[y] >> 4];
                tskey[tsoff++] = hextab[saltdec[y] & 0xf];
              }
              tskey[tsoff] = 0;
              JSLI(PV, TYPESALT(JOB_PBKDF1_SHA1), (unsigned char *)tskey);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_PBKDF1_SHA1]++; }
              Foundcnt[JOB_PBKDF1_SHA1]++;
              continue;
            }
          }
        }
      }
    }
    /* Microsoft Online Account (33700): $MSONLINEACCOUNT$0$ITER$64hex
     * Algorithm: PBKDF2-HMAC-SHA256(UTF16LE(pass), "", iter, 32) → AES-256-ECB decrypt
     * JudyJ key = full $MSONLINEACCOUNT$ string, Typesalt = "iter:hexdata" */
    if (lf[JOB_MSONLINE] && strncmp(line, "$MSONLINEACCOUNT$", 17) == 0 && inhashbuf) {
      char *p = line + 17;
      if (*p == '0' && *(p+1) == '$') {
        char *iter_s = p + 2;
        char *d1 = strchr(iter_s, '$');
        if (d1) {
          int iter = atoi(iter_s);
          char *datahex = d1 + 1;
          int datahexlen = len - (datahex - line);
          if (iter > 0 && datahexlen == 64) {
            int hok = 1; for (x = 0; x < 64; x++) if (!isxdigit((unsigned char)datahex[x])) { hok = 0; break; }
            if (hok) {
              char judykey[256];
              snprintf(judykey, sizeof(judykey), "%.*s", (int)len, line);
              JSLI(PV, JUDYJ(JOB_MSONLINE), (unsigned char *)judykey);
              char tskey[256];
              snprintf(tskey, sizeof(tskey), "%d:%.*s", iter, datahexlen, datahex);
              JSLI(PV, TYPESALT(JOB_MSONLINE), (unsigned char *)tskey);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_MSONLINE]++; }
              Foundcnt[JOB_MSONLINE]++;
              continue;
            }
          }
        }
      }
    }
    /* WBB4 (33800): bcrypt(bcrypt($pass)) — standard bcrypt format $2a$CC$22charsalt31charhash
     * JudyJ key = full bcrypt string, Typesalt = "$2a$CC$22charsalt" prefix */
    if (lf[JOB_WBB4] && len == 60 && (memcmp(line, "$2a$", 4) == 0 || memcmp(line, "$2b$", 4) == 0) && inhashbuf) {
      if (line[6] == '$') {
        char judykey[61];
        memcpy(judykey, line, 60); judykey[60] = 0;
        JSLI(PV, JUDYJ(JOB_WBB4), (unsigned char *)judykey);
        /* Typesalt = first 29 chars (prefix + salt) */
        char tskey[30];
        memcpy(tskey, line, 29); tskey[29] = 0;
        JSLI(PV, TYPESALT(JOB_WBB4), (unsigned char *)tskey);
        if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_WBB4]++; }
        Foundcnt[JOB_WBB4]++;
        continue;
      }
    }
    /* SAP CODVN H SHA512 (35000): {x-isSHA512, ITER}BASE64(hash64+salt)
     * Algorithm: SHA512(pass+salt) iterated, same as SAPCODVNH but SHA512
     * JudyJ key = 128-hex SHA512, Typesalt = "iter:hexsalt" */
    /* MONGODB (John "MongoDB" type 0): $mongodb$0$<user>$<32hex>
     * digest = md5(user . ":mongo:" . pass). JudyJ key is the whole stored form,
     * Typesalt the username. Type 1 is a sniffed challenge/response and is not
     * loaded here. */
    if (lf[JOB_MONGODB] && len > 14 && memcmp(line, "$mongodb$0$", 11) == 0) {
      char *mg_d = strchr(line + 11, '$');
      if (mg_d && (len - (int)(mg_d + 1 - line)) == 32) {
        int mg_ul = (int)(mg_d - line - 11);
        if (mg_ul > 0 && mg_ul < 200) {
          JSLI(PV, JUDYJ(JOB_MONGODB), (unsigned char *)line);
          Saltloaded[JOB_MONGODB] += store_typesalt(JOB_MONGODB, line + 11, mg_ul);
          Foundcnt[JOB_MONGODB]++;
          continue;
        }
      }
    }
    /* H3C: $h$6$<salt>$<base64 of 64-byte sha512>
     * digest = sha512(pass . NUL . salt . pass . NUL). */
    if (lf[JOB_H3C] && len > 10 && memcmp(line, "$h$6$", 5) == 0) {
      char *h3_d = strchr(line + 5, '$');
      if (h3_d) {
        int h3_sl = (int)(h3_d - line - 5);
        if (h3_sl > 0 && h3_sl < 200 && (len - h3_sl - 6) >= 40) {
          JSLI(PV, JUDYJ(JOB_H3C), (unsigned char *)line);
          Saltloaded[JOB_H3C] += store_typesalt(JOB_H3C, line + 5, h3_sl);
          Foundcnt[JOB_H3C]++;
          continue;
        }
      }
    }

      /* DragonFly BSD $3$/$4$: <tag><salt>$<crypt-b64 digest>.
       * digest = sha256/sha512(pass . magic . salt); magic is the tag plus
       * its NUL, and the 64-bit variants read 8 magic bytes rather than 4,
       * picking up four adjacent rodata bytes. The stored forms of the 32-
       * and 64-bit variants are IDENTICAL, so a line is loaded into both
       * types and each is tried separately. */
      if (line[0] == '$' && (line[1] == '3' || line[1] == '4') && line[2] == '$') {
        char *df_d = strchr(line + 3, '$');
        if (df_d) {
          int df_sl = (int)(df_d - line - 3);
          int df_hl = len - df_sl - 4;
          if (df_sl > 0 && df_sl <= 8 &&
              df_hl == ((line[1] == '3') ? 44 : 84)) {
            int df_a = (line[1] == '3') ? JOB_DRAGONFLY3_32 : JOB_DRAGONFLY4_32;
            int df_i, df_hit = 0;
            for (df_i = df_a; df_i <= df_a + 1; df_i++) {
              if (!lf[df_i]) continue;
              JSLI(PV, JUDYJ(df_i), (unsigned char *)line);
              Saltloaded[df_i] += store_typesalt(df_i, line + 3, df_sl);
              Foundcnt[df_i]++;
              df_hit = 1;
            }
            if (df_hit) continue;
          }
        }
      }

    /* SAP CODVN H SHA256: {x-isSHA256, ITER}BASE64(digest || salt).
     * Same shape as SAPCODVNH512, 32-byte digest. JudyJ key = 64-hex digest,
     * Typesalt = "iter:hexsalt". */
    if (lf[JOB_SAPCODVNH256] && len > 20 && memcmp(line, "{x-isSHA256, ", 13) == 0) {
      char *brace = strchr(line + 13, '}');
      if (brace) {
        int sh_iter = atoi(line + 13);
        if (sh_iter >= 1) {
          unsigned char sh_dec[400];
          int incount;
          int sh_dlen = b64_decode(brace + 1, (char *)sh_dec, &incount);
          if (sh_dlen >= 32+1) {
            char sh_jkey[64+1];
            prmd5_64(sh_dec, sh_jkey);
            sh_jkey[64] = 0;
            JSLI(PV, JUDYJ(JOB_SAPCODVNH256), (unsigned char *)sh_jkey);
            int sh_saltlen = sh_dlen - 32;
            char sh_ts[600];
            int sh_tsoff = snprintf(sh_ts, 20, "%d:", sh_iter);
            for (y = 0; y < sh_saltlen && sh_tsoff + 2 < (int)sizeof(sh_ts); y++) {
              sh_ts[sh_tsoff++] = hextab[sh_dec[32+y] >> 4];
              sh_ts[sh_tsoff++] = hextab[sh_dec[32+y] & 0xf];
            }
            sh_ts[sh_tsoff] = 0;
            Saltloaded[JOB_SAPCODVNH256] += store_typesalt(JOB_SAPCODVNH256, sh_ts, sh_tsoff);
            Foundcnt[JOB_SAPCODVNH256]++;
            continue;
          }
        }
      }
    }
    /* SAP CODVN H SHA384: {x-isSHA384, ITER}BASE64(digest || salt).
     * Same shape as SAPCODVNH512, 48-byte digest. JudyJ key = 96-hex digest,
     * Typesalt = "iter:hexsalt". */
    if (lf[JOB_SAPCODVNH384] && len > 20 && memcmp(line, "{x-isSHA384, ", 13) == 0) {
      char *brace = strchr(line + 13, '}');
      if (brace) {
        int sh_iter = atoi(line + 13);
        if (sh_iter >= 1) {
          unsigned char sh_dec[400];
          int incount;
          int sh_dlen = b64_decode(brace + 1, (char *)sh_dec, &incount);
          if (sh_dlen >= 48+1) {
            char sh_jkey[96+1];
            prmd5_96(sh_dec, sh_jkey);
            sh_jkey[96] = 0;
            JSLI(PV, JUDYJ(JOB_SAPCODVNH384), (unsigned char *)sh_jkey);
            int sh_saltlen = sh_dlen - 48;
            char sh_ts[600];
            int sh_tsoff = snprintf(sh_ts, 20, "%d:", sh_iter);
            for (y = 0; y < sh_saltlen && sh_tsoff + 2 < (int)sizeof(sh_ts); y++) {
              sh_ts[sh_tsoff++] = hextab[sh_dec[48+y] >> 4];
              sh_ts[sh_tsoff++] = hextab[sh_dec[48+y] & 0xf];
            }
            sh_ts[sh_tsoff] = 0;
            Saltloaded[JOB_SAPCODVNH384] += store_typesalt(JOB_SAPCODVNH384, sh_ts, sh_tsoff);
            Foundcnt[JOB_SAPCODVNH384]++;
            continue;
          }
        }
      }
    }
    if (lf[JOB_SAPCODVNH512] && len > 20 && memcmp(line, "{x-isSHA512, ", 13) == 0) {
      char *brace = strchr(line + 13, '}');
      if (brace) {
        int sh_iter = atoi(line + 13);
        if (sh_iter >= 1) {
          unsigned char sh_dec[400];
          int incount;
          int sh_dlen = b64_decode(brace + 1, (char *)sh_dec, &incount);
          if (sh_dlen >= 65) { /* 64 hash + at least 1 salt byte */
            char sh_jkey[129];
            prmd5_128(sh_dec, sh_jkey);
            sh_jkey[128] = 0;
            JSLI(PV, JUDYJ(JOB_SAPCODVNH512), (unsigned char *)sh_jkey);
            /* Typesalt = "iter:hex_salt" */
            int sh_saltlen = sh_dlen - 64;
            char sh_ts[600];
            int sh_tsoff = snprintf(sh_ts, 20, "%d:", sh_iter);
            for (y = 0; y < sh_saltlen && sh_tsoff + 2 < (int)sizeof(sh_ts); y++) {
              sh_ts[sh_tsoff++] = hextab[sh_dec[64+y] >> 4];
              sh_ts[sh_tsoff++] = hextab[sh_dec[64+y] & 0xf];
            }
            sh_ts[sh_tsoff] = 0;
            Saltloaded[JOB_SAPCODVNH512] += store_typesalt(JOB_SAPCODVNH512, sh_ts, sh_tsoff);
            Foundcnt[JOB_SAPCODVNH512]++;
            continue;
          }
        }
      }
    }
    /* AS/400 SSHA1 (35200): $as400$ssha1$*USER*40HEXHASH
     * Algorithm: SHA1(UTF16BE(UC(user_pad10)) + UTF16BE(pass))
     * JudyJ key = 40-hex UC hash, Typesalt = uppercase username */
    if (lf[JOB_AS400SSHA1] && strncmp(line, "$as400$ssha1$*", 14) == 0 && inhashbuf) {
      char *user = line + 14;
      char *d1 = strchr(user, '*');
      if (d1) {
        int userlen = d1 - user;
        char *hash = d1 + 1;
        int hashlen2 = len - (hash - line);
        if (userlen >= 1 && userlen <= 10 && hashlen2 == 40) {
          int hok = 1; for (x = 0; x < 40; x++) if (!isxdigit((unsigned char)hash[x])) { hok = 0; break; }
          if (hok) {
            char judykey[41];
            for (x = 0; x < 40; x++) judykey[x] = toupper(hash[x]);
            judykey[40] = 0;
            JSLI(PV, JUDYJ(JOB_AS400SSHA1), (unsigned char *)judykey);
            char tskey[11];
            for (x = 0; x < userlen; x++) tskey[x] = toupper(user[x]);
            tskey[userlen] = 0;
            JSLI(PV, TYPESALT(JOB_AS400SSHA1), (unsigned char *)tskey);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_AS400SSHA1]++; }
            Foundcnt[JOB_AS400SSHA1]++;
            continue;
          }
        }
      }
    }
    /* ARGON2 (34000): $argon2{d|i|id}$v=VER$m=MEM,t=ITER,p=PARA$B64SALT$B64HASH
     * JudyJ key = full format string (lowercase)
     * Typesalt = "type:ver:m:t:p:saltlen:salthex:hashlen" */
    if ((lf[JOB_ARGON2] || lf[JOB_ARGON2MD5]) && strncmp(line, "$argon2", 7) == 0 && inhashbuf) {
      int a2type = -1;
      char *p = line + 7;
      if (*p == 'd' && p[1] == '$') { a2type = 0; p += 2; }
      else if (*p == 'i' && p[1] == '$') { a2type = 1; p += 2; }
      else if (*p == 'i' && p[1] == 'd' && p[2] == '$') { a2type = 2; p += 3; }
      if (a2type >= 0 && strncmp(p, "v=", 2) == 0) {
        int ver = atoi(p + 2);
        char *d1 = strchr(p, '$');
        if (d1 && (ver == 19 || ver == 16)) {
          d1++;
          int m_cost = 0, t_cost = 0, parallelism = 0;
          if (sscanf(d1, "m=%d,t=%d,p=%d", &m_cost, &t_cost, &parallelism) == 3 &&
              m_cost >= 1 && t_cost >= 1 && parallelism >= 1 && parallelism <= 32) {
            char *d2 = strchr(d1, '$');
            if (d2) {
              d2++;
              char *d3 = strchr(d2, '$');
              if (d3 && (d3 - d2) >= 1 && (d3 - d2) <= 512) {
                int b64saltlen = d3 - d2;
                d3++;
                int b64hashlen = len - (d3 - line);
                if (b64hashlen >= 1 && b64hashlen <= 512) {
                  /* decode salt */
                  unsigned char saltbin[256];
                  int saltincount;
                  int saltbinlen = b64_decode(d2, (char *)saltbin, &saltincount);
                  if (saltincount == b64saltlen && saltbinlen >= 1 && saltbinlen <= 255) {
                    /* decode hash to get hashlen */
                    unsigned char hashbin[256];
                    int hashincount;
                    int hashbinlen = b64_decode(d3, (char *)hashbin, &hashincount);
                    if (hashincount == b64hashlen && hashbinlen >= 1 && hashbinlen <= 128) {
                      /* Store full line as JudyJ key (case-sensitive, base64 is case-sensitive) */
                      char judykey[MAXLINE];
                      memcpy(judykey, line, len);
                      judykey[len] = 0;
                      /* The wrapper does not say whether the password was pre-hashed,
                       * so a selected ARGON2MD5 gets its own copy of the same line. */
                      if (lf[JOB_ARGON2])    JSLI(PV, JUDYJ(JOB_ARGON2),    (unsigned char *)judykey);
                      if (lf[JOB_ARGON2MD5]) JSLI(PV, JUDYJ(JOB_ARGON2MD5), (unsigned char *)judykey);
                      /* Build Typesalt: type:ver:m:t:p:saltlen:salthex:hashlen */
                      char salthex[512];
                      prmd5(saltbin, salthex, saltbinlen * 2);
                      salthex[saltbinlen * 2] = 0;
                      /* Use JSLI directly (salt string can exceed 255) */
                      char tskey[MAXLINE];
                      snprintf(tskey, MAXLINE, "%d:%d:%d:%d:%d:%d:%s:%d",
                               a2type, ver, m_cost, t_cost, parallelism,
                               saltbinlen, salthex, hashbinlen);
                      /* Both types read the same wrapper, so whichever is
                       * selected needs its own salt entry; keying only ARGON2
                       * left a selected ARGON2MD5 with an empty snapshot and it
                       * computed nothing at all. */
                      if (lf[JOB_ARGON2]) {
                        JSLI(PV, TYPESALT(JOB_ARGON2), (unsigned char *)tskey);
                        if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_ARGON2]++; }
                        Foundcnt[JOB_ARGON2]++;
                      }
                      if (lf[JOB_ARGON2MD5]) {
                        JSLI(PV, TYPESALT(JOB_ARGON2MD5), (unsigned char *)tskey);
                        if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_ARGON2MD5]++; }
                        Foundcnt[JOB_ARGON2MD5]++;
                      }
                      /* Track max workspace needed */
                      { size_t needed = (size_t)m_cost * 1024;
                        if (needed > Argon2_maxmem) Argon2_maxmem = needed;
                      }
                      continue;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    /* ARGON2 Magento v2 format: HEXHASH:SALT:{2|3[_HASHLEN_ITER_MEMBYTES]}
     * Always argon2id, version 19, parallelism 1.
     * Salt = first 16 chars of salt field, used as raw ASCII bytes.
     * JudyJ key = standard $argon2id$... string, Typesalt = same format as above */
    if (lf[JOB_ARGON2] && len > 66 && inhashbuf) {
      /* Check for even-length hex hash followed by colon */
      int hexlen = 0;
      { char *fc = memchr(line, ':', len);
        if (fc) hexlen = fc - line;
      }
      if (hexlen >= 2 && hexlen <= 256 && (hexlen & 1) == 0 && line[hexlen] == ':') {
        int hexok = 1;
        for (x = 0; x < hexlen; x++)
          if (!isxdigit((unsigned char)line[x])) { hexok = 0; break; }
        if (hexok) {
          /* Find second colon: salt field */
          char *saltstart = line + hexlen + 1;
          char *colon2 = memchr(saltstart, ':', len - hexlen - 1);
          if (colon2) {
            char *verfield = colon2 + 1;
            int verlen = len - (verfield - line);
            if (verlen >= 1 && (verfield[0] == '2' || verfield[0] == '3') &&
                (verlen == 1 || verfield[1] == '_')) {
              /* Parse parameters */
              int mg_hashlen = hexlen / 2;
              int mg_t = 2, mg_m = 65536;  /* defaults */
              int mg_valid = 1;
              if (verlen > 1 && verfield[1] == '_') {
                int ph = 0, pt = 0;
                long long pm = 0;
                if (sscanf(verfield + 2, "%d_%d_%lld", &ph, &pt, &pm) == 3 &&
                    ph == mg_hashlen && pt >= 1 && pm >= 8192) {
                  mg_t = pt;
                  mg_m = (int)(pm / 1024);  /* bytes to KiB */
                } else mg_valid = 0;
              }
              if (mg_valid) {
                /* Salt: first 16 chars of salt field, as raw ASCII */
                int sflen = colon2 - saltstart;
                int saltlen = (sflen > 16) ? 16 : sflen;
                if (saltlen >= 1) {
                  /* Build Typesalt in salttmp: "2:19:m:t:1:saltlen:salthex:hashlen"
                   * Layout: [0..63] salthex, [64..319] tskey, [320..447] hashbin,
                   *         [448..511] b64salt, [512..703] b64hash, [704..] judykey */
                  char *salthex = salttmp;
                  prmd5((unsigned char *)saltstart, salthex, saltlen * 2);
                  snprintf(salttmp + 64, 256, "2:19:%d:%d:1:%d:%s:%d",
                           mg_m, mg_t, saltlen, salthex, mg_hashlen);
                  JSLI(PV, TYPESALT(JOB_ARGON2), (unsigned char *)(salttmp + 64));
                  if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_ARGON2]++; }
                  /* Build JudyJ key: standard $argon2id$ format
                   * Decode hex hash to binary, then base64-encode salt and hash */
                  unsigned char *hashbin = (unsigned char *)salttmp + 320;
                  int hblen = hexlen / 2;
                  get32(line, hashbin, hblen);
                  char *b64salt = salttmp + 448, *b64hash = salttmp + 512;
                  int b64slen = b64_encode((char *)saltstart, b64salt, saltlen);
                  while (b64slen > 0 && b64salt[b64slen-1] == '=') b64slen--;
                  b64salt[b64slen] = 0;
                  int b64hlen = b64_encode((char *)hashbin, b64hash, hblen);
                  while (b64hlen > 0 && b64hash[b64hlen-1] == '=') b64hlen--;
                  b64hash[b64hlen] = 0;
                  /* b64salt + b64hash here are bounded by the upstream salt/hash hex parse
                   * (typically <= 256 bytes each for Argon2 inputs); the MAXLINE-704
                   * destination far exceeds the realistic worst case. snprintf is bounded
                   * regardless — suppress the upper-bound informational warning. */
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
                  snprintf(salttmp + 704, MAXLINE - 704, "$argon2id$v=19$m=%d,t=%d,p=1$%s$%s",
                           mg_m, mg_t, b64salt, b64hash);
#pragma GCC diagnostic pop
                  JSLI(PV, JUDYJ(JOB_ARGON2), (unsigned char *)(salttmp + 704));
                  Foundcnt[JOB_ARGON2]++;
                  { size_t needed = (size_t)mg_m * 1024;
                    if (needed > Argon2_maxmem) Argon2_maxmem = needed;
                  }
                  continue;
                }
              }
            }
          }
        }
      }
    }
      /* SYMFONY256 (35800): 64hex:SALT -- Symfony Legacy SHA256.
       * No special case: <hex>:<salt> is exactly what the generic hex loader
       * consumes, putting the digest in the compact table and the remainder
       * in Typesalt. Claiming the line here and diverting it to JudyJ made
       * the type loadable only as a combined record through -F. */
    /* WPBCRYPT (35500): $wp$2y$NN$SALT22HASH31 — WordPress bcrypt(hmac-sha384)
     * Format: $wp$ + 2y$NN$salt22hash31 (59 chars after $wp$, 63 total)
     * JudyJ key = "$2y$NN$salt22hash31" (60 chars, standard bcrypt)
     * Typesalt = "$2y$NN$salt22" (29 chars, bcrypt setting) */
    if (lf[JOB_WPBCRYPT] && len == 63 && strncmp(line, "$wp$", 4) == 0 && inhashbuf) {
      char *bchash = line + 4; /* points at "2y$NN$..." (59 chars) */
      if (bchash[2] == '$' && bchash[5] == '$') {
        /* Reconstruct standard bcrypt: "$2y$NN$salt22hash31" (60 chars) */
        char judykey[61];
        judykey[0] = '$';
        memcpy(judykey + 1, bchash, 59);
        judykey[60] = 0;
        JSLI(PV, JUDYJ(JOB_WPBCRYPT), (unsigned char *)judykey);
        /* Setting = first 29 chars: "$2y$NN$salt22" */
        char tskey[30];
        tskey[0] = '$';
        memcpy(tskey + 1, bchash, 28);
        tskey[29] = 0;
        JSLI(PV, TYPESALT(JOB_WPBCRYPT), (unsigned char *)tskey);
        if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_WPBCRYPT]++; }
        Foundcnt[JOB_WPBCRYPT]++;
        continue;
      }
    }
    /* SUNMD5: $md5$[rounds=N$]SALT$HASH22 (salt = up to the LAST $) */
    if (lf[JOB_SUNMD5] && strncmp(line, "$md5", 4) == 0 && len > 26 && inhashbuf) {
      char *lastsep = NULL;
      for (x = len - 1; x > 3; x--) {
        if (line[x] == '$') { lastsep = line + x; break; }
      }
      if (lastsep && (len - (int)(lastsep + 1 - line)) == 22) {
        char judykey[512];
        snprintf(judykey, sizeof(judykey), "%.*s", (int)len, line);
        JSLI(PV, JUDYJ(JOB_SUNMD5), (unsigned char *)judykey);
        Foundcnt[JOB_SUNMD5]++;
        continue;
      }
    }
    /* GOST12256CRYPT: $gost12256hash$[rounds=N$]SALT$HASH43 */
    if (lf[JOB_GOST12256CRYPT] && strncmp(line, "$gost12256hash$", 15) == 0 && inhashbuf) {
      char *lastsep = NULL;
      for (x = len - 1; x > 14; x--) {
        if (line[x] == '$') { lastsep = line + x; break; }
      }
      if (lastsep && (len - (int)(lastsep + 1 - line)) == 43) {
        char judykey[512];
        snprintf(judykey, sizeof(judykey), "%.*s", (int)len, line);
        JSLI(PV, JUDYJ(JOB_GOST12256CRYPT), (unsigned char *)judykey);
        Foundcnt[JOB_GOST12256CRYPT]++;
        continue;
      }
    }
    /* GOST94CRYPT: $gost94hash$[rounds=N$]SALT$HASH43 */
    if (lf[JOB_GOST94CRYPT] && strncmp(line, "$gost94hash$", 12) == 0 && inhashbuf) {
      char *lastsep = NULL;
      for (x = len - 1; x > 11; x--) {
        if (line[x] == '$') { lastsep = line + x; break; }
      }
      if (lastsep && (len - (int)(lastsep + 1 - line)) == 43) {
        char judykey[512];
        snprintf(judykey, sizeof(judykey), "%.*s", (int)len, line);
        JSLI(PV, JUDYJ(JOB_GOST94CRYPT), (unsigned char *)judykey);
        Foundcnt[JOB_GOST94CRYPT]++;
        continue;
      }
    }
    /* GOST12512CRYPT (35600): $gost12512hash$[rounds=N$]SALT$HASH86
     * Algorithm: sha512crypt with GOST-2012 Streebog-512 replacing SHA-512
     * JudyJ key = full format string; bootstrap copies to Typesalt (like SHA512CRYPT) */
    if (lf[JOB_GOST12512CRYPT] && strncmp(line, "$gost12512hash$", 15) == 0 && inhashbuf) {
      /* Find the last $ — hash is after it (86 chars of itoa64) */
      char *lastsep = NULL;
      for (x = len - 1; x > 14; x--) {
        if (line[x] == '$') { lastsep = line + x; break; }
      }
      if (lastsep && (len - (int)(lastsep + 1 - line)) == 86) {
        char judykey[512];
        snprintf(judykey, sizeof(judykey), "%.*s", (int)len, line);
        JSLI(PV, JUDYJ(JOB_GOST12512CRYPT), (unsigned char *)judykey);
        Foundcnt[JOB_GOST12512CRYPT]++;
        continue;
      }
    }
    /* YESCRYPT (67000): $y$PARAMS$SALT$HASH (43 chars of itoa64)
     * JudyJ key = full format string; Typesalt = setting ($y$params$salt$) */
    if (lf[JOB_YESCRYPT] && strncmp(line, "$y$", 3) == 0 && inhashbuf) {
      /* Find params$salt$hash — need exactly 3 '$' after $y$ */
      char *p = line + 3;
      char *salt_start = strchr(p, '$');
      if (salt_start) {
        salt_start++;
        char *hash_start = strchr(salt_start, '$');
        if (hash_start && (len - (int)(hash_start + 1 - line)) == 43) {
          char judykey[512], saltkey[256];
          int setting_len = hash_start - line + 1; /* include trailing $ */
          snprintf(judykey, sizeof(judykey), "%.*s", (int)len, line);
          snprintf(saltkey, sizeof(saltkey), "%.*s", setting_len, line);
          JSLI(PV, JUDYJ(JOB_YESCRYPT), (unsigned char *)judykey);
          JSLI(PV, TYPESALT(JOB_YESCRYPT), (unsigned char *)saltkey);
          if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_YESCRYPT]++; }
          Foundcnt[JOB_YESCRYPT]++;
          continue;
        }
      }
    }
    /* GOST-YESCRYPT (46100): $gy$PARAMS$SALT$HASH (43 chars of itoa64)
     * JudyJ key = full format string; Typesalt = setting ($gy$params$salt$).
     *
     * The prefix test is exact, lowercase and anchored on purpose. The CMIYC
     * corpus salts three md5crypt decoys into the same file ($1$Gy$..., $1$gY$...)
     * that a substring or case-insensitive match would happily swallow. */
    if (lf[JOB_GOSTYESCRYPT] && strncmp(line, "$gy$", 4) == 0 && inhashbuf) {
      char *p = line + 4;
      char *salt_start = strchr(p, '$');
      if (salt_start) {
        salt_start++;
        char *hash_start = strchr(salt_start, '$');
        if (hash_start && (len - (int)(hash_start + 1 - line)) == 43) {
          char judykey[512], saltkey[256];
          int setting_len = hash_start - line + 1; /* include trailing $ */
          if (setting_len < (int)sizeof(saltkey) && len < (int)sizeof(judykey)) {
            snprintf(judykey, sizeof(judykey), "%.*s", (int)len, line);
            snprintf(saltkey, sizeof(saltkey), "%.*s", setting_len, line);
            JSLI(PV, JUDYJ(JOB_GOSTYESCRYPT), (unsigned char *)judykey);
            JSLI(PV, TYPESALT(JOB_GOSTYESCRYPT), (unsigned char *)saltkey);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_GOSTYESCRYPT]++; }
            Foundcnt[JOB_GOSTYESCRYPT]++;
            continue;
          }
        }
      }
    }
    /* CMIYC contest hash (e1001), from challenge_3 (AIX PowerPC binary):
     *   $cmiyc$<version>$<rounds>$<memlog>$<b64url salt>$<b64url hash>
     * b64url = A-Za-z0-9-_ with NO padding; salt 22 chars -> 16 bytes,
     * hash 43 chars -> 32 bytes. Parameter ranges from cmiyc_raw's own
     * argument validation: rounds 1..32, memlog 10..24.
     *
     * PARSING ONLY. The digest is NOT computed -- see the JOB_CMIYC case in
     * procjob for why (the buffer-expansion step is still undecoded). The
     * loader exists so the type number is pinned in both mdxfind and
     * hashpipe, and so the hashes can be counted and carried, without any
     * risk of reporting a wrong answer. */
    if (lf[JOB_CMIYC] && strncmp(line, "$cmiyc$", 7) == 0) {
      char *f[5];
      int nf = 0;
      char *sp = line + 7;
      f[nf++] = sp;
      while (*sp && nf < 5) {
        if (*sp == '$') { *sp = 0; f[nf++] = sp + 1; }
        sp++;
      }
      if (nf == 5) {
        long c_ver = strtol(f[0], NULL, 10);
        long c_rounds = strtol(f[1], NULL, 10);
        long c_memlog = strtol(f[2], NULL, 10);
        int slen = (int)strlen(f[3]), hlen = (int)strlen(f[4]);
        int okc = 1, q;
        for (q = 0; q < slen + hlen && okc; q++) {
          int ch2 = (q < slen) ? (unsigned char)f[3][q]
                               : (unsigned char)f[4][q - slen];
          if (!isalnum(ch2) && ch2 != '-' && ch2 != '_') okc = 0;
        }
        { int k; for (k = 0; k < nf - 1; k++) f[k][strlen(f[k])] = '$'; }
        if (okc && c_ver > 0 && c_rounds >= 1 && c_rounds <= 32 &&
            c_memlog >= 10 && c_memlog <= 24 && slen == 22 && hlen == 43) {
          char cjudy[256], csalt[128];
          int setting_len = (int)(f[4] - line);   /* through the salt's '$' */
          if (len < (int)sizeof(cjudy) && setting_len < (int)sizeof(csalt)) {
            snprintf(cjudy, sizeof(cjudy), "%.*s", (int)len, line);
            snprintf(csalt, sizeof(csalt), "%.*s", setting_len, line);
            JSLI(PV, JUDYJ(JOB_CMIYC), (unsigned char *)cjudy);
            JSLI(PV, TYPESALT(JOB_CMIYC), (unsigned char *)csalt);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_CMIYC]++; }
            Foundcnt[JOB_CMIYC]++;
            continue;
          }
        }
      }
    }
    /* 7-Zip 7zAES (11600):
     *   $7z$type$log2iter$saltlen$salt$ivlen$iv$crc$packedlen$unpackedlen$data
     *
     * The `data` field in stock 7z2john output is the ENTIRE encrypted stream
     * (112 KB for the test vector, 1.5 MB for the five challenge archives) --
     * far past MAXLINE, so such a line can never be read here. Stage 1 only
     * needs the final two ciphertext blocks, so mdxfind accepts a TRUNCATED
     * data field: anything with at least 64 hex chars, of which the LAST 64
     * (32 bytes) are used. tools/7z2mdx.py does the truncation. A short stock
     * line (small archive) therefore also parses unchanged.
     *
     * Typesalt holds the KDF GROUP (saltlen:salt:log2iter), not the archive --
     * see struct SevenZipRec. JudyJ holds the full line for dedup/output. */
    if (lf[JOB_SEVENZIP] && strncmp(line, "$7z$", 4) == 0) {
      /* After the "$7z$" prefix:
       *   f[0]=type f[1]=log2 f[2]=saltlen f[3]=salt f[4]=ivlen f[5]=iv
       *   f[6]=crc  f[7]=packedlen f[8]=unpackedlen f[9]=data
       *   f[10]=crc_len f[11]=coder_attributes        (types 1..127 only)
       *
       * The type field does NOT name the codec directly, and an earlier
       * comment here claiming "type 0 = Deflate64, type 1 = LZMA" was wrong.
       * Per 7z2john.pl's own definition: 0 means the data needs no
       * decompression to check the CRC32 -- which is ALSO what an
       * unrecognised codec degrades to, so Deflate64 arrives here as 0.
       * 1..127 pack a decompressor in the low nibble (1 LZMA1, 2 LZMA2,
       * 3 PPMd, 6 BZip2, 7 Deflate) and a BCJ-family preprocessor in the
       * high nibble, and carry the two extra fields. 128 means the data was
       * truncated to its final block, with the penultimate block moved into
       * the iv field. So the field count is 10 or 12, and both are exact.
       *
       * See contest/cmiyc26/challenge/MDXFIND-7Z-SPEC.md sections 2 and 8. */
      char *f[12];
      int flen[12];
      int nf = 0, k;
      char *sp = line + 4;
      const char *why = NULL;         /* non-NULL => reject, and this says why */
      /* MUST be taken before the splitter punches NULs over the separators;
       * afterwards strlen(line) only reaches the first '$'. */
      size_t rawlen = strlen(line);
      f[nf++] = sp;
      while (*sp && nf < 12) {
        if (*sp == '$') { *sp = 0; f[nf++] = sp + 1; }
        sp++;
      }
      /* Field lengths must be taken BEFORE the separators go back, because
       * afterwards every f[k] runs to end of line. */
      for (k = 0; k < nf; k++) flen[k] = (int)strlen(f[k]);
      { long sz_type = 0, sz_log2 = 0, sz_saltlen = 0, sz_ivlen = 0;
        long sz_packed = 0, sz_unpacked = 0, sz_crclen = 0;
        unsigned long sz_crc = 0;
        int dbytes = 0, padsize = 0, bits = 0, tier2 = 0, truncated = 0;
        long minbits = SevenZipMinBits;

        /* A line longer than the read buffer arrives here already cut short,
         * and the bytes at its end are NOT the end of the stream -- so the
         * pad check would run against the wrong ciphertext and could never
         * pass. Nothing downstream can detect that, so catch it here. */
        if (rawlen >= MAXLINE - 64)
          why = "line exceeds MAXLINE and was read truncated; shorten the data "
                "field with tools/7z2mdx.py";
        else if (nf != 10 && nf != 12) why = "field count is not 10 or 12";
        else {
          sz_type     = strtol(f[0], NULL, 10);
          sz_log2     = strtol(f[1], NULL, 10);
          sz_saltlen  = strtol(f[2], NULL, 10);
          sz_ivlen    = strtol(f[4], NULL, 10);
          sz_crc      = strtoul(f[6], NULL, 10);
          sz_packed   = strtol(f[7], NULL, 10);
          sz_unpacked = strtol(f[8], NULL, 10);
          if (nf == 12) sz_crclen = strtol(f[10], NULL, 10);
          dbytes  = flen[9] / 2;
          padsize = (int)(sz_packed - sz_unpacked);
          /* 7z2john encodes unpackedlen mod 16 when it truncates, so a
           * stream that was already block-aligned reports 16 -- which means
           * no padding at all, not sixteen bytes of it. */
          truncated = (sz_type == 128 && dbytes == 16 && sz_packed == 16);
          if (truncated && padsize == 16) padsize = 0;

          /* R1/R2: shape and declared-vs-actual field lengths. */
          if (flen[9] & 1)                     why = "data field has an odd hex length";
          else if (sz_saltlen < 0 || sz_saltlen > 64) why = "saltlen out of range";
          else if (flen[3] & 1)                why = "salt field has an odd hex length";
          else if (flen[3] < sz_saltlen * 2)   why = "salt field is shorter than saltlen declares";
          else if (sz_ivlen < 0 || sz_ivlen > 32)     why = "ivlen out of range";
          else if (flen[5] & 1)                why = "iv field has an odd hex length";
          else if (flen[5] < sz_ivlen * 2)     why = "iv field is shorter than ivlen declares";
          /* R3/R4: is the KDF reachable at all? */
          else if (sz_log2 == 63)              why = "log2=0x3F special KDF is not implemented";
          else if (sz_log2 < 0 || sz_log2 > SEVENZIP_MAX_LOG2)
                                               why = "iteration count 2^log2 is unreachable";
          /* R5 */
          else if (padsize < 0 || padsize > 16) why = "padsize out of range (packedlen < unpackedlen?)";
          /* R6: enough ciphertext to work with. A record carrying the WHOLE
           * stream is usable even when that stream is a single block -- the
           * archive IV supplies the CBC predecessor, and tier 2 can decide it
           * outright. Only a partial record needs two blocks of its own. */
          else {
            tier2 = !truncated && sz_packed > 0 && (long)dbytes == sz_packed;
            if (dbytes < 32 && dbytes != 16)
              why = "data holds fewer than two ciphertext blocks";
          }
        }

        /* restore the separators we punched out, so the JudyJ key and the
         * reported line are byte-identical to the input */
        for (k = 0; k < nf - 1; k++) f[k][flen[k]] = '$';

        if (!why) {
          /* tier2 was set above. It needs the WHOLE stream: 7z2john keeps
           * packedlen in step when it shortens, so a mismatch is the exact
           * signature of a tail-only record (what tools/7z2mdx.py produces).
           * Type 128 rewrites both fields to 16, so it never qualifies. */
          bits  = 8 * padsize;

          if (!tier2 && bits == 0)
            why = "padsize 0 and no complete stream: undecidable by any implemented tier";
          else if (!tier2 && bits < minbits && SevenZipStrict)
            why = "evidence below the strict threshold";
        }

        if (why) {
          sevenzip_reject(why, line);
          continue;                    /* never counted, never searched */
        }

        { unsigned char tail[32], ivbuf[16];
          char grp[96];
          struct SevenZipRec *r;
          const char *tailhex = f[9] + flen[9] - 64;
          int ivb = flen[5] / 2;

          /* 7z stores the iv length beside a field that is zero-padded out to
           * the AES block size, so the field -- not the declared length -- is
           * what the cipher actually consumes. Taking min(field,16) and
           * relying on the zeroed record for the rest reproduces both the
           * 7z2john form (ivlen 16, 32 hex) and John's (ivlen 8, 32 hex). */
          if (ivb > 16) ivb = 16;
          memset(ivbuf, 0, sizeof(ivbuf));
          if (ivb > 0 && get32(f[5], ivbuf, ivb) != ivb) {
            sevenzip_reject("bad hex in iv", line); continue;
          }

          if (dbytes == 16) {
            /* A single ciphertext block: its CBC predecessor is the iv field
             * -- the archive IV when the whole stream is one block, or the
             * penultimate block that 7z2john moves there when it truncates. */
            memcpy(tail, ivbuf, 16);
            if (get32(f[9], tail + 16, 16) != 16) {
              sevenzip_reject("bad hex in data", line); continue;
            }
          } else if (get32((char *)tailhex, tail, 32) != 32) {
            sevenzip_reject("bad hex in data", line); continue;
          }

          /* saltlen is authoritative: a producer may leave stale bytes in the
           * salt field beyond it (John does), and the KDF must not see them. */
          snprintf(grp, sizeof(grp), "%ld:%.*s:%ld",
                   sz_saltlen, (int)(sz_saltlen * 2), f[3], sz_log2);

          r = sevenzip_addrec();
          snprintf(r->group, sizeof(r->group), "%s", grp);
          memcpy(r->tail, tail, 32);
          r->type = (int)sz_type;
          r->padsize = padsize;
          r->bits = bits;
          r->packed = sz_packed;
          r->unpacked = sz_unpacked;
          r->crc = (unsigned int)sz_crc;
          r->crc_len = sz_crclen;
          r->tier2 = tier2;
          memcpy(r->iv, ivbuf, 16);
          /* head[] is only genuinely the START of the stream when the data
           * field holds the whole stream. On a tail-only record (what
           * tools/7z2mdx.py emits) the first 32 bytes ARE the last two
           * blocks, and running the tier-1 check on them rejects correct
           * candidates. So tier 1 is gated on the same predicate as tier 2. */
          if (tier2 && flen[9] >= 64) {
            if (get32(f[9], r->head, 32) == 32) r->havehead = 1;
          } else if (tier2 && flen[9] == 32) {
            /* whole stream is one block: that block IS the head */
            if (get32(f[9], r->head, 16) == 16) r->havehead = 1;
          }
          if (nf == 12 && flen[11] > 0 && (flen[11] & 1) == 0 && flen[11] <= 16) {
            r->attrs = malloc(flen[11] / 2);
            if (r->attrs && get32(f[11], r->attrs, flen[11] / 2) == flen[11] / 2)
              r->attrlen = flen[11] / 2;
            else { free(r->attrs); r->attrs = NULL; }
          }
          if (tier2) {
            r->data = malloc(dbytes);
            if (r->data && get32(f[9], r->data, dbytes) == dbytes)
              r->datalen = dbytes;
            else { free(r->data); r->data = NULL; r->tier2 = tier2 = 0; }
          }
          r->line = strdup(line);
          if (!r->line) { fprintf(stderr, "Failed to store 7zip hash line\n"); exit(1); }

          if (!tier2 && bits < minbits) {
            fprintf(stderr,
                    "7ZIP: padsize %d gives only %d bits of evidence; expect "
                    "~1 false hit per 2^%d candidates. Supply the complete "
                    "stream (do not run tools/7z2mdx.py on a line that already "
                    "fits) to enable decompress+CRC verification.\n",
                    padsize, bits, bits);
            SevenZipWarned++;
            SevenZipNoisy++;
          }

          JSLI(PV, JUDYJ(JOB_SEVENZIP), (unsigned char *)line);
          JSLI(PV, TYPESALT(JOB_SEVENZIP), (unsigned char *)grp);
          if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_SEVENZIP]++; }
          Foundcnt[JOB_SEVENZIP]++;
          continue;
        }
      }
    }
    /* SHA1CRYPT (15100): $sha1$ITERATIONS$SALT$DIGEST(28 chars)
     * JudyJ key = full format string; Typesalt = setting ($sha1$iter$salt$).
     * The salt is raw ASCII text, never base64-decoded. Iterations are
     * parsed, never assumed -- NetBSD randomises them by design (the CMIYC
     * corpus happens to be uniformly 5000, hashcat's example is 20000). */
    if (lf[JOB_SHA1CRYPT] && strncmp(line, "$sha1$", 6) == 0) {
      char *iter_start = line + 6;
      char *salt_start = strchr(iter_start, '$');
      if (salt_start && salt_start > iter_start) {
        int itok = 1;
        char *q;
        for (q = iter_start; q < salt_start; q++)
          if (!isdigit((unsigned char)*q)) { itok = 0; break; }
        if (itok) {
          salt_start++;
          char *dig_start = strchr(salt_start, '$');
          if (dig_start && (dig_start - salt_start) >= 1 &&
              (dig_start - salt_start) <= 64 &&
              (len - (int)(dig_start + 1 - line)) == 28) {
            char judykey[256], saltkey[128];
            unsigned char sc_raw[24];
            char sc_canon[32];
            int setting_len = dig_start - line + 1; /* include trailing $ */
            /* Liberal in what we accept, conformant in what we emit.
             *
             * The 28-char field encodes 21 bytes, but SHA-1 only produces 20:
             * the 21st is digest[0] wrapped around, so it carries no
             * information. Some producers put something else there --
             * hashcat's own published -m 15100 example uses 0x00 -- and such a
             * hash is perfectly crackable, it is just spelled differently.
             *
             * So decode the field, discard whatever the 21st byte was, and key
             * the Judy on the RE-ENCODED canonical form. A nonconformant input
             * therefore matches and is reported in the NetBSD/corpus spelling.
             * Note this also validates the alphabet, which the shape test above
             * does not. Two lines differing only in that byte collapse to one
             * unique hash, which is correct -- they are the same hash. */
            if (setting_len < (int)sizeof(saltkey) && len < (int)sizeof(judykey) &&
                sha1crypt_b64decode(dig_start + 1, sc_raw) == 0) {
              sha1crypt_b64encode(sc_raw, sc_canon);  /* re-wraps from byte 0 */
              snprintf(judykey, sizeof(judykey), "%.*s%s",
                       setting_len, line, sc_canon);
              snprintf(saltkey, sizeof(saltkey), "%.*s", setting_len, line);
              JSLI(PV, JUDYJ(JOB_SHA1CRYPT), (unsigned char *)judykey);
              JSLI(PV, TYPESALT(JOB_SHA1CRYPT), (unsigned char *)saltkey);
              if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_SHA1CRYPT]++; }
              Foundcnt[JOB_SHA1CRYPT]++;
              continue;
            }
          }
        }
      }
    }
    /* MURMUR3 (27800): 8hex:8hex (hash:seed) */
    if (lf[JOB_MURMUR3] && len == 17 && line[8] == ':' && inhashbuf) {
      int m3ok = 1;
      for (x = 0; x < 17; x++) { if (x == 8) continue; if (!isxdigit((unsigned char)line[x])) { m3ok = 0; break; } }
      if (m3ok) {
        char judykey[9], seedkey[9];
        for (x = 0; x < 8; x++) judykey[x] = tolower(line[x]);
        judykey[8] = 0;
        for (x = 0; x < 8; x++) seedkey[x] = tolower(line[x + 9]);
        seedkey[8] = 0;
        JSLI(PV, JUDYJ(JOB_MURMUR3), (unsigned char *)judykey);
        Saltloaded[JOB_MURMUR3] += store_typesalt(JOB_MURMUR3, seedkey, 8);
        Foundcnt[JOB_MURMUR3]++;
        continue;
      }
    }
    /* VEEAM-VBK (31200): $vbk$*128hex_salt*iter*32hex_ct */
    if (lf[JOB_VEEAM_VBK] && len > 20 && strncmp(line, "$vbk$*", 6) == 0 && inhashbuf) {
      memcpy(salttmp, line, len); salttmp[len] = 0;
      char *p = salttmp + 6;
      char *d1 = strchr(p, '*');
      if (d1 && (d1 - p) == 128) {
        int vok = 1; for (x = 0; x < 128; x++) if (!isxdigit((unsigned char)p[x])) { vok = 0; break; }
        if (vok) {
        char *iterp = d1 + 1;
        char *d2 = strchr(iterp, '*');
        if (d2) {
          int iter = atoi(iterp);
          char *ct = d2 + 1;
          int ctlen = strlen(ct);
          vok = 1; for (x = 0; x < ctlen; x++) if (!isxdigit((unsigned char)ct[x])) { vok = 0; break; }
          if (ctlen == 32 && vok && iter > 0) {
            /* JudyJ key = full $vbk$... line */
            char judykey[512];
            snprintf(judykey, sizeof(judykey), "%.*s", len, line);
            for (x = 0; judykey[x]; x++) judykey[x] = tolower(judykey[x]);
            JSLI(PV, JUDYJ(JOB_VEEAM_VBK), (unsigned char *)judykey);
            /* Typesalt = "iter:128hex_salt:32hex_ct" */
            char tskey[256];
            *d1 = 0;
            snprintf(tskey, sizeof(tskey), "%d:%s:%s", iter, p, ct);
            for (x = 0; tskey[x]; x++) tskey[x] = tolower(tskey[x]);
            JSLI(PV, TYPESALT(JOB_VEEAM_VBK), (unsigned char *)tskey);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_VEEAM_VBK]++; }
            Foundcnt[JOB_VEEAM_VBK]++;
            continue;
          }
        }
        }
      }
    }
    /* MSSNTP (31300): $sntp-ms$32hex_hash$96hex_salt */
    if (lf[JOB_MSSNTP] && len > 50 && strncmp(line, "$sntp-ms$", 9) == 0 && inhashbuf) {
      char *hash = (char *)line + 9;
      char *d1 = strchr(hash, '$');
      if (d1 && (d1 - hash) == 32) {
        int sok = 1; for (x = 0; x < 32; x++) if (!isxdigit((unsigned char)hash[x])) { sok = 0; break; }
        if (sok) {
        char *salt = d1 + 1;
        int slen = len - (salt - line);
        sok = 1; for (x = 0; x < slen; x++) if (!isxdigit((unsigned char)salt[x])) { sok = 0; break; }
        if (slen == 96 && sok) {
          /* JudyJ key = lowercase 32-hex hash */
          char judykey[33];
          for (x = 0; x < 32; x++) judykey[x] = tolower(hash[x]);
          judykey[32] = 0;
          JSLI(PV, JUDYJ(JOB_MSSNTP), (unsigned char *)judykey);
          /* Typesalt = 96-hex salt */
          char tskey[97];
          for (x = 0; x < 96; x++) tskey[x] = tolower(salt[x]);
          tskey[96] = 0;
          JSLI(PV, TYPESALT(JOB_MSSNTP), (unsigned char *)tskey);
          if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_MSSNTP]++; }
          Foundcnt[JOB_MSSNTP]++;
          continue;
        }
        }
      }
    }
    /* Domino 8 (9100): (H + 48 lotus64 + ) = 51 chars
     * Decodes to 36 bytes: salt16 + iter10 + chars2 + digest8 */
    if (lf[JOB_DOMINO8] && len == 51 && line[0] == '(' && line[1] == 'H' && line[50] == ')') {
      unsigned char d8_dec[40];
      int d8_ret = lotus64_decode(line + 2, d8_dec, 48);
      if (d8_ret >= 36) {
        d8_dec[3] -= 4; /* undo encoding quirk */
        /* validate iteration field (bytes 16-25 should be decimal digits) */
        int d8_ok = 1;
        for (y = 16; y < 26; y++)
          if (d8_dec[y] < '0' || d8_dec[y] > '9') { d8_ok = 0; break; }
        if (d8_ok) {
          /* JudyJ key = 16-hex digest (bytes 28-35) */
          char d8_jkey[17];
          prmd5_16(d8_dec + 28, d8_jkey);
          d8_jkey[16] = 0;
          JSLI(PV, JUDYJ(JOB_DOMINO8), (unsigned char *)d8_jkey);
          /* Typesalt = 32-hex salt16 + ":" + iter_str + ":" + 4-hex chars */
          char d8_salt[64];
          prmd5_32(d8_dec, d8_salt);
          d8_salt[32] = ':';
          memcpy(d8_salt + 33, d8_dec + 16, 10);
          d8_salt[43] = ':';
          prmd5(d8_dec + 26, d8_salt + 44, 4);
          d8_salt[48] = 0;
          Saltloaded[JOB_DOMINO8] += store_typesalt(JOB_DOMINO8, d8_salt, 48);
          Foundcnt[JOB_DOMINO8]++;
          continue;
        }
      }
    }
    /* SipHash (10100): 16hex:2:4:32hex = 53 chars
     * hash = 16-hex (LE bytes of u64), key = 32-hex (16 bytes)
     * JudyJ key = 16-hex hash, Typesalt = 32-hex key */
    if (lf[JOB_SIPHASH] && len == 53 && line[16] == ':' && line[18] == ':' &&
        line[20] == ':' && line[17] == '2' && line[19] == '4') {
      int sh_ok = 1;
      for (y = 0; y < 16; y++)
        if (trhex[(unsigned char)line[y]] > 15) { sh_ok = 0; break; }
      if (sh_ok) {
        for (y = 21; y < 53; y++)
          if (trhex[(unsigned char)line[y]] > 15) { sh_ok = 0; break; }
      }
      if (sh_ok) {
        char sh_jkey[17];
        memcpy(sh_jkey, line, 16);
        sh_jkey[16] = 0;
        /* lowercase the hash key for consistent lookup */
        for (y = 0; y < 16; y++)
          if (sh_jkey[y] >= 'A' && sh_jkey[y] <= 'F') sh_jkey[y] += 32;
        JSLI(PV, JUDYJ(JOB_SIPHASH), (unsigned char *)sh_jkey);
        char sh_salt[33];
        memcpy(sh_salt, line + 21, 32);
        sh_salt[32] = 0;
        Saltloaded[JOB_SIPHASH] += store_typesalt(JOB_SIPHASH, sh_salt, 32);
        Foundcnt[JOB_SIPHASH]++;
        continue;
      }
    }
    /* CRAM-MD5 (10200): $cram_md5$b64(challenge)$b64("user hmac_hex")
     * JudyJ key = 32-hex HMAC, Typesalt = b64challenge$b64response */
    if (lf[JOB_CRAMMD5] && len > 15 && memcmp(line, "$cram_md5$", 10) == 0) {
      char *sep = strchr(line + 10, '$');
      if (sep && sep[1]) {
        /* Decode response to extract 32-hex HMAC */
        unsigned char cm_resp[256];
        int incount;
        int cm_rlen = b64_decode(sep + 1, (char *)cm_resp, &incount);
        if (cm_rlen >= 34) {
          cm_resp[cm_rlen] = 0;
          /* Last 32 chars should be hex HMAC, preceded by space */
          char *cm_hex = (char *)cm_resp + cm_rlen - 32;
          if (cm_hex > (char *)cm_resp && cm_hex[-1] == ' ') {
            int cm_ok = 1;
            for (y = 0; y < 32; y++)
              if (trhex[(unsigned char)cm_hex[y]] > 15) { cm_ok = 0; break; }
            if (cm_ok) {
              char cm_jkey[33];
              memcpy(cm_jkey, cm_hex, 32);
              cm_jkey[32] = 0;
              for (y = 0; y < 32; y++)
                if (cm_jkey[y] >= 'A' && cm_jkey[y] <= 'F') cm_jkey[y] += 32;
              JSLI(PV, JUDYJ(JOB_CRAMMD5), (unsigned char *)cm_jkey);
              /* Typesalt = everything after $cram_md5$ */
              int cm_tslen = len - 10;
              Saltloaded[JOB_CRAMMD5] += store_typesalt(JOB_CRAMMD5, line + 10, cm_tslen);
              Foundcnt[JOB_CRAMMD5]++;
              continue;
            }
          }
        }
      }
    }
    /* SAP CODVN H (10300): {x-issha, ITER}BASE64(hash20 + salt)
     * JudyJ key = 40-hex SHA1, Typesalt = iter:hex_salt */
    if (lf[JOB_SAPCODVNH] && len > 20 && memcmp(line, "{x-issha, ", 10) == 0) {
      char *brace = strchr(line + 10, '}');
      if (brace) {
        int sh_iter = atoi(line + 10);
        if (sh_iter >= 1) {
          unsigned char sh_dec[256];
          int incount;
          int sh_dlen = b64_decode(brace + 1, (char *)sh_dec, &incount);
          if (sh_dlen >= 21) { /* 20 hash + at least 1 salt byte */
            char sh_jkey[41];
            prmd5_40(sh_dec, sh_jkey);
            sh_jkey[40] = 0;
            JSLI(PV, JUDYJ(JOB_SAPCODVNH), (unsigned char *)sh_jkey);
            /* Typesalt = "iter:hex_salt" */
            int sh_saltlen = sh_dlen - 20;
            char sh_ts[600];
            int sh_tsoff = snprintf(sh_ts, 20, "%d:", sh_iter);
            for (y = 0; y < sh_saltlen && sh_tsoff + 2 < (int)sizeof(sh_ts); y++) {
              sh_ts[sh_tsoff++] = hextab[sh_dec[20+y] >> 4];
              sh_ts[sh_tsoff++] = hextab[sh_dec[20+y] & 0xf];
            }
            sh_ts[sh_tsoff] = 0;
            Saltloaded[JOB_SAPCODVNH] += store_typesalt(JOB_SAPCODVNH, sh_ts, sh_tsoff);
            Foundcnt[JOB_SAPCODVNH]++;
            continue;
          }
        }
      }
    }
    /* RedHat 389-DS (10901): {PBKDF2_SHA256}BASE64(iter4_BE + salt64 + hash256) = 447 chars
     * JudyJ key = 64-hex (first 32 bytes of hash), Typesalt = iter:hex_salt128 */
    if (lf[JOB_REDHAT389DS] && len == 447 && memcmp(line, "{PBKDF2_SHA256}", 15) == 0) {
      unsigned char rh_dec[400];
      int incount;
      int rh_dlen = b64_decode(line + 15, (char *)rh_dec, &incount);
      if (rh_dlen == 324) {
        int rh_iter = ((int)rh_dec[0] << 24) | ((int)rh_dec[1] << 16) |
                      ((int)rh_dec[2] << 8) | (int)rh_dec[3];
        if (rh_iter >= 1 && rh_iter <= 1000000) {
          /* JudyJ key = first 32 bytes of hash (bytes 68-99) as 64 hex */
          char rh_jkey[65];
          prmd5_64(rh_dec + 68, rh_jkey);
          rh_jkey[64] = 0;
          JSLI(PV, JUDYJ(JOB_REDHAT389DS), (unsigned char *)rh_jkey);
          /* Typesalt = "iter:hex_salt128" */
          char rh_ts[160];
          int rh_off = snprintf(rh_ts, 20, "%d:", rh_iter);
          for (y = 0; y < 64; y++) {
            rh_ts[rh_off++] = hextab[rh_dec[4+y] >> 4];
            rh_ts[rh_off++] = hextab[rh_dec[4+y] & 0xf];
          }
          rh_ts[rh_off] = 0;
          Saltloaded[JOB_REDHAT389DS] += store_typesalt(JOB_REDHAT389DS, rh_ts, rh_off);
          Foundcnt[JOB_REDHAT389DS]++;
          continue;
        }
      }
    }
    /* PostgreSQL CRAM (11100): $postgres$USER*CHALLENGE*HASH
     * JudyJ key = 32-hex hash, Typesalt = user:challenge_hex */
    if (lf[JOB_POSTGRESCRAM] && len > 20 && memcmp(line, "$postgres$", 10) == 0) {
      char *star1 = strchr(line + 10, '*');
      if (star1) {
        char *star2 = strchr(star1 + 1, '*');
        if (star2 && star2 - star1 - 1 == 8 && mystrlen(star2 + 1) == 32) {
          /* Validate challenge is hex */
          int pg_ok = 1;
          for (y = 0; y < 8; y++)
            if (trhex[(unsigned char)star1[1+y]] > 15) { pg_ok = 0; break; }
          if (pg_ok) {
            /* Validate hash is hex */
            for (y = 0; y < 32; y++)
              if (trhex[(unsigned char)star2[1+y]] > 15) { pg_ok = 0; break; }
          }
          if (pg_ok) {
            char pg_jkey[33];
            memcpy(pg_jkey, star2 + 1, 32);
            pg_jkey[32] = 0;
            for (y = 0; y < 32; y++)
              if (pg_jkey[y] >= 'A' && pg_jkey[y] <= 'F') pg_jkey[y] += 32;
            JSLI(PV, JUDYJ(JOB_POSTGRESCRAM), (unsigned char *)pg_jkey);
            /* Typesalt = user:challenge_hex */
            int pg_ulen = star1 - (line + 10);
            char pg_ts[256];
            memcpy(pg_ts, line + 10, pg_ulen);
            pg_ts[pg_ulen] = ':';
            memcpy(pg_ts + pg_ulen + 1, star1 + 1, 8);
            pg_ts[pg_ulen + 9] = 0;
            /* lowercase the challenge hex */
            for (y = pg_ulen + 1; y < pg_ulen + 9; y++)
              if (pg_ts[y] >= 'A' && pg_ts[y] <= 'F') pg_ts[y] += 32;
            Saltloaded[JOB_POSTGRESCRAM] += store_typesalt(JOB_POSTGRESCRAM, pg_ts, pg_ulen + 9);
            Foundcnt[JOB_POSTGRESCRAM]++;
            continue;
          }
        }
      }
    }
    /* MySQL CRAM (11200): $mysqlna$NONCE_HEX*HASH_HEX
     * JudyJ key = 40-hex hash, Typesalt = 40-hex nonce */
    if (lf[JOB_MYSQLCRAM] && len == 90 && memcmp(line, "$mysqlna$", 9) == 0 && line[49] == '*') {
      int mc_ok = 1;
      for (y = 9; y < 49; y++)
        if (trhex[(unsigned char)line[y]] > 15) { mc_ok = 0; break; }
      if (mc_ok) {
        for (y = 50; y < 90; y++)
          if (trhex[(unsigned char)line[y]] > 15) { mc_ok = 0; break; }
      }
      if (mc_ok) {
        char mc_jkey[41];
        memcpy(mc_jkey, line + 50, 40);
        mc_jkey[40] = 0;
        for (y = 0; y < 40; y++)
          if (mc_jkey[y] >= 'A' && mc_jkey[y] <= 'F') mc_jkey[y] += 32;
        JSLI(PV, JUDYJ(JOB_MYSQLCRAM), (unsigned char *)mc_jkey);
        /* Typesalt = 40-hex nonce (lowercase) */
        char mc_ts[41];
        memcpy(mc_ts, line + 9, 40);
        mc_ts[40] = 0;
        for (y = 0; y < 40; y++)
          if (mc_ts[y] >= 'A' && mc_ts[y] <= 'F') mc_ts[y] += 32;
        Saltloaded[JOB_MYSQLCRAM] += store_typesalt(JOB_MYSQLCRAM, mc_ts, 40);
        Foundcnt[JOB_MYSQLCRAM]++;
        continue;
      }
    }
    /* Apache Shiro 1 (12150): $shiro1$SHA-512$ITER$B64SALT$B64HASH
     * JudyJ key = 64-hex (first 32 bytes of SHA-512), Typesalt = iter:hex_salt32 */
    if (lf[JOB_SHIRO1] && len > 30 && memcmp(line, "$shiro1$SHA-512$", 16) == 0) {
      char *iter_start = line + 16;
      char *dollar1 = strchr(iter_start, '$');
      if (dollar1 && dollar1 > iter_start) {
        int sh_iter = atoi(iter_start);
        if (sh_iter > 0) {
          char *b64salt = dollar1 + 1;
          char *dollar2 = strchr(b64salt, '$');
          if (dollar2) {
            /* Decode salt */
            unsigned char sh_salt[32];
            int incount;
            int sh_slen = b64_decode(b64salt, (char *)sh_salt, &incount);
            if (sh_slen == 16) {
              /* Decode hash */
              unsigned char sh_hash[80];
              int sh_hlen = b64_decode(dollar2 + 1, (char *)sh_hash, &incount);
              if (sh_hlen == 64) {
                /* JudyJ key = first 32 bytes as 64 hex */
                char sh_jkey[65];
                prmd5_64(sh_hash, sh_jkey);
                sh_jkey[64] = 0;
                JSLI(PV, JUDYJ(JOB_SHIRO1), (unsigned char *)sh_jkey);
                /* Typesalt = iter:hex_salt32 */
                char sh_ts[80];
                int sh_off = snprintf(sh_ts, 20, "%d:", sh_iter);
                for (y = 0; y < 16; y++) {
                  sh_ts[sh_off++] = hextab[sh_salt[y] >> 4];
                  sh_ts[sh_off++] = hextab[sh_salt[y] & 0xf];
                }
                sh_ts[sh_off] = 0;
                Saltloaded[JOB_SHIRO1] += store_typesalt(JOB_SHIRO1, sh_ts, sh_off);
                Foundcnt[JOB_SHIRO1]++;
                continue;
              }
            }
          }
        }
      }
    }
    /* eCryptfs (12200): $ecryptfs$0$1$HEX_SALT$HEX_HASH or $ecryptfs$0$HEX_HASH
     * JudyJ key = 16-hex hash, Typesalt = 16-hex salt */
    if (lf[JOB_ECRYPTFS] && len > 15 && memcmp(line, "$ecryptfs$0$", 12) == 0) {
      char *rest = line + 12;
      char ec_salt_hex[17], ec_hash_hex[17];
      int ec_ok = 0;
      if (rest[0] == '1' && rest[1] == '$') {
        /* $ecryptfs$0$1$SALT$HASH */
        char *sep = strchr(rest + 2, '$');
        if (sep && sep - (rest + 2) == 16 && mystrlen(sep + 1) == 16) {
          memcpy(ec_salt_hex, rest + 2, 16); ec_salt_hex[16] = 0;
          memcpy(ec_hash_hex, sep + 1, 16); ec_hash_hex[16] = 0;
          ec_ok = 1;
        }
      } else if (mystrlen(rest) == 16) {
        /* $ecryptfs$0$HASH — default salt 0011223344556677 */
        memcpy(ec_hash_hex, rest, 16); ec_hash_hex[16] = 0;
        memcpy(ec_salt_hex, "0011223344556677", 17);
        ec_ok = 1;
      }
      if (ec_ok) {
        /* Validate hex */
        int ec_valid = 1;
        for (y = 0; y < 16; y++)
          if (trhex[(unsigned char)ec_salt_hex[y]] > 15) { ec_valid = 0; break; }
        if (ec_valid)
          for (y = 0; y < 16; y++)
            if (trhex[(unsigned char)ec_hash_hex[y]] > 15) { ec_valid = 0; break; }
        if (ec_valid) {
          /* Lowercase hash for JudyJ key */
          for (y = 0; y < 16; y++)
            if (ec_hash_hex[y] >= 'A' && ec_hash_hex[y] <= 'F') ec_hash_hex[y] += 32;
          JSLI(PV, JUDYJ(JOB_ECRYPTFS), (unsigned char *)ec_hash_hex);
          /* Lowercase salt for Typesalt */
          for (y = 0; y < 16; y++)
            if (ec_salt_hex[y] >= 'A' && ec_salt_hex[y] <= 'F') ec_salt_hex[y] += 32;
          Saltloaded[JOB_ECRYPTFS] += store_typesalt(JOB_ECRYPTFS, ec_salt_hex, 16);
          Foundcnt[JOB_ECRYPTFS]++;
          continue;
        }
      }
    }
    /* Oracle 12+ (12300): 160 hex chars = 128 hash + 32 salt (uppercase)
     * JudyJ key = first 64 hex (lowercase), Typesalt = 32-hex salt (lowercase) */
    if (lf[JOB_ORACLE12]) hexform_cut(line, &len, 160, 0);
    if (lf[JOB_ORACLE12] && len == 160) {
      int or_ok = 1;
      for (y = 0; y < 160; y++)
        if (trhex[(unsigned char)line[y]] > 15) { or_ok = 0; break; }
      if (or_ok) {
        char or_jkey[65];
        memcpy(or_jkey, line, 64);
        or_jkey[64] = 0;
        for (y = 0; y < 64; y++)
          if (or_jkey[y] >= 'A' && or_jkey[y] <= 'F') or_jkey[y] += 32;
        JSLI(PV, JUDYJ(JOB_ORACLE12), (unsigned char *)or_jkey);
        char or_ts[33];
        memcpy(or_ts, line + 128, 32);
        or_ts[32] = 0;
        for (y = 0; y < 32; y++)
          if (or_ts[y] >= 'A' && or_ts[y] <= 'F') or_ts[y] += 32;
        Saltloaded[JOB_ORACLE12] += store_typesalt(JOB_ORACLE12, or_ts, 32);
        Foundcnt[JOB_ORACLE12]++;
        continue;
      }
    }
    /* ColdFusion 10+ (12600): SHA256(salt_text + UC(SHA1(pass)))
     * Format: 64hex_hash:salt_text
     * JudyJ = 64-hex hash (lc), Typesalt = salt text as-is */
    if (lf[JOB_COLDFUSION10] && len > 65 && line[64] == ':') {
      int cf_ok = 1;
      for (y = 0; y < 64; y++)
        if (trhex[(unsigned char)line[y]] > 15) { cf_ok = 0; break; }
      if (cf_ok) {
        char cf_jkey[65];
        memcpy(cf_jkey, line, 64); cf_jkey[64] = 0;
        for (y = 0; y < 64; y++)
          if (cf_jkey[y] >= 'A' && cf_jkey[y] <= 'F') cf_jkey[y] += 32;
        JSLI(PV, JUDYJ(JOB_COLDFUSION10), (unsigned char *)cf_jkey);
        int cf_slen = len - 65;
        Saltloaded[JOB_COLDFUSION10] += store_typesalt(JOB_COLDFUSION10, line + 65, cf_slen);
        Foundcnt[JOB_COLDFUSION10]++;
        continue;
      }
    }
    /* MS-AzureSync (12800): PBKDF2-SHA256(UTF16LE(UC(NTLM(pass))), salt, iter, 32)
     * Format: v1;PPH1_MD4,hex_salt,iter,hex_hash
     * JudyJ = 64-hex hash (lc), Typesalt = iter:hex_salt */
    if (lf[JOB_AZURESYNC] && len > 30 && memcmp(line, "v1;PPH1_MD4,", 12) == 0) {
      char *f1 = line + 12; /* salt_hex */
      char *c1 = strchr(f1, ',');
      if (c1) {
        char *f2 = c1 + 1; /* iter */
        char *c2 = strchr(f2, ',');
        if (c2) {
          char *f3 = c2 + 1; /* hash */
          int az_slen = c1 - f1;
          int az_iter = atoi(f2);
          int az_hlen = mystrlen(f3);
          if (az_slen == 20 && az_hlen == 64 && az_iter > 0) {
            int az_ok = 1;
            for (y = 0; y < 20; y++)
              if (trhex[(unsigned char)f1[y]] > 15) { az_ok = 0; break; }
            if (az_ok)
              for (y = 0; y < 64; y++)
                if (trhex[(unsigned char)f3[y]] > 15) { az_ok = 0; break; }
            if (az_ok) {
              char az_jkey[65];
              memcpy(az_jkey, f3, 64); az_jkey[64] = 0;
              for (y = 0; y < 64; y++)
                if (az_jkey[y] >= 'A' && az_jkey[y] <= 'F') az_jkey[y] += 32;
              JSLI(PV, JUDYJ(JOB_AZURESYNC), (unsigned char *)az_jkey);
              char az_ts[40];
              int az_off = snprintf(az_ts, 20, "%d:", az_iter);
              memcpy(az_ts + az_off, f1, 20);
              for (y = az_off; y < az_off + 20; y++)
                if (az_ts[y] >= 'A' && az_ts[y] <= 'F') az_ts[y] += 32;
              az_ts[az_off + 20] = 0;
              Saltloaded[JOB_AZURESYNC] += store_typesalt(JOB_AZURESYNC, az_ts, az_off + 20);
              Foundcnt[JOB_AZURESYNC]++;
              continue;
            }
          }
        }
      }
    }
    /* Android FDE Samsung DEK (12900): HMAC-SHA256(PBKDF2-SHA256(pass,salt,4096,32), salt2)
     * Format: 160 hex = salt2(64) + hmac(64) + salt(32)
     * JudyJ = 64-hex hmac (lc), Typesalt = salt2_hex:salt_hex */
    if (lf[JOB_ANDROIDFDE]) hexform_cut(line, &len, 160, 0);
    if (lf[JOB_ANDROIDFDE] && len == 160) {
      int af_ok = 1;
      for (y = 0; y < 160; y++)
        if (trhex[(unsigned char)line[y]] > 15) { af_ok = 0; break; }
      if (af_ok) {
        char af_jkey[65];
        memcpy(af_jkey, line + 64, 64); af_jkey[64] = 0;
        for (y = 0; y < 64; y++)
          if (af_jkey[y] >= 'A' && af_jkey[y] <= 'F') af_jkey[y] += 32;
        JSLI(PV, JUDYJ(JOB_ANDROIDFDE), (unsigned char *)af_jkey);
        /* Typesalt = salt2_hex(64):salt_hex(32) */
        char af_ts[98];
        memcpy(af_ts, line, 64); af_ts[64] = ':'; memcpy(af_ts + 65, line + 128, 32); af_ts[97] = 0;
        for (y = 0; y < 97; y++)
          if (af_ts[y] >= 'A' && af_ts[y] <= 'F') af_ts[y] += 32;
        Saltloaded[JOB_ANDROIDFDE] += store_typesalt(JOB_ANDROIDFDE, af_ts, 97);
        Foundcnt[JOB_ANDROIDFDE]++;
        continue;
      }
    }
    /* Kerberos 5 TGS-REP etype 23 (13100): NTLM -> HMAC-MD5(usage=2) -> RC4 -> ASN.1 verify
     * Format: $krb5tgs$23$*user$realm$spn*$checksum$edata2
     * JudyJ = full line, Typesalt = full line */
    if (lf[JOB_KRB5TGS23] && len > 30 && memcmp(line, "$krb5tgs$23$", 12) == 0) {
      /* Find *user$realm$spn*$checksum$edata2 — skip to closing * */
      char *star1 = strchr(line + 12, '*');
      if (star1) {
        char *star2 = strchr(star1 + 1, '*');
        if (star2) {
          char *dcheck = star2 + 1;
          if (*dcheck == '$') {
            dcheck++;
            int ck_ok = 1;
            if (mystrlen(dcheck) < 34) ck_ok = 0; /* at least 32hex checksum + $ + some edata */
            if (ck_ok) {
              for (y = 0; y < 32; y++)
                if (trhex[(unsigned char)dcheck[y]] > 15) { ck_ok = 0; break; }
            }
            if (ck_ok && dcheck[32] == '$' && mystrlen(dcheck + 33) >= 64) {
              JSLI(PV, JUDYJ(JOB_KRB5TGS23), (unsigned char *)line);
              Saltloaded[JOB_KRB5TGS23] += store_typesalt(JOB_KRB5TGS23, line, len);
              Foundcnt[JOB_KRB5TGS23]++;
              continue;
            }
          }
        }
      }
    }
    /* DCC2 / MS Cache 2 (2100): PBKDF2-HMAC-SHA1(MD4(MD4(UTF16(pass))+UTF16(lc(user))), lc(user), iter, 16)
     * Format: $DCC2$iter#username#hash32hex
     * JudyJ = hash32hex, Typesalt = iter:username */
    if (lf[JOB_DCC2] && len > 15 && memcmp(line, "$DCC2$", 6) == 0) {
      char *p = line + 6;
      char *hash_p = strchr(p, '#');
      if (hash_p) {
        char *hash_hex = strchr(hash_p + 1, '#');
        if (hash_hex && mystrlen(hash_hex + 1) == 32) {
          int dcc_ok = 1;
          for (y = 0; y < 32; y++)
            if (trhex[(unsigned char)hash_hex[1+y]] > 15) { dcc_ok = 0; break; }
          if (dcc_ok) {
            /* JudyJ key = lowercase hash */
            char dcc_tmp[256];
            for (y = 0; y < 32; y++)
              dcc_tmp[y] = tolower(hash_hex[1+y]);
            dcc_tmp[32] = 0;
            JSLI(PV, JUDYJ(JOB_DCC2), (unsigned char *)dcc_tmp);
            /* Typesalt = iter:username */
            int iter_val = atoi(p);
            if (iter_val < 1) iter_val = 10240;
            int ulen = hash_hex - hash_p - 1;
            int toff = snprintf(dcc_tmp, 250, "%d:", iter_val);
            /* lowercase username into salt */
            for (y = 0; y < ulen && toff + y < 250; y++)
              dcc_tmp[toff + y] = tolower(hash_p[1+y]);
            dcc_tmp[toff + ulen] = 0;
            Saltloaded[JOB_DCC2] += store_typesalt(JOB_DCC2, dcc_tmp, toff + ulen);
            Foundcnt[JOB_DCC2]++;
            continue;
          }
        }
      }
    }
    /* Password Safe v3 (5200): SHA256(pass+salt) iterated
     * Input: hex-encoded binary: PWS3(4)+salt(32)+iter_LE(4)+hash(32)+rest
     * JudyJ = hash64hex, Typesalt = iter:salt64hex */
    if (lf[JOB_PWSAFE3] && len >= 144 && memcmp(line, "50575333", 8) == 0) {
      /* Verify all hex */
      int pw_ok = 1;
      for (y = 0; y < 144; y++)
        if (trhex[(unsigned char)line[y]] > 15) { pw_ok = 0; break; }
      if (pw_ok) {
        /* salt = bytes 4-35 = hex chars 8-71 */
        /* iter = bytes 36-39 = hex chars 72-79 (little-endian u32) */
        /* hash = bytes 40-71 = hex chars 80-143 */
        unsigned char iter_bytes[4];
        for (y = 0; y < 4; y++)
          iter_bytes[y] = (trhex[(unsigned char)line[72+y*2]] << 4) | trhex[(unsigned char)line[73+y*2]];
        unsigned int pw_iter = iter_bytes[0] | (iter_bytes[1] << 8) | (iter_bytes[2] << 16) | (iter_bytes[3] << 24);
        char pw_tmp[256];
        /* JudyJ key = hash (64 hex, lowercase) */
        for (y = 0; y < 64; y++)
          pw_tmp[y] = tolower(line[80+y]);
        pw_tmp[64] = 0;
        JSLI(PV, JUDYJ(JOB_PWSAFE3), (unsigned char *)pw_tmp);
        /* Typesalt = iter:salt64hex */
        int toff = snprintf(pw_tmp, 250, "%u:", pw_iter);
        for (y = 0; y < 64; y++)
          pw_tmp[toff+y] = tolower(line[8+y]);
        pw_tmp[toff+64] = 0;
        Saltloaded[JOB_PWSAFE3] += store_typesalt(JOB_PWSAFE3, pw_tmp, toff+64);
        Foundcnt[JOB_PWSAFE3]++;
        continue;
      }
    }
    /* IKE-PSK MD5/SHA1 (5300/5400): HMAC(HMAC(PSK, nonces), msg)
     * Format: 9 colon-separated hex fields: msg0:msg1:msg2:msg3:msg4:msg5:nr0:nr1:hash
     * JudyJ = full line, Typesalt = full line */
    if ((lf[JOB_IKEPSK_MD5] || lf[JOB_IKEPSK_SHA1]) && len > 40) {
      /* Count colons — need exactly 8 */
      int ncol = 0;
      char *colpos[9];
      colpos[0] = line;
      for (y = 0; y < len; y++) {
        if (line[y] == ':') {
          ncol++;
          if (ncol < 9) colpos[ncol] = line + y + 1;
        }
      }
      if (ncol == 8) {
        /* Last field is the hash: 32 hex for MD5, 40 hex for SHA1 */
        char *hash_start = colpos[8];
        int hlen = len - (hash_start - line);
        int ike_job = 0;
        if (hlen == 32 && lf[JOB_IKEPSK_MD5]) ike_job = JOB_IKEPSK_MD5;
        else if (hlen == 40 && lf[JOB_IKEPSK_SHA1]) ike_job = JOB_IKEPSK_SHA1;
        if (ike_job) {
          /* Verify hash is hex */
          int ike_ok = 1;
          for (y = 0; y < hlen; y++)
            if (trhex[(unsigned char)hash_start[y]] > 15) { ike_ok = 0; break; }
          if (ike_ok) {
            JSLI(PV, JUDYJ(ike_job), (unsigned char *)line);
            /* Direct JSLI for Typesalt — line too long for store_typesalt */
            { Word_t *SPV;
              JSLI(SPV, TYPESALT(ike_job), (unsigned char *)line);
              if (SPV && (*SPV)++ == 0) {
                Saltloaded[ike_job]++;
                Typesaltcnt[ike_job]++;
                Typesaltbytes[ike_job] += len + 1;
              }
            }
            Foundcnt[ike_job]++;
            continue;
          }
        }
      }
    }
    /* AxCrypt (13200): SHA1(pass) XOR salt -> AES Key Unwrap(KEK, wrapped_key, iter)
     * Format: $axcrypt$*1*iter*salt_hex*wrapped_hex
     * JudyJ = full line, Typesalt = iter:salt_hex:wrapped_hex */
    if (lf[JOB_AXCRYPT] && len > 30 && memcmp(line, "$axcrypt$*1*", 12) == 0) {
      char *iter_start = line + 12;
      char *s1 = strchr(iter_start, '*');
      if (s1) {
        int ax_iter = atoi(iter_start);
        char *salt_start = s1 + 1;
        char *s2 = strchr(salt_start, '*');
        if (s2 && s2 - salt_start == 32 && mystrlen(s2 + 1) == 48 && ax_iter > 0) {
          int ax_ok = 1;
          for (y = 0; y < 32; y++)
            if (trhex[(unsigned char)salt_start[y]] > 15) { ax_ok = 0; break; }
          if (ax_ok)
            for (y = 0; y < 48; y++)
              if (trhex[(unsigned char)s2[1+y]] > 15) { ax_ok = 0; break; }
          if (ax_ok) {
            JSLI(PV, JUDYJ(JOB_AXCRYPT), (unsigned char *)line);
            /* Typesalt = iter:salt32:wrapped48 */
            char ax_ts[100];
            int ax_off = snprintf(ax_ts, 20, "%d:", ax_iter);
            memcpy(ax_ts + ax_off, salt_start, 32);
            ax_ts[ax_off + 32] = ':';
            memcpy(ax_ts + ax_off + 33, s2 + 1, 48);
            ax_ts[ax_off + 81] = 0;
            for (y = ax_off; y < ax_off + 81; y++)
              if (ax_ts[y] >= 'A' && ax_ts[y] <= 'F') ax_ts[y] += 32;
            Saltloaded[JOB_AXCRYPT] += store_typesalt(JOB_AXCRYPT, ax_ts, ax_off + 81);
            Foundcnt[JOB_AXCRYPT]++;
            continue;
          }
        }
      }
    }
    /* AxCrypt in-memory SHA1 (13300): SHA1(pass), truncated to 32 hex
     * Format: $axcrypt_sha1$hex_hash (32 or 40 hex chars)
     * JudyJ = 32-hex (first 16 bytes of SHA1) */
    if (lf[JOB_AXCRYPTSHA1] && len >= 46 && len <= 54 && memcmp(line, "$axcrypt_sha1$", 14) == 0) {
      int ax_hlen = len - 14;
      if (ax_hlen == 32 || ax_hlen == 40) {
        int ax_ok = 1;
        for (y = 14; y < len; y++)
          if (trhex[(unsigned char)line[y]] > 15) { ax_ok = 0; break; }
        if (ax_ok) {
          char ax_jkey[33];
          memcpy(ax_jkey, line + 14, 32); ax_jkey[32] = 0;
          for (y = 0; y < 32; y++)
            if (ax_jkey[y] >= 'A' && ax_jkey[y] <= 'F') ax_jkey[y] += 32;
          JSLI(PV, JUDYJ(JOB_AXCRYPTSHA1), (unsigned char *)ax_jkey);
          Foundcnt[JOB_AXCRYPTSHA1]++;
          continue;
        }
      }
    }
    /* NetScaler PBKDF2: 5 + 64-hex-salt + 64-hex-hash (129 chars) → Judy + Typesalt */
    if (lf[JOB_NETSCALER_PBKDF2]) hexform_cut(line, &len, 129, '5');
    if (lf[JOB_NETSCALER_PBKDF2] && len == 129 && line[0] == '5') {
      int ns_ok = 1;
      for (y = 1; y < 129; y++) {
        if (trhex[(unsigned char)line[y]] > 15) { ns_ok = 0; break; }
      }
      if (ns_ok) {
        JSLI(PV, JUDYJ(JOB_NETSCALER_PBKDF2), (unsigned char *)line);
        Saltloaded[JOB_NETSCALER_PBKDF2] += store_typesalt(JOB_NETSCALER_PBKDF2, line + 1, 64);
        Foundcnt[JOB_NETSCALER_PBKDF2]++;
        continue;
      }
    }
    /* NetScaler SHA512: 2SALTHEXHASH (1 + 8salt + 128hex = 137 chars) → reformat as hexhash:salt */
    if (lf[JOB_NETSCALER_SHA512]) hexform_cut(line, &len, 137, '2');
    if (lf[JOB_NETSCALER_SHA512] && mystrlen(line) == 137 && line[0] == '2') {
      char tmp[MAXLINE + 16];
      /* hex hash is at positions 9..136, salt at positions 1..8 */
      memcpy(tmp, line + 9, 128);
      tmp[128] = ':';
      memcpy(tmp + 129, line + 1, 8);
      tmp[137] = 0;
      strcpy(line, tmp);
      /* fall through to hex loader */
    }
    /* Werkzeug MD5: md5$salt$hexhash → reformat as hexhash:salt */
    if (lf[JOB_WERKZEUG_MD5] && strncmp(line, "md5$", 4) == 0) {
      char *wsalt = line + 4;
      char *whash = strchr(wsalt, '$');
      if (whash && mystrlen(whash + 1) >= 32) {
        int slen = whash - wsalt;
        if (slen > 0 && slen < MAXLINE) {
          char tmp[MAXLINE + 16];
          snprintf(tmp, sizeof(tmp), "%.32s:%.*s", whash + 1, slen, wsalt);
          strcpy(line, tmp);
          /* fall through to hex loader */
        }
      }
    }
    /* Werkzeug SHA256: sha256$salt$hexhash → reformat as hexhash:salt */
    if (lf[JOB_WERKZEUG_SHA256] && strncmp(line, "sha256$", 7) == 0) {
      char *wsalt = line + 7;
      char *whash = strchr(wsalt, '$');
      if (whash && mystrlen(whash + 1) >= 64) {
        int slen = whash - wsalt;
        if (slen > 0 && slen < MAXLINE) {
          char tmp[MAXLINE + 16];
          snprintf(tmp, sizeof(tmp), "%.64s:%.*s", whash + 1, slen, wsalt);
          strcpy(line, tmp);
          /* fall through to hex loader */
        }
      }
    }
    /* NetWitness: UCHEXHASH:BASE64SALT → decode base64 salt, reformat as hexhash:salt */
    if (lf[JOB_NETWITNESS] && mystrlen(line) > 65 && line[64] == ':') {
      int nok = 1;
      for (x = 0; x < 64; x++) {
        if (!isxdigit((unsigned char)line[x])) { nok = 0; break; }
      }
      if (nok) {
        char tmp[MAXLINE + 16];
        char decoded[MAXLINE];
        int incount;
        int dlen = b64_decode(line + 65, decoded, &incount);
        if (dlen > 0 && dlen < MAXLINE) {
          memcpy(tmp, line, 64);
          tmp[64] = ':';
          memcpy(tmp + 65, decoded, dlen);
          tmp[65 + dlen] = 0;
          strcpy(line, tmp);
          /* fall through to hex loader */
        }
      }
    }
    /* AuthMe: $SHA$salt$hexhash → reformat as hexhash:salt */
    if (lf[JOB_AUTHME] && strncmp(line, "$SHA$", 5) == 0) {
      char *asalt = line + 5;
      char *ahash = strchr(asalt, '$');
      if (ahash && mystrlen(ahash + 1) >= 64) {
        int slen = ahash - asalt;
        if (slen > 0 && slen < MAXLINE) {
          char tmp[MAXLINE + 16];
          snprintf(tmp, sizeof(tmp), "%.64s:%.*s", ahash + 1, slen, asalt);
          strcpy(line, tmp);
          /* fall through to hex loader */
        }
      }
    }
    /* $BLAKE2$hexhash[:salt] → strip prefix, fall through to hex loader */
    if ((lf[JOB_BLAKE2B512] || lf[JOB_BLAKE2B512PASSSALT] || lf[JOB_BLAKE2B512SALTPASS] ||
         lf[JOB_BLAKE2B256] || lf[JOB_BLAKE2B256PASSSALT] || lf[JOB_BLAKE2B256SALTPASS]) &&
        strncmp(line, "$BLAKE2$", 8) == 0) {
      memmove(line, line + 8, mystrlen(line + 8) + 1);
      /* fall through to hex loader */
    }
    /* Crypto++ Legacy/Default Encryptor (e1028/e1029). The whole line is one
     * hex record: salt(8) || CBC ciphertext. Uppercase in the wild; either
     * case is read, and the key is NORMALISED TO UPPERCASE here. The procjob
     * case compares and emits that same Judy key, so the normalisation has to
     * happen in exactly one place -- a case mismatch between loader and
     * compute presents as "hash not found" with a perfectly correct compute.
     *
     * PLACED LAST, immediately above the plain-hex loader, deliberately. The
     * claim is only "all hex, byte length a multiple of 8, at least 24
     * bytes", which is broad enough to swallow shapes belonging to a more
     * specific recognizer -- Oracle 12+ is exactly 160 hex characters, 80
     * bytes, and 80 is a legal Legacy length. Every structured recognizer
     * above therefore gets first refusal.
     *
     * ONE block serves BOTH types rather than two sequential recognizers,
     * because the two length sets overlap completely: Legacy is 8+8k, Default
     * is 8+16m, and every Default length is also a Legacy length. A 40-, 56-
     * or 72-byte record is a legal record of either type, so each selected
     * type whose block size can hold it gets it and the line is consumed
     * once. With two blocks the first `continue` would starve the second.
     * There is thus no ordering hazard BETWEEN the two types; the order that
     * IS load-bearing is this block against the plain-hex loader below, and
     * this block wins.
     *
     * Claim and consume: the value is ciphertext, and letting the hex loader
     * have it would file ciphertext as a digest of whatever type shares its
     * width. But a consumed line is otherwise invisible, so when the width is
     * also a standard digest width AND a plain-hex type is selected in the
     * same run, say so once on stderr. LoadHex is the available proxy for
     * "some plain-hex type is selected" -- mdxfind keeps no per-type digest
     * width table, so the note can name the widths but not prove that the
     * specific colliding type is among them. It over-warns rather than
     * staying silent. */
    if ((lf[JOB_CRYPTOPPLEGACY] || lf[JOB_CRYPTOPPDEFAULT]) && len >= 48) {
      /* Single-threaded: load_hash_file runs on the main thread, before any
       * procjob worker exists.
       *
       * THREE input shapes are accepted, and all three are the ordinary
       * salted convention rather than anything new:
       *
       *   <record>                       record only; keys come from -s/-S
       *   <record>:<sitekey>             key inline -- the normal form
       *   <record>:<sitekey>:<plaintext> mdxfind's own output, fed back
       *
       * The site key is a SALT of cardinality one, so it goes where a salt
       * goes: TYPESALT of each claiming type. The third field is the
       * recovered PLAINTEXT and mdxfind IGNORES it; it exists so that a
       * founds file verifies through hashpipe unchanged and so that a -z
       * generated line cracks back through -F. The record is pure hex and can
       * never itself contain a colon, so the split is unambiguous.
       *
       * The length test is on the LEADING HEX RUN, not on the whole line,
       * which is what makes the colon forms reachable at all. `len >= 48`
       * above only skips lines too short to hold the 24-byte minimum record
       * in hex; the real gate is cpe_hl below. */
      static int cppenc_collide_warned = 0;
      int cpe_hl = 0, cpe_bytes, cpe_claimed = 0, cpe_kl = 0;
      char *cpe_key = NULL, *cpe_c;
      while (cpe_hl < MAXLINE && isxdigit((unsigned char)line[cpe_hl])) cpe_hl++;
      cpe_bytes = cpe_hl / 2;
      if (cpe_hl >= 48 && (cpe_hl & 1) == 0 && (cpe_bytes % 8) == 0 &&
          cpe_hl < MAXLINE && (line[cpe_hl] == 0 || line[cpe_hl] == ':')) {
        if (line[cpe_hl] == ':') {
          cpe_key = line + cpe_hl + 1;
          cpe_c = strchr(cpe_key, ':');
          cpe_kl = cpe_c ? (int)(cpe_c - cpe_key) : (int)mystrlen(cpe_key);
          /* Terminate the key in place. `line` is this function's own buffer
           * and the block always `continue`s once it has claimed, so nothing
           * downstream reads past here. */
          if (cpe_kl <= 0) { cpe_key = NULL; cpe_kl = 0; }
          else cpe_key[cpe_kl] = 0;
        }
        for (x = 0; x < cpe_hl; x++)
          salttmp[x] = toupper((unsigned char)line[x]);
        salttmp[cpe_hl] = 0;
        if (lf[JOB_CRYPTOPPLEGACY] && cppenc_reclen_ok(&CPPENC_LEGACY, cpe_bytes)) {
          JSLI(PV, JUDYJ(JOB_CRYPTOPPLEGACY), (unsigned char *)salttmp);
          Foundcnt[JOB_CRYPTOPPLEGACY]++;
          cpe_claimed = 1;
          if (cpe_key) {
            JSLI(PV, TYPESALT(JOB_CRYPTOPPLEGACY), (unsigned char *)cpe_key);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_CRYPTOPPLEGACY]++; }
          }
        }
        if (lf[JOB_CRYPTOPPDEFAULT] && cppenc_reclen_ok(&CPPENC_DEFAULT, cpe_bytes)) {
          JSLI(PV, JUDYJ(JOB_CRYPTOPPDEFAULT), (unsigned char *)salttmp);
          Foundcnt[JOB_CRYPTOPPDEFAULT]++;
          cpe_claimed = 1;
          if (cpe_key) {
            JSLI(PV, TYPESALT(JOB_CRYPTOPPDEFAULT), (unsigned char *)cpe_key);
            if (PV) { if ((*PV)++ == 0) Saltloaded[JOB_CRYPTOPPDEFAULT]++; }
          }
        }
        if (cpe_claimed) {
          if (LoadHex && !cppenc_collide_warned) {
            const char *cpe_coll = NULL;
            switch (cpe_bytes) {
              case 24: cpe_coll = "Tiger and the other 192-bit digests"; break;
              case 32: cpe_coll = "SHA-256 and the other 256-bit digests"; break;
              case 48: cpe_coll = "SHA-384"; break;
              case 64: cpe_coll = "SHA-512 and Whirlpool"; break;
            }
            if (cpe_coll) {
              cppenc_collide_warned = 1;
              fprintf(stderr,
                "NOTE: %s: a %d-byte hex value was claimed as a Crypto++ "
                "encryptor record.\n"
                "      %d bytes is also the width of %s, and a plain-hex "
                "type is selected in this run.\n"
                "      The Crypto++ recognizer consumes the line, so the hex "
                "loader never sees it and\n"
                "      those digests will NOT load, nor will a trailing "
                ":salt reach another salted type.\n"
                "      Run the digest type in a separate pass, or drop "
                "CRYPTOPPLEGACY/CRYPTOPPDEFAULT\n"
                "      from the selection.\n",
                filename, cpe_bytes, cpe_bytes, cpe_coll);
            }
          }
          continue;
        }
      }
    }
    /* Plain hex hashes: load into compact table */
    if (LoadHex && inhashbuf) {
      int hlen = get32(line, inhashbuf + 2, 256);
      /* Floor is 8 bytes, not 16, to match the fast path above and the
       * storage layer beneath. The compact-table key is a single uint64_t
       * read off the hash bytes, so 8 is the real minimum; addhash()
       * enforces exactly that and zero-pads sub-16-byte entries to 16 so
       * the GPU 4xuint32 probe cannot spill into the next entry. This gate
       * read >= 16, so any digest shorter than 32 hex chars was dropped
       * silently whenever a salted or user-bearing type forced the slow
       * path (LoadSalt/LoadUser with -F). MYSQL3 (e456, 8 bytes) was
       * unloadable in exactly that combination while still being listed
       * in "Working on hash types". */
      if (hlen >= 8 && hlen <= 255) {
        commit_compact(hlen);
        /* DCC2 (e918) compares against its OWN JudyJ, not the compact table,
         * because the $DCC2$ loader files digests there. A plain hex list whose
         * usernames arrive via -u/-U must therefore be filed there too, or the
         * digests load, report on the receipt, and are never comparable -- a
         * clean zero indistinguishable from an honest one. 16 bytes only: that
         * is the sole width DCC2 emits. Additive; the $DCC2$ path is untouched
         * and a wrapper-loaded digest simply re-inserts the same key. */
        if (hlen == 16 && lf[JOB_DCC2]) {
          char _dh[33];
          int _dy;
          for (_dy = 0; _dy < 32; _dy++)
            _dh[_dy] = tolower((unsigned char)line[_dy]);
          _dh[32] = 0;
          JSLI(PV, JUDYJ(JOB_DCC2), (unsigned char *)_dh);
        }
        /* RORAILS-SHA1 (e962) compares against its OWN JudyJ, and until now
         * the only thing that filled it was the 40hex:40hex structured loader
         * above. Real lists of this type do not carry a 40-hex salt: the salt
         * is whatever the Rails app's token generator emitted, and the one
         * that produced the bulk of the published 27200 corpus emits a
         * 20-character mixed-case token. Those lines fall past the structured
         * loader to here, where the Dosalt[] capture below files the salt
         * correctly while the digest went only to the compact table -- which
         * this type never consults. The run then printed "found N unique salts
         * for RORAILS-SHA1", listed the type as working, and ran it with N
         * salts and ZERO hashes: every digest computed, every lookup missed,
         * exit clean. Nothing distinguished that from an honest negative, and
         * the salt line read like success. 20 bytes only, the sole width SHA-1
         * emits. Additive: a 40-hex salt is still claimed by the structured
         * loader and never reaches this line, so that path is byte-identical.
         *
         * The salt is filed by Dosalt[] VERBATIM, with no case folding. That
         * matters here and is why the fix does not simply widen the structured
         * loader: that loader lowercases its salt, which is free for 40 hex
         * and destroys a mixed-case token. */
        if (hlen == 20 && lf[JOB_RORAILS_SHA1]) {
          char _rh[41];
          int _ry;
          for (_ry = 0; _ry < 40; _ry++)
            _rh[_ry] = tolower((unsigned char)line[_ry]);
          _rh[40] = 0;
          JSLI(PV, JUDYJ(JOB_RORAILS_SHA1), (unsigned char *)_rh);
        }
        /* Check for :suffix on hex hashes (salt and/or user) */
        if ((LoadSalt || LoadUser) && line[hlen * 2] == ':' && line[hlen * 2 + 1]) {
          char *suffix = line + hlen * 2 + 1;
	  for (x=0;  x < Dosaltcnt; x++) {
	     JSLI(PV, TYPESALT(Dosalt[x]),(unsigned char *)suffix);
	     if (PV) { if ((*PV)++ == 0) {Saltloaded[Dosalt[x]]++;} }
	     /* Typehashsalt populated in main() after all salts loaded */
          }
	  if (Dousercnt && user_canon(suffix, (int)strlen(suffix), Ucanon) > 0)
	  for (x=0;  x < Dousercnt; x++) {
	     JSLI(PV, TYPEUSER(Douser[x]),(unsigned char *)Ucanon);
	     if (PV) { if ((*PV)++ == 0) {Userloaded[Douser[x]]++;} }
          }
        }
        continue;
      }
    }
    /* Nothing loaded this line.  Retry it without its trailing field.
     *
     * A structured hash followed by ":<anything>" -- most often a founds
     * file holding a WRONG solve, "hash:bad-password" -- is refused by every
     * recognizer that measures its hash, DESCRYPT's len == 13 among them.
     * The general rule is that once a hash has been fully accepted, what
     * follows it on the line is ignored.  That cannot be done by cutting
     * first, because the ordinary "<hash>:<salt>" form is the same shape and
     * a salt may be any length and contain anything.  So it is done by
     * cutting LAST: the whole line has already been offered to every
     * recognizer and to the hex loader, and only a line that loaded NOTHING
     * comes back here.  Every line that loads today loads exactly as before.
     *
     * The cut moves right to left through the colons, so the longest prefix
     * any selected type accepts wins.  That is what keeps a hash whose own
     * grammar uses the colon whole: "sha256:1000:<salt>:<digest>:bad" is
     * offered as "sha256:1000:<salt>:<digest>" before anything shorter.
     *
     * "Nothing loaded" is measured, not assumed: a few recognizers load and
     * fall through rather than continue, so the per-type counters are summed
     * before and after.  The original line is restored before each try
     * because a recognizer may write into the buffer and then decline.
     *
     * Known limit: a salted hash with a wrong solve appended,
     * "<hex>:<salt>:<bad>", already loads on the first pass, with
     * "<salt>:<bad>" as its salt.  That is the ambiguity this cannot
     * resolve, and that line is unchanged. */
    if (rt_cut > 0 && rt_tries < 32 &&
        claim_mark(Foundcnt, rt_sel, rt_nsel) == rt_mark) {
      int c = rt_cut - 1;
      while (c > 0 && rt_orig[c] != ':') c--;
      if (c > 0) {
        memcpy(line, rt_orig, (size_t)c);
        line[c] = 0;
        len = c;
        rt_cut = c;
        rt_tries++;
        goto reparse;
      }
    }
  }
  /* TYPENAME(), not Types[]: these walk the whole op range and a
   * user-defined type's op is past the end of Types[]. Reading it was
   * out of bounds the moment any user type loaded hashes. */
  for (x = 0; x < JOB_DONE; x++) {
    if (Foundcnt[x])
      fprintf(stderr, "%d %s hashes read from %s\n", Foundcnt[x],
              TYPENAME(x), filename);
  }
  for (x = 0; x < JOB_DONE; x++) {
    if (Saltloaded[x])
      fprintf(stderr, "%s: found %lu unique salts for %s\n",
              filename, Saltloaded[x], TYPENAME(x));
  }
  for (x = 0; x < JOB_DONE; x++) {
    if (Userloaded[x])
      fprintf(stderr, "%s: found %lu unique users for %s\n",
              filename, Userloaded[x], TYPENAME(x));
  }

  /* A receipt for EVERY file, including the empty one. The per-type lines
   * above print only what was found, so a file that yielded nothing said
   * nothing at all, and a run that loaded no hashes looked exactly like a run
   * that loaded them and cracked none. Both halves of the trap kpd hit on
   * 2026-09-05 -- salted hashes fed to -f, and -F placed ahead of -m -- end in
   * that same silence. */
 load_done:
  { unsigned long ts = 0, tu = 0, tf = 0, th;
    int nt = 0;
    for (x = 0; x < JOB_DONE; x++) {
      if (Foundcnt[x]) { tf += (unsigned long)Foundcnt[x]; nt++; }
      ts += Saltloaded[x];
      tu += Userloaded[x];
    }
    /* A structured format is recognised per type (Foundcnt) and a plain hex
     * line is committed straight to the input set (HashesThisFile); a line
     * counted both ways is one hash, so take the larger rather than the sum. */
    th = HashesThisFile > tf ? HashesThisFile : tf;
    (void)nt;
    { Word_t rb; JSLFA(rb, RcptSalts); (void)rb; }
    fprintf(stderr, "%s: %lu hashes, %lu salts, %lu users loaded\n",
            filename, th, ts, tu);
    /* A file of bare crypt settings loads salts and no hashes, which is
     * what it is for: those salts drive -z. */
    if (th == 0 && ts == 0)
      fprintf(stderr,
        "%s: NOTHING was loaded from this file. Check that the -m/-M selection\n"
        "  names the type these hashes are, that it comes BEFORE -F/-J, and that\n"
        "  hash:salt input is read with -F rather than -f.\n", filename);
  }
}
void *Stack;

/* gpu_op_advertise_for_h_listing -- predicate-based GPU-eligibility
 * check driving the [GPU] suffix on the `mdxfind -h` hash-type listing.
 *
 * Replaces a manually-maintained if-chain that listed every JOB literal
 * (added 2026-05-27). The chain wasn't extended when codegen-path
 * algorithms became GPU-eligible, so newly-eligible e120/e122/e157/e159/
 * e161/e163/e165/e167/e169/e171/e173/e347 entries silently lost their
 * [GPU] tag in the listing despite being live on the runtime dispatch
 * path. This helper composes the three actual admit predicates so the
 * display tracks runtime reality automatically:
 *
 *   1. gpu_ops[] legacy template-path scan -- every JOB enum whose
 *      presence in Dohash[] triggers need_gpu in the GPU init preamble.
 *      Same file-scope table referenced by the init preamble (no mirror).
 *   2. JOB_MD5MD5SALT (e347) -- codegen Phase 4 production path; absent
 *      from gpu_ops[] by design (need_gpu override at ~line 39200).
 *   3. gpu_codegen_kernelb_family_md5pass_eligible -- Phase 5a/5b
 *      MAKE_MD5PASS family entries; absent from gpu_ops[] by design
 *      (need_gpu override at ~line 39210).
 *
 * Cleanup 2026-05-27 (rev 1.511): the helper's parallel `gpu_ops_listing`
 * mirror is gone; both call-sites now iterate the single file-scope
 * gpu_ops[] table above. Future Phase 5b Tier 3/4 ships (haval/gost/
 * sne128/sne256 family members) auto-update this listing once
 * gpu_codegen_kernelb_family_md5pass_eligible widens -- no edit needed
 * here.
 *
 * Non-GPU builds (GPU_ENABLED undefined) unconditionally return 0,
 * preserving the prior iMac listing behavior.
 */
static int gpu_op_advertise_for_h_listing(int x) {
#ifdef GPU_ENABLED
    /* Predicate 1: shared file-scope gpu_ops[] linear scan. */
    for (int i = 0; gpu_ops[i] >= 0; i++) {
        if (gpu_ops[i] == x) return 1;
    }
    /* Predicate 2: Phase 4 codegen e347 (MD5MD5SALT). Absent from
     * gpu_ops[]; production dispatch via codegen kernel B. */
    if (x == JOB_MD5MD5SALT) return 1;
    /* Predicate 3: Phase 5a/5b MAKE_MD5PASS family codegen entries. */
    if (gpu_codegen_kernelb_family_md5pass_eligible(x)) return 1;
#else
    (void)x;
#endif
    return 0;
}

int main(int argc, char **argv) {
  struct stat sb;
  struct job *job;
  unsigned long long Totalfiles, Totallines;
unsigned long long Numbers = 0, Maxnumbers = 0;
  unsigned long long SkipLine = 0;
  size_t MemSize;
unsigned int val, maxtype;
  int len, x, y, z, ch, Printsource, maxt;
  char *tok, *s, *d;
  int doneprint, donedashh, Addlf, Dodigits;
  /* Digit-count -n/-N is sugar for a mask: -n 2 IS "?d?d".  Rewritten into
   * this buffer at getopt time so both spellings travel one code path.
   * Two characters per position; its size sets the digit limit. */
  char digitmask[2 * 16 + 1];
  /* A structured hash file read BEFORE any -m/-M/-h: -F and -J parse
   * their file at the moment getopt reaches them, so a selection that
   * arrives later cannot reach it. Remembered here so that selection can
   * refuse, rather than run to completion against nothing. */
  int structfile_preselect = 0;
  const char *structfile_preselect_name = NULL, *structfile_preselect_opt = NULL;
int HashesLoaded = 0;
  FILE *fi;
  gzFile input, zfi;
  int isstdin, isdone;
  Word_t lsi;
  char line[MAXLINE + 16], last[MAXLINE + 16], *cur;
  char workline[MAXLINE + 16];
  struct rule_workspace main_rule_ws;
char *infilename;
  char *Prog = argv[0];
union HashU curin;
  time_t start, end;
  Word_t RC, NextX;
  Word_t *PV = NULL, CurHash;
  /* Dohash, Doload, SaltArray, UserArray, KeyArray, RuleArray, NRuleArray,
     PepperArray, JUDYJ(JOB_PHPBB3), JUDYJ(JOB_APR1), JUDYJ(JOB_MD5CRYPT), JUDYJ(JOB_SHA256CRYPT), JUDYJ(JOB_SHA512CRYPT),
     JUDYJ(JOB_MSSQL2000) are now file-scope globals for load_hash_file() access */
  pcre *regex;
  int ovec[30], pcre_erroff;
  const char *pcre_err;
  unsigned long long curline, numline;    /* widened — wordlist line positions can exceed 4.29B */
  char *readbuf;
  struct LineInfo *readindex;
  unsigned int Linecount, Bcryptcnt, DEScryptcnt, BSDIcryptcnt, ProgressEnccnt;
  __attribute__((unused)) int gpu_batch_lines = 0;
  struct Hashchain *chain;
  int Recurse = 0;
  double wtime;
  struct RuleHist *rulehist;
  int DoCustom;

/*
    pthread_attr_t attr;
    size_t stacksize;

    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr,&stacksize);
    printf("Current pthread stack is %ld\n",stacksize);
    pthread_attr_setstacksize(&attr,512*1024);
    pthread_attr_getstacksize(&attr,&stacksize);
    printf("Current pthread stack is %ld\n",stacksize);
    printf("Mainline stack at %lx\n",&sb);
*/
  Memlock = new_lock(0);
  /* 7ZIP (e1000) load-time triage tunables. Read BEFORE getopt, because -F
   * loads its hash file at the moment getopt reaches it, which is when the
   * triage runs. */
  /* SevenZipMinBits and SevenZipStrict keep their compiled-in defaults.
   * MDXFIND_7Z_MIN_BITS and MDXFIND_7Z_STRICT used to set them here, before
   * getopt, because -F triages its hash file as getopt reaches it. */
  TsfLock = new_lock(0);   /* tsfprintf() stderr serialization (startup-phase diagnostics) */
  /* Seed the tsfprintf monotonic origin now so every later timestamp
   * is relative to a known mdxfind-start point. Without this the
   * first tsfprintf() (which may not happen for hundreds of ms during
   * argv parsing) would establish T+0, distorting Phase A timing. */
  clock_gettime(CLOCK_MONOTONIC, &tsfprintf_origin);
  tsfprintf_origin_set = 1;
  ServerState = new_lock(0);
  for (maxtype=0; Types[maxtype]; maxtype++);
  /* ---- Phase A: built-in / user-defined op address separation ------------
   *
   * The user-defined range used to start at a hand-maintained constant. When
   * the built-in range grew into it, a new built-in type silently ALIASED a
   * user-defined op -- both index the same [JOB_DONE]-sized per-op arrays
   * (JudyJ, Typesalt, TypeOpts, Totalfound, ...) and nothing complained. It
   * reached 999 against a base of 1000: zero slots left.
   *
   * The base is now DERIVED from the actual Types[] length, rounded up to
   * leave a visible gap, and handed to userdef.c BEFORE userdef_load()
   * assigns any ops. Adding a built-in type moves the base with it, so the
   * two ranges cannot collide however many built-ins are added. There is no
   * longer a constant for anyone to forget to raise.
   *
   * Op numbers never escape the process -- found lines carry TYPENAME(op) and
   * Hashchain.flags is in-memory only -- so the base moving between runs is
   * invisible externally. That is what makes deriving it safe.
   * ---------------------------------------------------------------------- */
  Userdef_base = ((unsigned)maxtype + 63u) & ~63u;   /* round up, gap is visible */
  if (Userdef_base + USERDEF_MAX > JOB_DONE) {
    fprintf(stderr,
      "FATAL: %u built-in hash types leaves no room for the user-defined\n"
      "       range (%u user slots) below JOB_DONE=%u. Raise JOB_DONE and the\n"
      "       [JOB_DONE]-sized per-op arrays together.\n",
      (unsigned)maxtype - 1, (unsigned)USERDEF_MAX, (unsigned)JOB_DONE);
    exit(1);
  }
  userdef_set_base(Userdef_base);


#if ARM > 6
#ifdef __aarch64__
  /* NEON is mandatory on AArch64 (ARMv8+) */
  Neon = 1;
#else
  fi = fopen("/proc/cpuinfo","r");
  if (fi) {
    while (fgets(line,MAXLINE,fi)) {
      if (strstr(line,"Features")) {
        if (strstr(line,"neon") || strstr(line,"simd"))
          Neon = 1;
        break;
      }
    }
    fclose(fi);
  }
#endif
  if (Neon)
    fprintf(stderr,"ARM Neon SIMD enabled\n");
#endif

  getcpuinfo();

#ifndef NOTINTEL
  if (HasSSSE3)
    fprintf(stderr, "SSSE3 hex conversion enabled\n");
  else
    fprintf(stderr, "SSE2 hex conversion enabled\n");
#endif
#if defined(ARM) && ARM >= 8 && defined(__aarch64__)
  { extern void arm_ce_detect(void); arm_ce_detect(); }
#endif
  /* Streaming SHA-256 dispatch. Defined on every arch (a no-op where there
   * is nothing to detect), called unconditionally and before any worker
   * thread starts, so the function pointer is read-only for the whole
   * threaded phase. arm_ce_detect() above cannot serve this role: it is
   * compiled and called only under the aarch64 guard. */
  { extern void mysha256_cpu_detect(void); mysha256_cpu_detect(); }

  Minhashlen = 256;
  JUDYJ(JOB_SHA512CRYPT) = JUDYJ(JOB_SHA256CRYPT) = JUDYJ(JOB_MD5CRYPT) = JUDYJ(JOB_APR1) = Dohash = NULL;
  Doload = NULL;
  JUDYJ(JOB_PHPBB3) = PepperArray = UserArray = KeyArray = NULL;
  memset(&Rulestore, 0, sizeof Rulestore);
  memset(&Rulefile,  0, sizeof Rulefile);
  RuleCnt = NULL;
  rhash_library_init();
  gosthash_init();
  time(&start);
  input = NULL;
  isstdin = 0;
  Debug = 0;
  Printsource = 0;
  atomic_store(&Totfound, (unsigned long long)0);
  Totalfiles = Totallines = 0;
  Maxiter = 1;
  maxt = get_nprocs();
  if (maxt < 1) maxt = 1;
  Addlf = 0;

  Readbuf = malloc_lock((size_t)NREADBUF * READBUFSZ + 16,"Read buffer");

  Readindex = malloc_lock((size_t)NREADBUF * RINDEXSIZE * sizeof(struct LineInfo) + 16,"Read index");

  for (x = 1; Types[x]; x++) {
    if (!TYPEOPTS(x)) {
      fprintf(stderr, "Program error: TYPEOPTS(%d) (%s) is zero\n", x, Types[x]);
      exit(1);
    }
  }
  Typesalt = (Pvoid_t *)malloc_lock(JOB_DONE * sizeof(Pvoid_t), "Typesalt");
  DashSSalt = (Pvoid_t *)malloc_lock(JOB_DONE * sizeof(Pvoid_t), "DashSSalt");
  UserTypesalt = (Pvoid_t *)malloc_lock(USERDEF_MAX * sizeof(Pvoid_t), "UserTypesalt");
  Typehashsalt = (Pvoid_t *)malloc_lock(JOB_DONE * sizeof(Pvoid_t), "Typehashsalt");
  Typesaltcnt = (int *)malloc_lock(JOB_DONE * sizeof(int), "Typesaltcnt");
  Typesaltbytes = (long long *)malloc_lock(JOB_DONE * sizeof(long long), "Typesaltbytes");
  Typesalt2 = (char **)malloc_lock(JOB_DONE * sizeof(char *), "Typesalt2");
  Typeuser = (void **)malloc_lock(JOB_DONE * sizeof(void *), "Typeuser");
  UserTypeuser = (void **)malloc_lock(USERDEF_MAX * sizeof(void *), "UserTypeuser");
  Typepepper = (char **)malloc_lock(JOB_DONE * sizeof(char *), "Typepepper");
  Livesalts = (volatile unsigned int *)malloc_lock(JOB_DONE * sizeof(unsigned int), "Livesalts");
  UserLivesalts = (volatile unsigned int *)malloc_lock(USERDEF_MAX * sizeof(unsigned int), "UserLivesalts");
  J1S(RC, Dohash, JOB_MD5);
  donedashh = 0;
  Dodigits = 0;
  infilename = "<STDIN>";
  Maxniter = 1;
  if (argc == 2 && strncmp(argv[1],"-h",2) == 0) {
        /* User-defined types are registered by userdef_load(), which runs
         * further down main() -- AFTER this block, which exits. Without this
         * the listing showed built-ins only, so a tool building a known-type
         * list from -h could never contain a USER_ type even though -m u<id>
         * and -c USER_<name> both work. Silent: userdef_verbose is 0 unless
         * -Y asked for the report. Cannot double-register: this block exits. */
        userdef_load(getenv("MDXFIND_CACHE"));
        z = 0;
        for (x = 1; Types[x]; x++) {
          y = strlen(Types[x]);
          if (y > z) z = y;
        }
        ch = 8; /* strlen("Internal") */
        for (x = 1; Types[x]; x++) {
          y = snprintf(NULL, 0, "e%d", x);
          if (y > ch) ch = y;
        }
        printf("%-*s  %-7s  %-*s  %s\n", ch, "Internal", "Options", z, "Hash name", "hashcat -m");
        printf("\n  Options -- how a hash of this type is fed to mdxfind:\n"
               "    f   plain hex, read with -f (or on stdin)\n"
               "    F   wrapped or structured hash, read with -F\n"
               "        (-J offers the file to every selected type instead of the -M set)\n"
               "    s   takes a salt: use -F with hash:salt lines, or -s <saltfile>\n"
               "        (-S <file> after -M loads salts for the selected types only)\n"
               "        CRYPTOPPLEGACY and CRYPTOPPDEFAULT are the exception that\n"
               "        matters here: those records are ENCRYPTED, not hashed, so\n"
               "        the salt is ONE SITE-WIDE KEY and THE WORDLIST CANNOT\n"
               "        AFFECT THE RESULT. Supply it inline as record:sitekey, or\n"
               "        put candidate keys in a -s/-S file to search them. Output\n"
               "        is record:sitekey:plaintext -- the LAST field is the\n"
               "        decrypted PLAINTEXT, never the key.\n"
               "    u   takes a username: -F with hash:user lines, or -u <userfile>\n"
               "    j   takes a pepper: -j <file>, or -P <file> after -M\n"
               "  The -m/-M selection MUST come BEFORE -F/-J on the command line;\n"
               "  -f and -s may appear anywhere.\n\n");
        for (x = 1; Types[x]; x++) {
          char optbuf[16];
          int oi = 0;
          if (TYPEOPTS(x) & TYPEOPT_NEEDSF) optbuf[oi++] = 'f';
          /* 'F', not 'J': this is the channel the user TYPES, and -F is the
           * option they reach for. The letter used to be J, the internal flag
           * name, so the one option that had to appear never did -- kpd asked
           * for the capital F on 2026-09-05 after losing an evening to it. */
          if (TYPEOPTS(x) & TYPEOPT_NEEDSJ) { if (oi) optbuf[oi++] = ','; optbuf[oi++] = 'F'; }
          if (TYPEOPTS(x) & TYPEOPT_NEEDSALT) { if (oi) optbuf[oi++] = ','; optbuf[oi++] = 's'; }
          if (TYPEOPTS(x) & TYPEOPT_NEEDUSER) { if (oi) optbuf[oi++] = ','; optbuf[oi++] = 'u'; }
          if (TYPEOPTS(x) & TYPEOPT_NEEDPEPPER) { if (oi) optbuf[oi++] = ','; optbuf[oi++] = 'j'; }
          optbuf[oi] = 0;
          { char hcbuf[64];
            int hci = 0;
            for (val = 0; Maphashcat[val].hc != 65535; val++) {
              if (Maphashcat[val].mdx == x) {
                if (hci) hci += snprintf(hcbuf + hci, sizeof(hcbuf) - hci, ",");
                hci += snprintf(hcbuf + hci, sizeof(hcbuf) - hci, "%d", Maphashcat[val].hc);
              }
            }
            { const char *gpu = "";
              /* 2026-05-27: replaced manual if-chain with predicate-based
               * detection via gpu_op_advertise_for_h_listing (defined
               * above main). The helper composes gpu_ops[] legacy
               * template scan, JOB_MD5MD5SALT (e347) codegen, and the
               * Phase 5a/5b MAKE_MD5PASS family eligibility predicate
               * so the listing auto-tracks runtime GPU admit reality
               * across all dispatch paths. Non-GPU builds get the
               * (void)x no-op path inside the helper. */
              if (gpu_op_advertise_for_h_listing(x))
                gpu = " [GPU]";
              if (hci)
                printf("e%-*d  %-7s  %-*s  %s%s\n", ch - 1, x, optbuf, z, Types[x], hcbuf, gpu);
              else
                printf("e%-*d  %-7s  %-*s  na%s\n", ch - 1, x, optbuf, z, Types[x], gpu);
            }
          }
        }
	printf("\nA type marked F does not load from -f at all. A type marked s does, but\n"
               "-f keeps only the hex hash and discards the salt after the first colon, so\n"
               "the run finds nothing and still exits 0. Use -F for hash:salt input.\n");
        {
          /* User-defined types live in a SEPARATE address space from the
           * built-ins, and are listed separately for that reason. They are
           * keyed on the DECLARED id -- what -m u<id> takes -- never on the
           * internal op: Userdef_base is derived from the built-in count at
           * startup and moves when built-ins are added, and op numbers are
           * documented as never escaping the process. Printing one here would
           * publish an unstable number. The name always carries the USER_
           * prefix, which is the public label for these types. */
          int _un = userdef_count(), _ui;
          if (_un > 0) {
            printf("\nUser-defined types, from $MDXFIND_CACHE/userdef.txt"
                   " (select with -m u<id>):\n\n");
            for (_ui = 0; _ui < _un; _ui++) {
              struct userdef_type *_ut = userdef_get_by_index(_ui);
              char _ub[16];
              int _uo = 0;
              if (!_ut) continue;
              /* Derived from the expression slot mask, not from TypeOpts:
               * that array is assembled further down main() and is still
               * empty here. Every user type reads plain hex, hence f. */
              _ub[_uo++] = 'f';
              if (_ut->form.npieces > 0) { _ub[_uo++] = ','; _ub[_uo++] = 'F'; }
              if (_ut->slot_mask & USERDEF_SLOT_SALT) { _ub[_uo++] = ','; _ub[_uo++] = 's'; }
              if (_ut->slot_mask & USERDEF_SLOT_USER) { _ub[_uo++] = ','; _ub[_uo++] = 'u'; }
              _ub[_uo] = 0;
              printf("u%-*s  %-7s  %-*s  na\n", ch - 1, _ut->idstr, _ub,
                     z, _ut->dispname);
            }
            printf("\nA user type is always labelled USER_<name> on output;"
                   " the bare name matches\nonly internally. Digest length and"
                   " expression are shown by -Y.\n");
          }
        }
	printf("For full usage, use -?, or supply no arguments\n");
        exit(1);
  }

  int alloc_maxt = maxt;  /* save for wait_for at exit */
  WorkWaiting = new_lock(0);
  FreeWaiting = new_lock(maxt * 32);
  for (x = 0; x < NREADBUF; x++) {
    ReadBufL[x] = new_lock(0);
    RBready[x] = new_lock(0);
  }
  TestVecWaiting = new_lock(0);
  HashWaiting = new_lock(0);
  Stats = new_lock(0);
  for (x = 0; x < (maxt * 32) + 1; x++) {
    job = malloc_lock(sizeof(struct job), "Job");
    if (FreeTail) {
      *FreeTail = job;
      FreeTail = &(job->next);
    } else {
      FreeHead = job;
      FreeTail = &(job->next);
    }
    job->op = 0;
  }
  WorkTail = &WorkHead;
  Insize = 0;
  launch(addhash, NULL);
  inhashstart = inhashbuf = inhash1 = malloc_lock(MAXADDHASHBUF + 16 + 512,"inhash");
  inhash2 = malloc_lock(MAXADDHASHBUF + 16 + 512,"inhash");
  inhash = 0;
  inhashcnt = 0;
  current_utc_time(&starthash);

  /*
   * User-defined hash types (Milestone 1): load $MDXFIND_CACHE/userdef.txt
   * before argv parsing so that `-m u<id>` selectors can resolve against
   * the registered synthetic ops.  Missing file / unset MDXFIND_CACHE is
   * fine -- user types are optional.
   */
  /*
   * -Y (validate userdef.txt and exit): the load runs before getopt so that
   * -m u<id> can resolve during arg parsing -- which means a getopt case would
   * flip the verbose flag too late to print the load report.  Pre-scan argv
   * for -Y here: if present, turn the load report on, load, and exit.  A pure
   * "did it read userdef.txt correctly?" mode that emits nothing otherwise.
   */
  {
    int _yi;
    for (_yi = 1; _yi < argc; _yi++)
      if (!strcmp(argv[_yi], "-Y")) {
        userdef_verbose = 1;
        userdef_load(getenv("MDXFIND_CACHE"));
        exit(0);
      }
  }
  userdef_load(getenv("MDXFIND_CACHE"));
  /*
   * Per loaded user type, assemble TypeOpts from the slot-inference mask
   * computed at load time (userdef.c::program_slot_mask).  Always
   * NEEDSF (plain-hex fast path through commit_compact); add NEEDSALT |
   * SALTJUDY if the expression references `salt`, and NEEDUSER |
   * USERJUDY if it references `user`.  The standard load_hash_file
   * generic hex[:salt][:user] handler then populates Typesalt[uop] /
   * Typeuser[uop] per-hash automatically.
   *
   * v1.527: salt2 + pepper still rejected at load (userdef.c), so those
   * bits never appear here.  Synthetic ops are assigned sequentially
   * from JOB_USERDEF_BASE, so iterate the contiguous range.
   */
  {
    int _uc = userdef_count(), _ui;
    for (_ui = 0; _ui < _uc; _ui++) {
      int _uop = (int)user_id_op(_ui);
      struct userdef_type *_ut = userdef_get(_uop);
      unsigned short _opts;
      if (_uop >= JOB_DONE || !_ut) continue;
      /*
       * User-defined types load through the STRUCTURED channel (-M/-F), not the
       * plain-hex fast path. Their stored form may carry a salt, a user, literals
       * or separators inside the field, none of which the hex loader can see -- it
       * would take the whole field as the digest.
       */
      _opts = TYPEOPT_NEEDSJ;
      if (_ut->slot_mask & USERDEF_SLOT_SALT)
        _opts |= TYPEOPT_NEEDSALT | TYPEOPT_SALTJUDY;
      if (_ut->slot_mask & USERDEF_SLOT_USER)
        _opts |= TYPEOPT_NEEDUSER | TYPEOPT_USERJUDY;
      TYPEOPTS(_uop) = _opts;
    }
  }

  /*
   * Which channel flags appear ANYWHERE in argv.
   *
   * The dual-channel dispatch in case 'm' has to know whether -s/-S or -u/-U
   * was supplied, and getopt reports options strictly left to right, so a test
   * written inside case 'm' would see only the flags that happened to precede
   * -m on the command line: `-s salts -m 4510` and `-m 4510 -s salts` would
   * then select different types, which is a silent wrong-type run either way
   * round.  Deferring the decision until after the loop is not available
   * either -- -F is handled inside it and needs Dohash already populated,
   * which is why the man page requires the selection to precede -F.
   *
   * So scan argv first.  Option ARGUMENTS are stepped over exactly as getopt
   * steps over them, so a salt file that happens to be named -u is not
   * mistaken for the flag.
   */
  {
    static const char takesarg[] = "GfghijkmnqrstuwxFJLNRMSUPWX";
    const char *p;
    int ai;
    for (ai = 1; ai < argc; ai++) {
      if (argv[ai][0] != '-' || argv[ai][1] == 0)
        continue;
      if (argv[ai][1] == '-' && argv[ai][2] == 0)
        break;                          /* -- ends the options */
      for (p = argv[ai] + 1; *p; p++) {
        if (*p == 'z') Arg_have_z = 1;
        if (!strchr(takesarg, *p))
          continue;                     /* flag letter, no argument */
        if (*p == 's' || *p == 'S') Arg_have_salt = 1;
        if (*p == 'u' || *p == 'U') Arg_have_user = 1;
        if (p[1] == 0) ai++;            /* argument is the next argv entry */
        break;                          /* rest of this token is the argument */
      }
    }
  }

  while ((ch = getopt(argc, argv, "?8abcde:lpvVyzZECG:f:g:h:i:j:k:m:n:q:r:s:t:u:w:x:F:J:L:N:R:M:S:U:P:W:X:")) != -1) {
    switch (ch) {
      case 'X':
        s = optarg;
        XOR_key_len = strlen(s);
        XOR_key = malloc_lock(XOR_key_len+1,"XOR Key");
        strncpy(XOR_key, s, XOR_key_len);
        s = XOR_key;
        if (strncmp(s, "$HEX[", 5) == 0) {
          /* Skip the '$'. The length has to follow, or the decode is handed
           * one character too many for this form. */
          s++;
          XOR_key_len--;
        }
        fprintf(stderr, "Setting XOR key to %s\n", s);
        if (strncmp(s, "HEX[", 4) == 0) {
          /* Decode into XOR_key, NOT into s. checkhash() reads the key from
           * XOR_key, so decoding into s left the key off by one for the
           * $HEX[..] form: byte 0 stayed the literal '$' that strncpy put
           * there, and the last decoded byte was never read. -X '$HEX[ffee]'
           * XORed with 24,ff rather than ff,ee, silently, while the bare
           * HEX[..] form was correct because there s == XOR_key. */
          XOR_key_len = get32(s + 4, (unsigned char *)XOR_key, XOR_key_len - 5);
          if (XOR_key_len == 0)
            XOR_key = NULL;
        }
        break;

      case 'm':
        /* FATAL: the file is already gone. -F and -J parse at the moment
         * getopt reaches them, against whatever selection exists THEN, so a
         * selection appearing later reads its file into nothing and the run
         * completes with 0 found and exit 0 -- indistinguishable from a genuine
         * negative. Reported by kpd 2026-09-05, who lost an evening to the
         * -f/-F half of the same trap. Every other input option (-f, -s) is
         * position-free, which is exactly why this one surprises. */
        if (structfile_preselect) {
          fprintf(stderr,
            "FATAL: %s %s was read BEFORE any -m/-M/-h type selection, so it was\n"
            "       parsed with no types to load into and nothing was kept. Put the\n"
            "       -m/-M selection BEFORE %s on the command line.\n",
            structfile_preselect_opt, structfile_preselect_name,
            structfile_preselect_opt);
          exit(1);
        }
        if (donedashh == 0) {
          J1FA(RC, Dohash);
          donedashh = 1;
        }
        s = optarg;
        while (*s) {
          if (*s == ',' || *s == ':' || *s == '.') {
            s++;
            continue;
          }
          if (*s == 'u') {
            /*
             * User-defined hash type selector: -m u<id>.  EXACT string-keyed
             * lookup (NOT the regex -h path) -- a freeform id may contain
             * regex metacharacters.  The id token runs to the next
             * separator (',' or end-of-string).
             */
            char idbuf[128];
            int il = 0, uop;
            s++;
            while (*s && *s != ',' && il < (int)sizeof(idbuf) - 1) {
              idbuf[il++] = *s++;
            }
            idbuf[il] = '\0';
            uop = userdef_lookup_by_id(idbuf);
            if (uop < 0) {
              /*
               * Sub-phase C2 (fatal-if-selected): a non-fatal load skip is
               * fatal when that exact id is the one SELECTED.  Surface the
               * recorded reason if the id was seen-but-skipped; otherwise
               * it simply was never defined.
               */
              const char *why = userdef_skip_reason(idbuf);
              if (why)
                fprintf(stderr, "User-defined hash type u%s failed to "
                        "load: %s\n", idbuf, why);
              else
                fprintf(stderr, "Unknown user-defined hash type u%s "
                        "(not found in $MDXFIND_CACHE/userdef.txt)\n", idbuf);
              exit(1);
            }
            /* Phase A: record the selection in the SEPARATE user bitfield
             * (keyed by user id), then mirror it into Dohash for iteration.
             * DoUser is authoritative for "is this user type selected"; a
             * built-in op can never appear in it, and a user id can never be
             * mistaken for a built-in op. */
            { Word_t _uid = (Word_t)op_user_id((unsigned)uop);
              J1S(RC, DoUser, _uid);
            }
            J1S(RC, Dohash, uop);
            continue;
          }
          if (*s == 'e') {
            s++;y=0;
	    while (*s >= '0' && *s <= '9') {
	      y = y * 10 + *s - '0';
	      s++;
	    }
	    if (y <= 0 || y >= (int)maxtype) {
	      /* Phase A: eN addresses BUILT-IN types only. maxtype is the real
	       * Types[] length, so this can no longer reach the user range even
	       * if that range moves. */
	      fprintf(stderr, "Internal type e%d out of range (valid: e1 to e%d)\n", y, maxtype - 1);
	      exit(1);
	    }
	    if (*s == '-') {
	      s++;
	      if (*s == 'e') s++;
	      z = 0;
	      while (*s >= '0' && *s <= '9') {
	        z = z * 10 + *s - '0';
	        s++;
	      }
	      if (z <= 0 || z >= (int)maxtype) {
	        fprintf(stderr, "Internal type e%d out of range (valid: e1 to e%d)\n", z, maxtype - 1);
	        exit(1);
	      }
	      if (z < y) {
	        fprintf(stderr, "Invalid range e%d-e%d: start exceeds end\n", y, z);
	        exit(1);
	      }
	      for (x = y; x <= z; x++)
	        J1S(RC, Dohash, x);
	    } else {
              J1S(RC, Dohash, y);
	    }
            continue;
	  }
          if (*s < '0' || *s > '9') {
            fprintf(stderr, "Invalid character %c in mode\n", *s);
            s++;
            continue;
          }
          y = 0;
          while (*s >= '0' && *s <= '9') {
            y = y * 10 + *s - '0';
            s++;
          }
          for (x = 0; Maphashcat[x].hc != 65535; x++)
            if (y == Maphashcat[x].hc)
              break;
          if (Maphashcat[x].mdx == 65535) {
            fprintf(stderr, "Unknown or unsupported hashcat mode %d\n", y);
            exit(1);
          } else {
            if ((y == 4500 || y == 2600 || y == 2630 || y == 34400 || y == 21000 || y == 21400) && Maxiter < 2)
              Maxiter = 2;
            if (y == 3500 && Maxiter < 3)
              Maxiter = 3;
            if (y == 14400 && Maxiter < 10)
              Maxiter = 10;
            /* Default salts for modes 24/124/125 deferred to post-load */
            /* Dual-channel modes: pick the member matching the channel the
             * operator actually fed.  See Mapdualchannel[] above. */
            { int d;
              for (d = 0; Mapdualchannel[d].hc; d++)
                if (Mapdualchannel[d].hc == (unsigned int)y)
                  break;
              if (Mapdualchannel[d].hc) {
                int got = 0;
                if (Arg_have_salt && Mapdualchannel[d].salt) {
                  J1S(RC, Dohash, Mapdualchannel[d].salt);
                  fprintf(stderr, "-m %d: -s/-S supplied, selecting e%d %s\n",
                          y, Mapdualchannel[d].salt,
                          Types[Mapdualchannel[d].salt]);
                  got = 1;
                }
                if (Arg_have_salt && !Mapdualchannel[d].salt)
                  fprintf(stderr, "-m %d: this mode's operand is read from the "
                          "userid channel (-u/-U); it has no salt-channel "
                          "type, so -s/-S does not feed it\n", y);
                if (Arg_have_user || !got) {
                  J1S(RC, Dohash, Mapdualchannel[d].user);
                  if (Arg_have_user)
                    fprintf(stderr, "-m %d: -u/-U supplied, selecting e%d %s\n",
                            y, Mapdualchannel[d].user,
                            Types[Mapdualchannel[d].user]);
                }
                continue;
              }
            }
            y = Maphashcat[x].mdx;
            J1S(RC, Dohash, y);
          }
        }
        break;
      case 'a':
        fprintf(stderr, "Email address munging on\n");
        Email = 1;
        break;

      /* -E: scope rules, masks and keys to the local part of an email address.
       * procrule spells this the same way; -A was not available there, being
       * its audit mode since well before this existed. */
      case 'E':
        fprintf(stderr, "Email local-part mode on: rules and masks apply to the local part\n");
        EmailLocal = 1;
        break;

      case 'b':
        fprintf(stderr, "Unicode expansion on\n");
        Unicode = 1;
        break;

      case 'c':
        fprintf(stderr, "Replacing specials with XML equivalents\n");
        XMLchar = 1;
        break;

      case '8':
        fprintf(stderr, "UTF-8 rule support enabled (UTF-32 engine available)\n");
        Utf32Rules = 1;
        break;

      case 'L':
        /* Implies -8, exactly as procrule's does. */
        Utf32Rules = 1;
        if (!strcasecmp(optarg, "tr") || !strcasecmp(optarg, "az") ||
            !strcasecmp(optarg, "turkish") || !strcasecmp(optarg, "azeri"))
          Utf32Locale = U32LOC_TR;
        else if (!strcasecmp(optarg, "c") || !strcasecmp(optarg, "root") ||
                 !strcasecmp(optarg, "posix") || !strcasecmp(optarg, "none"))
          Utf32Locale = U32LOC_ROOT;
        else {
          fprintf(stderr, "Unknown locale \"%s\" for -L.\n"
                  "  tr, az     Turkish and Azeri dotted/dotless I\n"
                  "  c, root    the mapping everyone else uses\n"
                  "  omit -L    try the default mapping and count the rest\n",
                  optarg);
          exit(1);
        }
        fprintf(stderr, "UTF-8 rule support enabled (UTF-32 engine available), "
                        "case mapping pinned to %s\n",
                Utf32Locale == U32LOC_TR ? "tr/az" : "c/root");
        break;

      case 'e':
        if (!(optarg[0] >= '0' && optarg[0] <= '2' && optarg[1] == 0)) {
          fprintf(stderr, "-e takes 0, 1 or 2 (got \"%s\"): 0 matches whole hashes only, "
                  "1 standard extended matching, 2 byte-by-byte. The former bare -e is -e 2.\n", optarg);
          exit(1);
        }
        Extmatch = optarg[0] - '0';
        Extmatch_set = 1;
        if (Extmatch == 2)
          fprintf(stderr, "Extended match: byte-by-byte\n");
        break;

      case 'w':
        SkipLine = atoll(optarg);
        fprintf(stderr, "Skipping first %llu lines on wordlist\n", SkipLine);
        break;

      case 'W':
        if (strcmp(optarg, "none") == 0) {
          AutoCountRequested = 0;
          EstimatedTotalLines = 0;
        } else if (strncmp(optarg, "auto", 4) == 0) {
          AutoCountRequested = 1;
          if (optarg[4] == ':' && optarg[5]) {
            strncpy(LineCountCachePath, optarg + 5, sizeof(LineCountCachePath) - 1);
            fprintf(stderr, "Wordlist ETA: automatic line counting, cache %s\n", LineCountCachePath);
          } else {
            fprintf(stderr, "Wordlist ETA: automatic line counting enabled\n");
          }
        } else {
          char *endp;
          unsigned long long wval = strtoull(optarg, &endp, 10);
          if (*endp == 'K' || *endp == 'k') { wval *= 1000; endp++; }
          else if (*endp == 'M' || *endp == 'm') { wval *= 1000000; endp++; }
          else if (*endp == 'G' || *endp == 'g') { wval *= 1000000000ULL; endp++; }
          if (wval == 0) {
            fprintf(stderr, "-W: invalid line count '%s' (use a number, 'auto', or 'none')\n", optarg);
            exit(1);
          }
          EstimatedTotalLines = wval;
          fprintf(stderr, "Wordlist ETA: estimated %llu total lines\n", wval);
        }
        break;

      case 'y':
        Recurse = 1;
        fprintf(stderr, "Directory recursion on\n");
        break;

      case 'z':
        fprintf(stderr, "Debug output - output all hashes, explore long hash chains\n");
        NoMarkSalt = Printall = 1;
        break;

      case 'Z':
        fprintf(stderr, "Rule Histogram enabled\n");
        DoHistogram = 1;
        break;

      case 'G':
        /* -G <list> selects specific GPU devices by index.
         * Format: comma-separated IDs or ranges: 0,2,4 or 0-2,5,7-9
         * -G none or -G n/N disables GPU entirely.
         * -G list: enumerate GPU devices and exit immediately. */
        if (optarg[0] == 'n' || optarg[0] == 'N') {
          fprintf(stderr, "GPU disabled\n");
          NoMetal = 1;
#ifdef GPU_ENABLED
        } else if (optarg[0] == 's' || optarg[0] == 'S') {
          /* -G serial / -G s — disable Memo C parallel device init.
           * Reverts gpu_opencl_init() to the prior single-threaded loop.
           * Per-device set_compact_table / set_overflow / set_rules loops
           * still parallelize (those are mdxfind.c-side, independent). */
#if defined(OPENCL_GPU)
          gpu_opencl_set_serial_init(1);
          fprintf(stderr, "GPU init: serial mode (parallel-init disabled)\n");
#endif
        } else if (optarg[0] == 'l' || optarg[0] == 'L') {
          fprintf(stderr, "Available GPU devices:\n");
#if defined(OPENCL_GPU)
          gpu_opencl_list_devices();
#elif defined(__APPLE__) && defined(METAL_GPU)
          /* Phase 1: single-device only — list the default Metal device */
          fprintf(stderr, "  Metal device 0 (default)\n");
#endif
          exit(0);
        } else if (optarg[0] == 'f' || optarg[0] == 'F') {
          extern int GPUForce;
          GPUForce = 1;
          fprintf(stderr, "GPU force: all eligible words dispatched to GPU\n");
        } else {
          /* Parse device list immediately — optarg is transient */
          extern int gpu_device_filter_set;
          extern int gpu_device_allowed[];
          gpu_device_filter_set = 1;
          memset(gpu_device_allowed, 0, sizeof(gpu_device_allowed));
          const char *s = optarg;
          while (*s) {
            while (*s == ',' || *s == ' ') s++;
            if (!*s) break;
            int lo = 0;
            while (*s >= '0' && *s <= '9') { lo = lo * 10 + *s - '0'; s++; }
            int hi = lo;
            if (*s == '-') {
              s++;
              hi = 0;
              while (*s >= '0' && *s <= '9') { hi = hi * 10 + *s - '0'; s++; }
            }
            for (int gi = lo; gi <= hi && gi < 64; gi++)
              gpu_device_allowed[gi] = 1;
          }
          fprintf(stderr, "GPU device filter: %s\n", optarg);
        }
#else
        } else {
          fprintf(stderr, "No GPU code present.\n");
          exit(1);
        }
#endif
        break;

      case 'q':
        Maxniter = atoi(optarg);
        fprintf(stderr, "Max n iterations set to %d\n", Maxniter);
        break;

      case 'g':
        Rotatehash = atoi(optarg);
        fprintf(stderr, "Rotate hashes by %d characters\n", Rotatehash);
        break;

      case 'r':
      case 'R':
      case 's':
      case 'u':
      case 'k':
      case 'j':
        goto Keys_start;

      case 'x':
        Hexkey = 1;
      Keys_start:
        fi = fopen(optarg, "rb");
        if (!fi) {
          perror(optarg);
          fprintf(stderr, "Can't open %s to read keys\n", optarg);
          exit(1);
        }
        fseek(fi, 0, SEEK_END);
        fseek(fi, 0, SEEK_SET);
        line[MAXLINE] = 0;
        x = 0;
        y = 0;
        while (fgets(line, MAXLINE, fi)) {
          cur = strchr(line, 13); /* if cr, zap it */
          if (cur) *cur = 0;
          cur = strchr(line, 10); /* if lf, zap it */
          if (cur) *cur = 0;
          len = mystrlen(line);
          switch (ch) {
	    case 'j':
              JSLI(PV, PepperArray, (unsigned char *)line);
              break;
            case 's':
              JSLI(PV, SaltArray, (unsigned char *)line);
              break;
            case 'u':
              if (len != 0 && user_canon(line, len, Ucanon) > 0) {
                JSLI(PV, UserArray, (unsigned char *)Ucanon);
              }
              break;
            case 'k':
            case 'x':
              JSLI(PV, KeyArray, (unsigned char *)line);
              break;
            case 'r':
            case 'R': {
              /* packedlen, NOT mystrlen(line): the packed bytecode is not a C
               * string.  An operand byte may legitimately be 0x00 and 0x00 is
               * never an opcode, so strlen truncates the rule at its first NUL
               * operand.  That truncation used to reach both the JudySL key and
               * the Rules[] copy, so `$\x00$\x42` and `$\x00$\x43` collapsed to
               * one key -- one of the two rules was silently DROPPED -- and the
               * survivor was stored short while its operand count still said 2,
               * leaving applyrule to read the missing operand out of the next
               * entry's length prefix. */
              int packedlen = 0;
              int packfail = 0;
              int u32_placeholder = 0;
              /* Block-static, not stack: MAXLINE is 40 KB and rule loading is
               * single-threaded at startup. */
              static char srcline[MAXLINE + 16];
              /* Capture the SOURCE TEXT before packrules_len compiles in place
               * (s = d = line) and destroys it.  The packed bytecode cannot be
               * decompiled back to a rule line -- the -Z decoder that tried
               * switched on ASCII opcode letters, so an OPERAND byte could
               * collide with an opcode case, and it read out of bounds.  -Z now
               * prints this text, which is the authored line and therefore
               * itself runnable. */
              int srclen = (int)mystrlen(line);
              if (srclen > (int)sizeof(srcline) - 1) srclen = (int)sizeof(srcline) - 1;
              memcpy(srcline, line, (size_t)srclen);
              srcline[srclen] = 0;
              u32_placeholder = 0;
              /* Ask the UTF-32 compiler FIRST, so the byte compiler can be run
               * quietly when its refusal is the expected answer rather than a
               * fault.  Under -8 both compilers run for every rule -- dispatch
               * is per (word, rule) and either engine may be chosen for the
               * same rule -- and a rule carrying a quoted or multi-codepoint
               * operand is one the BYTE compiler must refuse: that refusal IS
               * the UTF-32-REQUIRED tag.  Printing a parse error for it told
               * the user their rule file was broken when it was not; a 15-line
               * hand-authored file produced 14 errors and still loaded every
               * rule.  procrule has done this since 1.34 (procrule.c, the
               * Rulequiet assignment); mdxfind asked the two questions in the
               * opposite order and so had no answer to gate on yet.
               *
               * Still AUDIBLE when both compilers refuse: Rulequiet is only
               * set when the UTF-32 compiler accepted the line, so a genuinely
               * bad rule keeps its caret diagnostic, which is the only thing
               * that says why.  In byte mode Utf32Rules is 0 and && short
               * circuits before validrule_u32 is reached, so nothing changes
               * and nothing extra is spent.
               *
               * Cost under -8 is one validrule_u32 per rule at LOAD, never per
               * (word, rule) -- the same order as the per-rule call the Rules[]
               * pass below already makes. */
              Rulequiet = Utf32Rules && (validrule_u32(srcline) == 0);
              Ruleecho = srcline;
              packfail = packrules_len(line, &packedlen);
              Rulequiet = 0;
              Ruleecho = NULL;
              if (packfail) {
badrule:
                /* Reached from BOTH byte-compiler refusals: packrules_len
                 * failing, and the test-apply returning -3.  Either way the
                 * byte engine will not run this line.  srcline holds the source
                 * text -- packrules_len compiled `line` in place and destroyed
                 * it.  Never loops: the only path back here is the test-apply,
                 * which is skipped once u32_placeholder is set. */
                if (Utf32Rules && validrule_u32(srcline) == 0) {
                  /* U32-REQUIRED: the byte compiler refuses it and the UTF-32
                   * compiler accepts it, so under -8 it is a usable rule and
                   * must be ADMITTED rather than dropped.  It has no byte
                   * bytecode, and Rules[] entries ARE bytecode, so it carries a
                   * placeholder payload instead: 0x00 followed by the source
                   * text.
                   *
                   * 0x00 does three jobs at once and was chosen for all three.
                   * It is never an opcode, verified over 338,310 rule lines of
                   * HashMob.100k, all_gj and game.rule -- not one packed rule
                   * begins with it, and the lowest first byte seen was 0xc2 --
                   * so it identifies the entry unambiguously in the Rules[]
                   * build below, where Rule_byteok is cleared for it.  If the
                   * byte walker ever did reach one it would stop at once rather
                   * than misread the source text as bytecode.  And appending
                   * the source keeps the payload unique per distinct rule, so
                   * the store's bytecode dedup still collapses real duplicates
                   * and only those.
                   *
                   * pick_engine() guarantees this is never handed to
                   * applyrule(): byte_can is 0, so it returns UTF-32 or, for a
                   * word that cannot be decoded, -1. */
                  /* Bound the buffer being WRITTEN, which is `line`, not the
                   * one being read.  Both happen to be MAXLINE + 16 and fgets
                   * caps the content at MAXLINE - 1, so this cannot trip today;
                   * naming srcline here was accidentally correct, which is the
                   * kind of thing that stops being correct silently. */
                  if (srclen + 1 <= (int)sizeof(line)) {
                    line[0] = 0;
                    memcpy(line + 1, srcline, (size_t)srclen);
                    packedlen = srclen + 1;
                    u32_placeholder = 1;
                    Rule_u32_only++;
                  } else {
                    fprintf(stderr, "Rule file %s: line %d too long to admit\n",
                            optarg, x + 1);
                    packedlen = 0;
                  }
                } else {
                  if (Utf32Rules) Rule_both_bad++;
                  fprintf(stderr, "Rule file %s: bad line %d\n", optarg, x + 1);
                  packedlen = 0;
                }
              }
              strcpy(workline, "Password");
              if (!u32_placeholder && packedlen &&
                  applyrule(workline, last, 8, line, &main_rule_ws) == -3)
                goto badrule;
              /* packedlen == 0 covers a blank line, a whole-line comment and an
               * explicit `:`.  All three are skipped deliberately: the implicit
               * no-rule pass is foundational, and an explicit `:` must not add a
               * second one. */
              if (packedlen != 0) {
                if (rs_add_src(ch == 'R' ? &Rulefile : &Rulestore,
                               line, packedlen, srcline, srclen) == -2) {
                  fprintf(stderr,
                          "Rule file %s: out of memory storing rule at line %d\n",
                          optarg, x + 1);
                  exit(1);
                }
              }
              break;
            }

            default:
              break;
          }
          x++;
        }
        fclose(fi);
        tok = "keys";
	if (ch == 'j')
	  tok = "peppers";
        if (ch == 's')
          tok = "salts";
        if (ch == 'u')
          tok = "userids";
        if (ch == 'r' || ch == 'R')
          tok = "rules";
        if (ch == 'R') {
          /* Left-major product: the accumulated set varies slowest.  With no
           * prior -r the -R file simply becomes the set, which is what the old
           * `if (!RuleArray)` shortcut did. */
          if (Rulestore.n == 0) {
            rs_reset(&Rulestore);
            Rulestore = Rulefile;
            memset(&Rulefile, 0, sizeof Rulefile);
          } else {
            if (rs_product(&Rulestore, &Rulefile, MAXLINE) < 0) {
              fprintf(stderr, "Rule file %s: out of memory forming -R product\n",
                      optarg);
              exit(1);
            }
            rs_reset(&Rulefile);
          }
        }

        fprintf(stderr, "%d %s read from %s\n", x, tok, optarg);
        break;

      case 'F':
        { gzFile gf;
          if (strcmp(optarg, "stdin") == 0)
            gf = gzdopen(0, "rb");
          else
            gf = gzopen(optarg, "rb");
          if (!gf) {
            perror(optarg);
            fprintf(stderr, "Can't open %s to read hashes\n", optarg);
            exit(1);
          }
          gzbuffer(gf,128*1024);
	  HashesLoaded++;
          if (donedashh == 0) {
            structfile_preselect = 1;
            structfile_preselect_name = optarg;
            structfile_preselect_opt = "-F";
          }
          load_hash_file(gf, optarg, &Doload, /*hex_only=*/0);
          if (strcmp(optarg, "stdin") != 0)
            gzclose(gf);
        }
        break;

      case 'J':
        { gzFile gj;
          if (strcmp(optarg, "stdin") == 0)
            gj = gzdopen(0, "rb");
          else
            gj = gzopen(optarg, "rb");
          if (!gj) {
            perror(optarg);
            fprintf(stderr, "Can't open %s to read non-standard hashes\n", optarg);
            exit(1);
          }
          gzbuffer(gj,128*1024);
	  HashesLoaded++;
          if (donedashh == 0) {
            structfile_preselect = 1;
            structfile_preselect_name = optarg;
            structfile_preselect_opt = "-J";
          }
          load_hash_file(gj, optarg, NULL, /*hex_only=*/0);
          if (strcmp(optarg, "stdin") != 0)
            gzclose(gj);
        }
        break;

      case 'N':
        mask_init_classes();
        /* Same desugar as -n, see there: -N 3 is exactly "?d?d?d".  The
         * digit-count prepend form never accepted the x/X hex suffix, so
         * neither does this. */
        if (!strchr(optarg, '?') && !strchr(optarg, '[')) {
          /* Strict decimal digit count, see -n. No x/X: the digit-count
           * prepend form never accepted a hex suffix. */
          const char *q = optarg;
          int nd = 0;
          while (*q >= '0' && *q <= '9' && nd < 1000)
            nd = nd * 10 + (*q++ - '0');
          if (q != optarg && *q == 0 && nd >= 1) {
            if (nd > (int)(sizeof(digitmask) - 1) / 2) {
              fprintf(stderr, "Too many digits\n");
              exit(1);
            }
            for (x = 0; x < nd; x++)
              memcpy(&digitmask[2 * x], "?d", 2);
            digitmask[2 * nd] = 0;
            optarg = digitmask;
          }
        }
        if (strchr(optarg, '?') || strchr(optarg, '[')) {
          /* Mask prepend mode */
          if (parse_mask_into(optarg, MaskPrependPattern, &MaskPrependLen,
                              &MaskPrependTotal) < 0)
            exit(1);
          /* Also set legacy globals for backward compat */
          MaskPrepend = 1;
          memcpy(MaskPattern, MaskPrependPattern, sizeof(MaskPattern));
          MaskLen = MaskPrependLen;
          MaskTotal = MaskPrependTotal * MaskAppendTotal;
          Dodigits = MaskPrependLen + MaskAppendLen;
          Maxnumbers = MaskTotal;
          /* MaskChunkSize: rightmost non-literal class in the combined pattern */
          { int ci;
            MaskChunkSize = 1;
            /* Prefer append rightmost, then prepend rightmost */
            for (ci = MaskAppendLen - 1; ci >= 0; ci--) {
              if (MaskAppendPattern[ci].classid != MASK_LITERAL) {
                MaskChunkSize = MaskClasses[MaskAppendPattern[ci].classid].count;
                goto chunk_done_N;
              }
            }
            for (ci = MaskPrependLen - 1; ci >= 0; ci--) {
              if (MaskPrependPattern[ci].classid != MASK_LITERAL) {
                MaskChunkSize = MaskClasses[MaskPrependPattern[ci].classid].count;
                break;
              }
            }
          chunk_done_N: ;
          }
          fprintf(stderr, "Mask prepend: %s (%llu combinations)\n", optarg, MaskPrependTotal);
        } else {
          fprintf(stderr, "-N %s: expected a mask (?d, ?l, ?[abc], ...) or a "
                          "digit count 1-16\n", optarg);
          exit(1);
        }
        break;

      case 'n':
        mask_init_classes();
        /* Desugar the digit-count form into the mask that means the same
         * thing: -n 2 is exactly "?d?d", -n 2x is "?h?h", -n 2X is "?H?H".
         * Same candidates in the same order -- the sequences were compared
         * position by position and are identical, so no result changes.
         * What does change is that the mask path enumerates the WHOLE
         * keyspace (digit-count mode stopped after one batch of
         * Iter_Count[] candidates) and sets MaskTotal, which is what the
         * GPU dispatcher reads to size its keyspace. */
        if (!strchr(optarg, '?') && !strchr(optarg, '[')) {
          /* Accepted digit-count form is strict: decimal digits, then an
           * optional single x or X for hex, then end of string. Trailing
           * junk is REJECTED rather than silently ignored -- atoi() would
           * read "4i" as 4 and quietly run an old -n i (IPv4, removed) as
           * a four-digit sweep. Anything not matching falls through to the
           * diagnostic below. */
          const char *q = optarg;
          int nd = 0;
          while (*q >= '0' && *q <= '9' && nd < 1000)
            nd = nd * 10 + (*q++ - '0');
          if (q != optarg && nd >= 1 &&
              (*q == 0 || ((*q == 'x' || *q == 'X') && q[1] == 0))) {
            const char *cls = (*q == 'X') ? "?H" : (*q == 'x') ? "?h" : "?d";
            if (nd > (int)(sizeof(digitmask) - 1) / 2) {
              fprintf(stderr, "Too many digits, won't finish in your lifetime\n");
              exit(1);
            }
            for (x = 0; x < nd; x++)
              memcpy(&digitmask[2 * x], cls, 2);
            digitmask[2 * nd] = 0;
            optarg = digitmask;
          }
        }
        if (strchr(optarg, '?') || strchr(optarg, '[')) {
          /* Mask append mode */
          if (parse_mask_into(optarg, MaskAppendPattern, &MaskAppendLen,
                              &MaskAppendTotal) < 0)
            exit(1);
          /* Also set legacy globals for backward compat */
          MaskPrepend = 0;
          memcpy(MaskPattern, MaskAppendPattern, sizeof(MaskPattern));
          MaskLen = MaskAppendLen;
          MaskTotal = MaskPrependTotal * MaskAppendTotal;
          Dodigits = MaskPrependLen + MaskAppendLen;
          Maxnumbers = MaskTotal;
          /* MaskChunkSize: rightmost non-literal class */
          { int ci;
            MaskChunkSize = 1;
            for (ci = MaskAppendLen - 1; ci >= 0; ci--) {
              if (MaskAppendPattern[ci].classid != MASK_LITERAL) {
                MaskChunkSize = MaskClasses[MaskAppendPattern[ci].classid].count;
                goto chunk_done_n;
              }
            }
            for (ci = MaskPrependLen - 1; ci >= 0; ci--) {
              if (MaskPrependPattern[ci].classid != MASK_LITERAL) {
                MaskChunkSize = MaskClasses[MaskPrependPattern[ci].classid].count;
                break;
              }
            }
          chunk_done_n: ;
          }
          fprintf(stderr, "Mask append: %s (%llu combinations)\n", optarg, MaskAppendTotal);
        } else {
          /* Not a mask and not a digit count the desugar above accepted.
           * -n 0 appended nothing, and -n i (IPv4) was removed: an IP
           * range is trivial to generate externally and pipe in as a
           * wordlist, so it does not need to live in the option parser.
           * Fail loudly rather than run as though no -n were given. */
          fprintf(stderr, "-n %s: expected a mask (?d, ?l, ?[abc], ...) or a "
                          "digit count 1-16 with optional x/X for hex\n", optarg);
          exit(1);
        }
        break;

      case 'l':
        fprintf(stderr, "Adding NUL,CR,CR/LF, etc to passwords\n");
        Addlf = JOBFLAG_HEX;
        break;
      case 'd':
        fprintf(stderr, "Deduping candidate passwords\n");
        Dedupe = 1;
        break;
      case 'p':
        fprintf(stderr, "Print source of password\n");
        Printsource = JOBFLAG_PRINT;
        break;
      case 'f':
        input = gzopen(optarg, "rb");
	isstdin = 0;
        if (!input) {
          perror(optarg);
          exit(1);
        }
        gzbuffer(input,128*1024);
        fprintf(stderr, "Reading hash list from %s...\n",optarg);
        infilename = strdup(optarg);

        break;
      case 'i':
        Maxiter = atoi(optarg);
        if (Maxiter < 1) {
          fprintf(stderr, "Maximum iterations must be positive\n");
          exit(1);
        }
        fprintf(stderr, "Iterations set to %d\n", Maxiter);
        break;
      case 't':
        maxt = atoi(optarg);
        if (maxt < 1 || maxt > 32768) {
          fprintf(stderr, "Maxthreads invalid: %d\n", maxt);
          exit(1);
        }
        break;

      case 'h':
        /* FATAL: the file is already gone. -F and -J parse at the moment
         * getopt reaches them, against whatever selection exists THEN, so a
         * selection appearing later reads its file into nothing and the run
         * completes with 0 found and exit 0 -- indistinguishable from a genuine
         * negative. Reported by kpd 2026-09-05, who lost an evening to the
         * -f/-F half of the same trap. Every other input option (-f, -s) is
         * position-free, which is exactly why this one surprises. */
        if (structfile_preselect) {
          fprintf(stderr,
            "FATAL: %s %s was read BEFORE any -m/-M/-h type selection, so it was\n"
            "       parsed with no types to load into and nothing was kept. Put the\n"
            "       -m/-M selection BEFORE %s on the command line.\n",
            structfile_preselect_opt, structfile_preselect_name,
            structfile_preselect_opt);
          exit(1);
        }
        if (donedashh == 0) {
          J1FA(RC, Dohash);
          donedashh = 1;
        }
        tok = strtok(optarg, ",");
        while (tok) {
          if (strcmp(tok, "ALL") == 0 || strcmp(tok, ".") == 0) {
            /* Curated default set: ~22 commonly encountered hash types.
             * Covers the most frequent unsalted + salted types seen in
             * real-world combo lists. Use -h REGEX for specific types. */
            static const char *all_types[] = {
              /* Unsalted */
              "MD5", "MD4", "NTLM", "SHA1", "SHA224", "SHA256", "SHA384", "SHA512",
              "MYSQL3", "SQL5", "WRL", "RMD160", "BLAKE2S256", "KECCAK256",
              "MD5SHA1", "SHA1MD5",
              /* Salted */
              "MD5PASSSALT", "MD5SALTPASS",
              "SHA1PASSSALT", "SHA1SALTPASS",
              "SHA256PASSSALT", "SHA512SALTPASS",
              NULL
            };
            for (int ti = 0; all_types[ti]; ti++) {
              for (x = 1; Types[x]; x++) {
                if (strcmp(Types[x], all_types[ti]) == 0) {
                  J1S(RC, Dohash, x);
                  break;
                }
              }
            }

          } else {
            y = 0;
            if (tok[0] == '!') {
              y = 1;
              tok++;
            }
            regex = pcre_compile(tok, PCRE_CASELESS, &pcre_err, &pcre_erroff, NULL);
            if (regex) {
              for (x = 1; Types[x]; x++) {
                if (pcre_exec(regex, NULL, Types[x], mystrlen(Types[x]), 0, 0, ovec, 30) < 0)
                  continue;
                if (y) {
                  J1U(RC, Dohash, x);
                } else {
                  J1S(RC, Dohash, x);
                }
              }
            } else {
              x = matchhash(tok);
              if (x) {
                J1S(RC, Dohash, x);
              } else {
                fprintf(stderr, "Invalid hash type %s\n", tok);
              }
            }
          }
          tok = strtok(NULL, ",");
        }
        break;

      case 'v':
        NoMarkSalt = 1;
        fprintf(stderr, "Do not mark salts as found\n");
        break;

      case 'C':
        /* Implies -v.  A solved hash is no longer retired, so retiring the
         * SALT it was solved with would stop the very repeats -C exists to
         * report -- and would do it quietly, by shrinking the next snapshot
         * rather than by reporting anything. */
        Contfound = 1;
        NoMarkSalt = 1;
        fprintf(stderr, "Continue reporting hashes that are already solved\n");
        break;

      case 'V':
        printf("%s: %s\n", Prog, Version);
        exit(1);

      case 'M':
        /* FATAL: the file is already gone. -F and -J parse at the moment
         * getopt reaches them, against whatever selection exists THEN, so a
         * selection appearing later reads its file into nothing and the run
         * completes with 0 found and exit 0 -- indistinguishable from a genuine
         * negative. Reported by kpd 2026-09-05, who lost an evening to the
         * -f/-F half of the same trap. Every other input option (-f, -s) is
         * position-free, which is exactly why this one surprises. */
        if (structfile_preselect) {
          fprintf(stderr,
            "FATAL: %s %s was read BEFORE any -m/-M/-h type selection, so it was\n"
            "       parsed with no types to load into and nothing was kept. Put the\n"
            "       -m/-M selection BEFORE %s on the command line.\n",
            structfile_preselect_opt, structfile_preselect_name,
            structfile_preselect_opt);
          exit(1);
        }
        /* -M: select target types for subsequent -S/-U/-P loading */
        if (donedashh == 0) {
          J1FA(RC, Dohash);
          donedashh = 1;
        }
        J1FA(RC, Doload);
        Doload = NULL;
        s = optarg;
        if (strcmp(s, "RESET") == 0)
          break;
        if (*s == 'u') {
          /* -M u<id>[,u<id>...]: user-defined types, EXACT string-keyed lookup --
           * never the regex path below, because a freeform id may contain regex
           * metacharacters. Mirrors the -m u<id> branch, but ALSO records the op in
           * Doload, so -S/-U/-P can target a user type; -m sets only DoUser and
           * Dohash, which is why per-type salt, userid and pepper loading was
           * previously unreachable for user-defined types however they were
           * selected. */
          while (*s) {
            char idbuf[128];
            int il = 0, uop;
            if (*s == ',' || *s == ':' || *s == '.') { s++; continue; }
            if (*s == 'u') s++;
            while (*s && *s != ',' && il < (int)sizeof(idbuf) - 1) idbuf[il++] = *s++;
            idbuf[il] = '\0';
            if (!il) continue;
            uop = userdef_lookup_by_id(idbuf);
            if (uop < 0) {
              const char *why = userdef_skip_reason(idbuf);
              if (why)
                fprintf(stderr, "-M: user-defined hash type u%s failed to load: %s\n", idbuf, why);
              else
                fprintf(stderr, "-M: Unknown user-defined hash type u%s "
                        "(not found in $MDXFIND_CACHE/userdef.txt)\n", idbuf);
              exit(1);
            }
            { Word_t _uid = (Word_t)op_user_id((unsigned)uop);
              J1S(RC, DoUser, _uid); }
            J1S(RC, Dohash, uop);
            J1S(RC, Doload, uop);
          }
        } else if (*s == 'e') {
          /* internal type indices: eN, eN-eM, comma-separated */
          while (*s) {
            if (*s == ',' || *s == ':' || *s == '.') { s++; continue; }
            if (*s == 'e') {
              s++; y = 0;
              while (*s >= '0' && *s <= '9') { y = y * 10 + *s - '0'; s++; }
              if (y <= 0 || y >= (int)maxtype) {
                fprintf(stderr, "-M: Internal type e%d out of range (valid: e1 to e%d)\n", y, maxtype - 1);
                exit(1);
              }
              if (*s == '-') {
                s++;
                if (*s == 'e') s++;
                z = 0;
                while (*s >= '0' && *s <= '9') { z = z * 10 + *s - '0'; s++; }
                if (z <= 0 || z >= (int)maxtype) {
                  fprintf(stderr, "-M: Internal type e%d out of range (valid: e1 to e%d)\n", z, maxtype - 1);
                  exit(1);
                }
                if (z < y) {
                  fprintf(stderr, "-M: Invalid range e%d-e%d: start exceeds end\n", y, z);
                  exit(1);
                }
                for (x = y; x <= z; x++) {
                  J1S(RC, Dohash, x);
                  J1S(RC, Doload, x);
                }
              } else {
                J1S(RC, Dohash, y);
                J1S(RC, Doload, y);
              }
              continue;
            }
            fprintf(stderr, "-M: Invalid character '%c' in internal type spec\n", *s);
            s++;
          }
        } else if (*s >= '0' && *s <= '9') {
          /* hashcat mode numbers, comma-separated */
          while (*s) {
            if (*s == ',' || *s == ':' || *s == '.') { s++; continue; }
            if (*s < '0' || *s > '9') {
              fprintf(stderr, "-M: Invalid character '%c' in hashcat mode\n", *s);
              s++;
              continue;
            }
            y = 0;
            while (*s >= '0' && *s <= '9') { y = y * 10 + *s - '0'; s++; }
            for (x = 0; Maphashcat[x].hc != 65535; x++)
              if (y == Maphashcat[x].hc) break;
            if (Maphashcat[x].mdx == 65535) {
              fprintf(stderr, "-M: Unknown or unsupported hashcat mode %d\n", y);
              exit(1);
            }
            if ((y == 4500 || y == 2600 || y == 2630 || y == 34400 || y == 21000 || y == 21400) && Maxiter < 2)
              Maxiter = 2;
            if (y == 3500 && Maxiter < 3)
              Maxiter = 3;
            if (y == 14400 && Maxiter < 10)
              Maxiter = 10;
            /* Default salts for modes 24/124/125 deferred to post-load */
            /* 22000/16800: also enable WPA-EAPOL for type=02 lines */
            if (y == 22000 || y == 16800) {
              J1S(RC, Dohash, JOB_WPA_EAPOL);
              J1S(RC, Doload, JOB_WPA_EAPOL);
            }
            y = Maphashcat[x].mdx;
            J1S(RC, Dohash, y);
            J1S(RC, Doload, y);
          }
        } else {
          /* regex pattern(s), comma-separated, with ! for negation.
           * Initialise y here so the post-loop "!matched && !y" check
           * is well-defined even if the loop never executes (empty -M arg). */
          int matched = 0;
          y = 0;
          tok = strtok(s, ",");
          while (tok) {
            y = 0;
            if (tok[0] == '!') { y = 1; tok++; }
            regex = pcre_compile(tok, PCRE_CASELESS, &pcre_err, &pcre_erroff, NULL);
            if (regex) {
              /* User-defined types are searched FIRST; only if this token matches none
               * of them does it fall through to the built-in table below. They live in
               * a separate registry -- Types[] holds built-ins only and is
               * NULL-terminated well below Userdef_base -- so the built-in loop cannot
               * reach them however the pattern is written. Both the display name
               * (USER_<name>) and the raw id are offered to the pattern. */
              int umatched = 0;
              { int ui, un = userdef_count();
                for (ui = 0; ui < un; ui++) {
                  struct userdef_type *ut = userdef_get_by_index(ui);
                  Word_t _uid;
                  if (!ut) continue;
                  if (pcre_exec(regex, NULL, ut->dispname, mystrlen(ut->dispname), 0, 0, ovec, 30) < 0 &&
                      pcre_exec(regex, NULL, ut->idstr, mystrlen(ut->idstr), 0, 0, ovec, 30) < 0)
                    continue;
                  _uid = (Word_t)op_user_id((unsigned)ut->op);
                  if (y) { J1U(RC, DoUser, _uid); J1U(RC, Dohash, ut->op); J1U(RC, Doload, ut->op); }
                  else   { J1S(RC, DoUser, _uid); J1S(RC, Dohash, ut->op); J1S(RC, Doload, ut->op); matched++; }
                  umatched++;
                }
              }
              if (!umatched)
              for (x = 1; Types[x]; x++) {
                if (pcre_exec(regex, NULL, Types[x], mystrlen(Types[x]), 0, 0, ovec, 30) < 0)
                  continue;
                if (y) {
                  J1U(RC, Dohash, x);
                  J1U(RC, Doload, x);
                } else {
                  J1S(RC, Dohash, x);
                  J1S(RC, Doload, x);
                  matched++;
                }
              }
            } else {
              fprintf(stderr, "-M: Invalid regex '%s': %s\n", tok, pcre_err);
              exit(1);
            }
            tok = strtok(NULL, ",");
          }
          if (!matched && !y) {
            fprintf(stderr, "-M: No hash types matched pattern '%s'\n", optarg);
            exit(1);
          }
        }
        { Word_t cnt = 0;
          J1C(cnt, Doload, 0, -1);
          fprintf(stderr, "-M: Selected %lu type(s) for loading\n", (unsigned long)cnt);
        }
        break;

      case 'S':
        /* -S: load salt file; per-type if Doload set, else global SaltArray */
        {
          FILE *sf;
          Pvoid_t TmpSalt = NULL;
          int scnt = 0;
          if (strcmp(optarg, "stdin") == 0 || strcmp(optarg, "-") == 0)
            sf = stdin;
          else {
            sf = fopen(optarg, "rb");
            if (!sf) { perror(optarg); fprintf(stderr, "-S: Can't open salt file %s\n", optarg); exit(1); }
          }
          while (fgets(line, MAXLINE, sf)) {
            cur = strchr(line, 13); if (cur) *cur = 0;
            cur = strchr(line, 10); if (cur) *cur = 0;
            /* Count occurrences: a salt supplied N times is a salt that accounts
             * for N hashes, and that count becomes part of its reference count
             * below, so it retires once those hashes are solved.  JSLI
             * initialises a new value to 0, so ++ yields 1 on first sight.
             * Mirrors the -U handler.  Before this the file was only deduped and
             * every distinct salt got exactly 1 regardless. */
            JSLI(PV, TmpSalt, (unsigned char *)line);
            if (PV) (*PV)++;
            scnt++;
          }
          if (sf != stdin) fclose(sf);
          if (Doload) {
            Word_t ti = 0;
            J1F(RC, Doload, ti);
            while (RC) {
              {
                char tline[MAXLINE + 16];
                Word_t *TPV;
                tline[0] = 0;
                JSLF(PV, TmpSalt, (unsigned char *)tline);
                while (PV) {
                  /* Accumulate, never stamp: the hash file's own byte-exact
                   * count -- store_typesalt, one increment per hash line -- has
                   * to survive a -S in the same run.  Same reasoning as -U, and
                   * the same reason -U stopped writing 1000000 over it. */
                  JSLI(TPV, TYPESALT(ti), (unsigned char *)tline);
                  if (TPV) *TPV += *PV;
                  /* Remember that THIS salt came from -S so the floor pass in
                   * main() can raise it, and only it. */
                  if (ti < JOB_DONE) {
                    Word_t *DPV;
                    JSLI(DPV, DashSSalt[ti], (unsigned char *)tline);
                    if (DPV) *DPV += *PV;
                  }
                  JSLN(PV, TmpSalt, (unsigned char *)tline);
                }
              }
              J1N(RC, Doload, ti);
            }
            fprintf(stderr, "-S: Loaded %d salt(s) for selected types\n", scnt);
          } else {
            char tline[MAXLINE + 16];
            tline[0] = 0;
            JSLF(PV, TmpSalt, (unsigned char *)tline);
            while (PV) {
              JSLI(PV, SaltArray, (unsigned char *)tline);
              JSLN(PV, TmpSalt, (unsigned char *)tline);
            }
            fprintf(stderr, "-S: Loaded %d salt(s) into global SaltArray\n", scnt);
          }
          JSLFA(RC, TmpSalt);
        }
        break;

      case 'U':
        /* -U: load userid file; per-type if Doload set, else global UserArray */
        {
          FILE *uf;
          Pvoid_t TmpUser = NULL;
          int ucnt = 0;
          if (strcmp(optarg, "stdin") == 0 || strcmp(optarg, "-") == 0)
            uf = stdin;
          else {
            uf = fopen(optarg, "rb");
            if (!uf) { perror(optarg); fprintf(stderr, "-U: Can't open userid file %s\n", optarg); exit(1); }
          }
          while (fgets(line, MAXLINE, uf)) {
            cur = strchr(line, 13); if (cur) *cur = 0;
            cur = strchr(line, 10); if (cur) *cur = 0;
            if (mystrlen(line) > 0 &&
                user_canon(line, (int)mystrlen(line), Ucanon) > 0) {
              /* Count occurrences: a userid supplied N times is a userid that
               * accounts for N hashes, and that count becomes its reference
               * count below, so it retires when its hashes are solved. JSLI
               * initialises a new value to 0, so ++ yields 1 on first sight.
               * Canonical form, so a raw and a $HEX[] spelling count as one. */
              JSLI(PV, TmpUser, (unsigned char *)Ucanon);
              if (PV) (*PV)++;
              ucnt++;
            }
          }
          if (uf != stdin) fclose(uf);
          if (Doload) {
            Word_t ti = 0;
            J1F(RC, Doload, ti);
            while (RC) {
              { char tline[MAXLINE + 16];
                Word_t *TPV;
                tline[0] = 0;
                JSLF(PV, TmpUser, (unsigned char *)tline);
                while (PV) {
                  /* Accumulate the occurrence count; do NOT stamp a sentinel.
                   * Each occurrence in a -U file is a separate hash that userid
                   * accounts for, so the count is real and the userid retires
                   * once its hashes are solved. The old `= 1000000` also
                   * CLOBBERED the byte-exact count the hash file itself builds
                   * (loader: `if ((*_UV)++ == 0)` once per hash line), so a run
                   * combining hash:user input with -U lost the real count. */
                  JSLI(TPV, TYPEUSER(ti), (unsigned char *)tline);
                  if (TPV) *TPV += *PV;
                  JSLN(PV, TmpUser, (unsigned char *)tline);
                }
              }
              J1N(RC, Doload, ti);
            }
            fprintf(stderr, "-U: Loaded %d userid(s) for selected types\n", ucnt);
          } else {
            char tline[MAXLINE + 16];
            tline[0] = 0;
            JSLF(PV, TmpUser, (unsigned char *)tline);
            while (PV) {
              JSLI(PV, UserArray, (unsigned char *)tline);
              JSLN(PV, TmpUser, (unsigned char *)tline);
            }
            fprintf(stderr, "-U: Loaded %d userid(s) into global UserArray\n", ucnt);
          }
          JSLFA(RC, TmpUser);
        }
        break;

      case 'P':
        /* -P: load pepper file; per-type if Doload set, else global PepperArray */
        {
          FILE *pf;
          Pvoid_t TmpPepper = NULL;
          int pcnt = 0;
          if (strcmp(optarg, "stdin") == 0 || strcmp(optarg, "-") == 0)
            pf = stdin;
          else {
            pf = fopen(optarg, "rb");
            if (!pf) { perror(optarg); fprintf(stderr, "-P: Can't open pepper file %s\n", optarg); exit(1); }
          }
          while (fgets(line, MAXLINE, pf)) {
            cur = strchr(line, 13); if (cur) *cur = 0;
            cur = strchr(line, 10); if (cur) *cur = 0;
            JSLI(PV, TmpPepper, (unsigned char *)line);
            pcnt++;
          }
          if (pf != stdin) fclose(pf);
          if (Doload) {
            Word_t ti = 0;
            J1F(RC, Doload, ti);
            while (RC) {
              {
                char tline[MAXLINE + 16];
                Word_t *TPV;
                Pvoid_t pjudy = (Pvoid_t)Typepepper[ti];
                tline[0] = 0;
                JSLF(PV, TmpPepper, (unsigned char *)tline);
                while (PV) {
                  JSLI(TPV, pjudy, (unsigned char *)tline);
                  if (TPV) (*TPV)++;
                  JSLN(PV, TmpPepper, (unsigned char *)tline);
                }
                Typepepper[ti] = (char *)pjudy;
              }
              J1N(RC, Doload, ti);
            }
            fprintf(stderr, "-P: Loaded %d pepper(s) for selected types\n", pcnt);
          } else {
            char tline[MAXLINE + 16];
            tline[0] = 0;
            JSLF(PV, TmpPepper, (unsigned char *)tline);
            while (PV) {
              JSLI(PV, PepperArray, (unsigned char *)tline);
              JSLN(PV, TmpPepper, (unsigned char *)tline);
            }
            fprintf(stderr, "-P: Loaded %d pepper(s) into global PepperArray\n", pcnt);
          }
          JSLFA(RC, TmpPepper);
        }
        break;

      case '?':
        goto usage;

      default:
        printf("-h\tThe hash types:");
        for (x = 1; Types[x]; x++) printf("%s ", Types[x]);
        printf("\n");
        goto usage;
    }
  }
  /* -L tr/az and the GPU do not compose yet, so take the CPU and say so.
   *
   * The device UTF-32 walker carries the `turkish` parameter but every call
   * site passes 0 -- variant 0 only, by its own comment -- so a pinned
   * Turkish run would compute the tr mapping on the CPU and the default one
   * on the GPU.  The GPU claims the batch and the CPU does not recompute it,
   * so the difference would surface as a quiet under-recovery, which is the
   * one outcome this tool must not produce.
   *
   * -L c/root is variant 0 and therefore already exactly what the device
   * computes, so it runs on the GPU unchanged and costs nothing.
   *
   * Forced rather than refused: the run is correct on the CPU, and refusing
   * would turn a slower answer into no answer.  Loud, because losing GPU
   * acceleration silently is its own kind of wrong. */
  if (Utf32Locale == U32LOC_TR && !NoMetal) {
    fprintf(stderr,
        "-L tr/az: GPU disabled for this run. The device UTF-32 walker "
        "implements the default case mapping only, so leaving the GPU on "
        "would compute the Turkish mapping on the CPU and the default one on "
        "the GPU. Use -L c/root to keep the GPU, or accept CPU speed for "
        "Turkish output.\n");
    NoMetal = 1;
  }

  argc -= optind;
  argv += optind;
  J1T(RC, Dohash, JOB_ED2000);
  if (RC) {
    J1T(RC, Dohash, JOB_MD4);
    if (RC) {
      J1U(RC, Dohash, JOB_ED2000);
      fprintf(stderr, "MD4 and ED2K are the same - clearing ED2K\n");
    }
  }
  J1T(RC, Dohash, JOB_SHA1);
  if (RC) {
    J1T(RC, Dohash, JOB_AICH);
    if (RC) {
      J1U(RC, Dohash, JOB_AICH);
      fprintf(stderr, "SHA1 and AICH are the same - clearing AICH\n");
    }
  }
  J1T(RC, Dohash, JOB_MD6);
  if (RC) {
    J1T(RC, Dohash, JOB_MD6128);
    if (RC) {
      J1U(RC, Dohash, JOB_MD6128);
      fprintf(stderr, "MD6 and MD6128 are the same - clearing MD6128\n");
    }
  }
  J1T(RC, Dohash, JOB_SHA1RAW);
  if (RC) {
    J1T(RC, Dohash, JOB_SQL5);
    if (RC) {
      J1U(RC, Dohash, JOB_SHA1RAW);
      fprintf(stderr, "SHA1RAW and SQL5 are the same - clearing SHA1RAW\n");
    }
  }
  J1T(RC, Dohash, JOB_GOST2012_32);
  if (RC) {
    J1T(RC, Dohash, JOB_STREEBOG_32);
    if (RC) {
      J1U(RC, Dohash, JOB_GOST2012_32);
      fprintf(stderr, "GOST2012_32 and STREEBOG_32 are the same - clearing GOST2012_32\n");
    }
  }
  J1T(RC, Dohash, JOB_GOST2012_64);
  if (RC) {
    J1T(RC, Dohash, JOB_STREEBOG_64);
    if (RC) {
      J1U(RC, Dohash, JOB_GOST2012_64);
      fprintf(stderr, "GOST2012_64 and STREEBOG_64 are the same - clearing GOST2012_64\n");
    }
  }

  fprintf(stderr, "Working on hash types: ");
  for (y = 0, x = 1; Types[x]; x++) {
    TOTALFOUND(x - 1) = NULL;
    J1T(RC, Dohash, x);
    if (RC) {
      y++;
      fprintf(stderr, "%s ", Types[x]);
      TOTALFOUND(x - 1) = malloc_lock(sizeof(unsigned long long)*(Maxiter + 1),"totalfound");
      if (x == JOB_MD5SPECAM) {
        TOTALFOUND(JOB_MD5AM - 1) = malloc_lock(sizeof(unsigned long long)*(Maxiter + 1),"totalfound");
        TOTALFOUND(JOB_MD5AM2 - 1) = malloc_lock(sizeof(unsigned long long)*(Maxiter + 1),"totalfound");
      }
    }
  }
  /* User-defined hash types (Milestone 1): not in Types[], so handled in a
   * second pass over the synthetic-op range.  Same Totalfound allocation +
   * "Working on" listing so checkhash's per-op found counter is valid. */
  {
    int _uc = userdef_count(), _ui;
    for (_ui = 0; _ui < _uc; _ui++) {
      int _uop = (int)user_id_op(_ui);
      if (_uop >= JOB_DONE) break;
      TOTALFOUND(_uop - 1) = NULL;
      J1T(RC, Dohash, _uop);
      if (RC) {
        const char *_gs;
        struct userdef_type *_ut;
        y++;
        fprintf(stderr, "%s ", userdef_name(_uop));
        TOTALFOUND(_uop - 1) = malloc_lock(sizeof(unsigned long long)*(Maxiter + 1),"totalfound");
        /*
         * Sub-phase D4: honest GPU-eligibility status for the SELECTED
         * user type.  Enum-agnostic shape detection on the compiled
         * hx_program (no new flag -- piggybacks the invoke-time output).
         * GPU dispatch for user types is phase 2, so a supported shape
         * still reports "running on CPU".
         */
        _gs = userdef_gpu_status(_uop);
        _ut = userdef_get(_uop);
        if (_gs && _ut)
          fprintf(stderr, "\n-m u%s: %s\n", _ut->idstr, _gs);
      }
    }
  }
  fprintf(stderr, "\n");
  if (y == 0) {
    fprintf(stderr, "No hash types specified, exiting\n");
    exit(0);
  }

  line[0] = 0;
  JSLF(PV, PepperArray, (unsigned char *)line);
  MemSize = y = x = 0;
  while (PV) {
    x++;
    MemSize += mystrlen(line) + 2;
    JSLN(PV, PepperArray, (unsigned char *)line);
  }
  Numpeppers = x;
  if (Numpeppers) {
    char *s;
    s = combo_pepper = malloc_lock(MemSize + 4,"Peppers");
    line[0] = 0;
    JSLF(PV, PepperArray, (unsigned char *)line);
    while (PV) {
      x = mystrlen(line);
      if (x > 5 && strncmp(line, "$HEX[", 5) == 0) {
        y = get32(line + 5, (unsigned char *)(s + 1), (x - 6));
        if (y > 0 && y < 127) {
          *s++ = y;
          s += y;
          *s++ = 0;
        }
      } else {
        if (x <= 80) {
          *s++ = x;
          strcpy(s, line);
          s += x + 1;
        }
      }
      JSLN(PV, PepperArray, (unsigned char *)line);
    }
    *s++ = (char)255;
    *s = (char)255;
    fprintf(stderr, "%d total peppers in use\n", Numpeppers);
    Totsalts = Numsalts;
    JSLFA(RC, PepperArray);
  }


  /* Count salts provided by user (-s flag) */
  line[0] = 0;
  JSLF(PV, SaltArray, (unsigned char *)line);
  x = 0;
  while (PV) {
    x++;
    JSLN(PV, SaltArray, (unsigned char *)line);
  }
  Totsalts = x;  /* non-zero means user provided salts via -s */


  line[0] = 0;
  JSLF(PV, UserArray, (unsigned char *)line);
  MemSize = y = x = 0;
  while (PV) {
    x++;
    MemSize += mystrlen(line) + 1;
    JSLN(PV, UserArray, (unsigned char *)line);
  }
  J1T(RC, Dohash, JOB_MD5AM);
  if (RC == 0)
    J1T(RC, Dohash, JOB_MD5AM2);
  if (RC == 0)
    J1T(RC, Dohash, JOB_LEET);
  if (x) {
    Word_t *UPV;
    /* Build UseridJudy from UserArray with PV=1000000 */
    line[0] = 0;
    JSLF(PV, UserArray, (unsigned char *)line);
    while (PV) {
      x = mystrlen(line);
      if (x > 0 && x < 80) {
        JSLI(UPV, UseridJudy, (unsigned char *)line);
        if (UPV) *UPV = 1000000;
      }
      JSLN(PV, UserArray, (unsigned char *)line);
    }
    /* Count userids in UseridJudy */
    { int ucnt = 0;
      char uline[MAXLINE]; uline[0] = 0;
      Word_t *upv;
      JSLF(upv, UseridJudy, (unsigned char *)uline);
      while (upv) { ucnt++; JSLN(upv, UseridJudy, (unsigned char *)uline); }
      fprintf(stderr, "%d total userids in use\n", ucnt);
      Numsalts += ucnt;
    }
    JSLFA(RC, UserArray);
  }

  line[0] = 0;
  JSLF(PV, KeyArray, (unsigned char *)line);
  MemSize =y = x = 0;
  while (PV) {
    x++;
    MemSize += mystrlen(line) + 1;
    JSLN(PV, KeyArray, (unsigned char *)line);
  }
  if (x) {
    char *s;
    s = Keys = malloc_lock(MemSize + 4,"Keys");
    line[0] = 0;
    JSLF(PV, KeyArray, (unsigned char *)line);
    while (PV) {
      y = mystrlen(line);
      if (Hexkey) {
        y = get32(line, (unsigned char *)(s + 1), y);
        if (y > 255)
          y = 255;
        if (y > 0) {
          *s = y;
          s += y + 1;
        }
      } else {
        if (y > 5 && strncmp(line, "$HEX[", 5) == 0) {
          y = get32(line + 5, (unsigned char *)(s + 1), (y - 6));
          if (y > 255)
            y = 255;
          if (y > 0) {
            *s = y;
            s += y + 1;
          }
        } else {
          if (y > 255)
            y = 255;
          /* Pascal-style length-prefix copy: *s = y; bytes follow. No NUL. */
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
          strncpy(s + 1, line, y);
#pragma GCC diagnostic pop
          *s = y;
          s += y + 1;
        }
      }
      JSLN(PV, KeyArray, (unsigned char *)line);
    }
    *s++ = 0;
    *s = 0;
    /* s now points at the SECOND zero; the walker's break tests curkey[0] and
     * curkey[1], so the terminator it recognises starts one byte earlier. */
    Keys_end = s - 1;
    Keycount = x;
    /* Longest key, so the GPU pack site can reserve space for a whole word's
     * worth of (word, key) entries in ONE flush test and then emit them all
     * without a mid-loop flush.  Walk the finished stream rather than tracking
     * it during the build: the build has three branches (hex, $HEX[], plain)
     * and a fourth place to update is a fourth place to forget. */
    { const unsigned char *kw = (const unsigned char *)Keys;
      Keymaxlen = 0;
      while (!(kw[0] == 0 && kw[1] == 0)) {
        if ((int)kw[0] > Keymaxlen) Keymaxlen = (int)kw[0];
        kw += (size_t)kw[0] + 1;
      }
    }
    fprintf(stderr, "%d total keys in use\n", x);
    Numsalts += x;
    JSLFA(RC, KeyArray);
  }

  /* Build the length-prefixed Rules[] buffer from the rule store.
   *
   * Entry layout is UNCHANGED -- 2-byte length (= bytecode length + 1), then the
   * bytecode, then a NUL -- so the rule_ptrs walk, applyrule and the classifier
   * all keep working untouched.  Two things change: the bytecode is copied with
   * memcpy at its TRUE length instead of strcpy (which stopped at a 0x00 operand
   * and stored a short rule whose operand count still said N), and entries are
   * emitted in rs_ent order, which is INPUT-FILE order.  The old JudySL walk
   * emitted them in lexicographic order of the bytecode.
   *
   * The trailing NUL is retained deliberately: every in-tree bytecode walker
   * terminates on a 0 OPCODE and skips operands by explicit count, so a NUL
   * operand is never mistaken for the terminator. */
  x = Rulestore.n;
  MemSize = 0;
  for (y = 0; y < x; y++)
    MemSize += Rulestore.ent[y].len + 3;      /* 2 prefix + bytecode + NUL */
  if (x) {
    unsigned short int *si;
    size_t u32srcused = 0;
    Rulesize = MemSize + 4;
    RuleCnt = malloc_lock((x + 2)* sizeof(unsigned long long),"Rulecnt");
    /* Per-rule engine capability, 0-based like gpu_rule_membership[].  Every rule
     * that reaches here was accepted by the BYTE compiler -- packrules refusing it
     * is what sends it to the badrule path -- so byteok is 1 throughout and only
     * u32ok has to be computed.  That is why -8 still buys something on a file with
     * ZERO U32-REQUIRED rules: the dispatch that matters on real data is the
     * WORD's class, not the rule's. */
    Rule_byteok = malloc_lock(x + 2, "Rule_byteok");
    Rule_u32ok  = malloc_lock(x + 2, "Rule_u32ok");
    memset(Rule_byteok, 1, x + 2);
    memset(Rule_u32ok,  0, x + 2);
    if (Utf32Rules && Rulestore.srcused) {
      Rule_u32src    = malloc_lock(Rulestore.srcused + x + 2, "Rule_u32src");
      Rule_u32srcoff = malloc_lock((x + 2) * sizeof(uint32_t), "Rule_u32srcoff");
      memset(Rule_u32srcoff, 0, (x + 2) * sizeof(uint32_t));
    }
    d = Rules = malloc_lock(MemSize + 4,"Rules");
    /* Copy the retained source text out of the store, which rs_reset frees
     * below.  Indexed y + 1 so it lines up with RuleCnt[] and Ruleindex. */
    if (Rulestore.srcused) {
      Rulesrc    = malloc_lock(Rulestore.srcused + 1, "Rulesrc");
      Rulesrcoff = malloc_lock((x + 2) * sizeof(uint32_t), "Rulesrcoff");
      Rulesrclen = malloc_lock((x + 2) * sizeof(unsigned short), "Rulesrclen");
      memcpy(Rulesrc, Rulestore.src, Rulestore.srcused);
      Rulesrc[Rulestore.srcused] = 0;
      memset(Rulesrcoff, 0, (x + 2) * sizeof(uint32_t));
      memset(Rulesrclen, 0, (x + 2) * sizeof(unsigned short));
    }
    for (y = 0; y < x; y++) {
      int rlen = Rulestore.ent[y].len;
      si = (unsigned short int *) d;
      *si = rlen + 1;
      memcpy(d + 2, Rulestore.slab + Rulestore.ent[y].off, (size_t)rlen);
      d[2 + rlen] = 0;
      d += rlen + 1 + 2;
      if (Rulesrcoff) {
        Rulesrcoff[y + 1] = Rulestore.ent[y].srcoff;
        Rulesrclen[y + 1] = Rulestore.ent[y].srclen;
      }
        /* Ask the UTF-32 compiler about the SOURCE text, not the bytecode -- the
         * bytecode cannot be decompiled.  Only under -8: without it the byte engine is
         * the only engine and this answer is never read, and a second compile of every
         * rule in a 6.8-million-rule file is not free.  One compile per rule at LOAD,
         * never per (word, rule). */
        /* Terminated copy first, then ask the UTF-32 compiler about it.  The
         * bytecode cannot be decompiled, so the SOURCE is the only thing either
         * question can be asked of.  One compile per rule at LOAD, never per
         * (word, rule). */
        if (Rule_u32src && Rulestore.ent[y].srclen) {
          int sl = Rulestore.ent[y].srclen;
          char *dst = Rule_u32src + u32srcused;
          memcpy(dst, Rulestore.src + Rulestore.ent[y].srcoff, (size_t)sl);
          dst[sl] = 0;
          Rule_u32srcoff[y + 1] = (uint32_t)u32srcused;
          u32srcused += (size_t)sl + 1;
          Rule_u32ok[y] = (validrule_u32(dst) == 0) ? 1 : 0;
        }
        /* A placeholder payload marks a U32-REQUIRED rule: it has no byte bytecode,
         * so the byte engine must never be chosen for it.  Without this the ASCII
         * words took the byte engine onto the placeholder and produced NOTHING --
         * measured: 10 candidates missing against procrule on u32-ops.rule, every one
         * of them an ASCII word crossed with a UTF-32-only rule.
         *
         * 0x00 is never an opcode (checked over 338,310 rule lines; zero hits, lowest
         * first byte 0xc2) so the marker cannot collide with a real packed rule. */
        if (Rulestore.ent[y].len > 0 &&
            Rulestore.slab[Rulestore.ent[y].off] == 0)
          Rule_byteok[y] = 0;
    }
    *d++ = 0;
    *d++ = 0;
    y = d - Rules;
    if (y > Rulesize) {
      fprintf(stderr, "Rule processing failure: memory allocation failure\n");
      exit(1);
    }
    Rulesize = y;
    fprintf(stderr, "%d total rules in use\n", x);
    if (Utf32Rules) {
      /* Three classes, not two.  Counting only "u32ok" called every admitted
       * UTF-32-only rule "either engine", which is the opposite of true. */
      long both = 0, u32only = 0, byteonly = 0;
      for (y = 0; y < x; y++) {
        if (Rule_u32ok[y] && Rule_byteok[y])      both++;
        else if (Rule_u32ok[y])                   u32only++;
        else                                      byteonly++;
      }
      fprintf(stderr,
              "  -8: %ld rule(s) run on either engine, %ld UTF-32 only, "
              "%ld byte only\n", both, u32only, byteonly);
    }
    /* Only under -8, so byte-mode stderr is untouched. */
    if (Utf32Rules && (Rule_u32_only || Rule_both_bad))
      fprintf(stderr,
              "  -8: %ld rule(s) the byte engine refuses were ADMITTED for the "
              "UTF-32 engine, %ld bad in both and refused\n",
              Rule_u32_only, Rule_both_bad);
    Numrules = x;
    Numsalts += x;
    rs_reset(&Rulestore);
  }


	  J1T(RC, Dohash, JOB_SHA1_CUSTOMUSERSALT);
	  DoCustom = RC ? 1 : 0;
	  /* All non-hex formats load via -J/-F only */
	  x = 0;
	  J1T(RC, Dohash, JOB_MD5UCSALT);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_SHA1_MD5SALT);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_SHA1_MD5MD5SALT);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_SHA1_MD5UC_MD5SALT);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_MD5SALT);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_MD5MD5SALT);
	  if (x || RC) gen_salts();
	  /* Deferred default salts for hashcat modes 24/124/125: add only if
	   * -F didn't already load salts for these types */
	  { struct { int job; const char *salt; } hcdefaults[] = {
	      { JOB_MD5SALTPASS, "00" },          /* hashcat mode 24 */
	      { JOB_SHA1SALTPASS, "fe76b" },       /* hashcat mode 124 */
	      { JOB_ARUBAOS, "5387280701" },       /* hashcat mode 125 */
	      { 0, NULL }
	    };
	    int di;
	    for (di = 0; hcdefaults[di].salt; di++) {
	      int j = hcdefaults[di].job;
	      J1T(RC, Dohash, j);
	      if (RC && !TYPESALT(j) && !SaltArray) {
	        JSLI(PV, TYPESALT(j), (unsigned char *)hcdefaults[di].salt);
	        if (PV) *PV = 1000000;
	      }
	    }
	  }
	  init_default_salts();
	  x = 0;
	  J1T(RC, Dohash, JOB_MD5USERIDMD5);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_MD5MD5USER);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_MD5CAPMD5USER);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_MD5CAPMD5MD5USER);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_SHA1SHA1USER);
	  if (RC) x = 1;
	  J1T(RC, Dohash, JOB_MD5USERIDMD5MD5);
	  if (x || RC) gen_userid();
  J1T(RC, Dohash, JOB_MD5_MD5USERSHA1MD5PASS);
  if (RC) {
    char *d;
    char uline2[MAXLINE + 16];
    Pvoid_t uj = (Pvoid_t)UseridJudy;
    if (!uj) { gen_userid(); uj = (Pvoid_t)UseridJudy; }
    /* Count userids */
    x = 0;
    uline2[0] = 0;
    JSLF(PV, uj, (unsigned char *)uline2);
    while (PV) { x++; JSLN(PV, uj, (unsigned char *)uline2); }
    if (!x) x = 1;
    MD5_user = malloc_lock(33 * (x + 1), "MD5_user");
    d = MD5_user;
    uline2[0] = 0;
    JSLF(PV, uj, (unsigned char *)uline2);
    while (PV) {
      y = mystrlen(uline2);
      mymd5(uline2, y, curin.h);
      prmd5_32(curin.h, d);
      d += 33;
      JSLN(PV, uj, (unsigned char *)uline2);
    }
    *d++ = 0;
    *d++ = 0;
  }


  if (argc < 1) {
usage:
    printf("%s: %s\n", Prog, Version);
    printf("Search for MD5x values from a file\n");
    printf("use:  mdxfind [options] [list of text files] < input hash\n");
    printf("The text file stdin is special, and may be used if -f is\n");
    printf("supplied, to read candidate passwords from stdin\n");
    printf("-?\tShows this help message\n");
    printf("-h\tUse regex to select hash types, comma separated. OK to use multiple times\n");
    printf("-m\tSelect hash types by hashcat mode or internal index\n");
    printf("\t  -m 0          hashcat mode 0 (MD5)\n");
    printf("\t  -m 0,100      hashcat modes 0 (MD5) and 100 (SHA1)\n");
    printf("\t  -m e1         internal type e1 (MD5)\n");
    printf("\t  -m e1-e10     internal types e1 through e10\n");
    printf("\t  -m e1,1000    mix internal (e1=MD5) and hashcat (1000=NTLM)\n");
    printf("-a\tDo email address munging\n");
    printf("-E\tRules, masks and keys apply to the email local part only\n");
    printf("-b\tExpand each word into unicode, best effort\n");
    printf("-c\tReplace each special char (<>&, etc) with XML equivalents\n");
    printf("-d\tDe-duplicate wordlists, best effort\n");
    printf("-e\tExtended search for truncated hashes\n");
    printf("-p\tPrint source (filename) of found plaintexts\n");
    printf("-q\tInternal iteration counts for SHA1MD5x, and others\n");
    printf("-g\tRotate calculated hashes to attempt match to input hash\n");
    printf("-s\tFile to read salts from\n");
    printf("-S\tAs -s, for the types named by the preceding -M (global otherwise)\n");
    printf("-u\tFile to read Userid/Usernames from\n");
    printf("-U\tAs -u, for the types named by the preceding -M (global otherwise)\n");
    printf("-k\tFile to read suffixes from\n");
    printf("-x\tAs -k, but every line is raw hex with no $HEX[] wrapper\n");
    printf("-j\tFile of peppers for the GLOBAL pepper array (tried against every\n");
    printf("  \tselected type that takes one; not tied to any type)\n");
    printf("-P\tFile of peppers for the types named by the preceding -M.\n");
    printf("  \tWith no preceding -M, -P behaves exactly as -j\n");
    printf("-n\tAppend mask/digits to passwords (e.g. -n 2, -n 3x, -n '?l?d', -n '?[0-9a-f]?[0-9a-f]')\n");
    printf("-N\tPrepend mask/digits to passwords (same syntax as -n)\n");
    printf("-i\tThe number of iterations for each hash\n");
    printf("-t\tThe number of threads to run\n");
    printf("-G\tGPU device select: list, a comma-separated device list, or none\n");
    printf("-f\tfile to read hashes from, else stdin\n");
    printf("-F\tFile to read hashes with embedded salts (-M selects types)\n");
    printf("-M\tAs -m, and ALSO marks those types for loading by -F, -S, -U and -P.\n");
    printf("  \tMUST precede -F: -F parses its file at the moment getopt reaches\n");
    printf("  \tit, so a later -M cannot affect that file\n");
    printf("-J\tAs -F, but with no load-set restriction: every selected type is\n");
    printf("  \toffered the file. The name stdin reads standard input\n");
    printf("-l\tAppend CR/LF/CRLF and print in hex\n");
    printf("-r\tFile to read rules from (concatenated)\n");
    printf("-R\tFile to read rules from (dot-product form)\n");
    printf("-8\tInterpret rules in UTF-32 (codepoints and grapheme clusters)\n");
    printf("  \tas well as over bytes. The engine is chosen per (word,rule)\n");
    printf("  \tpair: an all-ASCII word runs on the byte engine either way\n");
    printf("-L\tPin the locale of an ambiguous case mapping. Implies -8.\n");
    printf("  \t  -L tr, -L az    Turkish/Azeri dotted and dotless I\n");
    printf("  \t  -L c, -L root   the mapping everyone else uses\n");
    printf("  \tomit -L to try the default mapping and count the rest\n");
    printf("-e\tExtended matching of hashes shorter than the digest: -e 0 whole\n");
    printf("  \thashes only, -e 1 standard windows (the default when such hashes\n");
    printf("  \tare loaded), -e 2 byte-by-byte. 1 and 2 are CPU-only\n");
    printf("-v\tDo not mark salts as found.\n");
    printf("-V\tDisplay version\n");
    printf("-Y\tLoad userdef.txt, print the load report, and exit\n");
    printf("-X\tXOR key applied to computed hashes. Accepts a bare string, or\n");
    printf("  \t$HEX[..] / HEX[..] for a hex key\n");
    printf("-w\tNumber of lines to skip from first wordlist\n");
    printf("-W\tETA display: -W <count>[K|M|G] set estimated lines,\n");
    printf("\t-W auto count in background (cached in mdxfind.db),\n");
    printf("\t-W auto:/path/to/db override cache location,\n");
    printf("\t-W none disable ETA (overrides MDXFIND_CACHE)\n");
    printf("\tMDXFIND_CACHE=/path/to/mdxfind.db enables -W auto automatically\n");
    printf("-y\tEnable directory recursion for wordlists\n");
    printf("-C\tContinue reporting a hash after it is solved, so every\n");
    printf("  \tcandidate that matches is reported and not just the first.\n");
    printf("  \tFor types whose stored digest is truncated, where many\n");
    printf("  \tplaintexts share one stored value. Implies -v. The per-type\n");
    printf("  \tcounts then count REPORTED LINES, not distinct hashes\n");
    printf("-z\tEnable debugging information/hash results\n");
    printf("-Z\tEnable histogram of rule hits\n");
    printf("\nUse -h with no other options to display the list of supported hash types\n");
    exit(1);
  }


  line[MAXLINE + 1] = 0;


  Dupcnt = 0;
  Maxdepth = 0;
  if (HashesLoaded == 0) {
    if (input == NULL) {
      fprintf(stderr, "Reading hash list from stdin...\n");
      input = gzdopen(0,"rb");
      if (!input) {
	 perror("stdin");
	 exit(1);
      }
      isstdin = 1;
    }

    load_hash_file(input, infilename ? infilename : "stdin", NULL,
                 /*hex_only=*/1);
    gzclose(input);
  } else {
    infilename = "previously named files";
  }

  RC = 0;
  if (inhashcnt) {
    possess(HashWaiting);
    wait_for(HashWaiting, TO_BE, 0);
    ToAdd.length = inhashcnt;
    ToAdd.work = (inhash) ? inhash2 : inhash1;
    twist(HashWaiting, BY, +1);
  }
  possess(HashWaiting);
  wait_for(HashWaiting, TO_BE, 0);
  ToAdd.length = -1;
  twist(HashWaiting, BY, +1);
  x = join_all();

  Minhashlen *= 2;
  /* -e 2 steps the extended windows one byte (two hex characters) at a time.
   * It is applied here, after loading, because addhash() takes its own copy
   * of Minhashlen when its thread starts -- before the options are parsed --
   * and writes the loaded minimum back at the end, which silently undid the
   * Minhashlen = 1 the bare -e used to set during parsing. */
  if (Extmatch == 2) Minhashlen = 2;
  current_utc_time(&current);
  wtime = (double) current.tv_sec + (double) (current.tv_nsec) / 1000000000.0;
  wtime -= (double) starthash.tv_sec + (double) (starthash.tv_nsec) / 1000000000.0;
  if (wtime <= 0) wtime = 0.0;
  tsfprintf(stderr, "Took %.2f seconds to read hashes\n", wtime);

  /* Build compact hash table from flat data */
  /* Extended matching and the pre-check.  Minhashlen is now the shortest
   * loaded hash in hex characters.  A selected type whose compare length is
   * longer than that is matched through the extended windows, which only the
   * CPU implements -- as are the XOR key and rotation -- so any of the three
   * keeps the run off the GPU. */
  {
    unsigned maxdig = 0, maxop = 0, op, d;
    int ext_active;
    for (op = 1; op < JOB_DONE; op++) {
      J1T(RC, Dohash, op);
      if (!RC) continue;
      d = TYPEDIGEST_LEN(typedigest(op));
      if (d > maxdig) { maxdig = d; maxop = op; }
    }
    ext_active = (Extmatch == 2) || (Extmatch == 1 && maxdig && Minhashlen < 2 * maxdig);
    if (ext_active && !Extmatch_set)
      fprintf(stderr, "Extended matching: hashes as short as %u hex characters are loaded, "
              "shorter than the %u-character %s digest. The GPU is not used and CPU "
              "search is slower; -e 0 matches whole hashes only.\n",
              Minhashlen, 2 * maxdig, TYPENAME(maxop));
    Precheck_off = (XOR_key_len || Rotatehash || Printall);
    if (ext_active || XOR_key_len || Rotatehash) {
#ifdef GPU_ENABLED
      if (!NoMetal)
        fprintf(stderr, "GPU disabled (%s is CPU-only)\n",
                XOR_key_len ? "the XOR key" : Rotatehash ? "hash rotation" : "extended matching");
#endif
      NoMetal = 1;
    }
  }
  build_compact_table();
  /* Fold the pre-check's validity into one threshold now that the compact
   * table exists: off for -X, -g and -z, and with no table; otherwise valid up
   * to the shortest loaded hash, or for any length under -e 0, where checkhash
   * looks at offset 0 only. */
  Precheck_maxlen = (Precheck_off || !CompactFP) ? -1
                  : (Extmatch == 0) ? 0x7fffffff : (int)Minhashlen;

#ifdef OPENCL_GPU
  /* GPU rule engine — Layer 3 sub-commit A: classify the loaded rules
   * into GPU-eligible vs CPU-only and upload the GPU subset to each
   * device's __constant rule_program buffer. The kernel + classifier
   * are already validated at 792/792 byte-exact and 99.99% HashMob.100k
   * coverage; here we wire in the classification pass + per-device
   * upload. Dispatch routing comes in sub-commits B/C/D.
   *
   * Placement: after build_compact_table(), because gpu_opencl_init()
   * runs inside build_compact_table() — the rule-load block at line
   * ~42437 fires before any GPU device exists, so set_rules() must
   * happen here instead. Rules and Numrules are stable by this point. */
  if (gpu_opencl_available()) {
   if (Numrules == 0) {
    /* Numrules == 0 path: synthesize a 1-byte rule program containing
     * just `\0`. The kernel reads the leading NUL → is_no_rule = 1 →
     * runs MD5+probe on the original word once (no rule applied). This
     * is the implicit no-rule pass, foundational mdxfind semantics
     * (feedback_no_rule_pass.md). Routes the iter-only / unsalted-no-
     * rules-no-mask case through the rules-engine kernel, replacing
     * the older slab path that hit CL_INVALID_KERNEL_ARGS /
     * CL_OUT_OF_RESOURCES on Win+NVIDIA when mask_desc / salt_off /
     * salt_len were never bound (GTX 1650 user crash). */
    unsigned char *prog = malloc(1);
    uint32_t      *offs = malloc(sizeof(uint32_t));
    int           *slot = malloc(sizeof(int));
    if (prog && offs && slot) {
      prog[0] = 0;
      offs[0] = 0;
      slot[0] = 0;
      int n_dev = gpu_opencl_num_devices();
      int n_uploaded = 0;
      {
        struct srl_arg srl_args[64];
        thread *srl_threads[64];
        int nlaunched = (n_dev > 64) ? 64 : n_dev;
        for (int dev = 0; dev < nlaunched; dev++) {
          srl_args[dev].di       = dev;
          srl_args[dev].prog     = prog;
          srl_args[dev].prog_len = 1;
          srl_args[dev].offs     = offs;
          srl_args[dev].n_rules  = 1;
          srl_args[dev].rc       = -1;
          srl_threads[dev] = launch(srl_thread_fn, &srl_args[dev]);
        }
        for (int dev = 0; dev < nlaunched; dev++) join(srl_threads[dev]);
        for (int dev = 0; dev < nlaunched; dev++)
          if (srl_args[dev].rc == 0) n_uploaded++;
      }
      if (n_uploaded > 0) {
        gpu_rule_program       = prog;
        gpu_rule_offsets       = offs;
        gpu_rule_slot        = slot;
        gpu_rule_program_len   = 1;
        gpu_rule_count         = 1;       /* synthetic `:` only */
        gpu_rule_membership    = NULL;    /* no Numrules array — not needed when 0 user rules */
        gpu_legacy_slot_unused = 1;       /* no CPU rules to walk; gate on Numrules in procjob */
        gpu_rules_engine_active = 1;
        /* synthetic-only no-rule pass upload progress emit retired 2026-05-09 —
         * uniform behavior post-slab-retirement; the upload-failure paths below
         * still emit since those represent actual user-visible problems. */
        (void)n_uploaded;
      } else {
        free(prog); free(offs); free(slot);
        fprintf(stderr,
          "GPU rule engine: synthetic-only program upload failed on all %d device(s) — "
          "no GPU acceleration for this workload\n", n_dev);
      }
    } else {
      free(prog); free(offs); free(slot);
      fprintf(stderr,
        "GPU rule engine: malloc failed for synthetic-only program — "
        "no GPU acceleration for this workload\n");
    }
   } else {
    /* Walk the length-prefixed Rules buffer to build a char** array
     * of bytecode pointers (skipping the 2-byte length prefix on each
     * entry). classify_rules wants pointers to the raw bytecode. */
    char **rule_ptrs = malloc(Numrules * sizeof(char *));
    unsigned short *rule_lens = malloc(Numrules * sizeof(unsigned short));
    if (rule_ptrs && rule_lens) {
      char *rp = Rules;
      int rn = 0;
      while (rn < Numrules) {
        unsigned short int rlen = *((unsigned short int *)rp);
        if (rlen == 0) break;       /* end-of-rules sentinel */
        rule_lens[rn]   = (unsigned short)(rlen - 1);  /* bytecode, no NUL */
        rule_ptrs[rn++] = rp + 2;   /* skip length prefix -> bytecode */
        rp += rlen + 2;
      }
      if (rn == Numrules) {
        struct rule_lists rl;
        /* Counted BEFORE the guard below, and the guard admits n_u32only on its own.
         * A rule file whose every line is U32-REQUIRED leaves rl.ngpu == 0 -- every
         * rule carries the 0x00 placeholder, which gpu_rule_safe_phase0 refuses -- so
         * a guard of rl.ngpu > 0 alone skipped this whole block and sent NOTHING to
         * the device, including the very rules this change exists to admit. */
        int n_u32only = 0;
        if (Utf32Rules && Rule_u32ok && Rule_byteok) {
          for (unsigned y = 0; y < Numrules; y++)
            if (!Rule_byteok[y] && Rule_u32ok[y]) n_u32only++;
        }
        if (classify_rules(rule_ptrs, rule_lens, Numrules, &rl) >= 0 && (rl.ngpu > 0 || n_u32only > 0)) {
          /* Pack the GPU-eligible subset into a contiguous bytecode
           * buffer + uint32 offset table.  Copied at the rule's TRUE length
           * plus its trailing NUL -- strlen would stop at a 0x00 operand and
           * hand the device a truncated rule whose operand count still said N,
           * which the kernel would then satisfy from the next rule's bytes. */
          uint32_t prog_cap = 0;
          for (int i = 0; i < rl.ngpu; i++) {
            prog_cap += (uint32_t)rl.gpulen[i] + 1;
          }
          prog_cap += (uint32_t)n_u32only * 2u;
          prog_cap += 1;
          unsigned char *prog = malloc(prog_cap);
          uint32_t      *offs = malloc((rl.ngpu + 1 + n_u32only) * sizeof(uint32_t));
          int           *slot = malloc((rl.ngpu + 1 + n_u32only) * sizeof(int));
          if (prog && offs && slot) {
            uint32_t pos = 0;
            for (int i = 0; i < rl.ngpu; i++) {
              offs[i] = pos;
              size_t blen = (size_t)rl.gpulen[i] + 1;   /* + trailing NUL */
              memcpy(prog + pos, rl.gpu[i], blen);
              pos += (uint32_t)blen;
               /* Original Rules[] index, straight from classify_rules, which
                * preserves order and records it.  This used to be recovered by
                * restarting a pointer scan at 0 for every entry -- about 5e9
                * compares at HashMob.100k scale and 2.3e13 at Hash-IT. */
               slot[i] = rl.gpuidx[i] + 1;   /* 1-BASED: RuleCnt[] index */
            }
            /* Synthetic `:` no-rule pass at gpu_rule_idx == rl.ngpu.
             * Empty bytecode (one NUL byte). slot sentinel = 0 so
             * the worker hit decoder knows to skip applyrule replay
             * and use the original word as plaintext directly. */
            offs[rl.ngpu] = pos;
            prog[pos++] = 0;
            slot[rl.ngpu] = 0;

            /* The U32-REQUIRED entries go AFTER the synthetic `:`, which stays at index
             * rl.ngpu exactly where it is today so nothing that special-cases it by
             * position moves.
             *
             * The byte stream is RULE_OP_NOOP then the terminator, NOT a bare NUL.  The
             * kernel detects the synthetic pass as (rule_program[byte_roff] == 0); a bare
             * NUL would make one of these indistinguishable from it, bypassing the no-op
             * detection and crediting the rule for the no-rule pass -- the -Z inflation
             * this codebase has already fixed once.  A two-byte no-op is also harmless if
             * the byte walker is ever handed the rule, which bit 30 says it must not be. */
            int n_devrules = rl.ngpu + 1;
            if (n_u32only) {
              for (unsigned y = 0; y < Numrules; y++) {
                if (Rule_byteok[y] || !Rule_u32ok[y]) continue;
                offs[n_devrules]   = pos;
                prog[pos++]   = (unsigned char)RULE_OP_NOOP;
                prog[pos++]   = 0;
                slot[n_devrules]   = (int)y + 1;   /* 1-BASED, as rl.gpuidx[i] + 1 is */
                n_devrules++;
              }
            }
            int n_rules_with_synth = n_devrules;
            /* Upload to every available GPU device — Memo C parallel.
             * Per-device clEnqueueWriteBuffer + bookkeeping fanned across
             * yarn threads; rule data is shared read-only. */
            int n_dev = gpu_opencl_num_devices();
            int n_uploaded = 0;
            {
              struct srl_arg srl_args[64];
              thread *srl_threads[64];
              int nlaunched = (n_dev > 64) ? 64 : n_dev;
              for (int dev = 0; dev < nlaunched; dev++) {
                srl_args[dev].di       = dev;
                srl_args[dev].prog     = prog;
                srl_args[dev].prog_len = pos;
                srl_args[dev].offs     = offs;
                srl_args[dev].n_rules  = n_rules_with_synth;
                srl_args[dev].rc       = -1;
                srl_threads[dev] = launch(srl_thread_fn, &srl_args[dev]);
              }
              for (int dev = 0; dev < nlaunched; dev++) join(srl_threads[dev]);
              for (int dev = 0; dev < nlaunched; dev++)
                if (srl_args[dev].rc == 0) n_uploaded++;
            }
            if (n_uploaded > 0) {
              gpu_rule_program     = prog;
              gpu_rule_offsets     = offs;
              gpu_rule_slot      = slot;
              gpu_rule_program_len = pos;
              gpu_rule_count       = n_rules_with_synth;  /* includes synthetic `:` */
              gpu_rule_membership = calloc(Numrules, 1);
              if (gpu_rule_membership) {
                  int member_count = 0;
                  for (int i = 0; i < n_rules_with_synth; i++) {
                      if (slot[i] >= 1 && slot[i] <= (int)Numrules) {
                          gpu_rule_membership[slot[i] - 1] = 1;
                          member_count++;
                  /* -8: a rule with no byte bytecode must never reach the
                   * device byte walker.  It cannot today -- such a rule carries
                   * a 0x00 placeholder payload, gpu_rule_safe_phase0() has no
                   * case for 0x00 and its default returns 0, so the rule
                   * classifies CPU-only and never enters rl.gpu.  This says so
                   * out loud instead of leaving the property incidental, since
                   * the cost of it silently ceasing to hold is the device
                   * executing source text as bytecode. */
                  /* A GPU-eligible rule with no byte bytecode is now the INTENDED state for a
                   * U32-REQUIRED rule, so warn only when NEITHER engine can run it -- which is
                   * the genuinely broken case.  The old form fired once per U32-only rule, 14
                   * times per run on the mixed fixture. */
                  if (Rule_byteok && Rule_u32ok &&
                      !Rule_byteok[slot[i] - 1] && !Rule_u32ok[slot[i] - 1])
                      fprintf(stderr,
                        "%s WARNING rule %d is GPU-eligible but neither engine "
                        "can run it\n", "GPU rule engine:", slot[i]);
                      }
                  }
                  /* Sanity: member_count must equal rl.ngpu if slot[] map is correct.
                   * Note: synthetic `:` (slot[rl.ngpu] = 0) is intentionally NOT
                   * counted — it has no Rules[] entry to mark. */
                  if (member_count != rl.ngpu + n_u32only)
                      fprintf(stderr,
                        "GPU rule engine: WARNING membership count %d != rl.ngpu %d\n",
                        member_count, rl.ngpu);
              }
              /* AFTER the membership fill, not before it.  This call clears membership for
               * any admitted rule the device walker cannot run -- a tier-3 string-operand
               * rule -- and it reads gpu_rule_membership to do so.  Placed before the
               * calloc above, the pointer was still NULL, the clawback guard skipped
               * SILENTLY, and the pre-filter reported success having changed nothing:
               * syntax.md5 under MODE=1 gave 8,446 of 13,435, the 6 of 14 tier-3 rules
               * walked by neither side.  Still long before any word is read. */
              int u32_clawed = gpu_u32_membership_prefilter(prog, pos, offs,
                                                          n_rules_with_synth);
              if (u32_clawed < 0) u32_clawed = 0;   /* -1 = could not apply */
              /* + u32_clawed, and this is not bookkeeping.  The flag gates the
               * currule = NULL short-circuit: when set, the outer do/while exits after the
               * no-rule pass and THE CPU RULE WALK DOES NOT HAPPEN.  Correct only while
               * every rule really is on the device.  The clawback above hands the tier-3
               * string-operand rules BACK to the CPU, so without counting them here the
               * short-circuit stops the CPU walking rules the device also skipped --
               * measured 8,446 of 13,435 on u32-ops.rule, and appending one byte-refused
               * rule (v2-) to raise rl.ncpu made the same run byte-exact, which is what
               * isolated it. */
              /* If all original rules are GPU-eligible, the legacy jobg slot
               * (my_jobg in the chokepoint) is never used: the no-rule pass
               * is now handled by the synthetic `:` on the GPU, and there
               * are no CPU-only rules to expand. Slot allocations can use
               * the smaller GPUBATCH_RULES_*_SIZE constants for ~96x memory
               * savings on the typical HashMob workload. */
              /* A U32-REQUIRED rule sits in rl.cpu ONLY because its placeholder is not byte
               * bytecode.  Now that it runs on the device it is not CPU residue, and if it
               * still counted as such it would hold gpu_legacy_slot_unused at 0 and disable
               * the currule = NULL short-circuit -- measured at roughly half the wall time. */
              gpu_legacy_slot_unused = ((rl.ncpu - n_u32only + u32_clawed) == 0) ? 1 : 0;
              gpu_rules_engine_active = 1;
              tsfprintf(stderr,
                "GPU rule engine: %d/%d rules eligible (%.1f%%), "
                "%u-byte program uploaded to %d/%d device(s)%s\n",
                rl.ngpu, Numrules,
                100.0 * rl.ngpu / Numrules,
                pos, n_uploaded, n_dev,
                gpu_legacy_slot_unused ? "; legacy jobg slot bypass enabled" : "");
            } else {
              /* All uploads failed — likely __constant size limit hit
               * (HashMob.100k ~500 KB > 64 KB on most NVIDIA). Free
               * the host-side copies; sub-commit C will fall back to
               * CPU rule expansion when gpu_rule_count == 0. */
              free(prog); free(offs); free(slot);
              fprintf(stderr,
                "GPU rule engine: %d eligible rules but upload failed "
                "on all %d device(s) — falling back to CPU rule path "
                "(rule program %u bytes likely exceeds device "
                "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)\n",
                rl.ngpu, n_dev, pos);
            }
          } else {
            free(prog); free(offs); free(slot);
            fprintf(stderr,
              "GPU rule engine: malloc failed for %d-rule program — "
              "falling back to CPU rule path\n", rl.ngpu);
          }
          rule_lists_free(&rl);
        }
      }
      free(rule_ptrs);
      free(rule_lens);
    }
   }  /* end Numrules > 0 branch */
  }  /* end gpu_opencl_available() */

  /* Pre-load overflow hashes to every GPU device NOW, before any
   * procjob thread launches. The worker thread used to do this lazily
   * on first dispatch; under heavy CPU contention from procjobs that
   * inflated the upload from microseconds to seconds (3s observed on
   * fpga 16-core, ~30s suspected on mmt 64-core via GAP DEBUG). Doing
   * it at session start eliminates the contention. */
  if (gpu_opencl_available()) {
    gpujob_overflow_preload_all();
  }
#endif

#if defined(__APPLE__) && defined(METAL_GPU)
  /* Phase 2a-cont: Metal-side rule globals init. Mirrors the OpenCL block
   * above (~44982) but strips the per-device upload threads — Metal is
   * single-device and lazy-uploads from gpu_metal.m on first dispatch. */
  if (gpu_metal_available()) {
    if (Numrules == 0) {
      /* Synthetic 1-byte no-rule pass; same semantics as OpenCL synth arm. */
      unsigned char *prog = malloc(1);
      uint32_t      *offs = malloc(sizeof(uint32_t));
      int           *slot = malloc(sizeof(int));
      if (prog && offs && slot) {
        prog[0] = 0;
        offs[0] = 0;
        slot[0] = 0;
        gpu_rule_program        = prog;
        gpu_rule_offsets        = offs;
        gpu_rule_slot         = slot;
        gpu_rule_program_len    = 1;
        gpu_rule_count          = 1;
        gpu_rule_membership     = NULL;
        gpu_legacy_slot_unused  = 1;
        gpu_rules_engine_active = 1;
      } else {
        free(prog); free(offs); free(slot);
      }
    } else {
      /* Numrules > 0: classify + pack the GPU-eligible subset. */
      char **rule_ptrs = malloc(Numrules * sizeof(char *));
      unsigned short *rule_lens = malloc(Numrules * sizeof(unsigned short));
      if (rule_ptrs && rule_lens) {
        char *rp = Rules;
        int rn = 0;
        while (rn < Numrules) {
          unsigned short int rlen = *((unsigned short int *)rp);
          if (rlen == 0) break;
          rule_lens[rn]   = (unsigned short)(rlen - 1);  /* bytecode, no NUL */
          rule_ptrs[rn++] = rp + 2;
          rp += rlen + 2;
        }
        if (rn == Numrules) {
          struct rule_lists rl;
          /* Counted BEFORE the guard below, and the guard admits n_u32only on its own.
           * A rule file whose every line is U32-REQUIRED leaves rl.ngpu == 0 -- every
           * rule carries the 0x00 placeholder, which gpu_rule_safe_phase0 refuses -- so
           * a guard of rl.ngpu > 0 alone skipped this whole block and sent NOTHING to
           * the device, including the very rules this change exists to admit. */
          int n_u32only = 0;
          if (Utf32Rules && Rule_u32ok && Rule_byteok) {
            for (unsigned y = 0; y < Numrules; y++)
              if (!Rule_byteok[y] && Rule_u32ok[y]) n_u32only++;
          }
          if (classify_rules(rule_ptrs, rule_lens, Numrules, &rl) >= 0 && (rl.ngpu > 0 || n_u32only > 0)) {
            uint32_t prog_cap = 0;
            for (int i = 0; i < rl.ngpu; i++) prog_cap += (uint32_t)rl.gpulen[i] + 1;
            prog_cap += (uint32_t)n_u32only * 2u;
            prog_cap += 1; /* synthetic `:` */
            unsigned char *prog = malloc(prog_cap);
            uint32_t      *offs = malloc((rl.ngpu + 1 + n_u32only) * sizeof(uint32_t));
            int           *slot = malloc((rl.ngpu + 1 + n_u32only) * sizeof(int));
            if (prog && offs && slot) {
              uint32_t pos = 0;
              for (int i = 0; i < rl.ngpu; i++) {
                offs[i] = pos;
                size_t blen = (size_t)rl.gpulen[i] + 1;   /* + trailing NUL */
                memcpy(prog + pos, rl.gpu[i], blen);
                pos += (uint32_t)blen;
                slot[i] = rl.gpuidx[i] + 1;   /* 1-BASED: RuleCnt[] index */
              }
              offs[rl.ngpu] = pos;
              prog[pos++] = 0;
              slot[rl.ngpu] = 0;
              /* The U32-REQUIRED entries go AFTER the synthetic `:`, which stays at index
               * rl.ngpu exactly where it is today so nothing that special-cases it by
               * position moves.
               *
               * The byte stream is RULE_OP_NOOP then the terminator, NOT a bare NUL.  The
               * kernel detects the synthetic pass as (rule_program[byte_roff] == 0); a bare
               * NUL would make one of these indistinguishable from it, bypassing the no-op
               * detection and crediting the rule for the no-rule pass -- the -Z inflation
               * this codebase has already fixed once.  A two-byte no-op is also harmless if
               * the byte walker is ever handed the rule, which bit 30 says it must not be. */
              int n_devrules = rl.ngpu + 1;
              if (n_u32only) {
                for (unsigned y = 0; y < Numrules; y++) {
                  if (Rule_byteok[y] || !Rule_u32ok[y]) continue;
                  offs[n_devrules]   = pos;
                  prog[pos++]   = (unsigned char)RULE_OP_NOOP;
                  prog[pos++]   = 0;
                  slot[n_devrules]   = (int)y + 1;   /* 1-BASED, as rl.gpuidx[i] + 1 is */
                  n_devrules++;
                }
              }
              int n_rules_with_synth = n_devrules;
              gpu_rule_program     = prog;
              gpu_rule_offsets     = offs;
              gpu_rule_slot      = slot;
              gpu_rule_program_len = pos;
              gpu_rule_count       = n_rules_with_synth;
              gpu_rule_membership  = calloc(Numrules, 1);
              if (gpu_rule_membership) {
                for (int i = 0; i < n_rules_with_synth; i++) {
                  if (slot[i] >= 1 && slot[i] <= (int)Numrules) {
                    gpu_rule_membership[slot[i] - 1] = 1;
                    /* Same guard as the OpenCL block above; see it for why. */
                    /* A GPU-eligible rule with no byte bytecode is now the INTENDED state for a
                     * U32-REQUIRED rule, so warn only when NEITHER engine can run it -- which is
                     * the genuinely broken case.  The old form fired once per U32-only rule, 14
                     * times per run on the mixed fixture. */
                    if (Rule_byteok && Rule_u32ok &&
                        !Rule_byteok[slot[i] - 1] && !Rule_u32ok[slot[i] - 1])
                        fprintf(stderr,
                          "%s WARNING rule %d is GPU-eligible but neither engine "
                          "can run it\n", "Metal: rule engine", slot[i]);
                  }
                }
              }
              /* AFTER the membership fill, not before it.  This call clears membership for
               * any admitted rule the device walker cannot run -- a tier-3 string-operand
               * rule -- and it reads gpu_rule_membership to do so.  Placed before the
               * calloc above, the pointer was still NULL, the clawback guard skipped
               * SILENTLY, and the pre-filter reported success having changed nothing:
               * syntax.md5 under MODE=1 gave 8,446 of 13,435, the 6 of 14 tier-3 rules
               * walked by neither side.  Still long before any word is read. */
              int u32_clawed = gpu_u32_membership_prefilter(prog, pos, offs,
                                                          n_rules_with_synth);
              if (u32_clawed < 0) u32_clawed = 0;   /* -1 = could not apply */
              /* + u32_clawed, and this is not bookkeeping.  The flag gates the
               * currule = NULL short-circuit: when set, the outer do/while exits after the
               * no-rule pass and THE CPU RULE WALK DOES NOT HAPPEN.  Correct only while
               * every rule really is on the device.  The clawback above hands the tier-3
               * string-operand rules BACK to the CPU, so without counting them here the
               * short-circuit stops the CPU walking rules the device also skipped --
               * measured 8,446 of 13,435 on u32-ops.rule, and appending one byte-refused
               * rule (v2-) to raise rl.ncpu made the same run byte-exact, which is what
               * isolated it. */
              /* A U32-REQUIRED rule sits in rl.cpu ONLY because its placeholder is not byte
               * bytecode.  Now that it runs on the device it is not CPU residue, and if it
               * still counted as such it would hold gpu_legacy_slot_unused at 0 and disable
               * the currule = NULL short-circuit -- measured at roughly half the wall time. */
              gpu_legacy_slot_unused  = ((rl.ncpu - n_u32only + u32_clawed) == 0) ? 1 : 0;
              gpu_rules_engine_active = 1;
              fprintf(stderr,
                "Metal: rule engine — %d/%d rules GPU-eligible, "
                "%u-byte program, %d total kernel rules (incl. synthetic ':')\n",
                rl.ngpu, (int)Numrules, pos, n_rules_with_synth);
            } else {
              free(prog); free(offs); free(slot);
            }
          }
          /* classify_rules allocates rl.gpu/rl.cpu internally — they're
           * stable pointers into Rules, no free needed. */
        }
        free(rule_ptrs);
        free(rule_lens);
      }
    }
    /* Pre-upload overflow chain to GPU — mirrors the OpenCL arm above. */
    gpujob_overflow_preload_all();
  }
#endif

  free(inhash1);
  free(inhash2);



  tsfprintf(stderr, "Searching through %s unique hex hashes from %s\n", commify(Insize), infilename);
  Bcryptcnt = 0;
  DEScryptcnt = 0;
  BSDIcryptcnt = 0;
  ProgressEnccnt = 0;
  {
    line[0] = 0;
    JSLF(PV, JUDYJ(JOB_PROGRESSENCODE), (unsigned char *)line);
    while (PV) {
      ProgressEnccnt++;
      JSLN(PV, JUDYJ(JOB_PROGRESSENCODE), (unsigned char *)line);
    }
    if (ProgressEnccnt) {
      fprintf(stderr, "Searching through %s unique PROGRESSENCODE hashes\n", commify(ProgressEnccnt));
    }

    line[0] = 0;
    JSLF(PV, JUDYJ(JOB_DESCRYPT),(unsigned char *)line);
    while (PV) {
      DEScryptcnt++;
      JSLN(PV, JUDYJ(JOB_DESCRYPT),(unsigned char *)line);
    }
    if (DEScryptcnt) {
      /* Standard descrypt only: all entries are 13-char/2-char-salt */
      int des_types[] = { JOB_DESCRYPT, JOB_MD5DESCRYPT, JOB_MD4DESCRYPT,
                          JOB_SHA1DESCRYPT, JOB_MD4UTF16DESCRYPT,
                          JOB_MD5USERDESCRYPT, 0 };
      int dt;
      Pvoid_t salt_counts = NULL;
      Word_t *SPV;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_DESCRYPT),(unsigned char *)line);
      while (PV) {
        y = mystrlen(line);
        if (y >= 10) {
          strncpy(last, line, 2);
          last[2] = 0;
          JSLI(SPV, salt_counts, (unsigned char *)last);
          if (SPV) (*SPV)++;
        }
        JSLN(PV, JUDYJ(JOB_DESCRYPT),(unsigned char *)line);
      }
      last[0] = 0;
      JSLF(PV, salt_counts, (unsigned char *)last);
      while (PV) {
        for (dt = 0; des_types[dt]; dt++) {
          J1T(RC, Dohash, des_types[dt]);
          if (RC) {
            JSLI(SPV, TYPESALT(des_types[dt]), (unsigned char *)last);
            if (SPV) *SPV += *PV;
          }
        }
        Numsalts++;
        JSLN(PV, salt_counts, (unsigned char *)last);
      }
      JSLFA(RC, salt_counts);
      fprintf(stderr, "Searching through %s unique DESCRYPT hashes\n", commify(DEScryptcnt));
    } else if (Printall) {
      /* -z mode: generate 4096 standard 2-char DES salts */
      int des_types[] = { JOB_DESCRYPT, JOB_MD5DESCRYPT, JOB_MD4DESCRYPT,
                          JOB_SHA1DESCRYPT, JOB_MD4UTF16DESCRYPT,
                          JOB_MD5USERDESCRYPT, 0 };
      int dt;
      char dsalt[4];
      for (dt = 0; des_types[dt]; dt++) {
        J1T(RC, Dohash, des_types[dt]);
        if (RC && !TYPESALT(des_types[dt])) {
          for (x = 0; x < 64; x++) {
            for (y = 0; y < 64; y++) {
              dsalt[0] = phpitoa64[x]; dsalt[1] = phpitoa64[y]; dsalt[2] = 0;
              Word_t *TPV;
              JSLI(TPV, TYPESALT(des_types[dt]), (unsigned char *)dsalt);
              if (TPV) *TPV = 1000000;
            }
          }
          Numsalts += 4096;
          fprintf(stderr, "Generated 4096 standard DES salts for %s\n", Types[des_types[dt]]);
        }
      }
    }

    /* MD5USERDESCRYPT (e1060) defaults.  The DES block above seeds 2-char salts
     * for every selected DES-family type, but only from salts it can see: the
     * prefixes of loaded DESCRYPT records, or the generated 4096 under -z.  This
     * type is normally reached with a plain 32-hex digest and NO descrypt record
     * anywhere in the input, so neither source fires and it would run with an
     * empty salt table and find nothing.  Waffle's specification: with no salts
     * supplied, generate the default DESCRYPT range; with no users supplied,
     * default to root.
     *
     * Deliberately NOT gated on Printall, which is the difference from the block
     * above -- -m e1060 with a plain hex file has to work, not just -z.  The cost
     * is visible in the receipt because 4096 salts means 4096 descrypt operations
     * per candidate, and descrypt is the expensive half of this construction.
     *
     * Both defaults are skipped when the channel already supplied the data, so an
     * -F record carrying its own salt and user is never overridden. */
    { J1T(RC, Dohash, JOB_MD5USERDESCRYPT);
      if (RC) {
        if (!TYPESALT(JOB_MD5USERDESCRYPT)) {
          char dsalt[4];
          Word_t *TPV;
          for (x = 0; x < 64; x++) {
            for (y = 0; y < 64; y++) {
              dsalt[0] = phpitoa64[x]; dsalt[1] = phpitoa64[y]; dsalt[2] = 0;
              JSLI(TPV, TYPESALT(JOB_MD5USERDESCRYPT), (unsigned char *)dsalt);
              if (TPV) *TPV = 1000000;
            }
          }
          Numsalts += 4096;
          fprintf(stderr, "Generated 4096 standard DES salts for %s\n",
                  TYPENAME(JOB_MD5USERDESCRYPT));
        }
        if (!TYPEUSER(JOB_MD5USERDESCRYPT) && !UseridJudy) {
          Word_t *UPV;
          JSLI(UPV, TYPEUSER(JOB_MD5USERDESCRYPT), (unsigned char *)"root");
          if (UPV) *UPV = 1000000;
          fprintf(stderr, "Using default user 'root' for %s\n",
                  TYPENAME(JOB_MD5USERDESCRYPT));
        }
      }
    }

    /* BSDICRYPT bootstrap: BSDi extended DES (_CCCCSSSS 9-char salt) */
    line[0] = 0;
    JSLF(PV, JUDYJ(JOB_BSDICRYPT),(unsigned char *)line);
    while (PV) {
      BSDIcryptcnt++;
      JSLN(PV, JUDYJ(JOB_BSDICRYPT),(unsigned char *)line);
    }
    if (BSDIcryptcnt) {
      fprintf(stderr, "Searching through %s unique BSDICRYPT hashes\n", commify(BSDIcryptcnt));
    } else if (Printall) {
      /* -z mode: generate 4096 extended DES salts (_J9..XXXX format, 725 rounds) */
      J1T(RC, Dohash, JOB_BSDICRYPT);
      if (RC && !TYPESALT(JOB_BSDICRYPT)) {
        char edsalt[10];
        Word_t *EPV;
        int a, b;
        edsalt[0] = '_'; edsalt[1] = 'J'; edsalt[2] = '9';
        edsalt[3] = '.'; edsalt[4] = '.';
        for (a = 0; a < 64; a++) {
          for (b = 0; b < 64; b++) {
            edsalt[5] = phpitoa64[a]; edsalt[6] = phpitoa64[b];
            edsalt[7] = '.'; edsalt[8] = '.'; edsalt[9] = 0;
            JSLI(EPV, TYPESALT(JOB_BSDICRYPT), (unsigned char *)edsalt);
            if (EPV) *EPV = 1000000;
          }
        }
        /* Dummy JudyJ entry so Printall lookup doesn't gate output */
        JSLI(EPV, JUDYJ(JOB_BSDICRYPT), (unsigned char *)"_J9.......zz");
        if (EPV) *EPV = 0;
        Numsalts += 4096;
        fprintf(stderr, "Generated 4096 extended DES salts for BSDICRYPT\n");
      }
    } else {
      J1U(RC, Dohash, JOB_BSDICRYPT);
    }

    line[0] = 0;
    JSLF(PV, JUDYJ(JOB_BCRYPT),(unsigned char *)line);
    while (PV) {
      Bcryptcnt++;
      JSLN(PV, JUDYJ(JOB_BCRYPT),(unsigned char *)line);
    }
    /* Bare bcrypt settings loaded through -F fill Typesalt with no MCF in
     * JudyJ.  Under -z they are the salts to generate with, so the default
     * below is only for a run that supplied none. */
    { static const int bs_z[] = { JOB_BCRYPT, JOB_BCRYPT256, JOB_BCRYPTMD5,
                                  JOB_BCRYPTSHA1, JOB_BCRYPTSHA512, -1 };
      int bz;
      long bset = 0;
      if (!Bcryptcnt && Printall) {
        for (bz = 0; bs_z[bz] >= 0; bz++) {
          long n;
          J1T(RC, Dohash, bs_z[bz]);
          if (!RC) continue;
          n = typesalt_entries(bs_z[bz]);
          if (n) fprintf(stderr, "%ld unique salts for %s\n", n, Types[bs_z[bz]]);
          if (n > bset) bset = n;
        }
        if (bset) { Bcryptcnt = (unsigned int)bset; Numsalts += bset; }
      }
    }
    if (Bcryptcnt) {
      if (JUDYJ(JOB_BCRYPT))
        fprintf(stderr, "Searching through %s unique BCRYPT hashes\n", commify(Bcryptcnt));
    } else if (Printall) {
      /* -z mode: generate a default bcrypt salt so all bcrypt variants can produce output */
      char bcrypt_defsalt[] = "$2a$05$RndSa1tRndSa1tRndSa1tuAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
      JSLI(PV, JUDYJ(JOB_BCRYPT),(unsigned char *)bcrypt_defsalt);
      /* Typesalt[JOB_BCRYPT], NOT Typesalt[job->op].  This is main(), where job
       * is the last struct off the free-list build loop and job->op is still 0,
       * so the default salt was seeded into type 0 and BCRYPT's own salt table
       * stayed empty.  procjob then walks TYPESALT(job->op) with op == JOB_BCRYPT,
       * found nothing, and -z produced no output at all for e450/e451/e452. */
      /* Seed Typesalt for EVERY selected bcrypt variant, mirroring the loader.
       * Two separate bugs lived here.  First, this is main(), where job is the
       * last struct off the free-list build loop and job->op is still 0, so
       * TYPESALT(job->op) seeded type 0 and BCRYPT's own table stayed empty.
       * Second, the compound variants walk their OWN TYPESALT(job->op) in
       * procjob, so seeding only JOB_BCRYPT left e451/e452/e453 with nothing
       * to iterate.  Between them, -z produced no output for any bcrypt type. */
      { static const int bc_z[] = { JOB_BCRYPT, JOB_BCRYPT256, JOB_BCRYPTMD5,
                                    JOB_BCRYPTSHA1, JOB_BCRYPTSHA512, -1 };
        int bz;
        for (bz = 0; bc_z[bz] >= 0; bz++) {
          J1T(RC, Dohash, bc_z[bz]);
          if (!RC) continue;
          JSLI(PV, TYPESALT(bc_z[bz]), (unsigned char *)"$2a$05$RndSa1tRndSa1tRndSa1tu");
          if (PV) (*PV)++;
          /* BCRYPT256 carries a SECOND construction selected by the salt form:
           * a $2k$ salt means passlib bcrypt_sha256, which is what the CMIYC
           * 2024 $2k$ hashes are.  With only a $2a$ default salt, bare -z could
           * never exercise that path.  A $2k$ salt is 28 characters -- the
           * 7-character prefix plus 21 of the 22 salt characters, the 22nd
           * being enumerated at generate time. */
          if (bc_z[bz] == JOB_BCRYPT256) {
            JSLI(PV, TYPESALT(JOB_BCRYPT256),
                 (unsigned char *)"$2k$05$RndSa1tRndSa1tRndSa1t");
            if (PV) (*PV)++;
          }
        }
      }
      Bcryptcnt = 1;
    } else {
      J1U(RC, Dohash, JOB_BCRYPT);
      J1U(RC, Dohash, JOB_BCRYPTMD5);
      J1U(RC, Dohash, JOB_BCRYPTSHA1);
      J1U(RC, Dohash, JOB_BCRYPTSHA512);
    }
    { long sshacnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSHA1BASE64), (unsigned char *)line);
      while (PV) {
        sshacnt++;
        JSLN(PV, JUDYJ(JOB_SSHA1BASE64), (unsigned char *)line);
      }
      if (sshacnt) {
        fprintf(stderr, "Searching through %ld unique SSHA hashes\n", sshacnt);
      } else {
        J1T(RC, Dohash, JOB_SSHA1BASE64);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SSHA1BASE64), (unsigned char *)"{SSHA}AAAAAAAAAAAAAAAAAAAAAAAAAABTYWx0");
          if (!TYPESALT(JOB_SSHA1BASE64)) {
            JSLI(PV, TYPESALT(JOB_SSHA1BASE64), (unsigned char *)"53616c74");
            if (PV) *PV = 1;
          }
          sshacnt = 1;
          fprintf(stderr, "Searching through 1 unique SSHA hashes\n");
        } else {
          J1U(RC, Dohash, JOB_SSHA1BASE64);
        }
      }
    }
    { long ssha256cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSHA256BASE64), (unsigned char *)line);
      while (PV) {
        ssha256cnt++;
        JSLN(PV, JUDYJ(JOB_SSHA256BASE64), (unsigned char *)line);
      }
      if (ssha256cnt) {
        fprintf(stderr, "Searching through %ld unique SSHA256 hashes\n", ssha256cnt);
      } else {
        J1T(RC, Dohash, JOB_SSHA256BASE64);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SSHA256BASE64), (unsigned char *)"{SSHA256}AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          if (!TYPESALT(JOB_SSHA256BASE64)) {
            JSLI(PV, TYPESALT(JOB_SSHA256BASE64), (unsigned char *)"00000000");
            if (PV) *PV = 1;
          }
          ssha256cnt = 1;
          fprintf(stderr, "Searching through 1 unique SSHA256 hashes\n");
        } else {
          J1U(RC, Dohash, JOB_SSHA256BASE64);
        }
      }
    }
    { long ssha512cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSHA512BASE64), (unsigned char *)line);
      while (PV) {
        ssha512cnt++;
        JSLN(PV, JUDYJ(JOB_SSHA512BASE64), (unsigned char *)line);
      }
      if (ssha512cnt) {
        fprintf(stderr, "Searching through %ld unique SSHA512 hashes\n", ssha512cnt);
      } else {
        J1T(RC, Dohash, JOB_SSHA512BASE64);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SSHA512BASE64), (unsigned char *)"{SSHA512}AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          if (!TYPESALT(JOB_SSHA512BASE64)) {
            JSLI(PV, TYPESALT(JOB_SSHA512BASE64), (unsigned char *)"0000000000000000000000");
            if (PV) *PV = 1;
          }
          ssha512cnt = 1;
          fprintf(stderr, "Searching through 1 unique SSHA512 hashes\n");
        } else {
          J1U(RC, Dohash, JOB_SSHA512BASE64);
        }
      }
    }
    /* Insert MSSQL/macOS embedded salts into Typesalt */
    { int mssql_types[] = { JOB_MSSQL2000, JOB_MSSQL2005, JOB_MSSQL2012,
                            JOB_MACOSX, JOB_MACOSX7, 0 };
      int dt;
      long mssql_total = 0;
      Word_t *SPV;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MSSQL2000), (unsigned char *)line);
      while (PV) {
        mssql_total++;
        for (dt = 0; mssql_types[dt]; dt++) {
          J1T(RC, Dohash, mssql_types[dt]);
          if (RC) {
            JSLI(SPV, TYPESALT(mssql_types[dt]), (unsigned char *)line);
            if (SPV && *SPV == 0) *SPV = *PV ? *PV : 1;
          }
        }
        JSLN(PV, JUDYJ(JOB_MSSQL2000), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_MSSQL2000));
      if (mssql_total) {
        Numsalts += mssql_total;
        fprintf(stderr, "%ld unique MSSQL/macOS salts loaded\n", mssql_total);
      }
      if (!mssql_total && Printall) {
        for (dt = 0; mssql_types[dt]; dt++) {
          J1T(RC, Dohash, mssql_types[dt]);
          if (RC && !TYPESALT(mssql_types[dt])) {
            JSLI(SPV, TYPESALT(mssql_types[dt]), (unsigned char *)"12345678");
            if (SPV) *SPV = 1000000;
            Numsalts++;
          }
        }
      }
      if (!mssql_total && !Printall) {
        J1U(RC, Dohash, JOB_MSSQL2000);
        J1U(RC, Dohash, JOB_MSSQL2005);
        J1U(RC, Dohash, JOB_MSSQL2012);
        if (!TYPESALT(JOB_MACOSX))
          J1U(RC, Dohash, JOB_MACOSX);
        if (!TYPESALT(JOB_MACOSX7))
          J1U(RC, Dohash, JOB_MACOSX7);
        if (!TYPESALT(JOB_ARUBAOS))
          J1U(RC, Dohash, JOB_ARUBAOS);
      }
    }
    /* JudyJ[JOB_JUNIPERSSG]: count entries, add default for -z, or disable */
    { long junipercnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_JUNIPERSSG), (unsigned char *)line);
      while (PV) {
        junipercnt++;
        JSLN(PV, JUDYJ(JOB_JUNIPERSSG), (unsigned char *)line);
      }
      if (junipercnt) {
        fprintf(stderr, "Searching through %ld unique JUNIPERSSG hashes\n", junipercnt);
      } else {
        J1T(RC, Dohash, JOB_JUNIPERSSG);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_JUNIPERSSG), (unsigned char *)"nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAn");
          junipercnt = 1;
        } else {
          J1U(RC, Dohash, JOB_JUNIPERSSG);
        }
      }
    }
    /* JudyJ[JOB_PEOPLESOFT]: count entries, add default for -z, or disable */
    { long psoftcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_PEOPLESOFT), (unsigned char *)line);
      while (PV) {
        psoftcnt++;
        JSLN(PV, JUDYJ(JOB_PEOPLESOFT), (unsigned char *)line);
      }
      if (psoftcnt) {
        fprintf(stderr, "Searching through %ld unique PEOPLESOFT hashes\n", psoftcnt);
      } else {
        J1T(RC, Dohash, JOB_PEOPLESOFT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_PEOPLESOFT), (unsigned char *)"AAAAAAAAAAAAAAAAAAAAAAAAAAA=");
          psoftcnt = 1;
        } else {
          J1U(RC, Dohash, JOB_PEOPLESOFT);
        }
      }
    }
    /* JudyJ[JOB_EPISERVER]: count entries, add default for -z, or disable */
    { long episervercnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_EPISERVER), (unsigned char *)line);
      while (PV) {
        episervercnt++;
        JSLN(PV, JUDYJ(JOB_EPISERVER), (unsigned char *)line);
      }
      if (episervercnt) {
        fprintf(stderr, "Searching through %ld unique EPISERVER hashes\n", episervercnt);
      } else {
        J1T(RC, Dohash, JOB_EPISERVER);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_EPISERVER), (unsigned char *)"$episerver$*0*RGVm*AAAAAAAAAAAAAAAAAAAAAAAAAAA");
          JSLI(PV, JUDYJ(JOB_EPISERVER), (unsigned char *)"$episerver$*1*RGVm*AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          if (!TYPESALT(JOB_EPISERVER)) {
            JSLI(PV, TYPESALT(JOB_EPISERVER), (unsigned char *)"0*RGVm");
            if (PV) (*PV)++;
            JSLI(PV, TYPESALT(JOB_EPISERVER), (unsigned char *)"1*RGVm");
            if (PV) (*PV)++;
          }
          episervercnt = 2;
        } else {
          J1U(RC, Dohash, JOB_EPISERVER);
        }
      }
    }
    /* hMailServer: count entries, add default for -z, or disable */
    { long hmcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_HMAILSERVER), (unsigned char *)line);
      while (PV) {
        hmcnt++;
        JSLN(PV, JUDYJ(JOB_HMAILSERVER), (unsigned char *)line);
      }
      if (hmcnt) {
        fprintf(stderr, "Searching through %ld unique HMAILSERVER hashes\n", hmcnt);
      } else {
        J1T(RC, Dohash, JOB_HMAILSERVER);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_HMAILSERVER), (unsigned char *)"123456AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          if (!TYPESALT(JOB_HMAILSERVER)) {
            JSLI(PV, TYPESALT(JOB_HMAILSERVER), (unsigned char *)"123456");
            if (PV) (*PV)++;
          }
          hmcnt = 1;
        } else {
          J1U(RC, Dohash, JOB_HMAILSERVER);
        }
      }
    }
    /* YAF-SHA1: count entries, add default for -z, or disable */
    { long yafcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_YAF_SHA1), (unsigned char *)line);
      while (PV) {
        yafcnt++;
        JSLN(PV, JUDYJ(JOB_YAF_SHA1), (unsigned char *)line);
      }
      if (yafcnt) {
        fprintf(stderr, "Searching through %ld unique YAF-SHA1 hashes\n", yafcnt);
      } else {
        J1T(RC, Dohash, JOB_YAF_SHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_YAF_SHA1), (unsigned char *)"AAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          if (!TYPESALT(JOB_YAF_SHA1)) {
            JSLI(PV, TYPESALT(JOB_YAF_SHA1), (unsigned char *)"RJF/o8S5AhK+DA9rkaiE2Q==");
            if (PV) (*PV)++;
          }
          yafcnt = 1;
        } else {
          J1U(RC, Dohash, JOB_YAF_SHA1);
        }
      }
    }
    /* CiscoISE: count entries, add default for -z, or disable */
    { long cisecnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_CISCOISE), (unsigned char *)line);
      while (PV) {
        cisecnt++;
        JSLN(PV, JUDYJ(JOB_CISCOISE), (unsigned char *)line);
      }
      if (cisecnt) {
        fprintf(stderr, "Searching through %ld unique CISCOISE hashes\n", cisecnt);
      } else {
        J1T(RC, Dohash, JOB_CISCOISE);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_CISCOISE), (unsigned char *)"00000000000000000000000000000000000000000000000000000000000000004d65737361676544696765737400000000000000000000000000000000000000");
          if (!TYPESALT(JOB_CISCOISE)) {
            JSLI(PV, TYPESALT(JOB_CISCOISE), (unsigned char *)"4d65737361676544696765737400000000000000000000000000000000000000");
            if (PV) (*PV)++;
          }
          cisecnt = 1;
        } else {
          J1U(RC, Dohash, JOB_CISCOISE);
        }
      }
    }
    /* MediaWiki: count entries, add default for -z, or disable */
    { long mwcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MEDIAWIKI), (unsigned char *)line);
      while (PV) {
        mwcnt++;
        JSLN(PV, JUDYJ(JOB_MEDIAWIKI), (unsigned char *)line);
      }
      if (mwcnt) {
        fprintf(stderr, "Searching through %ld unique MEDIAWIKI hashes\n", mwcnt);
      } else {
        J1T(RC, Dohash, JOB_MEDIAWIKI);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MEDIAWIKI), (unsigned char *)"$B$56668501$00000000000000000000000000000000");
          if (!TYPESALT(JOB_MEDIAWIKI)) {
            JSLI(PV, TYPESALT(JOB_MEDIAWIKI), (unsigned char *)"56668501");
            if (PV) (*PV)++;
          }
          mwcnt = 1;
        } else {
          J1U(RC, Dohash, JOB_MEDIAWIKI);
        }
      }
    }
    /* JudyJ[JOB_CISCOPIX]: count entries, add default for -z, or disable */
    { long pixcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_CISCOPIX), (unsigned char *)line);
      while (PV) {
        pixcnt++;
        JSLN(PV, JUDYJ(JOB_CISCOPIX), (unsigned char *)line);
      }
      if (pixcnt) {
        fprintf(stderr, "Searching through %ld unique CISCOPIX hashes\n", pixcnt);
      } else {
        J1T(RC, Dohash, JOB_CISCOPIX);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_CISCOPIX), (unsigned char *)"AAAAAAAAAAAAAAAA");
          pixcnt = 1;
        } else {
          J1U(RC, Dohash, JOB_CISCOPIX);
        }
      }
    }
    /* JudyJ[JOB_CISCOASA]: count entries, add default for -z, or disable */
    { long asacnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_CISCOASA), (unsigned char *)line);
      while (PV) {
        asacnt++;
        JSLN(PV, JUDYJ(JOB_CISCOASA), (unsigned char *)line);
      }
      if (asacnt) {
        fprintf(stderr, "Searching through %ld unique CISCOASA hashes\n", asacnt);
      } else {
        J1T(RC, Dohash, JOB_CISCOASA);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_CISCOASA), (unsigned char *)"AAAAAAAAAAAAAAAA:aa");
          JSLI(PV, TYPESALT(JOB_CISCOASA), (unsigned char *)"42");
          if (PV) *PV = 1000000;
          asacnt = 1;
        } else {
          J1U(RC, Dohash, JOB_CISCOASA);
        }
      }
    }
    /* JudyJ[JOB_CISCO4]: count entries, add default for -z, or disable */
    { long c4cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_CISCO4), (unsigned char *)line);
      while (PV) {
        c4cnt++;
        JSLN(PV, JUDYJ(JOB_CISCO4), (unsigned char *)line);
      }
      if (c4cnt) {
        fprintf(stderr, "Searching through %ld unique CISCO4 hashes\n", c4cnt);
      } else {
        J1T(RC, Dohash, JOB_CISCO4);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_CISCO4), (unsigned char *)"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          c4cnt = 1;
        } else {
          J1U(RC, Dohash, JOB_CISCO4);
        }
      }
    }
    /* Dahua: -z mode needs default salt and pepper */
    if (Printall) {
      J1T(RC, Dohash, JOB_DAHUA);
      if (RC) {
	JSLI(PV, TYPESALT(JOB_DAHUA), (unsigned char *)"229381927");
	if (PV) *PV = 1000000;
        if (!Typepepper[JOB_DAHUA])
          Typepepper[JOB_DAHUA] = memndup("\x09""182719643\0\xff", 12);
      }
    }
    /* AIX-MD5: insert JudyJ[JOB_AIXMD5] entries into Typesalt */
    { long aixmd5cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AIXMD5), (unsigned char *)line);
      while (PV) {
        Word_t *SPV;
        JSLI(SPV, TYPESALT(JOB_AIXMD5), (unsigned char *)line);
        if (SPV && *SPV == 0) *SPV = 1;
        aixmd5cnt++;
        JSLN(PV, JUDYJ(JOB_AIXMD5), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_AIXMD5));
      if (aixmd5cnt) {
        Numsalts += aixmd5cnt;
        fprintf(stderr, "Searching through %ld unique AIX-MD5 hashes\n", aixmd5cnt);
      } else {
        J1T(RC, Dohash, JOB_AIXMD5);
        if (RC && Printall) {
          Word_t *SPV;
          JSLI(SPV, TYPESALT(JOB_AIXMD5), (unsigned char *)"rndSa1t$");
          if (SPV) *SPV = 1000000;
          Numsalts++;
        } else {
          J1U(RC, Dohash, JOB_AIXMD5);
        }
      }
    }
    /* AIX-SHA1: count entries, add default for -z, or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AIXSHA1), (unsigned char *)line);
      while (PV) {
        cnt++;
        JSLN(PV, JUDYJ(JOB_AIXSHA1), (unsigned char *)line);
      }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AIX-SHA1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AIXSHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AIXSHA1), (unsigned char *)"06$rndSa1trndSa1tr$...........................");
	  JSLI(PV, TYPESALT(JOB_AIXSHA1), (unsigned char *)"06$rndSa1trndSa1tr");
	    if (PV) *PV = 1000000;
        } else {
          J1U(RC, Dohash, JOB_AIXSHA1);
        }
      }
    }
    /* AIX-SHA256: count entries, add default for -z, or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AIXSHA256), (unsigned char *)line);
      while (PV) {
        cnt++;
        JSLN(PV, JUDYJ(JOB_AIXSHA256), (unsigned char *)line);
      }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AIX-SHA256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AIXSHA256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AIXSHA256), (unsigned char *)"06$rndSa1trndSa1tr$...........................................");
	  JSLI(PV, TYPESALT(JOB_AIXSHA256), (unsigned char *)"06$rndSa1trndSa1tr");
	    if (PV) *PV = 1000000;
        } else {
          J1U(RC, Dohash, JOB_AIXSHA256);
        }
      }
    }
    /* AIX-SHA512: count entries, add default for -z, or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AIXSHA512), (unsigned char *)line);
      while (PV) {
        cnt++;
        JSLN(PV, JUDYJ(JOB_AIXSHA512), (unsigned char *)line);
      }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AIX-SHA512 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AIXSHA512);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AIXSHA512), (unsigned char *)"06$rndSa1trndSa1tr$....................................................................................");
	  JSLI(PV, TYPESALT(JOB_AIXSHA512), (unsigned char *)"06$rndSa1trndSa1tr");
	    if (PV) *PV = 1000000;
        } else {
          J1U(RC, Dohash, JOB_AIXSHA512);
        }
      }
    }
    /* IPMI2_SHA1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_IPMI_SHA1), (unsigned char *)line);
      while (PV) {
        cnt++;
        JSLN(PV, JUDYJ(JOB_IPMI_SHA1), (unsigned char *)line);
      }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique IPMI2 HMAC-SHA1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_IPMI_SHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_IPMI_SHA1), (unsigned char *)"0000000000000000000000000000000000000000:00");
          if (!TYPESALT(JOB_IPMI_SHA1)) {
            JSLI(PV, TYPESALT(JOB_IPMI_SHA1), (unsigned char *)"00");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_IPMI_SHA1);
        }
      }
    }
    /* IPMI2_MD5: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_IPMI_MD5), (unsigned char *)line);
      while (PV) {
        cnt++;
        JSLN(PV, JUDYJ(JOB_IPMI_MD5), (unsigned char *)line);
      }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique IPMI2 HMAC-MD5 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_IPMI_MD5);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_IPMI_MD5), (unsigned char *)"00000000000000000000000000000000:00");
          if (!TYPESALT(JOB_IPMI_MD5)) {
            JSLI(PV, TYPESALT(JOB_IPMI_MD5), (unsigned char *)"00");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_IPMI_MD5);
        }
      }
    }
    /* KRB5PA23: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_KRB5PA23), (unsigned char *)line);
      while (PV) {
        cnt++;
        JSLN(PV, JUDYJ(JOB_KRB5PA23), (unsigned char *)line);
      }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Kerberos 5 etype 23 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_KRB5PA23);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_KRB5PA23), (unsigned char *)"$krb5pa$23$x$x$x$000000000000000000000000000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_KRB5PA23)) {
            JSLI(PV, TYPESALT(JOB_KRB5PA23), (unsigned char *)"$krb5pa$23$x$x$x$000000000000000000000000000000000000000000000000000000000000000000000000");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_KRB5PA23);
        }
      }
    }
    /* MySQL $A$ sha256crypt: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MYSQL_SHA256CRYPT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MYSQL_SHA256CRYPT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MySQL $A$ sha256crypt hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MYSQL_SHA256CRYPT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MYSQL_SHA256CRYPT), (unsigned char *)"$mysql$A$005*0000000000*00000000000000000000000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_MYSQL_SHA256CRYPT)) {
            JSLI(PV, TYPESALT(JOB_MYSQL_SHA256CRYPT), (unsigned char *)"$mysql$A$005*0000000000");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_MYSQL_SHA256CRYPT);
        }
      }
    }
    /* JudyJ[JOB_DRUPAL7]: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_DRUPAL7), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DRUPAL7), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Drupal7 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_DRUPAL7);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_DRUPAL7), (unsigned char *)"$S$C00000000AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          if (!TYPESALT(JOB_DRUPAL7)) {
            JSLI(PV, TYPESALT(JOB_DRUPAL7), (unsigned char *)"C00000000");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_DRUPAL7);
        }
      }
    }
    /* Sybase ASE: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SYBASE_ASE), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SYBASE_ASE), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Sybase ASE hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SYBASE_ASE);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SYBASE_ASE), (unsigned char *)"0xc00700000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_SYBASE_ASE)) {
            JSLI(PV, TYPESALT(JOB_SYBASE_ASE), (unsigned char *)"0000000000000000");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_SYBASE_ASE);
        }
      }
    }
    /* NetScaler: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_NETSCALER), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_NETSCALER), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Citrix NetScaler hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_NETSCALER);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_NETSCALER), (unsigned char *)"10000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_NETSCALER)) {
            JSLI(PV, TYPESALT(JOB_NETSCALER), (unsigned char *)"00000000");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_NETSCALER);
        }
      }
    }
    /* NSEC3: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_NSEC3), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_NSEC3), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique NSEC3 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_NSEC3);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_NSEC3), (unsigned char *)"00000000000000000000000000000000:.x.net:1:00");
          if (!TYPESALT(JOB_NSEC3)) {
            JSLI(PV, TYPESALT(JOB_NSEC3), (unsigned char *)".x.net:1:00");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_NSEC3);
        }
      }
    }
    /* JudyJ[JOB_WBB3]: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_WBB3), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_WBB3), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique WBB3 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_WBB3);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_WBB3), (unsigned char *)"0000000000000000000000000000000000000000:0000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_WBB3)) {
            JSLI(PV, TYPESALT(JOB_WBB3), (unsigned char *)"0000000000000000000000000000000000000000");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_WBB3);
        }
      }
    }
    /* PHPS: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_PHPS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_PHPS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique PHPS hashes\n", cnt);
      } else if (!Printall) {
        J1U(RC, Dohash, JOB_PHPS);
      }
    }
    /* RACF: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_RACF), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_RACF), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique RACF hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_RACF);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_RACF), (unsigned char *)"$racf$*USER*0000000000000000");
          if (!TYPESALT(JOB_RACF)) {
            JSLI(PV, TYPESALT(JOB_RACF), (unsigned char *)"USER");
            if (PV) (*PV)++;
          }
        } else {
          J1U(RC, Dohash, JOB_RACF);
        }
      }
    }
    /* RACF-KDFAES: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_RACF_KDFAES), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_RACF_KDFAES), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique RACF-KDFAES hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_RACF_KDFAES);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_RACF_KDFAES), (unsigned char *)"$racf-kdfaes$*USER*E7D7E66D000180000008003200100010*00112233445566778899AABBCCDDEEFF*0000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_RACF_KDFAES);
        }
      }
    }
    /* TACACS: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_TACACS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_TACACS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique TACACS hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_TACACS);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_TACACS), (unsigned char *)"5fde8e68:4e13e8fb33df:c006");
        } else {
          J1U(RC, Dohash, JOB_TACACS);
        }
      }
    }
    /* APPLE-SECURE-NOTES: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_APPLE_SECURE_NOTES), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_APPLE_SECURE_NOTES), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique APPLE-SECURE-NOTES hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_APPLE_SECURE_NOTES);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_APPLE_SECURE_NOTES), (unsigned char *)"$ASN$*1*20000*80771171105233481004850004085037*d04b17af7f6b184346aad3efefe8bec0987ee73418291a41");
        } else {
          J1U(RC, Dohash, JOB_APPLE_SECURE_NOTES);
        }
      }
    }
    /* CRAMMD5-DOVECOT: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_CRAMMD5_DOVECOT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_CRAMMD5_DOVECOT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique CRAMMD5-DOVECOT hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_CRAMMD5_DOVECOT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_CRAMMD5_DOVECOT), (unsigned char *)"00000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_CRAMMD5_DOVECOT);
        }
      }
    }
    /* JWT: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_JWT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_JWT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique JWT hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_JWT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_JWT), (unsigned char *)"00000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_JWT);
        }
      }
    }
    /* QNX-MD5: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_QNX_MD5), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_QNX_MD5), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique QNX-MD5 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_QNX_MD5);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_QNX_MD5), (unsigned char *)"00000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_QNX_MD5);
        }
      }
    }
    /* QNX-SHA256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_QNX_SHA256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_QNX_SHA256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique QNX-SHA256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_QNX_SHA256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_QNX_SHA256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_QNX_SHA256);
        }
      }
    }
    /* QNX-SHA512: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_QNX_SHA512), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_QNX_SHA512), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique QNX-SHA512 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_QNX_SHA512);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_QNX_SHA512), (unsigned char *)"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_QNX_SHA512);
        }
      }
    }
    /* QNX7-SHA512: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_QNX7_SHA512), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_QNX7_SHA512), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique QNX7-SHA512 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_QNX7_SHA512);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_QNX7_SHA512), (unsigned char *)"dummy_qnx7");
        } else {
          J1U(RC, Dohash, JOB_QNX7_SHA512);
        }
      }
    }
    /* SHA1-S1PS2 and RAILS-RESTFUL use compact table + checkhashsalt2,
     * not JudyJ. No bootstrap needed — Typesalt via hex fallthrough or default_salts. */
    /* KRB5PA-17: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_KRB5PA_17), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_KRB5PA_17), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique KRB5PA-17 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_KRB5PA_17);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_KRB5PA_17), (unsigned char *)"$krb5pa$17$x$X$dummy");
        } else {
          J1U(RC, Dohash, JOB_KRB5PA_17);
        }
      }
    }
    /* KRB5PA-18: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_KRB5PA_18), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_KRB5PA_18), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique KRB5PA-18 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_KRB5PA_18);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_KRB5PA_18), (unsigned char *)"$krb5pa$18$x$X$dummy");
        } else {
          J1U(RC, Dohash, JOB_KRB5PA_18);
        }
      }
    }
    /* WPA-PMKID: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_WPA_PMKID), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_WPA_PMKID), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique WPA-PMKID hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_WPA_PMKID);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_WPA_PMKID), (unsigned char *)"00000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_WPA_PMKID);
        }
      }
    }
    /* WPA-EAPOL: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_WPA_EAPOL), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_WPA_EAPOL), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique WPA-EAPOL hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_WPA_EAPOL);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_WPA_EAPOL), (unsigned char *)"00000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_WPA_EAPOL);
        }
      }
    }
    /* ANSIBLE-VAULT: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_ANSIBLE_VAULT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_ANSIBLE_VAULT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique ANSIBLE-VAULT hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_ANSIBLE_VAULT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_ANSIBLE_VAULT), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_ANSIBLE_VAULT);
        }
      }
    }
    /* APFS: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_APFS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_APFS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique APFS hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_APFS);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_APFS), (unsigned char *)"$fvde$2$16$00000000000000000000000000000000$20000$0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_APFS);
        }
      }
    }
    /* OTM-SHA256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_OTM_SHA256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_OTM_SHA256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique OTM-SHA256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_OTM_SHA256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_OTM_SHA256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_OTM_SHA256);
        }
      }
    }
    /* TELEGRAM-SHA256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_TELEGRAM_SHA256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_TELEGRAM_SHA256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique TELEGRAM-SHA256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_TELEGRAM_SHA256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_TELEGRAM_SHA256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_TELEGRAM_SHA256);
        }
      }
    }
    /* WEB2PY-SHA512: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_WEB2PY_SHA512), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_WEB2PY_SHA512), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique WEB2PY-SHA512 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_WEB2PY_SHA512);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_WEB2PY_SHA512), (unsigned char *)"0000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_WEB2PY_SHA512);
        }
      }
    }
    /* SOLARWINDS: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SOLARWINDS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SOLARWINDS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SOLARWINDS hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SOLARWINDS);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SOLARWINDS), (unsigned char *)"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_SOLARWINDS);
        }
      }
    }
    /* SOLARWINDS2: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SOLARWINDS2), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SOLARWINDS2), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SOLARWINDS2 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SOLARWINDS2);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SOLARWINDS2), (unsigned char *)"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_SOLARWINDS2);
        }
      }
    }
    /* SIMPLACMS: no per-type gate. This type matches through the compact table,
     * so its hashes are counted there like every other bare-hex type and there
     * is nothing type-specific to count or disable here. The block that used to
     * sit here counted the per-type Judy array and, finding it empty, removed
     * the type from Dohash with J1U -- which is what made a bare 32-hex hash
     * plus -s/-S fail silently: only the old special-case loader ever populated
     * that array, so any other channel disabled the type outright before a
     * single candidate was hashed. */
    /* APPLE-KEYCHAIN: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_APPLE_KEYCHAIN), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_APPLE_KEYCHAIN), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique APPLE-KEYCHAIN hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_APPLE_KEYCHAIN);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_APPLE_KEYCHAIN), (unsigned char *)"cfc13fbe9043b537f82101e5ff7c1dcda68e546fa1d225004969db9aaa8ffa0a09391a6ea5acc434da322198378f3cd0");
        } else { J1U(RC, Dohash, JOB_APPLE_KEYCHAIN); }
      }
    }
    /* APPLE-IWORK: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_APPLE_IWORK), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_APPLE_IWORK), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique APPLE-IWORK hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_APPLE_IWORK);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_APPLE_IWORK), (unsigned char *)"be7d23232a5fc51d875ef4343586aff0603cf9df10f24e33c34edd813eb5e770ba0b6251af83186a097820a8fabc4d1effa198a5d4556e21a8a32d3ea60781db");
        } else { J1U(RC, Dohash, JOB_APPLE_IWORK); }
      }
    }
    /* BITWARDEN: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_BITWARDEN), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_BITWARDEN), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique BITWARDEN hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_BITWARDEN);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_BITWARDEN), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_BITWARDEN); }
      }
    }
    /* MONGODB-SHA1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MONGODB_SHA1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MONGODB_SHA1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MONGODB-SHA1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MONGODB_SHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MONGODB_SHA1), (unsigned char *)"0000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_MONGODB_SHA1); }
      }
    }
    /* MONGODB-SHA256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MONGODB_SHA256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MONGODB_SHA256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MONGODB-SHA256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MONGODB_SHA256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MONGODB_SHA256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_MONGODB_SHA256); }
      }
    }
    /* FORTIGATE256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_FORTIGATE256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_FORTIGATE256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique FortiGate256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_FORTIGATE256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_FORTIGATE256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_FORTIGATE256)) {
            JSLI(PV, TYPESALT(JOB_FORTIGATE256), (unsigned char *)"142221334214210545260703");
            if (PV) *PV = 1;
          }
        } else { J1U(RC, Dohash, JOB_FORTIGATE256); }
      }
    }
    /* UMBRACO: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_UMBRACO), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_UMBRACO), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Umbraco hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_UMBRACO);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_UMBRACO), (unsigned char *)"0000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_UMBRACO); }
      }
    }
    /* DAHUA-AUTH: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_DAHUA_AUTH), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DAHUA_AUTH), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Dahua Auth hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_DAHUA_AUTH);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_DAHUA_AUTH), (unsigned char *)"00000000");
        } else { J1U(RC, Dohash, JOB_DAHUA_AUTH); }
      }
    }
    /* BESDER-AUTH: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_BESDER_AUTH), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_BESDER_AUTH), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Besder Auth hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_BESDER_AUTH);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_BESDER_AUTH), (unsigned char *)"00000000");
        } else { J1U(RC, Dohash, JOB_BESDER_AUTH); }
      }
    }
    /* SQLCIPHER: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SQLCIPHER), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SQLCIPHER), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SQLCipher hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SQLCIPHER);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SQLCIPHER), (unsigned char *)"33265289beab040faf0d638861cba205");
        } else { J1U(RC, Dohash, JOB_SQLCIPHER); }
      }
    }
    /* SAP-PASSCODE: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAP_PASSCODE), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAP_PASSCODE), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAP-PASSCODE hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAP_PASSCODE);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAP_PASSCODE), (unsigned char *)"0000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_SAP_PASSCODE);
        }
      }
    }
    /* SAP-PASSCODE5: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAP_PASSCODE5), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAP_PASSCODE5), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAP-PASSCODE5 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAP_PASSCODE5);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAP_PASSCODE5), (unsigned char *)"0000000000000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_SAP_PASSCODE5);
        }
      }
    }
    /* SAP-BCODE: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAP_BCODE), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAP_BCODE), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAP-BCODE hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAP_BCODE);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAP_BCODE), (unsigned char *)"0000000000000000");
        } else {
          J1U(RC, Dohash, JOB_SAP_BCODE);
        }
      }
    }
    /* SAP-BCODE4: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAP_BCODE4), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAP_BCODE4), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAP-BCODE4 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAP_BCODE4);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAP_BCODE4), (unsigned char *)"0000000000000000");
        } else {
          J1U(RC, Dohash, JOB_SAP_BCODE4);
        }
      }
    }
    /* AS400-DES: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AS400_DES), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AS400_DES), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AS400-DES hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AS400_DES);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AS400_DES), (unsigned char *)"$as400$des$*USER*0000000000000000");
        } else {
          J1U(RC, Dohash, JOB_AS400_DES);
        }
      }
    }
    /* Domino 6: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_DOMINO6), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DOMINO6), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Lotus Domino 6 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_DOMINO6);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_DOMINO6), (unsigned char *)"(G0000000000000000000)");
          if (!TYPESALT(JOB_DOMINO6)) {
            JSLI(PV, TYPESALT(JOB_DOMINO6), (unsigned char *)"0000000000");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_DOMINO6);
        }
      }
    }
    /* scrypt: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SCRYPT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SCRYPT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique scrypt hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SCRYPT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SCRYPT), (unsigned char *)"SCRYPT:1024:1:1:AA==:AA==");
          if (!TYPESALT(JOB_SCRYPT)) {
            JSLI(PV, TYPESALT(JOB_SCRYPT), (unsigned char *)"SCRYPT:1024:1:1:AA==");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_SCRYPT);
        }
      }
    }
    /* ISCSI-CHAP: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_ISCSI_CHAP), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_ISCSI_CHAP), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique iSCSI CHAP hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_ISCSI_CHAP);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_ISCSI_CHAP), (unsigned char *)"0000000000000000000000000000000000:01020304050607080910111213141516:01");
          if (!TYPESALT(JOB_ISCSI_CHAP)) {
            JSLI(PV, TYPESALT(JOB_ISCSI_CHAP), (unsigned char *)"01020304050607080910111213141516:01");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_ISCSI_CHAP);
        }
      }
    }
    /* NetScaler PBKDF2: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_NETSCALER_PBKDF2), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_NETSCALER_PBKDF2), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique NetScaler PBKDF2 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_NETSCALER_PBKDF2);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_NETSCALER_PBKDF2), (unsigned char *)"500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_NETSCALER_PBKDF2)) {
            JSLI(PV, TYPESALT(JOB_NETSCALER_PBKDF2), (unsigned char *)"567243c55099b6b10a714a350db53beea8be6ac9c247fd40fea7e96d206a9f11");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_NETSCALER_PBKDF2);
        }
      }
    }
    /* NetNTLMv1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_NETNTLMV1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_NETNTLMV1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique NetNTLMv1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_NETNTLMV1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_NETNTLMV1), (unsigned char *)"::X:00000000000000000000000000000000000000000000000000:000000000000000000000000000000000000000000000000:0000000000000000");
          if (!TYPESALT(JOB_NETNTLMV1)) {
            JSLI(PV, TYPESALT(JOB_NETNTLMV1), (unsigned char *)"::5V4T:ada06359242920a500000000000000000000000000000000:0556d5297b5daa70eaffde82ef99293a3f3bb59b7c9704ea:9c23f6c094853920");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_NETNTLMV1);
        }
      }
    }
    /* NetNTLMv2: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_NETNTLMV2), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_NETNTLMV2), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique NetNTLMv2 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_NETNTLMV2);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_NETNTLMV2), (unsigned char *)"X::X:0000000000000000:00000000000000000000000000000000:0000");
          if (!TYPESALT(JOB_NETNTLMV2)) {
            JSLI(PV, TYPESALT(JOB_NETNTLMV2), (unsigned char *)"0UL5G37JOI0SX::6VB1IS0KA74:ebe1afa18b7fbfa6:aab8bf8675658dd2a939458a1077ba08:010100000000000031c8aa092510945398b9f7b7dde1a9fb00000000f7876f2b04b700");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_NETNTLMV2);
        }
      }
    }
    /* Oracle7: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_ORACLE7), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_ORACLE7), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Oracle7 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_ORACLE7);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_ORACLE7), (unsigned char *)"0000000000000000:DEFAULTUSER");
          if (!TYPESALT(JOB_ORACLE7)) {
            JSLI(PV, TYPESALT(JOB_ORACLE7), (unsigned char *)"7284616727");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_ORACLE7);
        }
      }
    }
    /* LastPass: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_LASTPASS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_LASTPASS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique LastPass hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_LASTPASS);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_LASTPASS), (unsigned char *)"02eb97e869e0ddc7dc760fc633b4b54d");
          if (!TYPESALT(JOB_LASTPASS)) {
            JSLI(PV, TYPESALT(JOB_LASTPASS), (unsigned char *)"100100:pmix@trash-mail.com:9b071db7b8e265d4cadd3eb65ac0864a");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_LASTPASS);
        }
      }
    }
    /* FortiGate: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_FORTIGATE), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_FORTIGATE), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique FortiGate hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_FORTIGATE);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_FORTIGATE), (unsigned char *)"1702c2322ec6a69770b51cccc83a45385c5da47f");
          if (!TYPESALT(JOB_FORTIGATE)) {
            JSLI(PV, TYPESALT(JOB_FORTIGATE), (unsigned char *)"142221334214210545260703");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_FORTIGATE);
        }
      }
    }
    /* Domino 8: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_DOMINO8), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DOMINO8), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Domino 8 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_DOMINO8);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_DOMINO8), (unsigned char *)"6285cb9d56551256");
          if (!TYPESALT(JOB_DOMINO8)) {
            JSLI(PV, TYPESALT(JOB_DOMINO8), (unsigned char *)"30313730353737313232383031313430:0000005000:3032");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_DOMINO8);
        }
      }
    }
    /* SipHash: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SIPHASH), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SIPHASH), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SipHash hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SIPHASH);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SIPHASH), (unsigned char *)"583e6f51e52ba296");
          if (!TYPESALT(JOB_SIPHASH)) {
            JSLI(PV, TYPESALT(JOB_SIPHASH), (unsigned char *)"47356410265714355482333327356688");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_SIPHASH);
        }
      }
    }
    /* CRAM-MD5: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_CRAMMD5), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_CRAMMD5), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique CRAM-MD5 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_CRAMMD5);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_CRAMMD5), (unsigned char *)"b8f0699114b057889223df084f82246f");
          if (!TYPESALT(JOB_CRAMMD5)) {
            JSLI(PV, TYPESALT(JOB_CRAMMD5), (unsigned char *)"MTI=$dXNlciBiOGYwNjk5MTE0YjA1Nzg4OTIyM2RmMDg0ZjgyMjQ2Zg==");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_CRAMMD5);
        }
      }
    }
    /* SAP CODVN H: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAPCODVNH), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAPCODVNH), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAP CODVN H hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAPCODVNH);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAPCODVNH), (unsigned char *)"0678d732a70d4f06b707376750e7f5880bba770d");
          if (!TYPESALT(JOB_SAPCODVNH)) {
            JSLI(PV, TYPESALT(JOB_SAPCODVNH), (unsigned char *)"1024:363735383331363130");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_SAPCODVNH);
        }
      }
    }
    /* RedHat 389-DS: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_REDHAT389DS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_REDHAT389DS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique RedHat 389-DS hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_REDHAT389DS);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_REDHAT389DS), (unsigned char *)"1da633927318853496cd727b13aa959d371a9d8dd970952e04dd89fc63f666d6");
          if (!TYPESALT(JOB_REDHAT389DS)) {
            JSLI(PV, TYPESALT(JOB_REDHAT389DS), (unsigned char *)"8192:41414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_REDHAT389DS);
        }
      }
    }
    /* PostgreSQL CRAM: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_POSTGRESCRAM), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_POSTGRESCRAM), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique PostgreSQL CRAM hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_POSTGRESCRAM);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_POSTGRESCRAM), (unsigned char *)"e189d76dcdaf3bb15764784e2f24eee3");
          if (!TYPESALT(JOB_POSTGRESCRAM)) {
            JSLI(PV, TYPESALT(JOB_POSTGRESCRAM), (unsigned char *)"postgres:41414141");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_POSTGRESCRAM);
        }
      }
    }
    /* MySQL CRAM: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MYSQLCRAM), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MYSQLCRAM), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MySQL CRAM hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MYSQLCRAM);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MYSQLCRAM), (unsigned char *)"024763fc7fa983ef4c424bb91c4c9ca58a4c6a20");
          if (!TYPESALT(JOB_MYSQLCRAM)) {
            JSLI(PV, TYPESALT(JOB_MYSQLCRAM), (unsigned char *)"4141414141414141414141414141414141414141");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_MYSQLCRAM);
        }
      }
    }
    /* Apache Shiro 1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SHIRO1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SHIRO1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Apache Shiro hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SHIRO1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SHIRO1), (unsigned char *)"3c6a09f379b4a6162e7358b5ae1ed5aafdbe924b92ab18045fe021d25b9310e5");
          if (!TYPESALT(JOB_SHIRO1)) {
            JSLI(PV, TYPESALT(JOB_SHIRO1), (unsigned char *)"1024:41414141414141414141414141414141");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_SHIRO1);
        }
      }
    }
    /* eCryptfs: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_ECRYPTFS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_ECRYPTFS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique eCryptfs hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_ECRYPTFS);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_ECRYPTFS), (unsigned char *)"a07ae5f28a1db69d");
          if (!TYPESALT(JOB_ECRYPTFS)) {
            JSLI(PV, TYPESALT(JOB_ECRYPTFS), (unsigned char *)"4141414141414141");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_ECRYPTFS);
        }
      }
    }
    /* Oracle 12+: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_ORACLE12), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_ORACLE12), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Oracle 12+ hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_ORACLE12);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_ORACLE12), (unsigned char *)"00a4881893c78287cd84eb39d3da49f05a4df2fc48a97a70b0a9fc941c3cb465");
          if (!TYPESALT(JOB_ORACLE12)) {
            JSLI(PV, TYPESALT(JOB_ORACLE12), (unsigned char *)"41414141414141414141414141414141");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_ORACLE12);
        }
      }
    }
    /* ColdFusion 10+: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_COLDFUSION10), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_COLDFUSION10), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique ColdFusion 10+ hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_COLDFUSION10);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_COLDFUSION10), (unsigned char *)"443120b78d8eede52f25e291ce6621499348ba363b0b563a4e98c69a7f24993b");
          if (!TYPESALT(JOB_COLDFUSION10)) {
            JSLI(PV, TYPESALT(JOB_COLDFUSION10), (unsigned char *)"1234567890123456789012345678901234567890123456789012345678901234");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_COLDFUSION10);
        }
      }
    }
    /* MS-AzureSync: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AZURESYNC), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AZURESYNC), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MS-AzureSync hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AZURESYNC);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AZURESYNC), (unsigned char *)"9bd986554189ab1bdf2c85eed4d381cc6be472be53b9a54e08816680e93987f9");
          if (!TYPESALT(JOB_AZURESYNC)) {
            JSLI(PV, TYPESALT(JOB_AZURESYNC), (unsigned char *)"100:41414141414141414141");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_AZURESYNC);
        }
      }
    }
    /* Android FDE: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_ANDROIDFDE), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_ANDROIDFDE), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Android FDE hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_ANDROIDFDE);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_ANDROIDFDE), (unsigned char *)"4eb9e9cc64cf5d82eb7aaa88119a411233a1b93c15db5fcf1cee1aecf9bf8442");
          if (!TYPESALT(JOB_ANDROIDFDE)) {
            JSLI(PV, TYPESALT(JOB_ANDROIDFDE), (unsigned char *)"4242424242424242424242424242424242424242424242424242424242424242:41414141414141414141414141414141");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_ANDROIDFDE);
        }
      }
    }
    /* KRB5TGS23: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_KRB5TGS23), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_KRB5TGS23), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Kerberos TGS-REP hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_KRB5TGS23);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_KRB5TGS23), (unsigned char *)"$krb5tgs$23$*x$x$x*$00000000000000000000000000000000$0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_KRB5TGS23)) {
            JSLI(PV, TYPESALT(JOB_KRB5TGS23), (unsigned char *)"$krb5tgs$23$*x$x$x*$00000000000000000000000000000000$0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
            if (PV) *PV = 1;
          }
        } else {
          J1U(RC, Dohash, JOB_KRB5TGS23);
        }
      }
    }
    /* AxCrypt: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AXCRYPT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AXCRYPT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AxCrypt hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AXCRYPT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AXCRYPT), (unsigned char *)"$axcrypt$*1*10467*41414141414141414141414141414141*000000000000000000000000000000000000000000000000");
          if (!TYPESALT(JOB_AXCRYPT)) {
            JSLI(PV, TYPESALT(JOB_AXCRYPT), (unsigned char *)"10467:41414141414141414141414141414141:000000000000000000000000000000000000000000000000");
            if (PV) *PV = 1000000;
          }
        } else {
          J1U(RC, Dohash, JOB_AXCRYPT);
        }
      }
    }
    /* AxCrypt SHA1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AXCRYPTSHA1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AXCRYPTSHA1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AxCrypt SHA1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AXCRYPTSHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AXCRYPTSHA1), (unsigned char *)"cbfdac6008f9cab4083784cbd1874f76");
        } else {
          J1U(RC, Dohash, JOB_AXCRYPTSHA1);
        }
      }
    }
    /* RORAILS-SHA1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_RORAILS_SHA1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_RORAILS_SHA1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Ruby on Rails Restful Auth hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_RORAILS_SHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_RORAILS_SHA1), (unsigned char *)"0000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_RORAILS_SHA1); }
      }
    }
    /* AES128-NOKDF: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AES128_NOKDF), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AES128_NOKDF), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AES-128-ECB NOKDF hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AES128_NOKDF);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AES128_NOKDF), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_AES128_NOKDF); }
      }
    }
    /* AES192-NOKDF: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AES192_NOKDF), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AES192_NOKDF), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AES-192-ECB NOKDF hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AES192_NOKDF);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AES192_NOKDF), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_AES192_NOKDF); }
      }
    }
    /* AES256-NOKDF: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AES256_NOKDF), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AES256_NOKDF), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AES-256-ECB NOKDF hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AES256_NOKDF);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AES256_NOKDF), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_AES256_NOKDF); }
      }
    }
    /* VMWARE-VMX: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_VMWARE_VMX), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_VMWARE_VMX), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique VMware VMX hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_VMWARE_VMX);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_VMWARE_VMX), (unsigned char *)"a6e18a778759e00378c50087feaf1552");
        } else { J1U(RC, Dohash, JOB_VMWARE_VMX); }
      }
    }
    /* POSTGRESSCRAM256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_POSTGRESSCRAM256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_POSTGRESSCRAM256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique PostgreSQL SCRAM-SHA-256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_POSTGRESSCRAM256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_POSTGRESSCRAM256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_POSTGRESSCRAM256); }
      }
    }
    /* AWSSIGV4: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AWSSIGV4), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AWSSIGV4), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AWS Signature v4 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AWSSIGV4);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AWSSIGV4), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_AWSSIGV4); }
      }
    }
    /* KRB5DB17: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_KRB5DB17), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_KRB5DB17), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Kerberos 5 DB etype 17 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_KRB5DB17);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_KRB5DB17), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_KRB5DB17); }
      }
    }
    /* KRB5DB18: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_KRB5DB18), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_KRB5DB18), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Kerberos 5 DB etype 18 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_KRB5DB18);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_KRB5DB18), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_KRB5DB18); }
      }
    }
    /* MURMUR3: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MURMUR3), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MURMUR3), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MurmurHash3 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MURMUR3);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MURMUR3), (unsigned char *)"00000000");
        } else { J1U(RC, Dohash, JOB_MURMUR3); }
      }
    }
    /* VEEAM-VBK: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_VEEAM_VBK), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_VEEAM_VBK), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Veeam VBK hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_VEEAM_VBK);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_VEEAM_VBK), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_VEEAM_VBK); }
      }
    }
    /* MSSNTP: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MSSNTP), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MSSNTP), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MS SNTP hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MSSNTP);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MSSNTP), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_MSSNTP); }
      }
    }
    /* SSPR-MD5: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSPR_MD5), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SSPR_MD5), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SSPR MD5 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SSPR_MD5);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SSPR_MD5), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_SSPR_MD5); }
      }
    }
    /* SSPR-SHA1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSPR_SHA1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SSPR_SHA1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SSPR SHA1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SSPR_SHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SSPR_SHA1), (unsigned char *)"0000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_SSPR_SHA1); }
      }
    }
    /* SSPR-SHA1S: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSPR_SHA1S), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SSPR_SHA1S), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SSPR SHA1 salted hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SSPR_SHA1S);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SSPR_SHA1S), (unsigned char *)"0000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_SSPR_SHA1S); }
      }
    }
    /* SSPR-SHA256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSPR_SHA256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SSPR_SHA256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SSPR SHA256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SSPR_SHA256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SSPR_SHA256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_SSPR_SHA256); }
      }
    }
    /* SSPR-SHA512: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SSPR_SHA512), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SSPR_SHA512), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SSPR SHA512 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SSPR_SHA512);
        if (RC && Printall) {
          char *dummy512 = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
          JSLI(PV, JUDYJ(JOB_SSPR_SHA512), (unsigned char *)dummy512);
        } else { J1U(RC, Dohash, JOB_SSPR_SHA512); }
      }
    }
    /* EMPIRECMS: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_EMPIRECMS), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_EMPIRECMS), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Empire CMS hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_EMPIRECMS);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_EMPIRECMS), (unsigned char *)"00000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_EMPIRECMS); }
      }
    }
    /* PBKDF1-SHA1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_PBKDF1_SHA1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_PBKDF1_SHA1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique PBKDF1-SHA1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_PBKDF1_SHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_PBKDF1_SHA1), (unsigned char *)"AAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        } else { J1U(RC, Dohash, JOB_PBKDF1_SHA1); }
      }
    }
    /* MSONLINE: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MSONLINE), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MSONLINE), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MSONLINE hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MSONLINE);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MSONLINE), (unsigned char *)"$MSONLINEACCOUNT$0$10000$0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_MSONLINE); }
      }
    }
    /* WBB4: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_WBB4), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_WBB4), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique WBB4 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_WBB4);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_WBB4), (unsigned char *)"$2a$08$aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        } else { J1U(RC, Dohash, JOB_WBB4); }
      }
    }
    /* MONGODB: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_MONGODB), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_MONGODB), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique MONGODB hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_MONGODB);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_MONGODB), (unsigned char *)"placeholder-MONGODB");
        } else { J1U(RC, Dohash, JOB_MONGODB); }
      }
    }
    /* H3C: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_H3C), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_H3C), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique H3C hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_H3C);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_H3C), (unsigned char *)"placeholder-H3C");
        } else { J1U(RC, Dohash, JOB_H3C); }
      }
    }
      /* DRAGONFLY3-32: count entries or disable */
      { long cnt = 0;
        line[0] = 0;
        JSLF(PV, JUDYJ(JOB_DRAGONFLY3_32), (unsigned char *)line);
        while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DRAGONFLY3_32), (unsigned char *)line); }
        if (cnt) {
          fprintf(stderr, "Searching through %ld unique DRAGONFLY3-32 hashes\n", cnt);
        } else {
          J1T(RC, Dohash, JOB_DRAGONFLY3_32);
          if (RC && Printall) {
            JSLI(PV, JUDYJ(JOB_DRAGONFLY3_32), (unsigned char *)"placeholder-DRAGONFLY3-32");
          } else { J1U(RC, Dohash, JOB_DRAGONFLY3_32); }
        }
      }
      /* DRAGONFLY3-64: count entries or disable */
      { long cnt = 0;
        line[0] = 0;
        JSLF(PV, JUDYJ(JOB_DRAGONFLY3_64), (unsigned char *)line);
        while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DRAGONFLY3_64), (unsigned char *)line); }
        if (cnt) {
          fprintf(stderr, "Searching through %ld unique DRAGONFLY3-64 hashes\n", cnt);
        } else {
          J1T(RC, Dohash, JOB_DRAGONFLY3_64);
          if (RC && Printall) {
            JSLI(PV, JUDYJ(JOB_DRAGONFLY3_64), (unsigned char *)"placeholder-DRAGONFLY3-64");
          } else { J1U(RC, Dohash, JOB_DRAGONFLY3_64); }
        }
      }
      /* DRAGONFLY4-32: count entries or disable */
      { long cnt = 0;
        line[0] = 0;
        JSLF(PV, JUDYJ(JOB_DRAGONFLY4_32), (unsigned char *)line);
        while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DRAGONFLY4_32), (unsigned char *)line); }
        if (cnt) {
          fprintf(stderr, "Searching through %ld unique DRAGONFLY4-32 hashes\n", cnt);
        } else {
          J1T(RC, Dohash, JOB_DRAGONFLY4_32);
          if (RC && Printall) {
            JSLI(PV, JUDYJ(JOB_DRAGONFLY4_32), (unsigned char *)"placeholder-DRAGONFLY4-32");
          } else { J1U(RC, Dohash, JOB_DRAGONFLY4_32); }
        }
      }
      /* DRAGONFLY4-64: count entries or disable */
      { long cnt = 0;
        line[0] = 0;
        JSLF(PV, JUDYJ(JOB_DRAGONFLY4_64), (unsigned char *)line);
        while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_DRAGONFLY4_64), (unsigned char *)line); }
        if (cnt) {
          fprintf(stderr, "Searching through %ld unique DRAGONFLY4-64 hashes\n", cnt);
        } else {
          J1T(RC, Dohash, JOB_DRAGONFLY4_64);
          if (RC && Printall) {
            JSLI(PV, JUDYJ(JOB_DRAGONFLY4_64), (unsigned char *)"placeholder-DRAGONFLY4-64");
          } else { J1U(RC, Dohash, JOB_DRAGONFLY4_64); }
        }
      }
    /* SAPCODVNH256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAPCODVNH256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAPCODVNH256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAPCODVNH256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAPCODVNH256);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAPCODVNH256), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_SAPCODVNH256); }
      }
    }
    /* SAPCODVNH384: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAPCODVNH384), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAPCODVNH384), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAPCODVNH384 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAPCODVNH384);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAPCODVNH384), (unsigned char *)"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_SAPCODVNH384); }
      }
    }
    /* SAPCODVNH512: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SAPCODVNH512), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_SAPCODVNH512), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique SAP CODVN H SHA512 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_SAPCODVNH512);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SAPCODVNH512), (unsigned char *)"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_SAPCODVNH512); }
      }
    }
    /* SM3CRYPT: insert JudyJ entries into Typesalt, free JudyJ */
    { long sm3cryptcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SM3CRYPT), (unsigned char *)line);
      while (PV) {
        Word_t *SPV;
        JSLI(SPV, TYPESALT(JOB_SM3CRYPT), (unsigned char *)line);
        if (SPV && *SPV == 0) *SPV = 1;
        sm3cryptcnt++;
        JSLN(PV, JUDYJ(JOB_SM3CRYPT), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_SM3CRYPT));
      if (sm3cryptcnt) {
        if (Bcryptcnt == 0)
          Bcryptcnt = sm3cryptcnt;
        Numsalts += sm3cryptcnt;
        fprintf(stderr, "Searching through %ld unique SM3CRYPT hashes\n", sm3cryptcnt);
      } else {
        J1T(RC, Dohash, JOB_SM3CRYPT);
        if (RC && Printall) {
          Word_t *SPV;
          JSLI(SPV, TYPESALT(JOB_SM3CRYPT), (unsigned char *)"$sm3$aaaaaaaaaaaaaaaa$");
          if (SPV) *SPV = 1000000;
          Numsalts++;
        } else {
          J1U(RC, Dohash, JOB_SM3CRYPT);
        }
      }
    }
    /* AS400SSHA1: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_AS400SSHA1), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_AS400SSHA1), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique AS/400 SSHA1 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_AS400SSHA1);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_AS400SSHA1), (unsigned char *)"0000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_AS400SSHA1); }
      }
    }
    /* ARGON2: count entries or disable, set Argon2_maxmem for -z */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_ARGON2), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_ARGON2), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique Argon2 hashes (max workspace %zu KB)\n",
                cnt, Argon2_maxmem / 1024);
      } else {
        J1T(RC, Dohash, JOB_ARGON2);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_ARGON2), (unsigned char *)"$argon2id$v=19$m=65536,t=3,p=1$aaaaaaaaaaaaaaaaaaaaaa$aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
          if (Argon2_maxmem < 65536 * 1024)
            Argon2_maxmem = 65536 * 1024;
        } else { J1U(RC, Dohash, JOB_ARGON2); }
      }
      /* ARGON2MD5 shares this workspace and the same default cost, but the floor
       * above sits inside the ARGON2 arm and is gated on ARGON2 itself being
       * selected.  Selecting ARGON2MD5 ALONE therefore left Argon2_maxmem at 0,
       * the workspace was malloc'd at 16 bytes, and argon2 wrote 64 MB into it. */
      J1T(RC, Dohash, JOB_ARGON2MD5);
      if (RC && Printall && Argon2_maxmem < 65536 * 1024)
        Argon2_maxmem = 65536 * 1024;
    }
    /* BCRYPTHMACSHA256: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_BCRYPTHMACSHA256), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_BCRYPTHMACSHA256), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique BCRYPTHMACSHA256 hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_BCRYPTHMACSHA256);
        if (RC && Printall) {
          /* Default: $2b$05$RndSa1tRndSa1tRndSa1tu + 31-char dummy hash */
          JSLI(PV, JUDYJ(JOB_BCRYPTHMACSHA256), (unsigned char *)"$2b$05$RndSa1tRndSa1tRndSa1tuAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
          /* Ensure shared bcrypt infrastructure has a salt too */
          if (Bcryptcnt == 0) {
            JSLI(PV, JUDYJ(JOB_BCRYPT),(unsigned char *)"$2a$05$RndSa1tRndSa1tRndSa1tuAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
            /* Typesalt[JOB_BCRYPT], NOT Typesalt[job->op].  This is main(), where job
             * is the last struct off the free-list build loop and job->op is still 0,
             * so the default salt was seeded into type 0 and BCRYPT's own salt table
             * stayed empty.  procjob then walks TYPESALT(job->op) with op == JOB_BCRYPT,
             * found nothing, and -z produced no output at all for e450/e451/e452. */
            /* Seed Typesalt for EVERY selected bcrypt variant, mirroring the loader.
             * Two separate bugs lived here.  First, this is main(), where job is the
             * last struct off the free-list build loop and job->op is still 0, so
             * TYPESALT(job->op) seeded type 0 and BCRYPT's own table stayed empty.
             * Second, the compound variants walk their OWN TYPESALT(job->op) in
             * procjob, so seeding only JOB_BCRYPT left e451/e452/e453 with nothing
             * to iterate.  Between them, -z produced no output for any bcrypt type. */
            { static const int bc_z[] = { JOB_BCRYPT, JOB_BCRYPT256, JOB_BCRYPTMD5,
                                          JOB_BCRYPTSHA1, JOB_BCRYPTSHA512, -1 };
              int bz;
              for (bz = 0; bc_z[bz] >= 0; bz++) {
                J1T(RC, Dohash, bc_z[bz]);
                if (!RC) continue;
                JSLI(PV, TYPESALT(bc_z[bz]), (unsigned char *)"$2a$05$RndSa1tRndSa1tRndSa1tu");
                if (PV) (*PV)++;
              }
            }
            Bcryptcnt = 1;
          }
        } else { J1U(RC, Dohash, JOB_BCRYPTHMACSHA256); }
      }
    }
    /* WPA_PMK: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_WPA_PMK), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_WPA_PMK), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique WPA-PMK hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_WPA_PMK);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_WPA_PMK), (unsigned char *)"00000000000000000000000000000000");
        } else {
          J1U(RC, Dohash, JOB_WPA_PMK);
        }
      }
    }
    /* MD5SALT1SALT2 uses compact table + checkhashsalt2, no bootstrap needed */
    /* SYMFONY256: no per-type gate -- matches through the compact table, so
     * its hashes are counted there like any other bare-hex type. The block
     * here used to remove the type from Dohash when its per-type Judy array
     * was empty, which only the old special-case loader ever filled. */
    /* WPBCRYPT: count entries or disable */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_WPBCRYPT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_WPBCRYPT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique WPBCRYPT hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_WPBCRYPT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_WPBCRYPT), (unsigned char *)"$2b$05$000000000000000000000000000000000000000000000000000000000000");
        } else { J1U(RC, Dohash, JOB_WPBCRYPT); }
      }
    }
    /* BCRYPTMD5PASSSALT: count entries or disable.  The type keeps its own
     * JudyJ of MCFs rather than joining the family's shared JudyJ[JOB_BCRYPT],
     * so it needs its own gate -- without one, a run that loaded no e1059
     * record would walk an empty array for every candidate.  Under -z the seed
     * is a cost-05 MCF so generating a vector stays cheap; the site salt comes
     * from default_salts[] via init_default_salts(), which has already run. */
    { long cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_BCRYPTMD5PASSSALT), (unsigned char *)line);
      while (PV) { cnt++; JSLN(PV, JUDYJ(JOB_BCRYPTMD5PASSSALT), (unsigned char *)line); }
      if (cnt) {
        fprintf(stderr, "Searching through %ld unique BCRYPTMD5PASSSALT hashes\n", cnt);
      } else {
        J1T(RC, Dohash, JOB_BCRYPTMD5PASSSALT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_BCRYPTMD5PASSSALT),
               (unsigned char *)"$2a$05$RndSa1tRndSa1tRndSa1tuAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        } else { J1U(RC, Dohash, JOB_BCRYPTMD5PASSSALT); }
      }
    }
        /* SUNMD5: copy JudyJ to Typesalt, free JudyJ */
        { long smcnt = 0;
          line[0] = 0;
          JSLF(PV, JUDYJ(JOB_SUNMD5), (unsigned char *)line);
          while (PV) {
            Word_t *SPV;
            JSLI(SPV, TYPESALT(JOB_SUNMD5), (unsigned char *)line);
            if (SPV && *SPV == 0) *SPV = 1;
            smcnt++;
            JSLN(PV, JUDYJ(JOB_SUNMD5), (unsigned char *)line);
          }
          JSLFA(RC, JUDYJ(JOB_SUNMD5));
          if (smcnt) {
            Numsalts += smcnt;
            fprintf(stderr, "Searching through %ld unique SUNMD5 hashes\n", smcnt);
          } else {
            J1T(RC, Dohash, JOB_SUNMD5);
            if (RC && Printall) {
              JSLI(PV, TYPESALT(JOB_SUNMD5), (unsigned char *)"$md5$rounds=904$defaultSalt$");
              if (PV && *PV == 0) *PV = 1;
            } else { J1U(RC, Dohash, JOB_SUNMD5); }
          }
        }
        /* CRYPTOPPLEGACY / CRYPTOPPDEFAULT: copy JudyJ to Typesalt, free
         * JudyJ, exactly as SUNMD5 above. The record IS the salt here: its
         * leading 8 bytes are the salt and the rest is the ciphertext the
         * oracle reads, so Typesalt carries the whole line and the refcount
         * of 1 is right by construction -- each key maps to exactly one
         * record, which is the condition the MSSQL salt-accounting fix
         * identified as making a hardcoded 1 safe.
         *
         * In -z mode there is normally nothing loaded and
         * init_default_salts() has already put the default record in
         * Typesalt; the insert below is the belt to that brace and is a
         * no-op when it is already there. */
        { int cpz;
          static const int cpz_ops[2] = { JOB_CRYPTOPPLEGACY, JOB_CRYPTOPPDEFAULT };
          static const char *cpz_def[2] = {
            "0123456789ABCDEF0F4127ADFA2E97A8E043BE067112BF796C053027329CB7D5",
            "0123456789ABCDEF325C10900B9C608A24F808F8A3CD67A5DE74E19206E90D899617BA6318EB8AB3" };
          for (cpz = 0; cpz < 2; cpz++) {
            long cpcnt = 0;
            line[0] = 0;
            JSLF(PV, JUDYJ(cpz_ops[cpz]), (unsigned char *)line);
            while (PV) {
              Word_t *SPV;
              JSLI(SPV, TYPESALT(cpz_ops[cpz]), (unsigned char *)line);
              if (SPV && *SPV == 0) *SPV = 1;
              cpcnt++;
              JSLN(PV, JUDYJ(cpz_ops[cpz]), (unsigned char *)line);
            }
            JSLFA(RC, JUDYJ(cpz_ops[cpz]));
            if (cpcnt) {
              Numsalts += cpcnt;
              fprintf(stderr, "Searching through %ld unique %s records\n",
                      cpcnt, TYPENAME(cpz_ops[cpz]));
            } else {
              J1T(RC, Dohash, cpz_ops[cpz]);
              if (RC && Printall) {
                JSLI(PV, TYPESALT(cpz_ops[cpz]), (unsigned char *)cpz_def[cpz]);
                if (PV && *PV == 0) *PV = 1;
              } else { J1U(RC, Dohash, cpz_ops[cpz]); }
            }
          }
        }
        /* GOST12256CRYPT: copy JudyJ to Typesalt, free JudyJ (like GOST12512CRYPT) */
        { long gcnt = 0;
          line[0] = 0;
          JSLF(PV, JUDYJ(JOB_GOST12256CRYPT), (unsigned char *)line);
          while (PV) {
            Word_t *SPV;
            JSLI(SPV, TYPESALT(JOB_GOST12256CRYPT), (unsigned char *)line);
            if (SPV && *SPV == 0) *SPV = 1;
            gcnt++;
            JSLN(PV, JUDYJ(JOB_GOST12256CRYPT), (unsigned char *)line);
          }
          JSLFA(RC, JUDYJ(JOB_GOST12256CRYPT));
          if (gcnt) {
            Numsalts += gcnt;
            fprintf(stderr, "Searching through %ld unique GOST12256CRYPT hashes\n", gcnt);
          } else {
            J1T(RC, Dohash, JOB_GOST12256CRYPT);
            if (RC && Printall) {
              JSLI(PV, TYPESALT(JOB_GOST12256CRYPT), (unsigned char *)"$gost12256hash$defaultS$");
              if (PV && *PV == 0) *PV = 1;
            } else { J1U(RC, Dohash, JOB_GOST12256CRYPT); }
          }
        }
        /* GOST94CRYPT: copy JudyJ to Typesalt, free JudyJ (like GOST12512CRYPT) */
        { long gcnt = 0;
          line[0] = 0;
          JSLF(PV, JUDYJ(JOB_GOST94CRYPT), (unsigned char *)line);
          while (PV) {
            Word_t *SPV;
            JSLI(SPV, TYPESALT(JOB_GOST94CRYPT), (unsigned char *)line);
            if (SPV && *SPV == 0) *SPV = 1;
            gcnt++;
            JSLN(PV, JUDYJ(JOB_GOST94CRYPT), (unsigned char *)line);
          }
          JSLFA(RC, JUDYJ(JOB_GOST94CRYPT));
          if (gcnt) {
            Numsalts += gcnt;
            fprintf(stderr, "Searching through %ld unique GOST94CRYPT hashes\n", gcnt);
          } else {
            J1T(RC, Dohash, JOB_GOST94CRYPT);
            if (RC && Printall) {
              JSLI(PV, TYPESALT(JOB_GOST94CRYPT), (unsigned char *)"$gost94hash$defaultS$");
              if (PV && *PV == 0) *PV = 1;
            } else { J1U(RC, Dohash, JOB_GOST94CRYPT); }
          }
        }
    /* GOST12512CRYPT: copy JudyJ → Typesalt, free JudyJ (like SHA512CRYPT) */
    { long gost12cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_GOST12512CRYPT), (unsigned char *)line);
      while (PV) {
        Word_t *SPV;
        JSLI(SPV, TYPESALT(JOB_GOST12512CRYPT), (unsigned char *)line);
        if (SPV && *SPV == 0) *SPV = 1;
        gost12cnt++;
        JSLN(PV, JUDYJ(JOB_GOST12512CRYPT), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_GOST12512CRYPT));
      if (gost12cnt) {
        Numsalts += gost12cnt;
        fprintf(stderr, "Searching through %ld unique GOST12512CRYPT hashes\n", gost12cnt);
      } else {
        J1T(RC, Dohash, JOB_GOST12512CRYPT);
        if (RC && Printall) {
          JSLI(PV, TYPESALT(JOB_GOST12512CRYPT), (unsigned char *)"$gost12512hash$defaultS$");
          if (PV && *PV == 0) *PV = 1;
        } else { J1U(RC, Dohash, JOB_GOST12512CRYPT); }
      }
    }
    /* YESCRYPT: count entries or disable */
    { long yecnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_YESCRYPT), (unsigned char *)line);
      while (PV) { yecnt++; JSLN(PV, JUDYJ(JOB_YESCRYPT), (unsigned char *)line); }
      if (yecnt) {
        fprintf(stderr, "Searching through %ld unique YESCRYPT hashes\n", yecnt);
      } else {
        J1T(RC, Dohash, JOB_YESCRYPT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_YESCRYPT), (unsigned char *)"$y$j9T$oJqQoBLMgF5$0000000000000000000000000000000000000000000");
          JSLI(PV, TYPESALT(JOB_YESCRYPT), (unsigned char *)"$y$j9T$oJqQoBLMgF5$");
          if (PV && *PV == 0) *PV = 1;
        } else { J1U(RC, Dohash, JOB_YESCRYPT); }
      }
    }
    /* GOST-YESCRYPT: count entries or disable */
    { long gycnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_GOSTYESCRYPT), (unsigned char *)line);
      while (PV) { gycnt++; JSLN(PV, JUDYJ(JOB_GOSTYESCRYPT), (unsigned char *)line); }
      if (gycnt) {
        fprintf(stderr, "Searching through %ld unique GOST-YESCRYPT hashes\n", gycnt);
      } else {
        J1T(RC, Dohash, JOB_GOSTYESCRYPT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_GOSTYESCRYPT), (unsigned char *)"$gy$j9T$2WmURad1wKLkIzjayhv/41$0000000000000000000000000000000000000000000");
          JSLI(PV, TYPESALT(JOB_GOSTYESCRYPT), (unsigned char *)"$gy$j9T$2WmURad1wKLkIzjayhv/41$");
          if (PV && *PV == 0) *PV = 1;
        } else { J1U(RC, Dohash, JOB_GOSTYESCRYPT); }
      }
    }
    /* SHA1CRYPT: count entries or disable */
    { long sccnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SHA1CRYPT), (unsigned char *)line);
      while (PV) { sccnt++; JSLN(PV, JUDYJ(JOB_SHA1CRYPT), (unsigned char *)line); }
      if (sccnt) {
        fprintf(stderr, "Searching through %ld unique SHA1CRYPT hashes\n", sccnt);
      } else {
        J1T(RC, Dohash, JOB_SHA1CRYPT);
        if (RC && Printall) {
          JSLI(PV, JUDYJ(JOB_SHA1CRYPT), (unsigned char *)"$sha1$5000$rndSa1t$0000000000000000000000000000");
          JSLI(PV, TYPESALT(JOB_SHA1CRYPT), (unsigned char *)"$sha1$5000$rndSa1t$");
          if (PV && *PV == 0) *PV = 1;
        } else { J1U(RC, Dohash, JOB_SHA1CRYPT); }
      }
    }
    /* CMIYC: count entries or disable */
    { long cmcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_CMIYC), (unsigned char *)line);
      while (PV) { cmcnt++; JSLN(PV, JUDYJ(JOB_CMIYC), (unsigned char *)line); }
      if (cmcnt) {
        fprintf(stderr, "Searching through %ld unique CMIYC hashes\n", cmcnt);
      } else {
        J1T(RC, Dohash, JOB_CMIYC);
        if (RC && Printall) {
          JSLI(PV, TYPESALT(JOB_CMIYC),
               (unsigned char *)"$cmiyc$2026$1$10$AAAAAAAAAAAAAAAAAAAAAA$");
          if (PV && *PV == 0) *PV = 1;
        } else { J1U(RC, Dohash, JOB_CMIYC); }
      }
    }
    /* 7ZIP: count entries or disable */
    { long szcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SEVENZIP), (unsigned char *)line);
      while (PV) { szcnt++; JSLN(PV, JUDYJ(JOB_SEVENZIP), (unsigned char *)line); }
      if (szcnt) {
        fprintf(stderr, "Searching through %ld unique 7ZIP archives\n", szcnt);
        if (SevenZipRejected || SevenZipWarned) {
          long dec = 0; int q;
          for (q = 0; q < SevenZipCount; q++)
            if (SevenZipRecs[q].tier2) dec++;
          fprintf(stderr,
                  "7ZIP triage: %ld loaded (%ld verifiable by decompress+CRC, "
                  "%ld pad-check only), %ld rejected, %ld below %ld bits\n",
                  (long)SevenZipCount, dec, (long)SevenZipCount - dec,
                  SevenZipRejected, SevenZipWarned, SevenZipMinBits);
        }
      } else {
        if (SevenZipRejected)
          fprintf(stderr,
                  "7ZIP: all %ld archive(s) were rejected at load; nothing to search\n",
                  SevenZipRejected);
        J1T(RC, Dohash, JOB_SEVENZIP);
        if (RC && Printall) {
          /* -z also needs a synthetic ARCHIVE record: the compute loop walks
           * SevenZipRecs, which is empty with no hash file, so a Typesalt
           * group alone produced no output at all. Tail bytes are arbitrary
           * (stage 1 cannot pass against them) -- under Printall the key and
           * decrypted final block are reported regardless, which is the point
           * of -z for this type. */
          static const unsigned char sz_ztail[32] = {
            0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
            0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,
            0x0f,0x1e,0x2d,0x3c,0x4b,0x5a,0x69,0x78,
            0x87,0x96,0xa5,0xb4,0xc3,0xd2,0xe1,0xf0 };
          JSLI(PV, TYPESALT(JOB_SEVENZIP), (unsigned char *)"0::19");
          if (PV && *PV == 0) *PV = 1;
          if (!SevenZipCount) {
            struct SevenZipRec *zr = sevenzip_addrec();
            snprintf(zr->group, sizeof(zr->group), "%s", "0::19");
            memcpy(zr->tail, sz_ztail, 32);
            memcpy(zr->head, sz_ztail, 32);
            zr->havehead = 1;
            zr->padsize = 16;
            zr->bits = 128;
            zr->packed = 16; zr->unpacked = 0;
            zr->line = strdup("$7z$0$19$0$$16$00000000000000000000000000000000$0$16$0$"
                              "00112233445566778899aabbccddeeff0f1e2d3c4b5a69788796a5b4c3d2e1f0");
            if (!zr->line) { fprintf(stderr, "Failed to store 7zip hash line\n"); exit(1); }
          }
        } else { J1U(RC, Dohash, JOB_SEVENZIP); }
      }
    }
    /* -z mode: ensure DES/3DES have 16-char salts in SaltArray */
    if (Printall) {
      int need_des = 0;
      J1T(RC, Dohash, JOB_DESENCRYPT); if (RC) need_des = 1;
      J1T(RC, Dohash, JOB_DES3ENCRYPT); if (RC) need_des = 1;
      if (need_des && !Totsalts) {
        JSLI(PV, SaltArray, (unsigned char *)"1172075784504605");
        JSLI(PV, SaltArray, (unsigned char *)"8152001061460743");
      }
    }
    /* phpBB3 family: the loader has already filed each member's own salts with
     * the real per-salt count, so there is nothing to copy here and no count to
     * invent. phpbb3cnt is the SUM across selected members, which is what makes
     * the disjoint case correct:
     *
     *     -M e455 -F one.txt -M e537 -F two.txt
     *
     * loads two different files into two different salt arrays, and the total
     * is their sum rather than the size of either one. */
    { long phpbb3cnt = 0;
      int fi;
      for (fi = 0; fi < PHPBB3FAMILY_N; fi++) {
        int fop = Phpbb3Family[fi];
        long fcnt = 0;
        J1T(RC, Dohash, fop);
        if (!RC) continue;
        line[0] = 0;
        JSLF(PV, TYPESALT(fop), (unsigned char *)line);
        while (PV) {
          fcnt++;
          JSLN(PV, TYPESALT(fop), (unsigned char *)line);
        }
        /* "salts", not "hashes": the value is the number of distinct Typesalt
         * entries, and with one salt shared by many hashes the old noun read
         * as a hash count that was 11 when 121 hashes were loaded. */
        if (fcnt)
          fprintf(stderr, "Searching through %ld unique %s salts\n",
                  fcnt, TYPENAME(fop));
        phpbb3cnt += fcnt;
      }
      Numsalts += phpbb3cnt;
      if (phpbb3cnt) {
        if (Bcryptcnt == 0)
          Bcryptcnt = phpbb3cnt;
        /* The second add is carried over unchanged from before this rework. It
         * reads like a double count of the same salts in Numsalts, but Numsalts
         * feeds progress and ETA arithmetic, and correcting it is a separate
         * question from the salt-array split. Left as it was, deliberately. */
        Numsalts += phpbb3cnt;
      } else {
        J1T(RC, Dohash, JOB_PHPBB3);
        if (RC && Printall) {
          /* -z mode: generate a random salt so PHPBB3/PHPBB3MD5 can produce output */
          Word_t *SPV;
          unsigned int seed = (unsigned int)time(NULL);
          char dsalt[16];
          dsalt[0] = '$'; dsalt[1] = 'H'; dsalt[2] = '$';
          dsalt[3] = '9'; /* i64hex['9'] = 11, iterations = 2^11 = 2048 */
          for (x = 0; x < 8; x++)
            dsalt[4 + x] = phpitoa64[(seed = seed * 1103515245 + 12345) >> 16 & 63];
          dsalt[12] = 0;
          for (fi = 0; fi < PHPBB3FAMILY_N; fi++) {
            int fop = Phpbb3Family[fi];
            J1T(RC, Dohash, fop);
            if (!RC) continue;
            JSLI(SPV, TYPESALT(fop), (unsigned char *)dsalt);
            if (SPV) *SPV = 1000000;
          }
          Numsalts++;
        } else {
          J1U(RC, Dohash, JOB_PHPBB3);
        }
      }
    }
    /* APR1: insert JudyJ[JOB_APR1] entries into Typesalt */
    { long apr1cnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_APR1), (unsigned char *)line);
      while (PV) {
        Word_t *SPV;
        JSLI(SPV, TYPESALT(JOB_APR1), (unsigned char *)line);
        if (SPV && *SPV == 0) *SPV = 1;
        apr1cnt++;
        JSLN(PV, JUDYJ(JOB_APR1), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_APR1));
      if (apr1cnt) {
        if (Bcryptcnt == 0)
          Bcryptcnt = apr1cnt;
        Numsalts += apr1cnt;
        fprintf(stderr, "Searching through %ld unique APR1 hashes\n", apr1cnt);
      } else {
        J1T(RC, Dohash, JOB_APR1);
        if (RC && Printall) {
          Word_t *SPV;
          JSLI(SPV, TYPESALT(JOB_APR1), (unsigned char *)"$apr1$rndSa1t$");
          if (SPV) *SPV = 1000000;
          Numsalts++;
        } else {
          J1U(RC, Dohash, JOB_APR1);
        }
      }
    }
    /* MD5CRYPT: loader already populated compact table and Typesalt.
     * Count salts and set up for -z mode if needed. */
    { long md5cryptcnt = 0;
      line[0] = 0;
      JSLF(PV, TYPESALT(JOB_MD5CRYPT), (unsigned char *)line);
      while (PV) { md5cryptcnt++; JSLN(PV, TYPESALT(JOB_MD5CRYPT), (unsigned char *)line); }
      Numsalts += md5cryptcnt;
      if (md5cryptcnt) {
        if (Bcryptcnt == 0)
          Bcryptcnt = md5cryptcnt;
        fprintf(stderr, "%ld unique salts for MD5CRYPT\n", md5cryptcnt);
      } else {
        J1T(RC, Dohash, JOB_MD5CRYPT);
        if (RC && Printall) {
          Word_t *SPV;
          JSLI(SPV, TYPESALT(JOB_MD5CRYPT), (unsigned char *)"$1$rndSa1t$");
          if (SPV) *SPV = 1000000;
          Numsalts++;
        } else {
          J1U(RC, Dohash, JOB_MD5CRYPT);
        }
      }
    }
    /* MD5CRYPTMD5: same stored format as MD5CRYPT, so the loader above has
     * already filled Typesalt[JOB_MD5CRYPTMD5] for every selected line.  This
     * mirrors the MD5CRYPT block for the -z / no-hash case. */
    { long md5cryptmd5cnt = 0;
      line[0] = 0;
      JSLF(PV, TYPESALT(JOB_MD5CRYPTMD5), (unsigned char *)line);
      while (PV) { md5cryptmd5cnt++; JSLN(PV, TYPESALT(JOB_MD5CRYPTMD5), (unsigned char *)line); }
      Numsalts += md5cryptmd5cnt;
      if (md5cryptmd5cnt) {
        if (Bcryptcnt == 0)
          Bcryptcnt = md5cryptmd5cnt;
        fprintf(stderr, "%ld unique salts for MD5CRYPTMD5\n", md5cryptmd5cnt);
      } else {
        J1T(RC, Dohash, JOB_MD5CRYPTMD5);
        if (RC && Printall) {
          Word_t *SPV;
          JSLI(SPV, TYPESALT(JOB_MD5CRYPTMD5), (unsigned char *)"$1$rndSa1t$");
          if (SPV) *SPV = 1000000;
          Numsalts++;
        } else {
          J1U(RC, Dohash, JOB_MD5CRYPTMD5);
        }
      }
    }
    /* JUNIPERIVE: insert JudyJ[JOB_JUNIPERIVE] entries into Typesalt */
    { long juniperivecnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_JUNIPERIVE), (unsigned char *)line);
      while (PV) {
        Word_t *SPV;
        JSLI(SPV, TYPESALT(JOB_JUNIPERIVE), (unsigned char *)line);
        if (SPV && *SPV == 0) *SPV = 1;
        juniperivecnt++;
        JSLN(PV, JUDYJ(JOB_JUNIPERIVE), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_JUNIPERIVE));
      if (juniperivecnt) {
        Numsalts += juniperivecnt;
        fprintf(stderr, "Searching through %ld unique JUNIPERIVE hashes\n", juniperivecnt);
      } else if (!Printall) {
        J1U(RC, Dohash, JOB_JUNIPERIVE);
      }
    }
    /* SHA256CRYPT: insert JudyJ[JOB_SHA256CRYPT] entries into Typesalt */
    { long sha256cryptcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SHA256CRYPT), (unsigned char *)line);
      while (PV) {
        Word_t *SPV;
        JSLI(SPV, TYPESALT(JOB_SHA256CRYPT), (unsigned char *)line);
        if (SPV && *SPV == 0) *SPV = 1;
        drop_bare_setting(JOB_SHA256CRYPT, (char *)line);
        sha256cryptcnt++;
        JSLN(PV, JUDYJ(JOB_SHA256CRYPT), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_SHA256CRYPT));
      if (sha256cryptcnt) {
        if (Bcryptcnt == 0)
          Bcryptcnt = sha256cryptcnt;
        Numsalts += sha256cryptcnt;
        fprintf(stderr, "Searching through %ld unique SHA256CRYPT hashes\n", sha256cryptcnt);
      } else {
        J1T(RC, Dohash, JOB_SHA256CRYPT);
        if (RC && Printall) {
          /* Bare settings loaded through -F are the salts to generate with;
           * the default is only for a -z run that supplied none. */
          long sset = typesalt_entries(JOB_SHA256CRYPT);
          if (sset) {
            Numsalts += sset;
            fprintf(stderr, "%ld unique salts for SHA256CRYPT\n", sset);
          } else {
            Word_t *SPV;
            JSLI(SPV, TYPESALT(JOB_SHA256CRYPT), (unsigned char *)"$5$rndSa1t$");
            if (SPV) *SPV = 1000000;
            Numsalts++;
          }
        } else {
          J1U(RC, Dohash, JOB_SHA256CRYPT);
        }
      }
    }
    /* SHA512CRYPT: insert JudyJ[JOB_SHA512CRYPT] entries into Typesalt */
    /* Also insert into Typesalt[JOB_SHA512CRYPTMD5] since it falls through to crypt_round */
    { long sha512cryptcnt = 0;
      line[0] = 0;
      JSLF(PV, JUDYJ(JOB_SHA512CRYPT), (unsigned char *)line);
      while (PV) {
        Word_t *SPV;
        JSLI(SPV, TYPESALT(JOB_SHA512CRYPT), (unsigned char *)line);
        if (SPV && *SPV == 0) *SPV = 1;
        drop_bare_setting(JOB_SHA512CRYPT, (char *)line);
        J1T(RC, Dohash, JOB_SHA512CRYPTMD5);
        if (RC) {
          JSLI(SPV, TYPESALT(JOB_SHA512CRYPTMD5), (unsigned char *)line);
          if (SPV && *SPV == 0) *SPV = 1;
          drop_bare_setting(JOB_SHA512CRYPTMD5, (char *)line);
        }
        sha512cryptcnt++;
        JSLN(PV, JUDYJ(JOB_SHA512CRYPT), (unsigned char *)line);
      }
      JSLFA(RC, JUDYJ(JOB_SHA512CRYPT));
      if (sha512cryptcnt) {
        if (Bcryptcnt == 0)
          Bcryptcnt = sha512cryptcnt;
        Numsalts += sha512cryptcnt;
        fprintf(stderr, "Searching through %ld unique SHA512CRYPT hashes\n", sha512cryptcnt);
      } else {
        J1T(RC, Dohash, JOB_SHA512CRYPT);
        if (RC && Printall) {
          /* As SHA256CRYPT above: supplied settings first, default only
           * when there are none. */
          long sset = typesalt_entries(JOB_SHA512CRYPT);
          if (sset) {
            Numsalts += sset;
            fprintf(stderr, "%ld unique salts for SHA512CRYPT\n", sset);
          } else {
            Word_t *SPV;
            JSLI(SPV, TYPESALT(JOB_SHA512CRYPT), (unsigned char *)"$6$rndSa1t$");
            if (SPV) *SPV = 1000000;
            Numsalts++;
          }
        } else {
          J1U(RC, Dohash, JOB_SHA512CRYPT);
        }
        J1T(RC, Dohash, JOB_SHA512CRYPTMD5);
        if (RC && Printall) {
          long sset = typesalt_entries(JOB_SHA512CRYPTMD5);
          if (sset) {
            Numsalts += sset;
            fprintf(stderr, "%ld unique salts for SHA512CRYPTMD5\n", sset);
          } else {
            Word_t *SPV;
            JSLI(SPV, TYPESALT(JOB_SHA512CRYPTMD5), (unsigned char *)"$6$rndSa1t$");
            if (SPV) *SPV = 1000000;
            Numsalts++;
          }
        } else if (!Printall) {
          J1U(RC, Dohash, JOB_SHA512CRYPTMD5);
        }
      }
    }
    line[0] = 0;x=0;
    JSLF(PV, JUDYJ(JOB_CISCO8),(unsigned char *)line);
    while (PV) {
      x++;
      JSLN(PV, JUDYJ(JOB_CISCO8),(unsigned char *)line);
    }
    if (x) {
      fprintf(stderr, "Searching through %s unique CISCO8 hashes\n", commify(x));
      if (Bcryptcnt == 0)
        Bcryptcnt = x;
    } else {
      J1T(RC, Dohash, JOB_CISCO8);
      if (RC && Printall) {
        char cisco8_def[] = "$8$rndSa1tRndSa1t$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        JSLI(PV, JUDYJ(JOB_CISCO8),(unsigned char *)cisco8_def);
        if (!TYPESALT(JOB_CISCO8)) {
          JSLI(PV, TYPESALT(JOB_CISCO8), (unsigned char *)"rndSa1tRndSa1t");
          if (PV) (*PV)++;
        }
        Bcryptcnt = 1;
      }
    }
    line[0] = 0;x=0;
    JSLF(PV, JUDYJ(JOB_CISCO9),(unsigned char *)line);
    while (PV) {
      x++;
      JSLN(PV, JUDYJ(JOB_CISCO9),(unsigned char *)line);
    }
    if (x) {
      fprintf(stderr, "Searching through %s unique CISCO9 hashes\n", commify(x));
      if (Bcryptcnt == 0)
        Bcryptcnt = x;
    } else {
      J1T(RC, Dohash, JOB_CISCO9);
      if (RC && Printall) {
        char cisco9_def[] = "$9$rndSa1tRndSa1t$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        JSLI(PV, JUDYJ(JOB_CISCO9),(unsigned char *)cisco9_def);
        if (!TYPESALT(JOB_CISCO9)) {
          JSLI(PV, TYPESALT(JOB_CISCO9), (unsigned char *)"rndSa1tRndSa1t");
          if (PV) (*PV)++;
        }
        Bcryptcnt = 1;
      }
    }
    line[0] = 0;x=0;
    JSLF(PV, JUDYJ(JOB_DCC2),(unsigned char *)line);
    while (PV) {
      x++;
      JSLN(PV, JUDYJ(JOB_DCC2),(unsigned char *)line);
    }
    if (x) {
      fprintf(stderr, "Searching through %s unique DCC2 hashes\n", commify(x));
      if (Bcryptcnt == 0)
        Bcryptcnt = x;
    } else {
      J1T(RC, Dohash, JOB_DCC2);
      if (RC && Printall) {
        JSLI(PV, JUDYJ(JOB_DCC2), (unsigned char *)"00000000000000000000000000000000");
        if (!TYPESALT(JOB_DCC2)) {
          JSLI(PV, TYPESALT(JOB_DCC2), (unsigned char *)"10240:6848");
          if (PV) (*PV)++;
        }
        Bcryptcnt = 1;
      }
    }
    line[0] = 0;x=0;
    JSLF(PV, JUDYJ(JOB_PWSAFE3),(unsigned char *)line);
    while (PV) {
      x++;
      JSLN(PV, JUDYJ(JOB_PWSAFE3),(unsigned char *)line);
    }
    if (x) {
      fprintf(stderr, "Searching through %s unique PWSAFE3 hashes\n", commify(x));
      if (Bcryptcnt == 0)
        Bcryptcnt = x;
    } else {
      J1T(RC, Dohash, JOB_PWSAFE3);
      if (RC && Printall) {
        JSLI(PV, JUDYJ(JOB_PWSAFE3), (unsigned char *)"0000000000000000000000000000000000000000000000000000000000000000");
        if (!TYPESALT(JOB_PWSAFE3)) {
          JSLI(PV, TYPESALT(JOB_PWSAFE3), (unsigned char *)"2048:e4e2a590a5e5c8269f57ec04a8a1c0c03da55b311c51236dab8c6b96b0afca02");
          if (PV) (*PV)++;
        }
        Bcryptcnt = 1;
      }
    }
    line[0] = 0;x=0;
    JSLF(PV, JUDYJ(JOB_IKEPSK_MD5),(unsigned char *)line);
    while (PV) {
      x++;
      JSLN(PV, JUDYJ(JOB_IKEPSK_MD5),(unsigned char *)line);
    }
    if (x) {
      fprintf(stderr, "Searching through %s unique IKEPSK-MD5 hashes\n", commify(x));
    } else {
      J1T(RC, Dohash, JOB_IKEPSK_MD5);
      if (RC && Printall) {
        /* default_salts already set up Typesalt; add JudyJ dummy */
        JSLI(PV, JUDYJ(JOB_IKEPSK_MD5), (unsigned char *)"dummy");
      } else if (RC) {
        J1U(RC, Dohash, JOB_IKEPSK_MD5);
      }
    }
    line[0] = 0;x=0;
    JSLF(PV, JUDYJ(JOB_IKEPSK_SHA1),(unsigned char *)line);
    while (PV) {
      x++;
      JSLN(PV, JUDYJ(JOB_IKEPSK_SHA1),(unsigned char *)line);
    }
    if (x) {
      fprintf(stderr, "Searching through %s unique IKEPSK-SHA1 hashes\n", commify(x));
    } else {
      J1T(RC, Dohash, JOB_IKEPSK_SHA1);
      if (RC && Printall) {
        JSLI(PV, JUDYJ(JOB_IKEPSK_SHA1), (unsigned char *)"dummy");
      } else if (RC) {
        J1U(RC, Dohash, JOB_IKEPSK_SHA1);
      }
    }
    line[0] = 0;x=0;
    JSLF(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)line);
    while (PV) {
      x++;
      JSLN(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)line);
    }
    if (x) {
      fprintf(stderr, "Searching through %s unique PBKDF2 hashes\n", commify(x));
      if (Bcryptcnt == 0)
        Bcryptcnt = x;
    } else {
      J1T(RC, Dohash, JOB_PBKDF2_SHA512);
      if (RC && Printall) {
        char pbkdf2_sha256[] = "sha256:100:dGVzdA:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        char pbkdf2_sha1[]   = "sha1:100:dGVzdA:AAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        char pbkdf2_md5[]    = "md5:100:dGVzdA:AAAAAAAAAAAAAAAAAAAAAA==";
        char pbkdf2_sha512[] = "sha512:100:dGVzdA:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";
        char pkcs5s2[]       = "{PKCS5S2}AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        JSLI(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)pbkdf2_sha256);
        JSLI(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)pbkdf2_sha1);
        JSLI(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)pbkdf2_md5);
        JSLI(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)pbkdf2_sha512);
        JSLI(PV, JUDYJ(JOB_PKCS5S2),(unsigned char *)pkcs5s2);
        if (!TYPESALT(JOB_PKCS5S2)) {
          JSLI(PV, TYPESALT(JOB_PKCS5S2), (unsigned char *)"AAAAAAAAAAAAAAAAAAAAAA==");
          if (PV) (*PV)++;
        }
        if (!TYPESALT(JOB_PBKDF2_SHA256)) {
          JSLI(PV, TYPESALT(JOB_PBKDF2_SHA256), (unsigned char *)"sha256:100:dGVzdA");
          if (PV) (*PV)++;
        }
        if (!TYPESALT(JOB_PBKDF2_SHA1)) {
          JSLI(PV, TYPESALT(JOB_PBKDF2_SHA1), (unsigned char *)"sha1:100:dGVzdA");
          if (PV) (*PV)++;
        }
        if (!TYPESALT(JOB_PBKDF2_MD5)) {
          JSLI(PV, TYPESALT(JOB_PBKDF2_MD5), (unsigned char *)"md5:100:dGVzdA");
          if (PV) (*PV)++;
        }
        if (!TYPESALT(JOB_PBKDF2_SHA512)) {
          JSLI(PV, TYPESALT(JOB_PBKDF2_SHA512), (unsigned char *)"sha512:100:dGVzdA");
          if (PV) (*PV)++;
        }
        Bcryptcnt = 5;
      }
    }
    if (Printall && DoCustom) {
      /* -z mode: insert a synthetic SHA1-CUSTOMUSERSALT entry into Typesalt Judy */
      char custom_def[] = "AAAAAAAAAAAAAAAAAAAAAAAAAAA:rndSa1t";
      line[0] = 0;
      JSLF(PV, TYPESALT(JOB_SHA1_CUSTOMUSERSALT), (unsigned char *)line);
      if (!PV) {
        JSLI(PV, TYPESALT(JOB_SHA1_CUSTOMUSERSALT), (unsigned char *)custom_def);
        if (PV) (*PV)++;
      }
    }

  }
  /* Default salts and users for -z mode are handled by
   * default_salts[] and default_users[] tables in init_default_salts() */
  /* Populate Typesalt[] Judy from SaltArray for all SALTJUDY types */
  {
    long Numsalts = 0;
    line[0] = 0;
    JSLF(PV, SaltArray, (unsigned char *)line);
    while (PV) {
      Numsalts++;
      JSLN(PV, SaltArray, (unsigned char *)line);
    }
    if (Numsalts) {
      fprintf(stderr, "%ld total salts in use\n", Numsalts);
      Totsalts = Numsalts;
    }
    if (SaltArray) {
      Word_t ti = 0;
      J1F(RC, Dohash, ti);
      while (RC) {
        /* !TYPESALT(ti) keeps -s from stamping over salts a -F file already
         * supplied. CRYPTOPPLEGACY/CRYPTOPPDEFAULT are exempt because for
         * them TYPESALT is ALSO the record store: the bootstrap above has
         * already copied the records into it out of JudyJ, so the guard would
         * see a non-empty Judy and silently drop every -s candidate key the
         * moment any record was loaded -- which is every real run. procjob
         * separates records from keys structurally, so a mixed Judy is the
         * intended state here, not an accident. */
        if (ti < JOB_DONE && (TYPEOPTS(ti) & TYPEOPT_SALTJUDY) &&
            (TYPEOPTS(ti) & TYPEOPT_NEEDSALT) &&
            (!TYPESALT(ti) || ti == JOB_CRYPTOPPLEGACY ||
             ti == JOB_CRYPTOPPDEFAULT)) {
          { int _sc = 0;
            int _sp = (Numsalts > 500000);
            line[0] = 0;
            JSLF(PV, SaltArray, (unsigned char *)line);
            while (PV) {
              Word_t *TPV;
              JSLI(TPV, TYPESALT(ti), (unsigned char *)line);
              if (TPV) *TPV = 1000000;
              if (_sp && (++_sc % 100000) == 0)
                fprintf(stderr, "Loading salts for %s: %d of %ld\r", TYPENAME(ti), _sc, Numsalts);
              JSLN(PV, SaltArray, (unsigned char *)line);
            }
            if (_sp)
              fprintf(stderr, "Loading salts for %s: %d done                    \n", TYPENAME(ti), _sc);
          }
        }
        J1N(RC, Dohash, ti);
      }
    }
    JSLFA(RC, SaltArray);
  }
  /* -S reference-count floor.  A -S salt carries no pairing information: the
   * hash file is loaded separately, so ANY loaded hash could be the one that
   * uses this salt.  Its occurrence count in the file is a lower bound, not the
   * answer, and the (*TPV)++ this replaced was not even that -- every distinct
   * salt got 1 however many hashes shared it, so the first solve took the count
   * to zero, build_salt_snapshot dropped the salt, and every other hash using
   * it was silently abandoned.  The rule that produced is exact rather than
   * probabilistic and has nothing to do with thread scheduling, which was the
   * first and wrong guess: -M with -S recovered ONE hash per DISTINCT salt and
   * never more.  Measured 2026-10-11 on e386 SHA512PASSSALT with 200 hashes
   * sharing one salt among 40 candidates, 1 of 200 before and 200 of 200 after,
   * identical at -t 1, 2, 4, 8 and default; and on a six-hash fixture over
   * three salts, 3 of 6 before and 6 of 6 after.  regress/regress.sh has
   * carried a comment describing the same arithmetic since 2026-09-05 and works
   * around it by routing each salt through -F attached to its own hash.
   *
   * Cost of the floor, measured on the same fixture: a run that solves nothing
   * does identical work, 8,000 hash calculations either way, because nothing
   * retires in either build.  The run that solves everything goes from 7,801 to
   * 8,000, so 2.6 percent more work recovers 199 hashes.
   *
   * The floor is the number of hashes that could use the salt times the search
   * depth, because -i makes a single hash hittable once per depth and each hit
   * decrements the salt.  That depth factor is the correction the e918 -U work
   * recorded the hard way: under-counting loses hits silently, over-counting
   * only delays retirement, so the estimate is deliberately generous.
   *
   * The hash count comes from the type's OWN JudyJ for a structured type and
   * from Insize, the compact table, for a plain-hex one; the larger of the two
   * is used because a type can be fed through either channel and over-counting
   * is the safe direction.  This cannot be done in the -S handler: -F loads
   * hashes inside getopt in option order, so at -S time the population is not
   * known yet, and -S is documented to come before -F.
   *
   * Applied only to the salts -S actually supplied, which is what DashSSalt is
   * for: a salt that arrived attached to a hash record keeps the exact count
   * store_typesalt built for it.  -s is deliberately untouched -- its 1000000
   * exists precisely so a pooled salt never retires early. */
  if (DashSSalt) {
    Word_t ti = 0;
    J1F(RC, Dohash, ti);
    while (RC) {
      /* Only types that actually consult a salt.  -S inserts into Typesalt for
       * every type the preceding -M named, salted or not, so without this a
       * run like -M e1 -S salts.txt printed a receipt announcing that MD5's
       * salt reference counts had been raised -- for salts MD5 never reads. */
      if (ti < JOB_DONE && DashSSalt[ti] && (TYPEOPTS(ti) & TYPEOPT_NEEDSALT)) {
        unsigned long long nh = 0, floorcnt;
        int depth = (Maxiter > 0) ? Maxiter : 1;
        long nraised = 0;
        line[0] = 0;
        JSLF(PV, JUDYJ(ti), (unsigned char *)line);
        while (PV) { nh++; JSLN(PV, JUDYJ(ti), (unsigned char *)line); }
        if (Insize > nh) nh = Insize;
        floorcnt = nh * (unsigned long long)depth;
        line[0] = 0;
        JSLF(PV, DashSSalt[ti], (unsigned char *)line);
        while (PV) {
          Word_t *TPV;
          JSLG(TPV, TYPESALT(ti), (unsigned char *)line);
          if (TPV && (unsigned long long)*TPV < floorcnt) {
            *TPV = (Word_t)floorcnt;
            nraised++;
          }
          JSLN(PV, DashSSalt[ti], (unsigned char *)line);
        }
        if (nraised)
          fprintf(stderr,
                  "-S: %s: %ld salt reference count(s) raised to %llu "
                  "(%llu hash(es) x %d depth(s))\n",
                  TYPENAME(ti), nraised, floorcnt, nh, depth);
      }
      J1N(RC, Dohash, ti);
    }
    /* Release every set, not just the ones the loop above acted on: a type -M
     * named but that reads no salt is skipped by the NEEDSALT gate, and a type
     * a count-or-disable gate removed from Dohash is never visited at all.
     * Judy allocates outside malloc_lock, so neither would show up in the
     * memory accounting -- the array's only job was to carry the -S salts to
     * this point and it is dead afterwards either way. */
    { int di; Word_t frc;
      for (di = 0; di < JOB_DONE; di++)
        if (DashSSalt[di]) JSLFA(frc, DashSSalt[di]);
    }
  }
  /* Count salts per type; precompute MD5(salt) for Typehashsalt types */
  {
    Word_t ti = 0;
    J1F(RC, Dohash, ti);
    while (RC) {
      if (ti < JOB_DONE && TYPESALT(ti)) {
        int cnt = 0;
        long long bytes = 0;
        int salt_progress = (Numsalts > 500000);
        line[0] = 0;
        JSLF(PV, TYPESALT(ti), (unsigned char *)line);
        while (PV) {
          if (salt_progress && (cnt % 100000) == 0 && cnt > 0)
            fprintf(stderr, "Preparing salts for %s: %d of %u\r", TYPENAME(ti), cnt, Numsalts);
          /* Precompute MD5(salt) for types that need it */
          if (ti == JOB_MD5_MD5SALTMD5PASS || ti == JOB_SHA1_MD5_MD5SALTMD5PASS ||
              ti == JOB_SHA1_MD5_MD5SALTMD5PASS_SALT || ti == JOB_SHA1_MD5PEPPER_MD5SALTMD5PASS) {
            Word_t *HPV;
            JSLG(HPV, Typehashsalt[ti], (unsigned char *)line);
            if (!HPV) {
              unsigned char shash[16]; char *shex; int lc;
              struct saltentry *se;
              se = malloc_lock(sizeof(struct saltentry), "salthash");
              if (strncmp(line, "$HEX[", 5) == 0) {
                lc = get32(line + 5, (unsigned char *)workline, MAXLINE);
                mymd5((char *)workline, lc, shash);
              } else {
                mymd5((char *)line, mystrlen(line), shash);
              }
              shex = (char *)malloc_lock(33, "salthash");
              prmd5_32(shash, shex);
              shex[32] = 0;
              { int _sl = mystrlen(line);
                se->salt = (char *)malloc_lock(_sl + 1, "salthash");
                memcpy(se->salt, line, _sl + 1);
              }
              se->PV = PV;
              se->saltlen = mystrlen(line);
              se->hashsalt = shex;
              se->hashlen = 32;
              JSLI(HPV, Typehashsalt[ti], (unsigned char *)line);
              if (HPV) *HPV = (Word_t)se;
            }
            cnt++; bytes += 33;
          } else {
            cnt++;
            bytes += mystrlen(line) + 1;
          }
          JSLN(PV, TYPESALT(ti), (unsigned char *)line);
        }
        if (salt_progress)
          fprintf(stderr, "Preparing salts for %s: %d done                    \n", Types[ti], cnt);
        Typesaltcnt[ti] = cnt;
        Typesaltbytes[ti] = bytes;
        LIVESALTS(ti) = cnt;
      }
      J1N(RC, Dohash, ti);
    }
  }
  if (UseridJudy) {
      Word_t ti = 0;
      J1F(RC, Dohash, ti);
      while (RC) {
          if (ti < JOB_DONE && (TYPEOPTS(ti) & TYPEOPT_NEEDUSER) && !TYPEUSER(ti)) {
              TYPEUSER(ti) = (void *)UseridJudy;
          }
          J1N(RC, Dohash, ti);
      }
  }

  /* DCC2 (e918) accepts its usernames through -u / -U as well as inside a
   * $DCC2$ wrapper. The hot loop keys on Typesalt entries of the form
   * "iter:username", so fold each supplied userid into Typesalt here, carrying
   * the userid's OWN reference count rather than inventing one:
   *
   *   -F   byte-exact: store_typesalt is called once per hash line
   *   -U   occurrence count from the userid file -- retires when solved
   *   -u   1000000 -- a generic pool, effectively non-retiring
   *
   * This runs after the -u binding above, so reading TYPEUSER alone covers both
   * channels. Counts are ADDED to any salt a $DCC2$ file already contributed,
   * so mixing -F and -U in one run sums rather than clobbers. Typical DCC2
   * lists carry one unique user per hash, which is why a -U count of 1 is
   * correct and lets the salt retire once its hash is solved. */
  { Word_t dti = JOB_DCC2;
    J1T(RC, Dohash, dti);
    if (RC && TYPEUSER(JOB_DCC2)) {
      Pvoid_t dcc_uj = (Pvoid_t)TYPEUSER(JOB_DCC2);
      /* Sized for any stored user key: a canonical $HEX[] key is longer than
       * the name it stands for. */
      char *dcc_u = malloc(MAXLINE), *dcc_b = malloc(MAXLINE);
      char *dcc_s = malloc(MAXLINE), *dcc_c = malloc(2 * MAXLINE + 16);
      Word_t *DUPV, *DSPV;
      int dcc_n = 0, dcc_skip = 0;
      if (!dcc_u || !dcc_b || !dcc_s || !dcc_c) {
        fprintf(stderr, "%s:%d: malloc failed folding DCC2 userids\n", __FILE__, __LINE__);
        exit(1);
      }
      dcc_u[0] = 0;
      JSLF(DUPV, dcc_uj, (unsigned char *)dcc_u);
      while (DUPV) {
        /* The userid's BYTES, then iter:lc(user) re-canonicalised, so a user
         * needing $HEX[] is stored as one whole wrapper that
         * build_salt_snapshot decodes back to iter:bytes. */
        int ul = user_decode(dcc_u, dcc_b), k, off, bad = 0;
        /* A control byte (CR, LF, NUL, ...) cannot appear in the
         * $DCC2$iter#user#hash construct, so such a user is invalid for
         * this type and is skipped, and counted (Waffle, 2026-11-05). */
        for (k = 0; k < ul; k++)
          if ((unsigned char)dcc_b[k] < 0x20 || dcc_b[k] == 0x7f) { bad = 1; break; }
        if (bad) dcc_skip++;
        else if (ul > 0 && ul < 200) {
          off = snprintf(dcc_s, 32, "%d:", DCC2_DEFAULT_ITER);
          for (k = 0; k < ul; k++)
            dcc_s[off + k] = tolower((unsigned char)dcc_b[k]);
          user_canon(dcc_s, off + ul, dcc_c);
          JSLI(DSPV, TYPESALT(JOB_DCC2), (unsigned char *)dcc_c);
          if (DSPV) { *DSPV += *DUPV; dcc_n++; }
        }
        JSLN(DUPV, dcc_uj, (unsigned char *)dcc_u);
      }
      free(dcc_u); free(dcc_b); free(dcc_s); free(dcc_c);
      if (dcc_n)
        fprintf(stderr, "DCC2: %d userid(s) from -u/-U folded into salts "
                        "at %d iterations\n", dcc_n, DCC2_DEFAULT_ITER);
      if (dcc_skip)
        fprintf(stderr, "DCC2: %d userid(s) skipped: a control byte is invalid "
                        "in $DCC2$iter#user#hash\n", dcc_skip);
    }
  }
  if (combo_pepper) {
      Word_t ti = 0;
      J1F(RC, Dohash, ti);
      while (RC) {
          if (ti < JOB_DONE && (TYPEOPTS(ti) & TYPEOPT_NEEDPEPPER) && !Typepepper[ti])
              Typepepper[ti] = combo_pepper;
          J1N(RC, Dohash, ti);
      }
  }



  if (Dupcnt)
    fprintf(stderr, "There were %s duplicate hashes on input\n", commify(Dupcnt));

  tsfprintf(stderr, "Maximum hash chain depth is %u\n", Maxdepth);
  if (Maxdepth > 1 && Printall) {
    unsigned char *hptr;
    printf("Chain exploration (overflow entries)\n");
    CurHash = 0;
    JLF(PV, OverflowHash, CurHash);
    while (PV) {
      chain = (struct Hashchain *) *PV;
      y = 1;
      if (chain && chain->next) {
        while (chain->next) {
          for (x = 0, hptr = (unsigned char *) &CurHash; x < 8; x++, hptr++)
            printf("%02x", *hptr);
          for (x = chain->len - 8, hptr = chain->hash; x > 0; x--, hptr++)
            printf("%02x", *hptr);
          printf(" chain %6d,len=%d\n", y++, chain->len);
          chain = chain->next;
        }
      }
      JLN(PV, OverflowHash, CurHash);
    }
  }

  tsfprintf(stderr, "Minimum hash length is %u characters\n", Minhashlen);
  tsfprintf(stderr, "Using %d cores\n", maxt);

  if (Rotatehash > (Minhashlen + 1)) {
    Rotatehash = Minhashlen;
    fprintf(stderr, "Rotatehash reset to %d\n", Rotatehash);
  }
  if (Recurse)
    argv = dirprocess(&argc, argv);
	 


  lsi = -1;
  J1L(RC, Dohash,lsi);
  if (!RC) {
    fprintf(stderr,"No final hash type found\n");
    exit(1);
  }
  lsi++;
  linehints_count = (int)lsi;
  linehints = malloc_lock(sizeof(struct Linehints)*lsi,"linehints");
  lsi = 0;
  J1F(RC,Dohash,lsi);
  x = 0;
  while (RC) {
    LINEHINTS(lsi).curline = LINEHINTS(lsi).numline = 0;
    LINEHINTS(lsi).gpu = 0;
    for (; bench_rates[x].idx < lsi; x++);

    if (lsi == bench_rates[x].idx) 
      LINEHINTS(lsi).rate = bench_rates[x].rate;
    else
      LINEHINTS(lsi).rate = 1000000LL;
    J1N(RC,Dohash,lsi);
  }


  current_utc_time(&starthash);


  Curfile = "<Unknown>";
  LowSkip = Lowline = 0;
#ifdef GPU_ENABLED
  {
    int gpu_avail = 0;
#if defined(__APPLE__) && defined(METAL_GPU)
    gpu_avail = gpu_metal_available();
#elif defined(CUDA_GPU)
    gpu_avail = cuda_md5salt_available();
#elif defined(OPENCL_GPU)
    gpu_avail = gpu_opencl_available();
#endif
    if (gpu_avail) {
      gpujob_init(maxt * 2 + maxt / 2);
      gpu_batch_lines = gpujob_batch_max();

      /* 2026-05-30 long-mask amendment: pack the per-side run-descriptor
       * streams for the A2/A3/A4 dispatchers ONCE here. Both OpenCL and
       * Metal A2/A3/A4 dispatchers consult the globals below
       * (gpu_kern_a_prep_desc[], gpu_kern_a_app_desc[],
       * gpu_kern_a_prep_dlen, gpu_kern_a_app_dlen,
       * gpu_kern_a_mask_eligible) to decide whether to upload + dispatch.
       *
       * The template/slab kernel path uses a separate upload wire
       * (gpu_opencl_set_mask / gpu_metal_set_mask) and a separate
       * eligibility flag (MAX_MASK_POS_GPU_SIDE check) -- those live
       * in the backend-specific blocks below. */
      if (MaskPrependLen > 0 || MaskAppendLen > 0) {
        int prep_vars = 0, prep_lit = 0;
        int app_vars  = 0, app_lit  = 0;
        gpu_mask_count_vars_lit_bytes(MaskPrependPattern, MaskPrependLen,
                                      &prep_vars, &prep_lit);
        gpu_mask_count_vars_lit_bytes(MaskAppendPattern,  MaskAppendLen,
                                      &app_vars,  &app_lit);
        int kern_a_mask_ok = (prep_vars <= GPU_MASK_VAR_CAP) &&
                             (prep_lit  <= GPU_MASK_LIT_BYTES_CAP) &&
                             (app_vars  <= GPU_MASK_VAR_CAP) &&
                             (app_lit   <= GPU_MASK_LIT_BYTES_CAP);
        if (!kern_a_mask_ok) {
          fprintf(stderr,
            "Mask exceeds A2/A3/A4 per-side cap "
            "(prep vars=%d/%d lit=%d/%d  app vars=%d/%d lit=%d/%d). "
            "A2/A3/A4 GPU path falls back to CPU; template-path "
            "eligibility checked separately. Gate fixture for the "
            "long-mask amendment: -N \"This is triumph, I'm making a "
            "note here: ?a?a?a?a?a?d?d\" (42 lit + 7 var) fits.\n",
            prep_vars, GPU_MASK_VAR_CAP, prep_lit, GPU_MASK_LIT_BYTES_CAP,
            app_vars,  GPU_MASK_VAR_CAP, app_lit,  GPU_MASK_LIT_BYTES_CAP);
        }
        gpu_kern_a_mask_eligible = 0;
        gpu_kern_a_prep_dlen     = 0;
        gpu_kern_a_app_dlen      = 0;
        if (kern_a_mask_ok) {
          int dprep = gpu_pack_mask_descriptor(MaskPrependPattern, MaskPrependLen,
                                               gpu_kern_a_prep_desc,
                                               GPU_MASK_DESC_BYTES_CAP);
          int dapp  = gpu_pack_mask_descriptor(MaskAppendPattern,  MaskAppendLen,
                                               gpu_kern_a_app_desc,
                                               GPU_MASK_DESC_BYTES_CAP);
          if (dprep < 0 || dapp < 0) {
            fprintf(stderr,
              "FATAL: %s:%d gpu_pack_mask_descriptor overflow despite cap "
              "check passed (dprep=%d dapp=%d). Host/kernel buffer-sizing "
              "skew bug; report to mdxfind maintainer.\n",
              __FILE__, __LINE__, dprep, dapp);
            exit(1);
          }
          gpu_kern_a_prep_dlen     = dprep;
          gpu_kern_a_app_dlen      = dapp;
          gpu_kern_a_mask_eligible = 1;
          fprintf(stderr,
            "GPU A2/A3/A4 mask wire-format packed (prep dlen=%d, app dlen=%d, "
            "prep vars=%d lit=%d, app vars=%d lit=%d)\n",
            dprep, dapp, prep_vars, prep_lit, app_vars, app_lit);
        }
      }
#ifdef OPENCL_GPU
      /* Compile GPU kernel families for active hash types */
      { unsigned int fam = 0, finalfam = 0;
        int probe_first_op = -1;
        lsi = 0;
        J1F(RC, Dohash, lsi);
        while (RC) {
	  if (probe_first_op < 0) probe_first_op = (int)lsi;
	  fam = 0;
          switch ((int)lsi) {
            case JOB_MD5SALT: case JOB_MD5UCSALT: case JOB_MD5revMD5SALT:
            case JOB_MD5sub8_24SALT: fam |= 1u << FAM_MD5SALT; break;
            case JOB_MD5SALTPASS: case JOB_MD5PASSSALT: fam |= 1u << FAM_MD5SALTPASS; break;
            /* B12 (2026-05-08): FAM_MD5ITER + FAM_MD5MASK slab kernels retired.
             * FAM_MD5UNSALTED also retired in B9 — kept here as no-op for ABI
             * stability of the FAM_* enum; family_source[] NULL-skips it. */
            case JOB_MD5: case JOB_MD5UC: fam |= (1u << FAM_MD5UNSALTED); break;
            case JOB_MD5RAW: fam |= 1u << FAM_MD5UNSALTED; break;
            case JOB_MD4: case JOB_NTLMH: case JOB_NTLM: case JOB_MD4UTF16: fam |= 1u << FAM_MD4UNSALTED; break;
            case JOB_SHA1: case JOB_SHA1RAW: fam |= 1u << FAM_SHA1UNSALTED; break;
            case JOB_SHA256: case JOB_SHA256RAW: fam |= 1u << FAM_SHA256UNSALTED; break;
            case JOB_SHA224: fam |= 1u << FAM_SHA256UNSALTED; break;
            case JOB_SHA512: case JOB_SHA512RAW: fam |= 1u << FAM_SHA512UNSALTED; break;
            case JOB_SHA384: case JOB_SHA384RAW: fam |= 1u << FAM_SHA512UNSALTED; break;
            case JOB_WRL: fam |= 1u << FAM_WRLUNSALTED; break;
            case JOB_MD6256: fam |= 1u << FAM_MD6256UNSALTED; break;
            case JOB_KECCAK224: case JOB_KECCAK256: case JOB_KECCAK384: case JOB_KECCAK512:
            case JOB_SHA3_224: case JOB_SHA3_256: case JOB_SHA3_384: case JOB_SHA3_512:
              fam |= 1u << FAM_KECCAKUNSALTED; break;
            case JOB_SQL5: fam |= 1u << FAM_SHA1UNSALTED; break;
            case JOB_MD5SHA256: fam |= 1u << FAM_MD5UNSALTED; break;
            case JOB_MYSQL3: fam |= 1u << FAM_MYSQL3UNSALTED; break;
            case JOB_HMAC_MD5: case JOB_HMAC_MD5_KPASS:
              fam |= 1u << FAM_MD5SALT; break;
            case JOB_HMAC_SHA1: case JOB_HMAC_SHA1_KPASS:
              fam |= 1u << FAM_SHA1; break;
            case JOB_HMAC_SHA224: case JOB_HMAC_SHA224_KPASS:
            case JOB_HMAC_SHA256: case JOB_HMAC_SHA256_KPASS:
              fam |= 1u << FAM_SHA256; break;
            case JOB_HMAC_SHA384: case JOB_HMAC_SHA384_KPASS:
            case JOB_HMAC_SHA512: case JOB_HMAC_SHA512_KPASS:
              fam |= 1u << FAM_HMAC_SHA512; break;
            case JOB_HMAC_RMD160: case JOB_HMAC_RMD160_KPASS:
              fam |= 1u << FAM_HMAC_RMD160; break;
            case JOB_HMAC_RMD320: case JOB_HMAC_RMD320_KPASS:
              fam |= 1u << FAM_HMAC_RMD320; break;
            case JOB_HMAC_BLAKE2S:
              fam |= 1u << FAM_HMAC_BLAKE2S; break;
            case JOB_STREEBOG_32: case JOB_STREEBOG_64:
            case JOB_HMAC_STREEBOG256_KPASS: case JOB_HMAC_STREEBOG256_KSALT:
            case JOB_HMAC_STREEBOG512_KPASS: case JOB_HMAC_STREEBOG512_KSALT:
              fam |= 1u << FAM_STREEBOG; break;
            case JOB_PHPBB3: fam |= 1u << FAM_PHPBB3; break;
            case JOB_MD5CRYPT: fam |= 1u << FAM_MD5CRYPT; break;
            case JOB_MD5_MD5SALTMD5PASS: fam |= 1u << FAM_MD5_MD5SALTMD5PASS; break;
            case JOB_SHA1PASSSALT: case JOB_SHA1SALTPASS: case JOB_SHA1DRU: fam |= 1u << FAM_SHA1; break;
            case JOB_SHA256PASSSALT: case JOB_SHA256SALTPASS: fam |= 1u << FAM_SHA256; break;
            /* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS uses sha256_block
             * compression, family is FAM_SHA256 (timing/family bookkeeping). */
            case JOB_SHA224SALTPASS: fam |= 1u << FAM_SHA256; break;
            case JOB_SHA512PASSSALT: case JOB_SHA512SALTPASS: fam |= 1u << FAM_HMAC_SHA512; break;
            /* Phase 2d.5.7-followup (2026-05-13): SHA384SALTPASS uses
             * sha512_block compression (truncated to 12 uint32 digest);
             * family is FAM_HMAC_SHA512 for timing/family bookkeeping,
             * matching SHA512SALTPASS / SHA512PASSSALT siblings. */
            case JOB_SHA384SALTPASS: fam |= 1u << FAM_HMAC_SHA512; break;
            case JOB_SHA512CRYPT: case JOB_SHA512CRYPTMD5:
              fam |= 1u << FAM_SHA512CRYPT; break;
            case JOB_SHA256CRYPT: fam |= 1u << FAM_SHA256CRYPT; break;
            case JOB_DESCRYPT: fam |= 1u << FAM_DESCRYPT; break;
            case JOB_BCRYPT: fam |= 1u << FAM_BCRYPT; break;
            case JOB_RMD160: fam |= 1u << FAM_RMD160UNSALTED; break;
            /* Phase 2d.6.2 (2026-05-16): JOB_RMD320 (816) maps to FAM_RMD160-
             * UNSALTED for timing/family bookkeeping. RMD-160 and RMD-320
             * share the RIPEMD compression family (rmd160_block / rmd320_block
             * both live in metal_common.metal / gpu_common.cl); the bitmask
             * is a coarse family-grouping signal for compile_families pre-
             * compile, not a per-op dispatch key. The OpenCL twin's
             * gpu_family_for_op_internal lets JOB_RMD320 fall through to
             * default -> -1 (compiled lazily via gpu_opencl_template_resolve_-
             * kernel on first dispatch); pre-compile via this bitmask is
             * a Metal-side speed-up only. */
            case JOB_RMD320: fam |= 1u << FAM_RMD160UNSALTED; break;
            case JOB_BLAKE2S256: fam |= 1u << FAM_BLAKE2S256UNSALTED; break;
            /* Phase 2d.7a (2026-05-16): JOB_BLAKE2B256 (845) and JOB_BLAKE2B512
             * (841) map to FAM_BLAKE2S256UNSALTED for compile-families
             * bitmap bookkeeping. Same coarse-family-grouping pattern as
             * JOB_RMD320 -> FAM_RMD160UNSALTED above: not a per-op dispatch
             * key, just a Metal eager-compile pre-compile signal. The OpenCL
             * twin's gpu_family_for_op_internal doesn't route either op
             * (lazy-compiled via gpu_opencl_template_resolve_kernel on first
             * dispatch); the Metal twin compile_families iterates registered
             * families and admits both regardless. */
            case JOB_BLAKE2B256: fam |= 1u << FAM_BLAKE2S256UNSALTED; break;
            case JOB_BLAKE2B512: fam |= 1u << FAM_BLAKE2S256UNSALTED; break;
          }
	  if (fam & FAM_MD5SALT) {
	    LINEHINTS(lsi).lineswanted = UINT_MAX;
	    LINEHINTS(lsi).gpu = 1;
	  }
	  finalfam |= fam;
          J1N(RC, Dohash, lsi);
        }
        if (finalfam) gpu_opencl_compile_families(finalfam);
        /* Phase 6.2: kick the multi-GPU warm-probe as early as possible.
         * gpu_opencl_warm_probe_async uses a synthetic compact table and
         * synthetic ?l^8 mask when the real ones aren't uploaded yet, so
         * the probe runs concurrently with the rest of mdxfind setup
         * (mask parse, BF detection, set_mask). bf_partition_setup waits
         * via gpu_opencl_warm_probe_wait() before polling rates. */
        if (finalfam && probe_first_op >= 0)
          gpu_opencl_warm_probe_async(probe_first_op);
      }
      /* Upload mask descriptor to GPU if mask mode is active (OpenCL).
       *
       * 2026-05-30 long-mask amendment: A2/A3/A4 eligibility + run-
       * descriptor pack is performed ABOVE this block (in the common
       * gpu_avail branch) so both OpenCL and Metal blocks see the same
       * gpu_kern_a_{prep,app}_desc + gpu_kern_a_mask_eligible globals.
       * This block remains the template/slab-path uploader; that path
       * has its own MAX_MASK_POS_GPU_SIDE=16 cap and is unaffected by
       * the amendment. */
      if ((MaskPrependLen > 0 || MaskAppendLen > 0 || MaskLen > 0) && gpujob_available()) {
        int gpu_mask_ok = 1;
        if (MaskPrependLen > MAX_MASK_POS_GPU_SIDE ||
            MaskAppendLen > MAX_MASK_POS_GPU_SIDE ||
            MaskLen > MAX_MASK_POS_GPU_SIDE) {
          fprintf(stderr,
            "Mask exceeds template-path per-side cap (MaskPrependLen=%d "
            "MaskAppendLen=%d MaskLen=%d, template cap=%d). Template/slab "
            "GPU path skipped; A2/A3/A4 eligibility computed separately.\n",
            MaskPrependLen, MaskAppendLen, MaskLen, MAX_MASK_POS_GPU_SIDE);
          gpu_mask_ok = 0;
        }
        static uint8_t mask_sizes[MAX_MASK_POS_GPU_SIDE * 2];
        static uint8_t mask_tables[MAX_MASK_POS_GPU_SIDE * 2][256];
        int npre = 0, napp = 0;
        if (gpu_mask_ok) {

        /* Build prepend tables from MaskClasses */
        for (int mi = 0; mi < MaskPrependLen; mi++) {
          int cid = MaskPrependPattern[mi].classid;
          if (cid == MASK_LITERAL) {
            /* Literal: 1-character table */
            mask_sizes[npre] = 1;
            memset(mask_tables[npre], 0, 256);
            mask_tables[npre][0] = MaskPrependPattern[mi].literal;
            npre++;
          } else if (cid >= 0 && cid < MASK_MAX_CLASSES && MaskClasses[cid].count > 0) {
            mask_sizes[npre] = (uint8_t)MaskClasses[cid].count;
            memcpy(mask_tables[npre], MaskClasses[cid].chars, 256);
            npre++;
          } else {
            gpu_mask_ok = 0; break;
          }
        }
        /* Build append tables */
        if (gpu_mask_ok) {
          for (int mi = 0; mi < MaskAppendLen; mi++) {
            int cid = MaskAppendPattern[mi].classid;
            int ti = npre + napp;
            if (cid == MASK_LITERAL) {
              mask_sizes[ti] = 1;
              memset(mask_tables[ti], 0, 256);
              mask_tables[ti][0] = MaskAppendPattern[mi].literal;
              napp++;
            } else if (cid >= 0 && cid < MASK_MAX_CLASSES && MaskClasses[cid].count > 0) {
              mask_sizes[ti] = (uint8_t)MaskClasses[cid].count;
              memcpy(mask_tables[ti], MaskClasses[cid].chars, 256);
              napp++;
            } else {
              gpu_mask_ok = 0; break;
            }
          }
        }
        /* Legacy single-mask fallback */
        if (gpu_mask_ok && npre == 0 && napp == 0 && MaskLen > 0) {
          for (int mi = 0; mi < MaskLen; mi++) {
            int cid = MaskPattern[mi].classid;
            int ti;
            if (cid == MASK_LITERAL) {
              ti = MaskPrepend ? npre : (npre + napp);
              mask_sizes[ti] = 1;
              memset(mask_tables[ti], 0, 256);
              mask_tables[ti][0] = MaskPattern[mi].literal;
              if (MaskPrepend) npre++; else napp++;
            } else if (cid >= 0 && cid < MASK_MAX_CLASSES && MaskClasses[cid].count > 0) {
              ti = MaskPrepend ? npre : (npre + napp);
              mask_sizes[ti] = (uint8_t)MaskClasses[cid].count;
              memcpy(mask_tables[ti], MaskClasses[cid].chars, 256);
              if (MaskPrepend) npre++; else napp++;
            } else {
              gpu_mask_ok = 0; break;
            }
          }
        }
        if (gpu_mask_ok && (npre > 0 || napp > 0)) {
#if defined(OPENCL_GPU)
          gpu_opencl_set_mask(mask_sizes, mask_tables, npre, napp);
#endif
        }
        }  /* end if (gpu_mask_ok) — 2026-05-26 outer guard for GPU per-side cap */
      }
#endif
#if defined(__APPLE__) && defined(METAL_GPU)
      /* Phase 1 Metal: the metallib was loaded at gpu_metal_init(); no
       * lazy family compile is needed. Call the stub for symmetry with
       * the OpenCL path and to keep mdxfind.c's GPU init flow uniform. */
      gpu_metal_compile_families((unsigned int)-1);
      /* Phase 2c Metal: parallel linehints loop mirroring the OpenCL switch
       * at lines 48225-48297. Without this, JOB_MD5SALT (e31) falls through
       * to the rate/Livesalts floor of 512 words per dispatch (lines
       * 48995-49007), capping batch at 512 instead of 16384 and causing
       * ~32x dispatch overhead at high salt counts. Validated against
       * BENCHMARK.md sm-saltfull (1M salts) 2026-05-13 — without this
       * loop, dev3 M2 Max regressed from 2.65 G/s baseline to ~41 Mh/s.
       * Extend ops list as Phase 2d+ salted siblings land. */
      { Word_t lsi_mt = 0; int RC_mt;
        J1F(RC_mt, Dohash, lsi_mt);
        while (RC_mt) {
          if ((int)lsi_mt == JOB_MD5SALT) {
            LINEHINTS(lsi_mt).lineswanted = UINT_MAX;
            LINEHINTS(lsi_mt).gpu = 1;
          }
          J1N(RC_mt, Dohash, lsi_mt);
        }
      }
      /* Phase 2b Metal mask upload — mirrors the OpenCL mask classify+upload
       * block at lines ~48301-48370. The classification logic is
       * identical (MaskClasses → mask_sizes + mask_tables); only the
       * final upload call differs (gpu_metal_set_mask vs gpu_opencl_set_mask).
       * Kept as a parallel block rather than restructuring the OPENCL_GPU
       * outer gate at line 48216, which wraps family-compile + warm-probe
       * machinery that is OpenCL-only. */
      if ((MaskPrependLen > 0 || MaskAppendLen > 0 || MaskLen > 0) && gpujob_available()) {
        int m_gpu_ok = 1;
        if (MaskPrependLen > MAX_MASK_POS_GPU_SIDE ||
            MaskAppendLen > MAX_MASK_POS_GPU_SIDE ||
            MaskLen > MAX_MASK_POS_GPU_SIDE) {
          fprintf(stderr,
            "Mask exceeds Metal template-path per-side cap (MaskPrependLen=%d "
            "MaskAppendLen=%d MaskLen=%d, template cap=%d). Template/slab "
            "GPU path skipped; A2/A3/A4 eligibility computed separately.\n",
            MaskPrependLen, MaskAppendLen, MaskLen, MAX_MASK_POS_GPU_SIDE);
          m_gpu_ok = 0;
        }
        /* Per-position charset sizes cross to the GPU as uint8_t, so a
         * 256-member class -- ?b, the only built-in with 256 -- is sent as 0
         * and the reader restores it via gpu_mask_wire_size(). The truncation
         * below is therefore DELIBERATE and the cast says so; do not "fix" it
         * by widening this array alone, which would send 256 as a byte and
         * silently wrap it again somewhere else. */
        static uint8_t m_sizes[MAX_MASK_POS_GPU_SIDE * 2];
        static uint8_t m_tables[MAX_MASK_POS_GPU_SIDE * 2][256];
        int npre = 0, napp = 0;
        if (m_gpu_ok) for (int mi = 0; mi < MaskPrependLen; mi++) {
          int cid = MaskPrependPattern[mi].classid;
          if (cid == MASK_LITERAL) {
            m_sizes[npre] = 1; memset(m_tables[npre], 0, 256);
            m_tables[npre][0] = MaskPrependPattern[mi].literal; npre++;
          } else if (cid >= 0 && cid < MASK_MAX_CLASSES && MaskClasses[cid].count > 0) {
            m_sizes[npre] = (uint8_t)MaskClasses[cid].count;
            memcpy(m_tables[npre], MaskClasses[cid].chars, 256); npre++;
          } else { m_gpu_ok = 0; break; }
        }
        if (m_gpu_ok) {
          for (int mi = 0; mi < MaskAppendLen; mi++) {
            int cid = MaskAppendPattern[mi].classid;
            int ti = npre + napp;
            if (cid == MASK_LITERAL) {
              m_sizes[ti] = 1; memset(m_tables[ti], 0, 256);
              m_tables[ti][0] = MaskAppendPattern[mi].literal; napp++;
            } else if (cid >= 0 && cid < MASK_MAX_CLASSES && MaskClasses[cid].count > 0) {
              m_sizes[ti] = (uint8_t)MaskClasses[cid].count;
              memcpy(m_tables[ti], MaskClasses[cid].chars, 256); napp++;
            } else { m_gpu_ok = 0; break; }
          }
        }
        if (m_gpu_ok && npre == 0 && napp == 0 && MaskLen > 0) {
          for (int mi = 0; mi < MaskLen; mi++) {
            int cid = MaskPattern[mi].classid;
            int ti = MaskPrepend ? npre : (npre + napp);
            if (cid == MASK_LITERAL) {
              m_sizes[ti] = 1; memset(m_tables[ti], 0, 256);
              m_tables[ti][0] = MaskPattern[mi].literal;
              if (MaskPrepend) npre++; else napp++;
            } else if (cid >= 0 && cid < MASK_MAX_CLASSES && MaskClasses[cid].count > 0) {
              m_sizes[ti] = (uint8_t)MaskClasses[cid].count;
              memcpy(m_tables[ti], MaskClasses[cid].chars, 256);
              if (MaskPrepend) npre++; else napp++;
            } else { m_gpu_ok = 0; break; }
          }
        }
        if (m_gpu_ok && (npre > 0 || napp > 0)) {
          gpu_metal_set_mask(m_sizes, m_tables, npre, napp);
        }
      }
#endif
    }
  }
#endif

  /* Phase 1a sub-phase 1a.4 (2026-05-21): kernel A4 (brute-force)
   * harness fixture env-mode. Per feedback_metal_bf_chunk_producer_gap.md,
   * the BF chunk producer at line 48890 is gated #if defined(OPENCL_GPU),
   * so the normal `mdxfind -n "?d?d"` CLI invocation cannot exercise A4
   * end-to-end on Metal builds (g->bf_chunk never gets set). This shim
   * synthesizes a jobg with bf_chunk=1 + geometry, then calls the
   * backend dispatcher DIRECTLY. Bypasses the producer entirely so the
   * Python harness sees identical behavior on both backends.
   *
   * Env var: MDXFIND_KERNEL_A_FIXTURE_BF=<num_words>,<bf_mask_start>,
   *          <bf_offset_per_word>,<bf_num_masks>
   *
   * Preconditions when this fires:
   *   - MDXFIND_KERNEL_A_PROTO=1 + MDXFIND_KERNEL_A_VARIANT=4
   *   - -n "<mask>" was parsed during getopt -> MaskAppendPattern + Mask-
   *     Classes[] populated
   *   - GPU subsystem initialized (build_compact_table ran during getopt
   *     processing)
   *   - gpu_*_set_mask uploaded MaskAppendPattern to the slab-style
   *     buffers (used by template kernels, not by A4; A4 uses its own
   *     b_kern_a_mask_* buffers via gpu_opencl_kernel_a_upload_mask_buffers
   *     which lazy-uploads from MaskAppendPattern directly)
   *
   * Behavior: synthesize jobg, dispatch, exit. NO wordlist reading, NO
   * hash file parsing, NO procjob threads. The kernel writes to the
   * MDXFIND_KERNEL_A_TRACE files; harness picks up from there. */
#ifdef GPU_ENABLED
  {
    const char *bf_fixture = NULL;   /* was MDXFIND_KERNEL_A_FIXTURE_BF */
    if (bf_fixture != NULL && *bf_fixture != '\0') {
      uint32_t fx_num_words = 0;
      uint64_t fx_bf_start  = 0;
      uint32_t fx_bf_off    = 0;
      uint32_t fx_bf_nmasks = 0;
      if (sscanf(bf_fixture, "%u,%llu,%u,%u",
                 &fx_num_words,
                 (unsigned long long *)&fx_bf_start,
                 &fx_bf_off, &fx_bf_nmasks) != 4) {
        fprintf(stderr,
            "FATAL: %s:%d MDXFIND_KERNEL_A_FIXTURE_BF parse error: '%s' "
            "(expected num_words,bf_mask_start,bf_offset_per_word,bf_num_masks)\n",
            __FILE__, __LINE__, bf_fixture);
        exit(1);
      }
      if (MaskAppendLen == 0) {
        fprintf(stderr,
            "FATAL: %s:%d MDXFIND_KERNEL_A_FIXTURE_BF requires -n <mask> "
            "(MaskAppendLen=0)\n",
            __FILE__, __LINE__);
        exit(1);
      }
      fprintf(stderr,
          "[fixture-bf] env-mode: num_words=%u bf_mask_start=%llu "
          "bf_offset_per_word=%u bf_num_masks=%u MaskAppendLen=%d\n",
          fx_num_words, (unsigned long long)fx_bf_start,
          fx_bf_off, fx_bf_nmasks, MaskAppendLen);

      /* Acquire a jobg slot. gpujob_get_free_rules blocks until the
       * pool has a free slot; called once here so no contention. */
      struct jobg *fxg = gpujob_get_free_rules("fixture-bf", 0);
      if (fxg == NULL) {
        fprintf(stderr,
            "FATAL: %s:%d MDXFIND_KERNEL_A_FIXTURE_BF: gpujob_get_free_rules "
            "returned NULL\n",
            __FILE__, __LINE__);
        exit(1);
      }
      if (fxg->packed_buf == NULL) {
        fxg->packed_buf = (char *)calloc(1, fxg->packed_buf_size);
        if (fxg->packed_buf == NULL) {
          fprintf(stderr, "FATAL: %s:%d fixture-bf packed_buf calloc failed\n",
                  __FILE__, __LINE__);
          exit(1);
        }
        fxg->word_offset = (uint32_t *)calloc(fxg->word_offset_entries,
                                              sizeof(uint32_t));
        if (fxg->word_offset == NULL) {
          fprintf(stderr, "FATAL: %s:%d fixture-bf word_offset calloc failed\n",
                  __FILE__, __LINE__);
          exit(1);
        }
      }
      /* Synthetic single empty plaintext (mirrors procjob short-circuit
       * at mdxfind.c:10487). All lanes point at the same zero-length
       * slot; A4 ignores packed_buf/word_offset entirely (BF candidate
       * IS the mask expansion). */
      fxg->packed_buf[0]    = 0;   /* plen = 0, low byte  */
      fxg->packed_buf[1]    = 0;   /* plen = 0, high byte (2-byte header) */
      if (fx_num_words > fxg->word_offset_entries) {
        fprintf(stderr,
            "FATAL: %s:%d fixture-bf num_words=%u exceeds word_offset_entries=%u\n",
            __FILE__, __LINE__, fx_num_words, fxg->word_offset_entries);
        exit(1);
      }
      for (uint32_t i = 0; i < fx_num_words; i++) fxg->word_offset[i] = 0;
      fxg->packed_count     = fx_num_words;
      /* Two bytes, per the 2-byte plen=0 header written above; see the
       * BF chunk producer's note on why packed_pos must cover the whole
       * header or the hit replay discards every hit. */
      fxg->packed_pos       = 2;
      fxg->op               = JOB_MD5;          /* arbitrary; A4 ignores */
      fxg->filename         = "fixture-bf";
      fxg->flags            = JOBFLAG_NUMBERS | JOBFLAG_BRUTEFORCE | JOBFLAG_BF_CHUNK;
      fxg->doneprint        = &doneprint;
      fxg->packed           = 1;
      fxg->rules_engine     = 1;
      fxg->bf_chunk            = 1;
      fxg->bf_mask_start       = fx_bf_start;
      fxg->bf_offset_per_word  = fx_bf_off;
      fxg->bf_num_masks        = fx_bf_nmasks;
      fxg->bf_inner_iter       = 1;             /* A4 v1 invariant */
      fxg->bf_fast_eligible    = 0;             /* A4 path, not BF-fast */

      int fx_nhits = 0;
      uint32_t *fx_res = NULL;
#if defined(__APPLE__) && defined(METAL_GPU)
      fx_res = gpu_metal_kernelA_bruteforce_dispatch(0, fxg, &fx_nhits);
#elif defined(OPENCL_GPU)
      fx_res = gpu_opencl_kernel_a_bruteforce_dispatch(0, fxg, &fx_nhits);
#else
      fprintf(stderr, "FATAL: %s:%d MDXFIND_KERNEL_A_FIXTURE_BF: no GPU backend compiled in\n",
              __FILE__, __LINE__);
      exit(1);
#endif
      fprintf(stderr,
          "[fixture-bf] dispatch returned res=%p nhits=%d (trace=%s)\n",
          (void *)fx_res, fx_nhits,
          "(removed)");
      /* Clean exit. The trace files have been written by the dispatcher's
       * gpu_*_kernel_a_trace_dump() call. */
      exit(0);
    }
  }
#endif

#ifndef _WIN32
  signal(SIGUSR1, sig_pause);
  signal(SIGUSR2, sig_resume);
#ifdef GPU_ENABLED
  signal(SIGINT, sig_gpu_exit);
  signal(SIGTERM, sig_gpu_exit);
#endif
#else
  win_pause_init();
#endif
  /* Validator harness phase boundary: rule loader test-applies (which run
   * applyrule(rule,"Password",8,...) at -r/-R parse time, mdxfind.c:41928)
   * fire BEFORE this point. Main word-processing fires AFTER. Harness drops
   * everything before this marker so loader-time CPU records do not appear
   * as CPU-only divergences against the GPU validator. Gated; production
   * stderr is identical when MDXFIND_RULE_VALIDATOR is unset. */
  launch(ReportStats, NULL);

  /* Auto-enable -W auto if MDXFIND_CACHE is set and -W was not specified */
  if (!AutoCountRequested && !EstimatedTotalLines && getenv("MDXFIND_CACHE")) {
    AutoCountRequested = 1;
    fprintf(stderr, "Wordlist ETA: automatic (MDXFIND_CACHE set)\n");
  }

  /* Launch background line counter for -W auto */
  if (AutoCountRequested && argc > 0) {
    LineCountArgv = argv;
    LineCountArgc = argc;
    LineCountDB = linecount_cache_open();  /* open+prepare on main thread (SQLite needs stack) */
    if (LineCountDB) {
      sqlite3_prepare_v2(LineCountDB,
          "SELECT lines FROM linecount WHERE filepath=? AND filesize=? AND mtime=?",
          -1, &LC_stmt_lookup, NULL);
      sqlite3_prepare_v2(LineCountDB,
          "INSERT OR REPLACE INTO linecount (filepath, filesize, mtime, lines)"
          " VALUES (?, ?, ?, ?)",
          -1, &LC_stmt_store, NULL);
    }
    launch(linecount_thread, NULL);
  }

  /* Brute-force detection: if exactly one argument and it is not a file,
   * try parsing it as a mask pattern. */
  if (argc == 1 && strcmp(argv[0], "stdin") != 0) {
    struct stat bf_sb;
    if (stat(argv[0], &bf_sb) != 0) {
      mask_init_classes();
      if (parse_mask_into(argv[0], MaskAppendPattern, &MaskAppendLen,
                          &MaskAppendTotal) == 0) {
        MaskPrependLen = 0; MaskPrependTotal = 1;
        memcpy(MaskPattern, MaskAppendPattern, sizeof(MaskPattern));
        MaskLen = MaskAppendLen;
        MaskTotal = MaskAppendTotal;
        Dodigits = MaskAppendLen;
        Maxnumbers = MaskTotal;
        { int ci;
          MaskChunkSize = 1;
          for (ci = MaskAppendLen - 1; ci >= 0; ci--) {
            if (MaskAppendPattern[ci].classid != MASK_LITERAL) {
              MaskChunkSize = MaskClasses[MaskAppendPattern[ci].classid].count;
              break;
            }
          }
        }
        if (MaskTotal > 1) {
          fprintf(stderr, "Brute force processing detected: %llu candidates from mask '%s'\n",
                  MaskTotal, argv[0]);
          BruteForceMode = 1;
          BruteForceTotal = MaskTotal;
        } else {
          BruteForceMode = 1;  /* single candidate test mode */
          BruteForceTotal = 1;
        }

        /* Set up GPU mask tables if GPU is available.
         *
         * 2026-05-26: per-side cap enforced. If the brute-force append
         * mask exceeds MAX_MASK_POS_GPU_SIDE, skip the upload entirely
         * and route via CPU. The CPU iterator handles up to
         * MAX_MASK_POS=256 per pattern.
         *
         * Why brute force does not get the 2026-05-30 long-mask amendment
         * (investigated 2026-10-26, do not re-derive): that amendment
         * decoupled the budgets into 16 placeholders + 224 literal bytes
         * and both BF kernels have carried uchar appbuf[240] and the
         * descriptor walker since.  "?a?a?a?a?a@gmail10p.com" is 5 vars +
         * 13 literal bytes and IS accepted under -n / -N, which prints
         * "app vars=5 lit=13".  Two things keep BF off it, and only the
         * first looks fixable from here:
         *
         *   1. The gate and pack serving -n / -N sit at the gpu_avail init
         *      path under "if (MaskPrependLen > 0 || MaskAppendLen > 0)",
         *      which runs BEFORE a brute-force mask exists -- a BF mask is
         *      the bare argv[0], parsed in this block, so MaskAppendLen is
         *      still 0 up there and the whole gate is skipped.
         *   2. The decisive one: the BF chunk short-circuit submits through
         *      the RULES-ENGINE slot (gpujob_get_free_rules), and that
         *      kernel reads the per-position tables below, not a descriptor.
         *      Packing a descriptor here and opening the gate therefore
         *      hands the GPU a job it cannot execute.  Measured on dev1 M1,
         *      2026-10-26: all 81,453,056 candidates of a 4-var/13-lit mask
         *      hashed, 0 hits, while -G none found all three planted
         *      answers -- the same shape of silent GPU-ate-the-keyspace
         *      failure that BfMaskGpuUsable exists to prevent.
         *
         * Lifting this needs BF chunks dispatched to the A4 kernel
         * (cand_bruteforce_phase0) rather than the rules-engine kernel, on
         * both backends.  That is a dispatcher change, not a host gate. */
#ifdef GPU_ENABLED
        if (gpujob_available()) {
          static uint8_t bf_mask_sizes[GPU_MASK_SLOT_CAP * 2];
          static uint8_t bf_mask_tables[GPU_MASK_SLOT_CAP * 2][256];
          int bf_napp = 0;
          int bf_ok = 1;
          if (MaskAppendLen > GPU_MASK_SLOT_CAP) {
            /* Name the path and the unit. The count is POSITIONS, literals
             * included, because the brute-force GPU wire packs one table
             * slot per position with literals as degenerate one-character
             * tables -- so "?a?a?a?a?a@gmail10p.com" is 18, not the 5
             * placeholders a reader counts, and shortening the variable
             * part will not help.
             *
             * This is NOT the A2/A3/A4 descriptor wire, which does decouple
             * the two budgets (16 placeholders + 224 literal bytes) and
             * which accepts this very mask under -n / -N. Brute force
             * cannot reach it: the BF chunk short-circuit submits through
             * the RULES-ENGINE slot (gpujob_get_free_rules), and the
             * rules-engine kernel reads the per-position tables uploaded
             * by gpu_opencl_set_mask / gpu_metal_set_mask. Packing a
             * descriptor here and clearing this gate is not sufficient and
             * is actively dangerous -- measured on dev1 M1, 2026-10-26:
             * the GPU accepted the job, hashed all 81,453,056 candidates
             * of a 4-var/13-lit mask and returned 0 hits while the CPU
             * found all three planted answers. Moving BF onto the
             * descriptor wire means dispatching BF chunks to the A4 kernel
             * (cand_bruteforce_phase0) instead of the rules-engine kernel,
             * on both backends. Until then this gate stands. */
            fprintf(stderr,
              "Brute-force mask exceeds the brute-force GPU mask-slot cap "
              "(%d slots needed, one per position incl. literals; cap %d). "
              "BF runs on CPU, across all cores.\n",
              MaskAppendLen, GPU_MASK_SLOT_CAP);
            bf_ok = 0;
          }
          if (bf_ok) for (int mi = 0; mi < MaskAppendLen; mi++) {
            int cid = MaskAppendPattern[mi].classid;
            if (cid == MASK_LITERAL) {
              bf_mask_sizes[bf_napp] = 1;
              memset(bf_mask_tables[bf_napp], 0, 256);
              bf_mask_tables[bf_napp][0] = MaskAppendPattern[mi].literal;
              bf_napp++;
            } else if (cid >= 0 && cid < MASK_MAX_CLASSES && MaskClasses[cid].count > 0) {
              bf_mask_sizes[bf_napp] = (uint8_t)MaskClasses[cid].count;
              memcpy(bf_mask_tables[bf_napp], MaskClasses[cid].chars, 256);
              bf_napp++;
            } else { bf_ok = 0; break; }
          }
          BfMaskGpuUsable = (bf_ok && bf_napp > 0) ? 1 : 0;
          if (bf_ok && bf_napp > 0) {
#if defined(OPENCL_GPU)
            gpu_opencl_set_mask(bf_mask_sizes, bf_mask_tables, 0, bf_napp);
            /* Warm-probe was already kicked off right after
             * gpu_opencl_compile_families (Phase 6.2). It is running with
             * a synthetic mask; bf_partition_setup waits for it. */
#elif defined(__APPLE__) && defined(METAL_GPU)
            gpu_metal_set_mask(bf_mask_sizes, bf_mask_tables, 0, bf_napp);
#endif
          }
        }
#endif

        /* BF Phase 3 (2026-05-10): bf_partition_setup retired. The multi-GPU
         * atomic-cursor partition that this scaffolding fed is gone; per-op
         * chunk-as-job production (below) feeds every active GPU via the
         * normal rules-engine free-list — no pre-partitioning needed. RCS
         * history retains the prior Dohash scan + bf_partition_setup call
         * (mdxfind.c rev 1.449). */
        Totalfiles = 1;
        Curfile = argv[0];
        doneprint = 0;
        Readindex[0].offset = 0;
        Readindex[0].len = 0;
        Readbuf[0] = 0;
        { Word_t NextX = 0; int RC;
          J1F(RC, Dohash, NextX);
          while (RC) {
            x = NextX;
            if (!TYPEDONE(x)) {
              /* BF chunk-as-job + Phase 1.4 adaptive servo (2026-05-10):
               * for ALL GPU_CAT_MASK ops (unsalted + salted post-Phase-2),
               * produce N chunk-jobs per op. Each chunk is sized via
               * adaptive_bf_chunk_size — the rate-EMA servo reads per-
               * device wall_us feedback (written by gpu/gpujob_opencl.c
               * after each clFinish) and sizes the next chunk to target
               * ~5s wall on the slowest active GPU. Warmup ramps target_-
               * seconds 1s -> 5s over the first 10 chunks. Falls back to
               * legacy per-op single-job activation for GPU-NONE ops or
               * when sizing fails.
               *
               * Phase 2 (2026-05-10): unified path. Salted ops use the
               * SAME chunk producer; the salt-axis cap inside adaptive_-
               * bf_chunk_size keeps per-dispatch lanes
               * (chunk_total * salts_per_page) under the 2^31 NDRange
               * ceiling. Salt snapshot is built by gpujob worker from
               * gpu_salt_judy(g->op); BF chunk synthetic plaintext flows
               * through is_salted_pack pagination naturally. */
              int use_bf_chunks = 0;
              int num_devs_v = 1;  /* restored: deleted in rev 1.477, re-added 1.481 */
#if defined(GPU_BF_CHUNK_PRODUCER)
              /* BfMaskGpuUsable: a BF mask the GPU could not be given must not be
               * dispatched to it. Falling through to the legacy arm below is the
               * documented CPU-only path. */
              if (gpujob_available() && BfMaskGpuUsable &&
                  gpu_op_category(x) == GPU_CAT_MASK) {
                num_devs_v = mdx_gpu_num_devices();
                if (num_devs_v < 1) num_devs_v = 1;
                /* Per-op servo reset: rate-EMA seed + chunk counter so
                 * warmup ramp restarts. Different ops may have very
                 * different rates (post-Phase-2 salted families slower);
                 * stale rate from previous op would mis-size the first
                 * chunk of the next op. */
                bf_rate_ema = 1000000000ull;     /* 1 GH/s seed */
                bf_chunks_produced = 0;
                /* Phase 1.4b: per-op clear of the bootstrap-wait flag so
                 * each new op blocks once for chunk-0 feedback before
                 * sizing chunk 1. */
                atomic_store_explicit(&bf_first_feedback_seen, 0,
                                      memory_order_relaxed);
                for (int _d = 0; _d < BF_MAX_GPU_SLOTS; _d++) {
                    atomic_store_explicit(&bf_dev_wall_us[_d], 0,
                                          memory_order_relaxed);
                    atomic_store_explicit(&bf_dev_chunk_total[_d], 0,
                                          memory_order_relaxed);
                    /* Phase 1.7c: clear per-slot first-dispatch flag so
                     * the JIT-contaminated wall_us of the first dispatch
                     * per slot is excluded from the rate-EMA again on
                     * each new BF op. */
                    atomic_store_explicit(&bf_dev_first_dispatch_done[_d], 0,
                                          memory_order_relaxed);
                }
                /* BF Phase 1.6 (2026-05-09): seed bf_rate_ema from
                 * persisted per-(device, op) sidecar(s). MIN across all
                 * device readings is conservative — under-seed grows fast
                 * via EMA, over-seed risks oversize first chunk and big
                 * retry penalties. Bootstrap wait is intentionally NOT
                 * skipped (defensive: disk value might be stale; let the
                 * first-feedback poll fire as usual; the EMA adjusts
                 * upward from the disk seed within a few chunks). */
                {
                    uint64_t disk_min = 0;
                    int hits = 0;
                    for (int _d = 0; _d < num_devs_v && _d < BF_MAX_GPU_SLOTS; _d++) {
                        char p[1024];
                        if (bf_sidecar_path(_d, x, p, sizeof(p)) != 0) continue;
                        uint64_t r = 0;
                        if (bf_sidecar_read_rate(p, &r) != 0) continue;
                        if (hits == 0 || r < disk_min) disk_min = r;
                        hits++;
                    }
                    if (hits > 0 && disk_min >= BF_SIDECAR_RATE_MIN) {
                        bf_rate_ema = disk_min;
                        fprintf(stderr,
                            "[bf-servo] sidecar seed: op=%d devs_seen=%d "
                            "rate_ema_MHs=%.2f (MIN-of-disk)\n",
                            x, hits, (double)bf_rate_ema / 1e6);
                    }
                }
                use_bf_chunks = 1;
                fprintf(stderr,
                    "BF chunk producer (adaptive): op=%d mask_total=%llu "
                    "num_devs=%d (Phase 1.4 rate-EMA servo)\n",
                    x, (unsigned long long)MaskTotal, num_devs_v);
              }
#endif
              if (use_bf_chunks) {
                /* Produce one job per chunk; recompute chunk size before
                 * each submission via the rate-EMA servo. Each chunk
                 * advances the cursor by chunk_total candidates (clamped
                 * on tail). */
                uint64_t chunk_cursor = 0;
                while (chunk_cursor < MaskTotal) {
                  uint64_t this_chunk = 0;
                  uint32_t this_nwords = 0;
                  uint32_t this_mspw   = 0;
                  uint32_t this_inner_iter = 1u;
#if defined(GPU_BF_CHUNK_PRODUCER)
                  if (adaptive_bf_chunk_size(MaskTotal, chunk_cursor,
                                             num_devs_v, x,
                                             &this_chunk,
                                             &this_nwords,
                                             &this_mspw,
                                             &this_inner_iter) != 0 ||
                      this_chunk == 0 || this_nwords == 0 || this_mspw == 0) {
                    /* Sizing failed mid-stream — break and let the goto
                     * bf_done logic flush the queue with what we already
                     * submitted. */
                    fprintf(stderr,
                        "BF servo: adaptive_bf_chunk_size failed at "
                        "cursor=%llu; halting chunk production for op=%d\n",
                        (unsigned long long)chunk_cursor, x);
                    break;
                  }
#endif

                  possess(FreeWaiting);
                  wait_for(FreeWaiting, NOT_TO_BE, 0);
                  job = FreeHead;
                  if (!job) { fprintf(stderr, "Job null on freehead!\n"); exit(1); }
                  FreeHead = job->next;
                  job->next = NULL;
                  if (FreeHead == NULL) FreeTail = &FreeHead;
                  twist(FreeWaiting, BY, -1);

                  job->op = x;
                  job->flags = JOBFLAG_NUMBERS | JOBFLAG_BRUTEFORCE |
                               JOBFLAG_BF_CHUNK;
                  job->Numbers = 0;
                  job->digits = Dodigits;
                  job->MaskIndex = chunk_cursor;     /* bf_mask_start */
                  job->MaskCount = this_chunk;       /* chunk's candidate count */
                  /* BF Phase 1.8 (2026-05-10): bf_offset_per_word is the
                   * per-word total mask coverage (== mspw_per_iter * inner_iter).
                   * bf_num_masks is the per-iter mask range (kernel mask_size).
                   * For inner_iter==1 these match (bit-identical to pre-1.8). */
                  job->bf_offset_per_word = this_mspw * this_inner_iter;
                  job->bf_num_masks       = this_mspw;
                  job->bf_inner_iter      = this_inner_iter;
                  /* Phase 1.9 Tranche A1 (2026-05-10): pre-qualify this
                   * BF chunk for the gpu_md5_bf.cl fast template kernel
                   * if all conditions hold:
                   *   - op == JOB_MD5 (unsalted MD5 only in A1)
                   *   - Numrules <= 1 (no real user rules; synthetic
                   *     NUL-only rule program is the "no-rule" pass)
                   *   - unsalted (Typesalt[op] == NULL)
                   *   - append-only mask: MaskPrependLen == 0 AND
                   *     MaskAppendLen in [1, 8]
                   *   - env override MDXFIND_GPU_FAST_DISABLE unset
                   * The dispatch site (gpu_opencl_dispatch_md5_rules)
                   * additionally requires the kernel resolver to land
                   * on d->kern_template_phase0 (so it doesn't displace
                   * a salted or non-MD5 kernel) and the BF-fast compile
                   * + lazy to succeed; otherwise it silently keeps the
                   * slow template kernel.
                   *
                   * A2-A4 widen this only via in-kernel optimizations;
                   * the host-side gate stays this restrictive (BF +
                   * unsalted MD5 + no real rules + small append mask is
                   * the dominant raw-MD5 BF workload we are tuning). */
                  {
                      int _bf_fast_ok =
                          (x == JOB_MD5) &&
                          (Numrules <= 1) &&
                          (TYPESALT(x) == NULL) &&
                          (MaskPrependLen == 0) &&
                          (MaskAppendLen >= 1) && (MaskAppendLen <= 8);
                      /* MDXFIND_GPU_FAST_DISABLE used to veto the fast path
                       * here. */
                      job->bf_fast_eligible = (unsigned int)(_bf_fast_ok ? 1 : 0);
                  }
                  /* num_words derived in procjob from MaskCount/bf_num_masks
                   * (see procjob short-circuit); this_nwords is computed
                   * by the servo for documentation/trace consistency. */
                  (void)this_nwords;
                  job->doneprint = &doneprint;
                  job->filename = argv[0];
                  job->startline = 0;
                  job->numline = 1;
                  job->readbuf = Readbuf;
                  job->readindex = Readindex;
                  possess(ReadBufL[0]);
                  twist(ReadBufL[0], BY, +1);
                  if (ThreadCount < maxt) {
                    launch(procjob, NULL);
                    ThreadCount++;
                  }
                  possess(WorkWaiting);
                  *WorkTail = job;
                  WorkTail = &(job->next);
                  twist(WorkWaiting, BY, +1);

                  chunk_cursor += this_chunk;
                }
#if defined(GPU_BF_CHUNK_PRODUCER)
                /* BF Phase 1.6 (2026-05-09): persist converged per-device
                 * rate to sidecar so the next run cold-starts with a real
                 * seed instead of the 1 GH/s default. Only write when the
                 * servo has had time to converge (>= BF_SIDECAR_MIN_CHUNKS
                 * dispatched). Per-device atomic samples are the LAST
                 * completion observed; in-flight chunks at this point may
                 * leave some devices stale, but we skip rate==0 cases
                 * (no completion yet) and atomic-rename means partial
                 * coverage is acceptable.
                 *
                 * OPENCL_GPU gate added 2026-05-10: pairs with the sidecar
                 * READ block at ~line 48518 which is already guarded.
                 * Without this guard, Win ARM64 / Metal-only builds fail to
                 * link (BF_SIDECAR_MIN_CHUNKS, BF_SIDECAR_RATE_MIN/MAX,
                 * bf_sidecar_path, bf_sidecar_write_rate undeclared). */
                if (bf_chunks_produced > BF_SIDECAR_MIN_CHUNKS) {
                    int wrote = 0;
                    for (int _d = 0; _d < num_devs_v && _d < BF_MAX_GPU_SLOTS; _d++) {
                        uint64_t w = atomic_load_explicit(&bf_dev_wall_us[_d],
                                                         memory_order_relaxed);
                        uint64_t c = atomic_load_explicit(&bf_dev_chunk_total[_d],
                                                         memory_order_relaxed);
                        if (w == 0 || c == 0) continue;
                        /* rate_d = candidates / wall_seconds */
                        double rate_d = (double)c * 1e6 / (double)w;
                        if (rate_d <= 0.0) continue;
                        uint64_t rate_u = (uint64_t)rate_d;
                        if (rate_u < BF_SIDECAR_RATE_MIN ||
                            rate_u > BF_SIDECAR_RATE_MAX) continue;
                        char p[1024];
                        if (bf_sidecar_path(_d, x, p, sizeof(p)) != 0) continue;
                        if (bf_sidecar_write_rate(p, rate_u) == 0) wrote++;
                    }
                    if (wrote > 0) {
                        fprintf(stderr,
                            "[bf-servo] sidecar write: op=%d devs_written=%d "
                            "chunks_produced=%u\n",
                            x, wrote, bf_chunks_produced);
                    }
                }
#endif
              } else {
                /* Legacy per-op CPU activation (non-GPU ops, or sizing
                 * failure). Post-Phase-2 (2026-05-10) salted
                 * GPU_CAT_MASK ops use the chunk producer above; this
                 * arm now only fires when gpujob is unavailable, the
                 * op has GPU_CAT_NONE, or adaptive_bf_chunk_size
                 * failed. CPU procjob handles iteration.
                 *
                 * Fan the keyspace out over chunk jobs rather than
                 * submitting one job of MaskTotal.  One job is one
                 * procjob thread, so a CPU-only brute force ran on a
                 * single core -- a 22-position mask over 1e9 candidates
                 * took 215s at 4.65 MH/s on fpga with fifteen of its
                 * sixteen cores idle.  It also reported nothing while it
                 * ran: procjob's only per-second checkpoint sits at the
                 * top of the per-line loop and a brute-force job carries
                 * numline == 1, so the body runs once with the counters
                 * still zero and the whole mask is walked inside the
                 * inner do/while, which has no checkpoint of its own.
                 * Tothash, Totfound, TotLines and job->outbuf were
                 * therefore all held until the job ended: the display
                 * read 0.0/1.0G at 0.00 h/s with Found=0 and an ETA of
                 * done for the entire run, and a hash recovered at mask
                 * index 500 was not printed for 215 seconds.  Chunking
                 * fixes both, because every chunk completion flushes all
                 * four through the job-exit accounting -- no change to
                 * the hot loop, which is why the chunk size is bounded
                 * in candidates rather than left proportional.
                 *
                 * The worker already supports this shape: mask_base
                 * captures job->MaskIndex once per job and the three
                 * rule-advance sites restart there, so a chunk owns
                 * [MaskIndex, MaskIndex + MaskCount) under rules too.
                 * The wordlist+mask path fans out the same way and the
                 * GPU arm above does it per device.
                 *
                 * Fan out ONLY where no mask-iterating GPU dispatch can
                 * follow, which is the question the FRAGILE block at the
                 * wordlist fan-out insists on asking: can this op reach a
                 * GPU kernel?  Three of this arm's routes answer no, and
                 * they are the whole of the reported problem:
                 *   - no gpujob at all, so nothing can be dispatched;
                 *   - BfMaskGpuUsable clear, which means the mask passed
                 *     MAX_MASK_POS_GPU_SIDE and therefore also fails
                 *     mask_b71_eligible's MaskAppendLen <= 16, so no GPU
                 *     mask path will take it either (the reported case);
                 *   - GPU_CAT_NONE, documented in gpujob.h as "not GPU-
                 *     capable", the FRAGILE block's fan-out case.
                 * A GPU-capable non-MASK category reaching this arm with a
                 * short mask keeps the single job, so its behaviour is
                 * unchanged: on Metal the mask-iterating path ignores the
                 * chunk range and uses the full gpu_mask_total, which is
                 * why 1.600 confined the NDRange cap to OPENCL_GPU, and
                 * chunking into it would multiply the work by the chunk
                 * count instead of dividing it.  Widening to that case
                 * needs a Metal host to validate on, not an argument. */
                int bf_fanout_ok = 1;
#if defined(GPU_BF_CHUNK_PRODUCER)
                bf_fanout_ok = (!gpujob_available() || !BfMaskGpuUsable ||
                                gpu_op_category(x) == GPU_CAT_NONE);
#endif
                unsigned long long bf_chunk, bf_start;
                /* Enough chunks per core to balance the tail and to keep
                 * the per-completion flush frequent, capped so the
                 * reporting interval does not grow with the keyspace. */
                bf_chunk = MaskTotal / ((unsigned long long)maxt * 32ULL);
                if (bf_chunk == 0ULL) bf_chunk = 1ULL;
                if (bf_chunk > (1ULL << 22)) bf_chunk = 1ULL << 22;
                if (!bf_fanout_ok) bf_chunk = MaskTotal;  /* 1.640 behaviour */
                for (bf_start = 0ULL; bf_start < MaskTotal;
                     bf_start += bf_chunk) {
                  unsigned long long bf_size = MaskTotal - bf_start;
                  if (bf_size > bf_chunk) bf_size = bf_chunk;

                  possess(FreeWaiting);
                  wait_for(FreeWaiting, NOT_TO_BE, 0);
                  job = FreeHead;
                  if (!job) { fprintf(stderr, "Job null on freehead!\n"); exit(1); }
                  FreeHead = job->next;
                  job->next = NULL;
                  if (FreeHead == NULL) FreeTail = &FreeHead;
                  twist(FreeWaiting, BY, -1);

                  job->op = x;
                  job->flags = JOBFLAG_NUMBERS | JOBFLAG_BRUTEFORCE;
                  /* Chunk 0 owns the single seed line; the rest must not
                   * re-retire it, or TotLines reports the chunk count
                   * instead of 1. */
                  if (bf_start > 0ULL)
                    job->flags |= JOBFLAG_MASK_SUBCHUNK;
                  job->Numbers = 0;
                  job->digits = Dodigits;
                  job->MaskIndex = bf_start;
                  job->MaskCount = bf_size;
                  job->bf_offset_per_word = 0;
                  job->bf_num_masks       = 0;
                  /* Phase 1.9 Tranche A1: CPU-only path; no GPU dispatch
                   * happens here, so the flag is informational only --
                   * kept zero for defensive consistency. */
                  job->bf_fast_eligible   = 0;
                  job->doneprint = &doneprint;
                  job->filename = argv[0];
                  job->startline = 0;
                  job->numline = 1;
                  job->readbuf = Readbuf;
                  job->readindex = Readindex;
                  possess(ReadBufL[0]);
                  twist(ReadBufL[0], BY, +1);
                  if (ThreadCount < maxt) {
                    launch(procjob, NULL);
                    ThreadCount++;
                  }
                  possess(WorkWaiting);
                  *WorkTail = job;
                  WorkTail = &(job->next);
                  twist(WorkWaiting, BY, +1);
                }
              }
            }
            J1N(RC, Dohash, NextX);
          }
        }
        Totallines = 1;
        goto bf_done;
      } else {
        fprintf(stderr, "Could not parse '%s' as a mask pattern, exiting\n", argv[0]);
        exit(1);
      }
    }
  }

  for (y = 0; y < argc; y++) {
    doneprint = 0;
    if (isstdin == 0 && strcmp(argv[y], "stdin") == 0) {
      zfi = gzdopen(0,"rb");
      Curfile = "stdin";
    } else {
      if (stat(argv[y], &sb)) {
        perror(argv[y]);
        continue;
      }
      if (S_ISDIR(sb.st_mode)) {
        fprintf(stderr, "File %s is a directory, ignored\n", argv[y]);
        continue;
      }
      creader_reset();
      zfi = gzopen(argv[y], "rb");
      Curfile = argv[y];
      if (S_ISREG(sb.st_mode))
        strncpy(Creader.filename, argv[y], sizeof(Creader.filename) - 1);
    }
    if (!zfi) {
      perror(argv[y]);
      fprintf(stderr, "Can't open %s, skipping\n", argv[y]);
      continue;
    }
    gzbuffer(zfi,128*1024);
    Totalfiles++;
    Fileline = 0;
    LowSkip = Lowline = 0;
    /* -w skip-N-lines perf instrumentation: capture start timestamp at file
     * open if SkipLine > 0; emit elapsed + read rate when SkipLine drops
     * to 0 inside the cacheline read loop. Bytes read via CurfileBytesRead
     * (atomic, accumulated by cacheline). */
    struct timespec skip_start_ts = {0};
    int skip_was_active = 0;
    if (SkipLine) {
        clock_gettime(CLOCK_MONOTONIC, &skip_start_ts);
        skip_was_active = 1;
    }
    /* Per-file flag: fire the cache-expunge DELETE at most once per file even
     * if the detection block below triggers across many cacheline batches.
     * Reset on every new file (we want to expunge a poisoned row for each
     * file we discover stale). */
    int cache_expunged = 0;
    /* Live byte-position tracking for ETA self-correction. CurfileBytesTotal
     * = file size from stat above (sb is populated in the non-stdin branch);
     * 0 for stdin since size is unknown. CurfileBytesRead is reset to 0 and
     * accumulated by cacheline() as it reads chunks. */
    {
      unsigned long long fsize = 0;
      if (Curfile && strcmp(Curfile, "stdin") != 0) fsize = (unsigned long long)sb.st_size;
      atomic_store_explicit(&CurfileBytesTotal, fsize, memory_order_relaxed);
      atomic_store_explicit(&CurfileBytesRead,  0,     memory_order_relaxed);
    }
    while ((Linecount = cacheline(zfi, &readbuf, &readindex))) {
      unsigned long long chunk_skip_offset = 0;
      curline = 0;
      Totallines += Linecount;
      Fileline += Linecount;
      /* If we've passed the cached line-count estimate (poisoned partial
       * count from a prior session), project a new estimate from byte
       * position and bump the global so ReportStats's ETA stays sensible. */
      {
        unsigned long long auto_now = AutoCountTotalLines;
        /* COMMENSURABLE COMPARISON. Fileline is reset per file at the top of
         * this loop; AutoCountTotalLines accumulates across every file the
         * counter thread has reached. Comparing them directly mixed a
         * file-local counter with a multi-file total: on file 1 of a set, with
         * the global still 0, file 1's projection landed in the global and the
         * thread then added all N files on top; once the thread had finished,
         * the test was false even where a projection was wanted. Project the
         * CURRENT file and add the lines already completed in earlier files,
         * which is the same quantity the counter is accumulating. */
        unsigned long long prior_files = (Totallines >= Fileline)
                                       ? (Totallines - Fileline) : 0ULL;
        if (Fileline > 0) {
          unsigned long long fsize = atomic_load_explicit(&CurfileBytesTotal, memory_order_relaxed);
          unsigned long long fread = atomic_load_explicit(&CurfileBytesRead,  memory_order_relaxed);
          unsigned long long projected;
          if (fsize > 0 && fread > 0 && fread < fsize)
            projected = (unsigned long long)((double)Fileline * (double)fsize / (double)fread);
          else
            projected = Fileline;   /* no byte info — at minimum we know there are Fileline lines */
          projected += prior_files;
          /* Grow the PROJECTION monotonically. Never touches
           * AutoCountTotalLines: that belongs to the counter thread alone. */
          while (1) {
            unsigned long long old = ProjectedTotalLines;
            if (projected <= old) break;
            if (__sync_bool_compare_and_swap(&ProjectedTotalLines, old, projected)) break;
          }
          /* Expunge the poisoned cache row so the next session re-counts
           * from scratch instead of reusing the stale value. Gated to fire
           * once per file (cache_expunged sticky flag). Only acts when:
           * (a) cache lookup succeeded in this session — i.e. auto_now > 0
           *     (a fresh recompute path has auto_now == projected and
           *     wouldn't reach this block);
           * (b) the projection is materially larger (>= 2x) than the cached
           *     value — avoids fluttering on near-equality;
           * (c) Curfile is a real file path (not "stdin"). */
          /* AutoCountDone added 2026-10-06: auto_now is a RUNNING SUM while
           * the counter thread works, so comparing a projection against it
           * mid-accumulation could expunge a perfectly good cache row. Only
           * judge a cached value once the count has settled.
           * KNOWN LIMIT, not fixed here: auto_now is still the multi-file
           * total while `projected` concerns one file, so on a multi-file run
           * this test remains the wrong comparison. Making it right needs the
           * per-file cached value in this loop, which it does not have. */
          if (!cache_expunged && AutoCountDone && auto_now > 0 &&
              projected >= 2ULL * auto_now &&
              Curfile && strcmp(Curfile, "stdin") != 0) {
            char resolved[PATH_MAX];
#ifdef _WIN32
            const char *expunge_path = _fullpath(resolved, Curfile, PATH_MAX) ? resolved : Curfile;
#else
            const char *expunge_path = realpath(Curfile, resolved) ? resolved : Curfile;
#endif
            linecount_cache_delete(expunge_path);
            fprintf(stderr,
                "linecount cache: row for %s expunged (stale: %llu cached vs %llu projected, %.1fx)\n",
                expunge_path, auto_now, projected,
                (double)projected / (double)auto_now);
            cache_expunged = 1;
          }
        }
      }
      if (SkipLine) {
        if (Fileline < SkipLine)
          continue;
        Totallines -= SkipLine;
        chunk_skip_offset = SkipLine - (Fileline - Linecount);
        if (chunk_skip_offset > Linecount)
          chunk_skip_offset = Linecount;
	LowSkip = SkipLine;
        SkipLine = 0;
        if (skip_was_active) {
          struct timespec skip_end_ts;
          clock_gettime(CLOCK_MONOTONIC, &skip_end_ts);
          double skip_ms = (skip_end_ts.tv_sec - skip_start_ts.tv_sec) * 1000.0
                         + (skip_end_ts.tv_nsec - skip_start_ts.tv_nsec) / 1e6;
          unsigned long long bytes = atomic_load_explicit(&CurfileBytesRead,
                                                          memory_order_relaxed);
          double mb = (double)bytes / (1024.0 * 1024.0);
          double mbps = (skip_ms > 0.0) ? mb * 1000.0 / skip_ms : 0.0;
          tsfprintf(stderr,
              "-w skip: %llu lines (%.2f MB) in %.0fms = %.1f MB/s\n",
              LowSkip, mb, skip_ms, mbps);
          skip_was_active = 0;
        }
      }
      Numbers = 0;

      /* Reset per-type line cursors for this chunk */
      lsi = 0;
      J1F(RC, Dohash, lsi);
      while (RC) {
	LINEHINTS(lsi).curline = chunk_skip_offset;
	/* NOTE: retired_line is NOT reset here — it must remain monotonic across
	 * chunks for the rate calculation in the 15s tick at 38866 to work
	 * (rate = (now - lasttick)/15s would go negative on reset, dropping to
	 * 0 and falling through to the broken bootstrap path). Reset only
	 * happens implicitly at start-of-run when malloc_lock zero-inits. */
	J1N(RC, Dohash, lsi);
      }
      Lowline = 0;
      isdone = 0;
      while (!isdone) {
        Numbers = 0;
        NextX = 0;

	isdone = 1;
        J1F(RC, Dohash, NextX);
        while (RC) {
          x = NextX;
	  curline = LINEHINTS(x).curline;
          if (TYPEDONE(x) || curline >= Linecount) { 
	    J1N(RC, Dohash, NextX); 
	    continue; 
	  }
	  if (LINEHINTS(x).gpu && !NoMetal && !Rules) {
	    /* Phase 2f experiment 2026-05-18: cap numline at 32K for GPU+salted
	     * to preserve in-order jobg slot packing. Was ULLONG_MAX (one big
	     * job per chunk → procjob #2 picks up the NEXT chunk concurrently,
	     * breaking line order in jobg). 32K matches jobg slot size so each
	     * procjob job produces exactly one ordered slot. Per user 2026-05-18
	     * dispatch goal: approach 1000s on dev1 e31+sm-saltfull+rockyou
	     * (baseline ~2400s). */
	    LINEHINTS(x).lineswanted = 32768;
	    numline = 32768;
	  } else {
	    LINEHINTS(x).lineswanted = LINEHINTS(x).rate;
	    if (maxt > 1)
	      LINEHINTS(x).lineswanted /= maxt;
	    if (LIVESALTS(x))
	      LINEHINTS(x).lineswanted /= LIVESALTS(x);
	    if (MaskTotal > 1)
	      LINEHINTS(x).lineswanted /= MaskTotal;
	    /* Numrules division removed: packed GPU buffers handle all rules per word */
	    /* Gate the one-word fan-out on work per WORD, not on the raw rate.
	     *
	     * bench_rates[] counts candidate x salt operations per second, so a
	     * type's cost per wordlist line is rate/Livesalts. Testing the raw
	     * rate missed every expensive-because-many-salts type: SCRYPT at
	     * 3403 h/s against 1800 salts is ~1 word/sec, but 3403 >= 100 sent
	     * it to the 512 floor -- which is worse than useless here, because
	     * lineswanted had ALREADY been divided down to 0 by the salt count.
	     * The code computed "one word is a full dispatch" and then overrode
	     * it back up to 512, so a 290-word run became a single dispatch on
	     * a single core with the other 15 idle.
	     *
	     * Same failure hit SHA1CRYPT (618 h/s / 8262 salts) and would hit
	     * any future high-salt-count type. BCRYPT/ARGON2/GOST-YESCRYPT only
	     * escaped it because their raw rates happen to sit under 100 too.
	     *
	     * The 512 floor still applies to genuinely cheap types, where per-
	     * dispatch overhead would otherwise dominate. Fixed 2026-08-08. */
	    /* ================= FRAGILE -- READ BEFORE EDITING =================
	     *
	     * This gate has now caused two production regressions in opposite
	     * directions. It looks like a two-line heuristic; it is not. Any
	     * change here MUST be validated against BOTH fixtures below, on a
	     * host with a working GPU, before it ships. A change that fixes one
	     * case and is merely "reasoned about" for the other WILL regress.
	     *
	     *   Case A -- GPU hybrid, large wordlist. PHPBB3 e455:
	     *     mdxfind -M 400 -F <5000 hashes> <20000 words>
	     *     fpga GTX 1080: 4.9s correct, 55.9s regressed (11.4x).
	     *     Source: docs/BENCHMARK.md sm-salt400 (bench fixtures live on
	     *     fpga.local:~/src/mdfind/sm-salt400*).
	     *
	     *   Case B -- CPU-only, small wordlist, many salts. SCRYPT e884:
	     *     ~290 words against ~1800 salted hashes.
	     *     Correct = every core busy. Regressed = ONE core busy.
	     *
	     * Why it is fragile: the two cases pull opposite ways and cost per
	     * word cannot separate them. PHPBB3 at 2931 h/s over 5000 salts is
	     * 0.59 words/sec -- MORE expensive per word than SCRYPT's ~2 -- yet
	     * PHPBB3 must keep the 512 floor and SCRYPT must not. Any heuristic
	     * built on rate, salt count, or their ratio alone will get one of
	     * them wrong. Ask instead: can this op reach a GPU kernel?
	     *
	     *   - op in gpu_ops[]  -> gpu_try_pack accumulates words and
	     *     dispatches them to the device together. One word per dispatch
	     *     starves it. KEEP THE FLOOR. (RCS 1.263/1.264 built the PHPBB3
	     *     hybrid around exactly this: "GPU assists while CPU processes
	     *     concurrently".)
	     *   - op absent from gpu_ops[] -> pure CPU. With many salts a single
	     *     word is already a full dispatch, and the floor collapses a
	     *     short wordlist onto one core. FAN OUT.
	     *
	     * History: pre-1.531 tested the raw rate, which sent every
	     * expensive-because-many-salts CPU type to the floor (Case B broken).
	     * 1.531 switched to cost per word, which sent every GPU hybrid to
	     * one word per dispatch (Case A broken, reported by a user against
	     * -M 400). Both were single-case changes.
	     *
	     * Note the comparison is by multiplication, not division: rate /
	     * salts truncates to 0 whenever rate < salts, which is most salted
	     * types at scale, so the divided form silently discards the very
	     * magnitude it is testing.
	     * ================================================================= */
	    { long long lh_rate  = LINEHINTS(x).rate;
	      long long lh_salts = LIVESALTS(x) ? (long long)LIVESALTS(x) : 1;
	      int lh_gpu = op_can_reach_gpu((int)x);
	      if (!lh_gpu && lh_rate < 100LL * lh_salts)
	        LINEHINTS(x).lineswanted = 1;
	      else if (LINEHINTS(x).lineswanted < 512)
	        LINEHINTS(x).lineswanted = 512;
	    }
	    numline = LINEHINTS(x).lineswanted;
	    /* -a: one address per job is right on the CPU, where a single address
	     * IS 2^N candidates of work.  On the GPU it is ruinous, because the
	     * end-of-job flush dispatches the rules-engine batch -- so one line
	     * per job means ONE WORD PER DISPATCH.  Measured on fpga (GTX 1080,
	     * 50,000 addresses, locals 8-11): 50,000 batches for 50,000 words at
	     * ~5 Mh/s on a card that does MD5 in GH/s, with length-sorted and
	     * shuffled input indistinguishable (2.38 s vs 2.37 s busy) because no
	     * batch ever held two words to order.  Per-arm counters put all 50,000
	     * dispatches on the end-of-job flush and none on the slot cap, the
	     * buffer bound or the width split.
	     *
	     * Hand the GPU a job that can fill one batch exactly; the width-keyed
	     * split (plan 15.1) is then the only thing that fragments, which is
	     * the point of keying it.  op_can_reach_gpu is the same predicate the
	     * lineswanted gate above uses, and returns 0 when no GPU backend is
	     * compiled in, so the CPU value is reached by the honest route. */
	    if (Email) {
	      /* GPU_RULES_MAX_WORDS_PER_BATCH lives in gpujob.h, whose whole body
	       * sits behind the same backend test that defines GPU_ENABLED -- so
	       * the constant does not exist in a plain non-GPU build and naming it
	       * unguarded broke the local macOS x86_64 target while .205's OpenCL
	       * build stayed green.  Same shape as the 1.533 break that
	       * op_can_reach_gpu was introduced to fix. */
#ifdef GPU_ENABLED
	      numline = op_can_reach_gpu((int)x)
	                ? (unsigned long long)GPU_RULES_MAX_WORDS_PER_BATCH
	                : 1ULL;
#else
	      numline = 1ULL;
#endif
	    }
	  }
	  /* JOB_NULL and JOB_PARALLEL exist to measure the line-distribution
	   * path itself (NULL adds a compact-table probe per line), so they
	   * always split the buffer into one equal share per thread.  A fixed
	   * rate cannot do this: the buffer fills by BYTES, so its line count
	   * varies with line length -- 12-byte lines give about 2.2M of the
	   * 3,276,800-line index, and 3,276,800/maxt then made 11 jobs for 16
	   * cores.  The share is taken from this buffer's actual Linecount,
	   * rounded up so the last job is short by less than maxt lines. */
	  if (x == JOB_NULL || x == JOB_PARALLEL) {
	    unsigned long long share = ((unsigned long long)Linecount + maxt - 1) / (unsigned long long)maxt;
	    if (share < 1) share = 1;
	    numline = share;
	    LINEHINTS(x).lineswanted = (unsigned int)share;
	  }
	  if ((curline + numline) > Linecount)
	    numline = Linecount - curline;
	  LINEHINTS(x).curline += numline;
	  isdone = 0;
          /* Mask-iteration thread fanout (2026-05-26): when a mask is configured
           * AND MaskTotal is large AND maxt > 1 AND no real rules are active, split
           * the per-line job into N chunked sub-jobs covering disjoint MaskIndex
           * ranges. Each launches its own procjob worker (up to maxt), enabling
           * true parallelism for large -n/-N sweeps. Without chunking, a 1-line
           * seed-file sweep produces 1 job and only 1 worker thread runs the
           * entire MaskTotal sequentially (the -t flag is structurally ignored
           * because additional launches happen only on additional job submits).
           * Chunking WAS gated on Numrules<=1, because the per-rule MaskIndex
           * reset restarted the cursor at 0 and so discarded a chunk's base
           * when real rules were walked. The rule-advance block now restarts
           * it at mask_base, the value the job was submitted with, so the gate
           * is gone and rules+mask jobs chunk like any other. */
          {
            unsigned long long _chunks = 1ULL;
            unsigned long long _chunk_threshold = (unsigned long long)maxt * 65536ULL;
            if ((MaskLen || MaskPrependLen || MaskAppendLen) &&
                MaskTotal >= _chunk_threshold &&
                maxt > 1) {
              _chunks = (unsigned long long)maxt;
            }
            /* NDRange ceiling (2026-09-23).  The dispatch is one work-item per
             * (word, rule, mask) triple and the kernel bounds itself with a
             * 32-bit total = n_words * n_rules * mask_size.  The fan-out above
             * sizes the MASK axis alone, so the WORD axis entered that product
             * unbounded: 9 lines x (95^5 / 16) is 4,352,517,774, which wraps to
             * 57,550,478 and retires 98.7 percent of the dispatch.  The host
             * counters are 64-bit and still report the full keyspace, so the
             * run reconciles exactly and exits clean while finding almost
             * nothing.  Measured on a GTX 1080: 9 lines of ?a?a?a?a?a found 0
             * of 2 known answers, 8 lines found 2 of 2, and -t 17 found 2 of 2
             * where -t 16 found none.
             *
             * Cap the chunk so the lane count stays under the ceiling, which is
             * what project_bf_chunk_as_job.md Q1 decided on 2026-05-09: chunk
             * total at or below the NDRange bound per chunk, the absolute mask
             * offset accumulated in 64 bits across chunks, the kernel left
             * seeing a 32-bit in-chunk index, and the producer watching for
             * overflow.  adaptive_bf_chunk_size has honoured that for brute
             * force since Phase 1.8; this producer did not. */
#if defined(OPENCL_GPU)
            /* OpenCL only.  The cap exists because the OpenCL grid is
             * num_words * n_rules * mask_size and the kernel bounds itself
             * with a 32-bit total.  Metal's grid omits the mask axis
             * (gpu_metal.m ~6856 sizes it num_words or num_words * n_rules)
             * and walks rules and masks in an inner double loop per thread
             * (gpu/metal_template.metal ~390 and ~494), so it cannot overflow
             * on the mask axis and gains nothing here.  Worse,
             * Metal ignores the chunk range entirely -- gpujob_metal.m passes
             * literal 0 for mask_start and bf_num_masks and uses the full
             * gpu_mask_total -- so every extra chunk recomputes the WHOLE
             * mask.  Measured on an M1, 1 word x ?l^4: one chunk 456,976
             * hashes, two chunks 913,952.  Applying the cap there multiplies
             * the work by the chunk count rather than dividing it. */
            /* Applies to rules+mask too, now that the rule-advance block
             * restarts the mask cursor at mask_base instead of 0.  1.602 gated
             * this on Numrules <= 1 because without that restart a chunked
             * rules job walked the wrong ranges -- 9 words x 3 rules x a
             * six-position lowercase mask found 1 of 2 planted answers.  With
             * the restart the gate is unnecessary, and lifting it is what
             * closes the 32-bit lane exposure for rules+mask and gives that
             * combination the thread fan-out back. */
            if (MaskLen || MaskPrependLen || MaskAppendLen) {
              unsigned long long _lanes_per_mask =
                  (unsigned long long)numline * ((unsigned long long)Numrules + 1ULL);
              if (_lanes_per_mask < 1ULL)
                _lanes_per_mask = 1ULL;
              if (_lanes_per_mask >= MDX_NDRANGE_HARD) {
                /* One mask index already exceeds the ceiling, so the mask axis
                 * cannot be chunked small enough.  Silence here is what this
                 * whole comment is about, so refuse rather than truncate. */
                fprintf(stderr,
                    "FATAL %s:%d: %llu lines x %d rules is %llu work-items per "
                    "mask index, at or past the %llu NDRange ceiling; the mask "
                    "axis cannot be chunked below one index.\n",
                    __FILE__, __LINE__, (unsigned long long)numline, Numrules,
                    _lanes_per_mask, MDX_NDRANGE_HARD);
                exit(1);
              }
              {
                unsigned long long _masks_cap = MDX_NDRANGE_HARD / _lanes_per_mask;
                unsigned long long _need =
                    (MaskTotal + _masks_cap - 1ULL) / _masks_cap;
                if (_need > _chunks)
                  _chunks = _need;
              }
            }
#endif  /* OPENCL_GPU */
            unsigned long long _base_size = MaskTotal / _chunks;
            unsigned long long _remainder = MaskTotal - (_base_size * _chunks);
            unsigned long long _chunk_start = 0ULL;
            unsigned long long _ci;
            for (_ci = 0; _ci < _chunks; _ci++) {
              unsigned long long _chunk_size = _base_size + (_ci < _remainder ? 1ULL : 0ULL);
              possess(FreeWaiting);
              wait_for(FreeWaiting, NOT_TO_BE, 0);
              job = FreeHead;
              if (!job) {
                fprintf(stderr, "Job null on freehead!\n");
                exit(1);
              }
              FreeHead = job->next;
              job->next = NULL;
              if (FreeHead == NULL)
                FreeTail = &FreeHead;
              twist(FreeWaiting, BY, -1);

              job->op = x;
              job->flags = (Printsource | Addlf);
              /* Chunk 0 owns the seed lines; later chunks re-use them and must
               * not re-retire them.  Mirrors the InflightLines guard below. */
              if (_ci > 0ULL)
                job->flags |= JOBFLAG_MASK_SUBCHUNK;
              if (Dodigits)
                job->flags |= JOBFLAG_NUMBERS;
              if (MaskPrepend || MaskPrependLen > 0)
                job->flags |= JOBFLAG_PREPEND;
              job->Numbers = Numbers;
              job->digits = Dodigits;
              if (MaskLen || MaskPrependLen || MaskAppendLen) {
                job->MaskIndex = _chunk_start;
                job->MaskCount = (_chunks > 1ULL) ? _chunk_size : MaskTotal;
              } else {
                job->MaskIndex = Numbers;
                job->MaskCount = 0;
              }
              job->doneprint = &doneprint;
              job->filename = argv[y];
              job->startline = curline;
              job->numline = numline;
              /* word-retirement: only first sub-job counts the line as inflight;
               * later chunks share the same seed line(s). */
              if (_ci == 0ULL)
                __sync_fetch_and_add(&InflightLines, numline);
              job->readindex = readindex;
              job->readbuf = readbuf;
              { lock *rl = readbuf_lock(readbuf); possess(rl); twist(rl, BY, +1); }
              if (ThreadCount < maxt) {
                /* Experimental: stagger second procjob() startup so the first
                 * thread fills jobg with line-ordered buffers before sibling
                 * threads start consuming. Env-gated, fires once when
                 * launching the second worker, gpu-active only. See
                 * project_launch_delay_experiment.md. */
#ifdef GPU_ENABLED
                if (ThreadCount == 1 && gpujob_available()) {
                  sleep(2);
                }
#endif
                launch(procjob, NULL);
                ThreadCount++;
              }
              possess(WorkWaiting);
              *WorkTail = job;
              WorkTail = &(job->next);
              twist(WorkWaiting, BY, +1);
              _chunk_start += _chunk_size;
            }
          }
          J1N(RC, Dohash, NextX);
        }
      }
    }
    if (!Creader.gz_closed) gzclose(zfi);
    creader_reset();
    /* Persist the AUTHORITATIVE line count (Fileline = total lines just
     * consumed) to the SQLite cache, overwriting any prior partial-count
     * entry. Skip stdin (no path/stat) and degenerate runs (Fileline==0). */
    if (Fileline > 0 && Curfile && strcmp(Curfile, "stdin") != 0) {
      char resolved[4096];
#ifdef _WIN32
      char *rp = _fullpath(resolved, argv[y], sizeof(resolved));
#else
      char *rp = realpath(argv[y], resolved);
#endif
      if (!rp) { strncpy(resolved, argv[y], sizeof(resolved) - 1); resolved[sizeof(resolved)-1] = 0; }
      linecount_cache_finalize(resolved,
                               (long long)sb.st_size, (long long)sb.st_mtime,
                               Fileline);
    }
  }
bf_done:
  possess(FreeWaiting);
  wait_for(FreeWaiting, TO_BE, alloc_maxt * 32);
  job = FreeHead;
  FreeHead = job->next;
  job->next = NULL;
  twist(FreeWaiting, BY, -1);
  if (ThreadCount) {
    possess(WorkWaiting);
    job->op = JOB_DONE;
    *WorkTail = job;
    WorkTail = &(job->next);
    twist(WorkWaiting, BY, +1);
#ifdef GPU_ENABLED
    gpujob_shutdown(); /* wait for GPU to finish all work */
    /* BF Phase 3 (2026-05-10): gpu_opencl_bf_stop retired alongside the
     * atomic-cursor BF machinery. Nothing to signal at shutdown anymore. */
#endif
    possess(HashWaiting);
    twist(HashWaiting, TO, 1);  /* now signal ReportStats to exit */
    stats_wake();
    x = join_all();
#ifdef OPENCL_GPU
    gpu_opencl_shutdown(); /* explicit — atexit races with NVIDIA driver teardown */
#endif
#if defined(__APPLE__) && defined(METAL_GPU)
    gpu_metal_shutdown();
#endif
#if (defined(__APPLE__) && defined(METAL_GPU)) || defined(CUDA_GPU) || defined(OPENCL_GPU)
    /* Per-device dispatch share — makes multi-GPU asymmetry obvious at
     * a glance. Reads accumulated counters from gpujob_opencl.c. Safe to
     * call after gpujob_shutdown(); does nothing on single-GPU runs. */
    gpujob_print_share_line(stderr);
#endif
    fprintf(stderr, "\nDone - %d threads caught\n", x - 1);
  }
  current_utc_time(&current);
  time(&end);
  start = end - start;
  fprintf(stderr, "%s lines processed in %llu seconds\n", commify(Totallines), (unsigned long long) start);
  {
    unsigned long long _gpu_rules_total = atomic_load(&Totrules_gpu);
    /* -8 accounting, reported for the same reason procrule reports it: a pair no
     * engine can run, and a pair whose UTF-32 answer was locale-ambiguous, are
     * both real losses of coverage, and a silent one reads as "the wordlist did
     * not contain it".  Printed only when non-zero, and only under -8. */
    if (Utf32Rules && Unreachable)
      fprintf(stderr, "%s (word,rule) pairs skipped: rule needs UTF-32 and the "
                      "word is not valid UTF-8\n", commify(Unreachable));
    /* -a width waste, per plan section 9 decision 9.  The clamp is silent
     * otherwise, and a sweep that covered 24 of a 30-character local looks
     * exactly like one that covered all of it. */
    if (Email && Email_batch_splits) {
      fprintf(stderr,
        "-a GPU batching: %s batch(es) dispatched early because the local-part "
        "length changed", commify(Email_batch_splits));
      if (Email_unsorted)
        fprintf(stderr,
          ", %s of them where the length DECREASED -- the input is not sorted "
          "shortest-first, so batches fragment and the GPU runs far below its "
          "rate. utils/emailprep orders a list correctly",
          commify(Email_unsorted));
      fprintf(stderr, ".\n");
    }
#ifdef GPU_ENABLED
    /* The clamp costs SPEED, not coverage, and the reason is an invariant
     * rather than luck.  The dispatch clamps only when a single word's full
     * width will not fit the grid -- the pack site bounds the word axis per
     * width (plan 15.4), so any batch it assembles fits by construction.  The
     * interlock that lets the CPU skip the dot axis is gated on the SAME
     * inequality, so for exactly the words that provoke a clamp it does not
     * fire and the CPU sweeps them in full.  Clamp and interlock are therefore
     * complementary: whenever one engages the other stands down, and no
     * candidate is lost either way.
     *
     * An earlier version of this message said the run was INCOMPLETE.  That
     * was wrong, and a wrong diagnostic is worse than none -- it would have
     * sent someone looking for lost candidates in a run that had computed all
     * of them, slowly.  What the operator needs to know is that the rule count
     * is too high for that local-part width to stay on the device. */
    if (Email && gpu_dot_width_asked > gpu_dot_width_used)
      fprintf(stderr,
        "-a GPU dot width narrowed to %d of the %d some batch asked for: with "
        "this rule count a local part that wide does not fit the GPU grid, so "
        "those addresses were swept on the CPU instead. Coverage is complete; "
        "the run is slower than it looks. Fewer rules per pass, or a narrower "
        "local-part width, keeps the whole sweep on the device.\n",
        gpu_dot_width_used, gpu_dot_width_asked);
#endif
    if (Email && Email_cpu_cheap)
      fprintf(stderr,
        "-a: %s word(s) kept on the CPU because the dot sweep was narrower than "
        "%d positions -- below that a GPU dispatch costs more than it saves. "
        "That is expected at the START of an emailprep-ordered list, which is "
        "sorted shortest-first.\n",
        commify(Email_cpu_cheap), EMAIL_GPU_MIN_DOT_W);
    if (Email && Email_clamped)
      fprintf(stderr,
        "%s word(s) had a local part longer than %d characters; the dot sweep "
        "covered only the first %d positions of each. Dottings beyond that were "
        "not tried.\n",
        commify(Email_clamped), EMAIL_DOT_WIDTH_MAX, EMAIL_DOT_WIDTH_MAX);
    if (Utf32Rules && Variants_dropped)
      fprintf(stderr, "%s (word,rule) pairs had more than one UTF-32 locale "
                      "variant; only the first was tried (use -L to pin one)\n",
              commify(Variants_dropped));
    if (Totrules || _gpu_rules_total) {
      fprintf(stderr, "%s total rule-generated passwords tested\n",
              commify(Totrules + Totallines + _gpu_rules_total));
    }
  }
  if (start == 0)
    start = 1;
  fprintf(stderr, "%.2f lines per second\n", (double) Totallines / (double) start);

  wtime = (double) current.tv_sec + (double) (current.tv_nsec) / 1000000000.0;
  wtime -= (double) starthash.tv_sec + (double) (starthash.tv_nsec) / 1000000000.0;
  if (wtime <= 0)
    wtime = 1.0;
  fprintf(stderr, "%.2f seconds hashing, %s total hash calculations\n", wtime, commify(Tothash));
  { double hps; char *mult;
    format_rate((double) Tothash / wtime, &hps, &mult);
    fprintf(stderr, "%.2f%s hashes per second (approx)\n", hps, mult);
  }

  fprintf(stderr, "%s total files\n", commify(Totalfiles));
  for (y = 1; Types[y]; y++) {
    if (!TOTALFOUND(y - 1))
      continue;
    for (x = 0; x < Maxiter; x++) {
      if (TOTALFOUND(y - 1)[x]) fprintf(stderr, "%s %sx%02d hashes found\n", commify(TOTALFOUND(y - 1)[x]), Types[y], x + 1);
    }
  }
  /* User-defined hash types (Milestone 1): per-op found counts. */
  {
    int _uc = userdef_count(), _ui;
    for (_ui = 0; _ui < _uc; _ui++) {
      int _uop = (int)user_id_op(_ui);
      if (_uop >= JOB_DONE) break;
      if (!TOTALFOUND(_uop - 1)) continue;
      for (x = 0; x < Maxiter; x++) {
        if (TOTALFOUND(_uop - 1)[x])
          fprintf(stderr, "%s %sx%02d hashes found\n",
                  commify(TOTALFOUND(_uop - 1)[x]), userdef_name(_uop), x + 1);
      }
    }
  }
  if (Totfound == 0)
    fprintf(stderr, "None found, sorry!\n");
  else
    fprintf(stderr, "%s Total hashes found\n", commify(Totfound));

  if (DoHistogram && RuleCnt) {
    for (z = x = 0; x <= Numrules; x++) {
      if (RuleCnt[x]) z++;
    }

    rulehist = malloc_lock((z + 2)*sizeof(struct RuleHist),"rulehist");
    y = 0;
    if (RuleCnt[0]) {
      rulehist[y].rule  = "No rule";
      rulehist[y].rlen  = 7;
      rulehist[y++].val = RuleCnt[0];
    }
    for (x = 1; x <= Numrules; x++) {
      if (!RuleCnt[x]) continue;
      /* The rule's SOURCE TEXT, captured at the -r/-R load site before
       * packrules compiled the line in place.  There is deliberately no decode
       * step here: packed bytecode is not decompilable, because an operand byte
       * can hold the same value as an opcode letter.  The decoder that tried it
       * printed `$z` as \345z^@^B and then walked past the entry NUL into the
       * next rule -- an out-of-bounds read, and the reason -Z output could not
       * be trusted for rule attribution. */
      if (Rulesrclen && Rulesrclen[x]) {
        rulehist[y].rule = Rulesrc + Rulesrcoff[x];
        rulehist[y].rlen = Rulesrclen[x];
      } else {
        rulehist[y].rule = "(source not retained)";
        rulehist[y].rlen = 21;
      }
      rulehist[y++].val = RuleCnt[x];
    }
    z = y;
    if (y > 0) {
      fprintf(stderr, "\nRule Histogram\n");
      qsort(rulehist, z, sizeof(struct RuleHist), rulecmp);
      y = 7;
      if (Totfound > 999999)
        y = 11;
      if (Totfound > 999999999L)
        y = 15;
      if (Totfound > 999999999999L)
        y = 19;
      fprintf(stderr, "%*s  Rule\n", y, "Count");
      for (x = 0; x < z; x++)
        fprintf(stderr, "%*s  %.*s\n", y, commify(rulehist[x].val),
                (int)rulehist[x].rlen, rulehist[x].rule);
    }
  }


  return (0);
}

	    

