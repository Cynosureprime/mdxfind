#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#ifndef _AIX
#include <getopt.h>
#endif
#include <math.h>
#include <Judy.h>

#ifdef _WIN32
/*
 * getline() is POSIX 2008 and is absent from the Windows CRT entirely; the
 * llvm-mingw cross-compiler this project ships with rejects it as an
 * undeclared function.  Every other libc call in this file compiles there
 * unchanged.
 *
 * The replacement lives HERE rather than in the release script, for the same
 * reason bertillon.h carries its own copy: the published source then
 * cross-compiles on its own, without reproducing release machinery that a
 * person cloning the repository cannot see.
 */
#include <sys/types.h>
#include <errno.h>
static ssize_t ep_getline(char **lineptr, size_t *n, FILE *stream)
{
    size_t used = 0;
    int c;
    if (!lineptr || !n || !stream) { errno = EINVAL; return -1; }
    if (!*lineptr || *n == 0) {
        size_t cap = 128;
        char *p = (char *)realloc(*lineptr, cap);
        if (!p) return -1;
        *lineptr = p; *n = cap;
    }
    for (;;) {
        c = fgetc(stream);
        if (c == EOF) { if (used == 0) return -1; break; }
        if (used + 2 > *n) {                 /* room for the byte and the NUL */
            size_t cap = *n * 2;
            char *p = (char *)realloc(*lineptr, cap);
            if (!p) return -1;
            *lineptr = p; *n = cap;
        }
        (*lineptr)[used++] = (char)c;
        if (c == '\n') break;               /* the terminator is KEPT, as POSIX */
    }
    (*lineptr)[used] = '\0';
    return (ssize_t)used;
}
#define getline ep_getline
#endif

/*
 * emailprep prepares a list of email addresses for an mdxfind -E or -a run.
 *
 * It does three things, and they have to happen in this order:
 *
 *   1. normalize -- remove '.' from the local part (the text before the first
 *      '@').  Gmail and googlemail ignore dots in the local part, so
 *      f.r.esh@gmail.com and fresh@gmail.com are one account; the dotless
 *      spelling is the seed that mdxfind -a sweeps back out to every dotting.
 *   2. dedup -- MUST follow normalization, because that is where the collapse
 *      happens.  Measured over a 10.2M-plaintext corpus, 27,179 distinct
 *      spellings of one local part normalize to a single seed.  Undeduped,
 *      each of those spellings sweeps its own 2^23 candidates: a 27,179x
 *      multiplier off one account.
 *   3. order by local-part length, shortest first -- MUST follow dedup.  The
 *      sweep for a local of length L costs 2^L candidates, so cost per input
 *      line spans eight orders of magnitude inside one unsorted file.
 *      Shortest-first spends the early budget where candidates-per-find is
 *      lowest, which is what matters on a run that gets stopped rather than
 *      completed -- and that is every run of this kind.
 *
 * A shell pipeline cannot get all three right in one pass.  rling's dedup is a
 * lexical sort-and-collapse and its -L is applied afterwards for exactly that
 * reason, so ordering is a third pass; and rling -L measures WHOLE-LINE bytes,
 * which is only the right sort key once the input is single-domain.  Holding
 * the set is what makes the correct order reachable, and the set is small
 * enough to hold.  Measured on the iMac, 2,000,000 synthetic gmail addresses of
 * local length 3 to 30 (40% carrying dots): 1.55 s wall, 292 MB peak RSS,
 * 1,940,346 written after 59,654 duplicates collapsed.  The arena itself is
 * only about 47 MB of that -- the rest is the JudyHS dedup index and the record
 * array, which is why an arena-sized estimate understates it six-fold.
 *
 * emailprep is not a replacement for rling(1).  Subtraction, intersection and
 * the -q statistics all still belong there; this tool does the one job rling
 * cannot express.
 *
 * $HEX[] input is decoded on the way in, the way every other tool in the suite
 * handles it.  The wrapper is ASCII and carries no literal '@', so a wrapped
 * address measured as written has no domain at all and lands in a length bucket
 * the width of the WRAPPER rather than of the local part -- so decoding is not
 * a convenience here, it is the difference between a correct ordering and a
 * meaningless one.  Everything downstream of the decode -- the '@' scan, the -d
 * match, normalization, the dedup key and the length key -- sees decoded bytes.
 * The decoder matches rehex(1) -u exactly (rehex.c:260-264): the prefix test is
 * a literal case-sensitive "$HEX[", and decoding stops at the first byte pair
 * that is not hex, which is the closing ']'.
 *
 * On output a line that arrived wrapped leaves wrapped and a line that arrived
 * raw leaves raw, so the representation is never changed -- only the content.
 * A decoded payload can hold a NUL or a newline, which no raw line can carry,
 * which is why that is a safety rule and not a stylistic one.  For the same
 * reason dedup uses JudyHS rather than JudySL: a JudySL key stops at the first
 * NUL, so $HEX[006a] and $HEX[0062] would collide on an empty key.
 *
 * Splitting on the FIRST '@' is a shared invariant with mdxfind -E.  If the two
 * ever disagree, the ordering this tool produces is no longer the ordering
 * mdxfind's sweep cost follows, and the whole reason for running it quietly
 * stops holding.  regress/regress-email.sh checks the two against one fixture.
 */

static char *Version = "$Header: /Users/dlr/src/mdfind/utils/RCS/emailprep.c,v 1.5 2026/11/01 05:30:17 dlr Exp dlr $";

/*
 * $Log: emailprep.c,v $
 * Revision 1.5  2026/11/01 05:30:17  dlr
 * Carry a getline replacement for Windows. getline is POSIX 2008 and the Windows CRT has none: the llvm-mingw cross-compilers this project ships with reject it outright as an undeclared function on x86_64 and aarch64, while the i686 gcc only warned and then linked a binary that file(1) reports as data rather than a PE image. Every other libc call in this file compiles there unchanged. The replacement sits in the source rather than in mdxfind-release, for the reason bertillon.h states for its own copy: the published source then cross-compiles on its own, without reproducing release machinery a person cloning the repository cannot see. Guarded by ifdef _WIN32, so the POSIX path is untouched and verified byte-identical on the same fixture. emailprep now builds on all eleven release platforms and is bundled into each one.
 *
 * Revision 1.4  2026/10/31 14:48:06  dlr
 * Re-measure after the JudyHS switch. 2,000,000 synthetic gmail addresses, local length 3 to 30 with 40 percent dotted: 1.55 s wall and 292 MB peak RSS, against 2.12 s and 241 MB under JudySL. Faster and larger, both expected -- JudyHS hashes a counted key instead of walking a NUL-terminated one, and it stores the length. Output unchanged at 1,940,346 records from 59,654 collapses. The arena is still about 47 MB, so the arena-only estimate this replaces understated the footprint six-fold rather than five.
 *
 * Revision 1.3  2026/10/31 14:47:19  dlr
 * Process $HEX[] natively, as every other tool in the suite does (Waffle, 2026-10-31). The 1.2 behaviour -- count and warn, leave decoding to rehex -u -- was wrong for this family. The decode now happens FIRST, so the @ scan, the -d match, normalization, the dedup key and the length key all see the bytes the address actually consists of. That is not a convenience: the wrapper is ASCII and carries no literal @, so a wrapped address measured as written has no domain at all and buckets at the width of the WRAPPER rather than of the local part, which makes the ordering meaningless for exactly the input it was supposed to order. The decoder matches rehex -u exactly (rehex.c:260-264): a literal case-sensitive $HEX[ prefix test, and decoding stops at the first byte pair that is not hex, which is the closing bracket. The nibble table is built at startup rather than written as a 256-entry initializer, because the literal form has to be counted by hand to audit and a miscount puts a valid nibble where a reject belongs, which decodes silently wrong. Three consequences. Dedup moved from JudySL to JudyHS: a JudySL key stops at the first NUL, so two decoded payloads that both begin with one collide on an empty key and the second is dropped as a duplicate; JudyHS is length-aware. struct rec gained an explicit byte length for the same reason, since strlen cannot measure a record holding a NUL. And output now preserves the REPRESENTATION while changing only the content -- a line that arrived wrapped leaves wrapped, re-encoded from the normalized bytes, a line that arrived raw leaves raw -- with one override that is a safety rule and not a stylistic one, since a decoded payload can hold a NUL or a newline that no raw line can carry. Separately, a trailing CR is now STRIPPED rather than preserved, matching rehex -u (rehex.c:259) and rling (rling.c:1067). The 1.1 reasoning was byte fidelity argued from first principles without checking what the sibling tools do; this is a line-oriented tool in a family that agrees on the point, and a CRLF wordlist silently yielding candidates that end in CR would cost real recoveries. The count is still reported so the operator learns the input was CRLF. Also compares -d domains by length and strncasecmp rather than strcasecmp, which would stop at a NUL in a decoded payload.
 *
 * Revision 1.2  2026/10/31 14:34:25  dlr
 * Replace the memory estimate with a measurement. The header comment and README both said 1.9M normalized gmail addresses is about 45 MB, which is the size of the text arena alone and ignores the Judy dedup index and the record array. Measured on the iMac over 2,000,000 synthetic gmail addresses of local length 3 to 30 with 40 percent carrying dots: 2.12 s wall, 241 MB peak RSS, 1,940,346 written after 59,654 duplicates collapsed. The arena is about 47 MB of that, so the original figure understated the real footprint five-fold. Also adds the $HEX[] warning: a wrapper is ASCII and carries no literal @, so a wrapped address measures as a domainless plain the length of the wrapper and lands in the wrong length bucket entirely, which is the same hazard splitlen documents for its own -u. Counted and reported rather than decoded, since rehex -u is the decoder and the input should arrive decoded; silently unwrapping here would alter bytes that get hashed downstream. And the no-@ count moved out of the disposition list in the report, because a line with no @ is also inside the -d filtered count when -d is given, so listing it alongside read/filtered/duplicate/written read as a fourth disjoint bucket and the numbers did not add up.
 *
 * Revision 1.1  2026/10/31 14:32:02  dlr
 * Initial revision: emailprep(1), the first tool in the new distributed utils/ directory. Normalizes the local part of an email address by removing dots, deduplicates, and orders by local-part length shortest first. The three steps are strictly ordered and that is why this is a tool rather than a shell pipeline: dedup must follow normalization, because normalization is where the collapse happens -- measured over a 10.2M-plaintext corpus, 27,179 distinct spellings of one local part normalize to a single seed, and undeduped each sweeps its own 2^23 candidates. Ordering must follow dedup. rling cannot express the result in one pass: its dedup is a lexical sort-and-collapse with -L applied afterwards, so ordering becomes a third pass, and -L measures whole-line bytes, which is the right sort key only once the input is single-domain. Holding the set is what makes the correct order reachable and the set is small enough to hold, about 45 MB for 1.9M normalized gmail addresses. Dedup is a JudySL insert-if-absent rather than a new hash table. Ordering is a qsort on (local length, input sequence); comparing the sequence is what makes it stable, since qsort is not, and without the tiebreak two equal-length lines could swap between runs and a diff of the same input would show spurious changes. Three deliberate non-behaviours, each reporting rather than altering: it does not lowercase, because the hash is over the bytes as typed and rule l covers it when wanted; it does not strip a trailing CR, counting and warning instead; it does not decode a $HEX[] wrapper, which is ASCII and carries no literal @, so a wrapped address would measure as a domainless plain the length of the wrapper and land in the wrong bucket entirely -- the same hazard splitlen documents for its own -u, with rehex -u as the decoder. -d matches a domain ignoring case but preserves its bytes. Splitting on the FIRST @ is a shared invariant with the planned mdxfind -E; if the two disagree the ordering this tool produces stops being the ordering mdxfind sweep cost follows, so regress/regress-email.sh checks them against one fixture.
 *
 */

#define MAXDOMAINS 64

static char      *Arena;        /* accepted lines, NUL-separated */
static size_t     ArenaUsed, ArenaCap;

struct rec {
    uint64_t off;               /* offset into Arena */
    uint32_t llen;              /* normalized local-part length, in bytes */
    uint32_t seq;               /* input order; makes the sort stable */
    uint32_t len;               /* whole record length; a decoded payload may
                                 * contain a NUL, so strlen is not usable */
    uint8_t  hex;               /* emit $HEX[] wrapped */
};
static struct rec *Recs;
static size_t      RecsUsed, RecsCap;

static const char *Domains[MAXDOMAINS];
static int         Ndomains;

static int Nonorm, Nodedup, Quiet;

/* Hex nibble table and decoder, semantics identical to rehex(1)'s get32():
 * consume hex PAIRS, stop at the first pair that is not hex (the ']'), return
 * the byte count.  dest may alias src as long as it trails it, which it does
 * here because the payload starts five bytes into the line. */
/* Built at startup rather than written out as a 256-entry initializer: the
 * literal form has to be counted by hand to audit, and a miscount puts a
 * valid nibble where a reject belongs, which decodes silently wrong. */
static unsigned char Trhex[256];
static void hexinit(void)
{
    int i;
    memset(Trhex, 99, sizeof Trhex);
    for (i = '0'; i <= '9'; i++) Trhex[i] = (unsigned char)(i - '0');
    for (i = 'a'; i <= 'f'; i++) Trhex[i] = (unsigned char)(i - 'a' + 10);
    for (i = 'A'; i <= 'F'; i++) Trhex[i] = (unsigned char)(i - 'A' + 10);
}

static size_t hexdecode(const char *src, size_t srclen, char *dest)
{
    size_t i, cnt = 0;
    unsigned char hi, lo;
    for (i = 0; i + 1 < srclen; i += 2) {
        hi = Trhex[(unsigned char)src[i]];
        lo = Trhex[(unsigned char)src[i+1]];
        if (hi > 15 || lo > 15) break;
        dest[cnt++] = (char)((hi << 4) | lo);
    }
    return cnt;
}

/* Can these bytes be written as a plain line and read back unchanged? */
static int line_safe(const char *s, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++)
        if (s[i] == 0 || s[i] == '\n' || s[i] == '\r') return 0;
    return !(n >= 5 && memcmp(s, "$HEX[", 5) == 0);
}

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n);
    if (!q) {
        fprintf(stderr, "emailprep: out of memory at %zu bytes\n", n);
        exit(1);
    }
    return q;
}

static void arena_put(const char *s, size_t n)
{
    if (ArenaUsed + n + 1 > ArenaCap) {
        size_t cap = ArenaCap ? ArenaCap * 2 : (1u << 20);
        while (cap < ArenaUsed + n + 1) cap *= 2;
        Arena = xrealloc(Arena, cap);
        ArenaCap = cap;
    }
    memcpy(Arena + ArenaUsed, s, n);
    Arena[ArenaUsed + n] = 0;
    ArenaUsed += n + 1;
}

static void rec_put(uint64_t off, uint32_t llen)
{
    if (RecsUsed == RecsCap) {
        size_t cap = RecsCap ? RecsCap * 2 : 65536;
        Recs = xrealloc(Recs, cap * sizeof(*Recs));
        RecsCap = cap;
    }
    Recs[RecsUsed].off  = off;
    Recs[RecsUsed].llen = llen;
    Recs[RecsUsed].seq  = (uint32_t)RecsUsed;
    RecsUsed++;
}

/* Order by local-part length, then by input order.  Comparing seq is what
 * makes this stable: qsort(3) is not, and without the tiebreak two equal-length
 * lines could swap between runs, so a diff against a previous run of the same
 * input would show spurious changes. */
static int reccmp(const void *a, const void *b)
{
    const struct rec *x = a, *y = b;
    if (x->llen != y->llen) return x->llen < y->llen ? -1 : 1;
    return x->seq < y->seq ? -1 : 1;
}

static void usage(void)
{
    fprintf(stderr,
      "emailprep: normalize, dedup and order email addresses by sweep cost\n"
      "usage: emailprep [-d domain]... [-N] [-u] [-q] input output\n"
      "  -d domain   keep only addresses whose domain matches this, ignoring\n"
      "              case; repeatable, up to %d times.  Lines with no '@'\n"
      "              have no domain and are excluded when -d is given.\n"
      "  -N          do not normalize (dedup and order only)\n"
      "  -u          do not dedup (normalize and order only)\n"
      "  -q          suppress the stderr histogram\n"
      "input and output may each be '-' for stdin / stdout.\n"
      "Order is by the length of the NORMALIZED LOCAL PART, shortest first,\n"
      "stable within a length.  See emailprep(1).\n", MAXDOMAINS);
    exit(1);
}

int main(int argc, char **argv)
{
    int c;
    FILE *in, *out;
    char *line = NULL;
    size_t linecap = 0;
    ssize_t n;
    Pvoid_t seen = (Pvoid_t)NULL;
    char *norm = NULL, *hexbuf = NULL;
    size_t normcap = 0, hexcap = 0;
    uint64_t nread = 0, nfiltered = 0, ndup = 0, ncr = 0, noat = 0, nhex = 0;
    size_t i;

    while ((c = getopt(argc, argv, "?hd:Nuq")) != -1) {
        switch (c) {
          case 'd':
            if (Ndomains >= MAXDOMAINS) {
                fprintf(stderr, "emailprep: more than %d -d domains\n", MAXDOMAINS);
                exit(1);
            }
            Domains[Ndomains++] = optarg;
            break;
          case 'N': Nonorm  = 1; break;
          case 'u': Nodedup = 1; break;
          case 'q': Quiet   = 1; break;
          default:  usage();
        }
    }
    if (argc - optind != 2) usage();
    hexinit();

    in  = strcmp(argv[optind],   "-") ? fopen(argv[optind],   "r") : stdin;
    if (!in)  { perror(argv[optind]);   exit(1); }
    out = strcmp(argv[optind+1], "-") ? fopen(argv[optind+1], "w") : stdout;
    if (!out) { perror(argv[optind+1]); exit(1); }

    while ((n = getline(&line, &linecap, in)) > 0) {
        char *rec, *at;
        size_t len, head, j, k, normlen;
        int washex = 0;

        nread++;
        if (line[n-1] == '\n') n--;
        /* A trailing CR is stripped, matching rehex(1) -u (rehex.c:259) and
         * rling(1) (rling.c:1067).  This is a line-oriented tool in a family
         * that agrees on the point, and a CRLF wordlist that silently produced
         * candidates ending in CR would cost real recoveries.  Counted so the
         * operator still learns the input was CRLF. */
        if (n > 0 && line[n-1] == '\r') { ncr++; n--; }
        len = (size_t)n;
        if (!len) continue;

        /* Decode $HEX[] FIRST, so every decision below is made on the bytes the
         * address actually consists of.  Measured as written, a wrapped address
         * has no '@' at all and buckets by the width of the wrapper. */
        rec = line;
        if (len > 5 && memcmp(line, "$HEX[", 5) == 0) {
            if (hexcap < len + 1) { hexcap = len + 1; hexbuf = xrealloc(hexbuf, hexcap); }
            len = hexdecode(line + 5, len - 5, hexbuf);
            rec = hexbuf;
            washex = 1;
            nhex++;
            if (!len) continue;                 /* $HEX[] with no payload */
        }

        at   = memchr(rec, '@', len);           /* FIRST '@'; see header */
        head = at ? (size_t)(at - rec) : len;
        if (!at) noat++;

        if (Ndomains) {
            int ok = 0;
            /* Matched ignoring case, because a leak carries @Gmail.com as well
             * as @gmail.com and they are one account -- but the BYTES are
             * preserved: the hash is over what was stored.  Compared by length
             * rather than with strcasecmp, since a decoded payload may hold a
             * NUL and strcasecmp would stop there. */
            if (at) {
                size_t dlen = len - head - 1;
                for (j = 0; j < (size_t)Ndomains; j++)
                    if (strlen(Domains[j]) == dlen &&
                        strncasecmp(rec + head + 1, Domains[j], dlen) == 0) {
                        ok = 1; break;
                    }
            }
            if (!ok) { nfiltered++; continue; }
        }

        if (normcap < len + 1) {
            normcap = len + 1;
            norm = xrealloc(norm, normcap);     /* normalizing only shrinks */
        }
        /* Build norm = <local><tail>, where tail runs from the '@' to the end
         * of the line and is copied VERBATIM -- the domain is never touched.
         * k ends as the local-part length that the ordering key needs.
         * When there is no '@', head == len, so taillen is 0 and the same two
         * statements handle a plain word with no second path to get wrong. */
        {
            size_t taillen = len - head;
            if (Nonorm) {
                memcpy(norm, rec, len);
                k = head;
            } else {
                for (j = k = 0; j < head; j++)
                    if (rec[j] != '.') norm[k++] = rec[j];
                memcpy(norm + k, rec + head, taillen);
            }
            normlen = k + taillen;
            norm[normlen] = 0;
        }

        /* JudyHS, not JudySL: a JudySL key stops at the first NUL, so two
         * different decoded payloads that both begin with one would collide on
         * an empty key and the second would be dropped as a duplicate. */
        if (!Nodedup) {
            Word_t *PV;
            JHSI(PV, seen, norm, (Word_t)normlen);
            if (!PV) { fprintf(stderr, "emailprep: Judy insert failed\n"); exit(1); }
            if (*PV) { ndup++; continue; }
            *PV = 1;
        }

        arena_put(norm, normlen);
        rec_put((uint64_t)(ArenaUsed - normlen - 1), (uint32_t)k);
        Recs[RecsUsed-1].len  = (uint32_t)normlen;
        Recs[RecsUsed-1].hex  = (uint8_t)(washex || !line_safe(norm, normlen));
    }
    if (in != stdin) fclose(in);

    qsort(Recs, RecsUsed, sizeof(*Recs), reccmp);
    /* A line that arrived wrapped leaves wrapped and a line that arrived raw
     * leaves raw, so the representation is never changed -- only the content.
     * The override is a safety rule, not a style one: a decoded payload can
     * hold a NUL or a newline, and writing one raw would corrupt the file. */
    for (i = 0; i < RecsUsed; i++) {
        const char *p = Arena + Recs[i].off;
        size_t plen = Recs[i].len, b;
        if (Recs[i].hex) {
            fputs("$HEX[", out);
            for (b = 0; b < plen; b++)
                fprintf(out, "%02x", (unsigned char)p[b]);
            fputs("]\n", out);
        } else {
            fwrite(p, 1, plen, out);
            fputc('\n', out);
        }
    }
    if (out != stdout) fclose(out); else fflush(stdout);

    if (!Quiet) {
        long double total = 0.0L;
        uint32_t cur;
        size_t run;

        fprintf(stderr, "emailprep: %" PRIu64 " read", nread);
        if (nfiltered) fprintf(stderr, ", %" PRIu64 " filtered by -d", nfiltered);
        if (ndup)      fprintf(stderr, ", %" PRIu64 " duplicate", ndup);
        fprintf(stderr, ", %zu written\n", RecsUsed);
        /* Reported separately, not as a disposition: a line with no '@' is
         * counted here AND may already be inside the -d filtered count, since
         * it has no domain to match.  Listing it beside the dispositions reads
         * as a fourth, disjoint bucket and the numbers then fail to add up. */
        if (noat)
            fprintf(stderr, "emailprep: %" PRIu64 " input line(s) had no '@'\n", noat);
        if (ncr)
            fprintf(stderr, "emailprep: %" PRIu64 " line(s) ended with CR; input looks "
                            "like CRLF and the CR was stripped, as rehex -u does\n", ncr);
        if (nhex)
            fprintf(stderr, "emailprep: %" PRIu64 " line(s) were $HEX[] wrapped; decoded "
                            "before measuring, and re-wrapped on output\n", nhex);

        /* Per-length census with the sweep cost it implies, so a run can be
         * sized before it is launched.  Recs is already length-major. */
        fprintf(stderr, "  local len      lines        2^len sweep\n");
        for (i = 0; i < RecsUsed; i = run) {
            cur = Recs[i].llen;
            for (run = i; run < RecsUsed && Recs[run].llen == cur; run++)
                ;
            total += (long double)(run - i) * powl(2.0L, (long double)cur);
            fprintf(stderr, "  %9u  %9zu  %17.3Lg\n",
                    cur, run - i, (long double)(run - i) * powl(2.0L, (long double)cur));
        }
        fprintf(stderr, "  total sweep at full width: %.3Lg candidates\n", total);
    }
    return 0;
}
