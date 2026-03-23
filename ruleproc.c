#include <stdio.h>

#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <wctype.h>
#include <sys/types.h>
#include <errno.h>
#include <stdint.h>

#ifdef POWERPC
#define NOTINTEL 1
#endif
#ifdef ARM
#define NOTINTEL 1
#if ARM > 6
#include <arm_neon.h>
extern int Neon;
#endif
#endif
#ifdef SPARC
#define NOTINTEL 1
#endif
#ifdef AIX
#define NOTINTEL 1
#endif

#ifndef NOTINTEL
#include <emmintrin.h>
#include <xmmintrin.h>
#include <cpuid.h>
int IntelSSE;
#endif

#include "mdxfind.h"

extern unsigned char trhex[];
extern int b64_encode(char *clrstr, char *b64dst, int inlen);

static char *Version = "$Header: /Users/dlr/src/mdfind/RCS/ruleproc.c,v 1.11 2025/11/28 18:24:48 dlr Exp dlr $";
/*
 * $Log: ruleproc.c,v $
 * Revision 1.11  2025/11/28 18:24:48  dlr
 * replace local memory copy with memcpy, will revisit.
 * Add control-b rule for base64 conversion
 *
 * Revision 1.10  2025/11/10 21:11:09  dlr
 * Fix potential "start of buffer" overwrite when processing multiple ^ rules
 * This does not affect most hashes, but can cause a problem with parallel
 * processing on  MD5 and others.
 *
 * Revision 1.9  2025/10/21 18:11:28  dlr
 * Fix dup line
 *
 * Revision 1.8  2025/10/21 16:19:00  dlr
 * Make v rule a tiny bit faster
 *
 * Revision 1.7  2025/10/16 14:30:10  dlr
 * change rule 9 to v, change order from char, count to count, char
 *
 * Revision 1.6  2025/10/10 19:42:48  dlr
 * Add 9, h and H rules
 *
 * Revision 1.5  2025/08/11 14:19:41  dlr
 * add parserules()
 *
 * Revision 1.4  2020/03/11 02:49:29  dlr
 * SSSE modifications complete.  About to start on fastrule
 *
 * Revision 1.3  2020/03/08 07:12:31  dlr
 * Improve rule processing
 *
 */

extern char *Rulepos;


#ifdef NOTDEF
void print128(char *s,__m128i v)
{
    unsigned char *z = (unsigned char *)&v;
    int x;
    fprintf(stderr,"%s",s);
    for (x=0; x < 16; x++)
       fprintf(stderr,"%02x",z[x]);
    fprintf(stderr,"\n");
}
#endif

static inline unsigned char positiontranslate(char c) {
   char *res;
   res = strchr(Rulepos,c);
   if (!res) {
       fprintf(stderr,"Invalid position %c in rules",c);
       return(1);
    }
   return(((res - Rulepos) & 0xff)+1);
}

#ifdef SPARC
#define NOUNALIGN 1
static inline int lfastcmp(void *dest,void *src,int len) {
  unsigned char *d = (unsigned char *) dest;
  unsigned char *s = (unsigned char *) src;
  while (len--) {
    if (*s++ != *d++)
      return(1);
  }
  return(0);
}
#else
#ifdef AIX
#define NOUNALIGN 1
static inline int lfastcmp(void *dest,void *src,int len) {
  unsigned char *d = (unsigned char *) dest;
  unsigned char *s = (unsigned char *) src;
  while (len--) {
    if (*s++ != *d++)
      return(1);
  }
  return(0);
}
#else

static inline int lfastcmp(void *dest, void *src, int len) {
  unsigned long *d = (unsigned long *) dest;
  unsigned long *s = (unsigned long *) src;
  int l = (len / sizeof(unsigned long));
  if ((l * sizeof(unsigned long)) != len)
    l++;
  while (l--) {
    if (*s++ != *d++)
      return (1);
  }
  return (0);
}

#endif
#endif


void getcpuinfo() {
    int a,b,c,d;
#ifndef NOTINTEL
    IntelSSE = a = b = c = d = 0;
    __cpuid(1,a,b,c,d);
    if (c & bit_SSE3)
	IntelSSE = 30;
    if (c & bit_SSE4_1)
    	IntelSSE = 41;
    if (c & bit_SSE4_2)
    	IntelSSE = 42;
#endif
}

#define PARSEHEX \
	c1 = *t++;\
	if (c1 && c1 == '\\') {\
	    c1 = *t++;\
	    switch (c1) {\
		case '3':\
		case '2':\
		case '1':\
		case '0':\
		    if (*t != 'x' && *t != 'X') {\
			c1 -= '0';\
			while (*t >= '0' && *t <= '7') {\
			    c1 = c1 << 3;\
			    c1 |= (*t - '0');\
			    t++;\
			}\
			lbuf[x++] = c1;\
			break;\
		    }\
		    /* fall through */\
\
		case 'x':\
		case 'X':\
		    c1 = (char)trhex[*(unsigned char *)t++];\
		    c1 = (c1 << 4) + (char)trhex[*(unsigned char *)t++];\
		    lbuf[x++] = c1;\
		    break;\
		case '\\':\
		    lbuf[x++] = c1;\
		    break;\
\
		case '\000':\
		    lbuf[x++] = '\\';\
		    break;\
		default:\
		    t--;\
		    lbuf[x++] = '\\';\
		    break;\
	    }\
	} else { lbuf[x++]=c1;}

int packrules(char *line) {
  char *t, *s, *d, c, lbuf[10240], n,c1;
  int x, y, rulefail = 0;


  s = d = line;
  while ((c = *s++)) {
    if (c == '#' || c == '\r' || c == '\n')
      break;
    if (c <= 0 || c > 126) {rulefail=1; break;}
    switch (c) {

      case '[':
        if (s[0] == '^') {
	   *d++ = 'o';
	   *d++ = 1;
	   *d++ = s[1];
	   s += 2;
	} else {
	   *d++ = c;
	}
	break;
      case '$':
      case '^':
        t = s;
        x = 0;
	PARSEHEX;
        while (*t) {
          if (*t == c && t[1] && x < 254) {
	    t++;
	    PARSEHEX;
            continue;
          }
          if (*t == ' ' || *t == '\t' || *t == ':') {
            t++;
            continue;
          }
          break;
        }
        switch (x) {
          case 0:
	    *d++ = c;
	    *d++ = c1;
            break;

          case 1:
            *d++ = c;
            *d++ = lbuf[0];
            s = t;
            break;

          default:
            switch (c) {
              case '^':
                *d++ = 0xfe;
		*d++ = (unsigned char) x;
                for (y = x - 1; y >= 0; y--)
                  *d++ = lbuf[y];
                break;
              case '$':
                *d++ = 0xff;
		*d++ = (unsigned char) x;
                for (y = 0; y < x; y++)
                  *d++ = lbuf[y];
                break;
              default:
                fprintf(stderr, "impossible\n");
                exit(1);
                break;
            }
            s = t;
            break;
        }
        break;


      case '@':
      case 'e':
      case '!':
        *d++ = c;
        *d++ = *s++;
        break;

      case 'D':
      case '\'':
      case 'Z':
      case 'z':
      case '/':
      case '(':
      case ')':
      case '_':
      case '<':
      case '>':
      case '+':
      case '-':
      case '.':
      case ',':
      case 'T':
      case 'L':
      case 'R':
      case 'y':
      case 'Y':
      case 'p':
	*d++ = c;
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
	  fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
	break;


      case 'v':
        *d++ = c;
	n = *s++;
        if ((n < '1') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
	  fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n)-1;
        *d++ = *s++;
	break;

      case 's':
        *d++ = c;
        *d++ = *s++;
        *d++ = *s++;
	break;

      case '=':
      case '%':
        *d++ = c;
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
        *d++ = *s++;
        break;

      case 'i':
      case 'o':
        *d++ = c;
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
        *d++ = *s++;
        break;

      case 'O':
      case 'x':
      case '*':
        *d++ = c;
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
	break;

      case 'X':
        *d++ = c;
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
	break;

      case ' ':
      case '\t':
      case ':':
        break;


      default:
        *d++ = c;
        break;
    }
  }
  *d++ = 0;
  return (rulefail);
}

char * parserules(char *line) {
  char *t, *s, c, lbuf[10240], n,c1;
  char *lastvalid;
  int x, y, rulefail = 0;


  lastvalid = s = line;
  while ((c = *s)) {
    if (c == '#' || c == '\r' || c == '\n')
      break;
    if (c <= 0 || c > 126) {rulefail=1; break;}
    if (c == ' ' || c == '\t' || c == ':') {
        s++;
        continue;
    }
    lastvalid = s++;
    switch (c) {

      case '[':
        if (s[0] == '^') {
	   s += 2;
	} else {
	}
	break;
      case '$':
      case '^':
        t = s;
        x = 0;
	PARSEHEX;
        while (*t) {
          if (*t == c && t[1] && x < 254) {
 	    lastvalid = t;
	    t++;
	    PARSEHEX;
            continue;
          }
          if (*t == ' ' || *t == '\t' || *t == ':') {
            t++;
            continue;
          }
          break;
        }
        switch (x) {
          case 0:
            break;

          case 1:
            s = t;
            break;

          default:
            switch (c) {
              case '^':
                break;
              case '$':
                break;
              default:
                fprintf(stderr, "impossible\n");
                exit(1);
                break;
            }
            s = t;
            break;
        }
        break;


      case '@':
      case 'e':
      case '!':
        s++;
        break;

      case 'D':
      case '\'':
      case 'Z':
      case 'z':
      case '/':
      case '(':
      case ')':
      case '_':
      case '<':
      case '>':
      case '+':
      case '-':
      case '.':
      case ',':
      case 'T':
      case 'L':
      case 'R':
      case 'y':
      case 'Y':
      case 'p':
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
	break;


      case 'v':
        n = *s++;
        if ((n < '1') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
	  fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
	n = *s++;
	break;

      case 's':
	break;

      case '=':
      case '%':
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        s++;
        break;

      case 'i':
      case 'o':
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        s++;
        break;

      case 'O':
      case 'x':
      case '*':
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
	break;

      case 'X':
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
	break;

      case ' ':
      case '\t':
      case ':':
        break;


      default:
        break;
    }
  }
  if (rulefail) return(NULL);
  return (lastvalid);
}


/* Apply rules to current word.  Basic error checking only.
   line points to the original input word - no touching this.
   pass points to the word we will be altering.
   len is the length of the line.  You cannot assume null terminates the line.
   rule points to the input rule, null terminated.
*/
#define FASTLEN 32

int applyrule(char *line, char *pass, int len, char *rule) {
    char *s, *d, *t, r, *cpass;
    unsigned char c, c1;
    char *orule = rule;
    unsigned long *s1, *d1;
    int x, y, z, clen, tlen;
#ifndef NOTINTEL
    __m128i *p128,*q128, a128,b128,c128,d128;
#endif
    char Memory[MAXLINE+16], Base64buf[MAXLINE+16];
    static char *hextab = "0123456789abcdef";
    static char *Hextab = "0123456789ABCDEF";
    int memlen;

    memlen = 0;
    Memory[0] = 0;
  if (len > MAXLINE) 
     return(-2);

if (len < FASTLEN) {

  cpass = pass+512;
  memcpy(cpass,line,len);

  cpass[len] = 0;
  clen = len;
  rule = orule;

  while ((c = *rule++)) {
    if (cpass < (pass+FASTLEN)) goto slowrule;
    /* fprintf(stderr,"rule=%c%s len=%d curpass=%s\n",c,rule,clen,cpass);   */
    switch (c) {
      case 0x02: /* control B */
	goto slowrule;

      default:
        /*
	      fprintf(stderr,"Unknown rule --> %c <--- in %s\n",c,orule);
        return(-1);
        */
        break;
      case 'h':
      case 'H':
	  goto slowrule;
	  break;
      case 0xff:
	x = *rule++ & 0xff;
	s = rule;
	rule += x;
	if ((clen + x) > FASTLEN)
	   goto slowrule;
        memcpy(cpass+clen,s,x);
	clen += x;
        break;

      case 0xfe:
	x = *rule++ & 0xff;
	t = rule + x;
	if ((x+clen) > FASTLEN)
	   goto slowrule;
	cpass -= x;
	for (y=0; y < x; y++)
	    cpass[y] = rule[y];
        clen += x;
        rule = t;
        break;


      case 'M':
	memcpy(Memory,cpass,clen);
        memlen = clen;
	break;

      case '4':
	y = memlen;
	if ((clen + memlen) > FASTLEN)
	   goto slowrule;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	memcpy(cpass+clen,Memory,y);
	clen += y;
	break;

    case '6':
	y = memlen;
	if ((clen + memlen) > FASTLEN)
	   goto slowrule;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	cpass -= y;
	memcpy(cpass,Memory,y);
	clen += y;
	break;

    case 'Q':
        if (memlen == clen && memcmp(cpass,Memory,memlen) == 0)
	    return(-1);
	break;

    case 'X':
        y = *rule++ - 1;
	tlen = *rule++ - 1;
	z = *rule++ - 1;
	if ((clen + tlen) > FASTLEN) 
	    goto slowrule;
	if (tlen > memlen)
	    tlen = memlen;
	for (x=clen; x >= z; x--)
	   cpass[x+tlen] = cpass[x];
	for (x=0; x < tlen; x++) 
	   cpass[x+z] = Memory[x];
	clen += tlen;
	break;

        


      case '_':
        y = *rule++ - 1;
	if (y != len)
	    return(-1);
	break;
      case '<':
        y = *rule++ - 1; 
        if (clen < y)
          return (-1);
        break;
      case '>':
        y = *rule++ - 1; 
        if (clen > y)
          return (-1);
        break;

      case '!':
        c = *rule++;
	for (x=0; x < clen; x++)
	    if (cpass[x] == c) return (-1);
        break;

      case '/':
        c = *rule++;
	for (x=0; x < clen; x++)
	   if (cpass[x] == c) break;
        if (x >= clen )
          return (-1);
        break;

      case '(':
        c = *rule++;
        if (clen > 0 && cpass[0] != c)
          return (-1);
        break;
      case ')':
        c = *rule++;
        if (clen > 0 && cpass[clen - 1] != c)
          return (-1);
        break;


      case 'S':
        for (x = 0; x < clen; x++) {
          if (cpass[x] == 'a' || cpass[x] == 'A')
            cpass[x] = 0xa;
        }
        break;

      case '#':
        goto fast_exit;
        break;

      case ':':
      case ' ':
      case '\t':
        break;

      case 'l':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if (c >= 'A' && c <= 'Z')
            cpass[x] = c ^ 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long)t & 15)  && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case 'u':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if (c >= 'a' && c <= 'z')
            cpass[x] = c ^ 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));

	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case 'c':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'a' && c <= 'z')
              cpass[x] = c - 0x20;
            z = 1;
            continue;
          }
          if (c >= 'A' && c <= 'Z')
            cpass[x] = c + 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = cpass[x];
	    if (c >= 'a' && c <= 'z') {
	        cpass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case 'C':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'A' && c <= 'Z')
              cpass[x] = c + 0x20;
            z = 1;
            continue;
          }
          if (c >= 'a' && c <= 'z')
            cpass[x] = c - 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = cpass[x];
	    if (c >= 'A' && c <= 'Z') {
	        cpass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case 't':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
            cpass[x] = c ^ 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    c128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    b128 = _mm_and_si128(b128,c128);
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case 'T':
        y = *rule++ - 1;
        c = cpass[y];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
          cpass[y] = c ^ 0x20;
        break;

      case 'r':
        for (x = 0; x < clen / 2; x++) {
          c = cpass[x];
          cpass[x] = cpass[clen - x - 1];
          cpass[clen - x - 1] = c;
        }
        break;

      case 'd':
        tlen = clen;
        if ((tlen + clen) > FASTLEN)
          goto slowrule;
        if (tlen > 0) {
	  memcpy(cpass+clen,cpass,tlen);
          clen += tlen;
	}
        break;

      case 'f':
        tlen = clen;
        if ((tlen + clen) > FASTLEN)
          goto slowrule;
        if (tlen < 0)
          tlen = 0;
        for (x = 0; x < tlen; x++)
          cpass[clen + tlen - x - 1] = cpass[x];
        clen += tlen;
        break;

      case '{':
        if (clen > 0) {
          y = 1;
          while (*rule == '{' && y < clen) {
            y++;
            rule++;
          }
          for (x = 0; x < y; x++)
            cpass[x + clen] = cpass[x];
          for (; x < (clen + y); x++)
            cpass[x - y] = cpass[x];
        }
        break;

      case '}':
        if (clen > 0) {
          y = 1;
          while (*rule == '}' && y < clen) {
            y++;
            rule++;
          }
          for (x = clen - 1; x >= 0; x--)
            cpass[x + y] = cpass[x];
          for (x = 0; x < y; x++)
            cpass[x] = cpass[x + clen];
        }
        break;

      case '$':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in append at %s\n", orule);
          return (-3);
        }
	if ((clen+1) < FASTLEN) 
	  cpass[clen++] = c;
	else
	  goto slowrule;
        break;

      case '^':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in insert at %s\n", orule);
          return (-3);
        }
	if ((clen+1) > FASTLEN)
	  goto slowrule;
	cpass--;
	cpass[0] = c;
	clen++;
        break;

      case '[':
        if (clen > 0) {
          y = 1;
          while (*rule == '[' && y < clen) {
            y++;
            rule++;
          }
	  cpass += y;
          clen -= y;
        }
        break;

      case ']':
        if (clen > 0) {
          y = 1;
          while (*rule == ']' && y < clen) {
            y++;
            rule++;
          }
          clen -= y;
        }
        break;

      case 'D':
        y = *rule++ - 1;
        if (y < clen) {
          for (x = y + 1; x < clen; x++)
            cpass[x - 1] = cpass[x];

          clen--;
        }
        break;

      case 'x':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y) {
          for (x = 0; x < z && ((y + x) < clen); x++) {
            cpass[x] = cpass[y + x];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case 'O':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y && (y + z) <= clen) {
          for (x = y; x < clen && (x + z) < clen; x++) {
            cpass[x] = cpass[x + z];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case 'i':
        y = *rule++ - 1;
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Invalid insert character in rule %s\n", orule);
          return (-3);
        }
        if (clen > y) {
	  if ((clen+1) > FASTLEN)
	      goto slowrule;
	   for (x = clen; x >= y && x > 0; x--)
	      cpass[x] = cpass[x - 1];
	    clen++;
	    cpass[y] = c;
        }
        break;
      case 'o':
        y = *rule++ - 1;
        c = *rule++;
	if (c == 0) {
	    fprintf(stderr,"Invalid character in o rule: %x\n",c);
	    return(-3);
	}
        if (y < clen)
          cpass[y] = c;
	if (y == 0 && clen == 0) {
	   cpass[0] = c; clen++;
	}
        break;
      case '\'':
        y = *rule++ - 1;
        if (y < clen)
          clen = y;
        break;

      case 'v':
	x = *rule++;
	c1 = *rule++;
	if (x <=0) {
	  fprintf(stderr,"Invalid count %d in rule: %c\n",x,c);
	  return(-3);
	}
	if (clen < x) break;
        if ((clen + clen/x) >= FASTLEN) goto slowrule;
        y = clen / x;
	s = &cpass[clen-1];
	d = s + y;
        for (y = clen; y > 0; y--) {
	  if ((y%x) == 0) {
 	    *d-- = c1;
	    if (s == d) break;
	  }
	  *d-- = *s--;
        }
	clen += clen / x;
	cpass[clen] = 0;
	break;

      case 's':
	c = *rule++;
        r = *rule++;
        if (!c || !r) {
          fprintf(stderr, "Invalid replace in rule: %c, %c\n", c,r);
          return (-3);
        }
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          if (cpass[x] == c)
            cpass[x] = r;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	    if (*t == c)
	        *t = r;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_cmpeq_epi8(d128,_mm_set1_epi8((char)c));
	    b128 = _mm_and_si128(a128,_mm_set1_epi8((char)(c^r)));
	    *p128++ = _mm_xor_si128(d128,b128);
	}
#endif
        break;

      case '@':
        c = *rule++;

        if (!c) {
          fprintf(stderr, "Invalid purge in rule %s\n", orule);
          return (-3);
        }
        d = cpass;
        s = cpass;
        for (x = 0; x < clen; x++) {
          if (*s != c)
            *d++ = *s;
          s++;
        }
        clen -= (s - d);
	if (clen < 0)
	  clen = 0;
        break;

      case 'z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((clen+y) > FASTLEN)
	    goto slowrule;
          for (x = clen - 1; x > 0; x--)
            cpass[x + y] = cpass[x];
          for (x = 1; x <= y; x++)
            cpass[x] = cpass[0];
          clen += y;
        }
        break;

      case 'Z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((y + clen) > FASTLEN)
	    goto slowrule;
          for (x = 0; x < y; x++)
            cpass[x + clen] = cpass[clen - 1];
          clen += y;
        }
        break;

      case 'q':
        tlen = clen * 2;
        if (tlen > FASTLEN)
          goto slowrule;
        for (x = clen * 2; x > 0; x -= 2) {
          cpass[x - 1] = cpass[x / 2 - 1];
          cpass[x - 2] = cpass[x / 2 - 1];
        }
        clen += clen;
        break;

      case 'p':
        y = *rule++ - 1;
        if (clen > 0 && y > 0) {
          d = &cpass[clen];
          z = y;
          tlen = clen;
          for (; y; y--) {
            if ((clen + tlen) > FASTLEN)
              goto slowrule;
            for (x = 0; x < tlen; x++)
              *d++ = cpass[x];
            clen += tlen;
          }
        }
        break;

      case 'k':
        if (clen >1) {
	   c = cpass[0];
	   cpass[0] = cpass[1];
	   cpass[1] = c;
	}
	break;

      case 'K':
        if (clen > 1) {
          c = cpass[clen - 2];
          cpass[clen - 2] = cpass[clen - 1];
          cpass[clen - 1] = c;
        }
        break;

      case '*':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (y < clen && z < clen) {
          c = cpass[y];
          cpass[y] = cpass[z];
          cpass[z] = c;
        }
        break;

      case 'L':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y] = cpass[y] << 1;
        break;

      case 'R':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y] = cpass[y] >> 1;
        break;

      case '+':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y]++;
        break;

      case '-':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y]--;
        break;

      case '.':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y] = cpass[y + 1];
        break;

      case ',':
        y = *rule++ - 1;
        if (y < clen && y > 0)
          cpass[y] = cpass[y - 1];
        break;

      case 'y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > FASTLEN)
	     goto slowrule;
          memmove(cpass + y, cpass, clen);
          clen += y;
        }
        break;
      
      case 'Y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > FASTLEN)
	    goto slowrule;
          memmove(cpass + clen, cpass + (clen - y), y);
          clen += y;
        }
        break;

      case 'E':
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (c == ' ')
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            cpass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            cpass[x] = c ^ 0x20;
        }
        break;
      case 'e':
	c1 = *rule++;
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (c == c1)
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            cpass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            cpass[x] = c ^ 0x20;
        }
        break;
    }
  }
fast_exit:
  memmove(pass,cpass,clen);
  goto app_exit;
 
}
slowrule:
  memcpy(pass,line,len);
  pass[len] = 0;

  clen = len;
  rule = orule;

  while ((c = *rule++)) {
    /* printf("rule=%c%s len=%d curpass=%s\n",c,rule,clen,pass);   */
    switch (c) {
      default:
        /*
	      fprintf(stderr,"Unknown rule --> %c <--- in %s\n",c,orule);
        return(-1);
        */
        break;
      case 0x02: /* Control B */
	if (clen > (MAXLINE*4/3)) break;
	clen = b64_encode(pass, Base64buf, clen);
	memcpy(pass,Base64buf,clen); pass[clen] = 0;
	break;

      case 'h':
      case 'H':
        d = hextab;
	if (c == 'H') d = Hextab;
        x = clen;
        if ((clen +x) > MAXLINE)
          x = MAXLINE - clen;
	clen = clen + x;
        for (x--; x >=0; x--) {
          c = pass[x];
	  pass[x*2] = d[(c>>4)&0xf];
	  pass[(x*2)+1] = d[c & 0xf];
	}
	pass[clen] = 0;
	break;
	  
 
           
      case 0xff:
	x = *rule++ & 0xff;
	s = rule;
	rule += x;
	if ((clen + x) > MAXLINE)
	   x = MAXLINE - clen;
	memcpy(pass+clen,s,x);
	clen += x;
        break;

      case 0xfe:
	x = *rule++ & 0xff;
	t = rule + x;
	if ((x+clen) > MAXLINE)
	   x = MAXLINE-clen;
	memmove(pass+x,pass,clen);
	for (y=0; y < x; y++)
	    pass[y] = rule[y];
        clen += x;
        rule = t;
        break;


      case 'M':
	memcpy(Memory,pass,clen);
        memlen = clen;
	break;

      case '4':
	y = memlen;
	if ((clen + memlen) > MAXLINE)
	   y = MAXLINE - clen;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	memcpy(pass+clen,Memory,y);
	clen += y;
	break;

    case '6':
	y = memlen;
	if ((clen + memlen) > MAXLINE)
	   y = MAXLINE - clen;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	memmove(pass+y,pass,clen);
	memcpy(pass,Memory,y);
	clen += y;
	break;

    case 'Q':
        if (memlen == clen && memcmp(pass,Memory,memlen) == 0)
	    return(-1);
	break;

    case 'X':
        y = *rule++ - 1;
	tlen = *rule++ - 1;
	z = *rule++ - 1;
	if ((clen + tlen) > MAXLINE) 
	    tlen = MAXLINE - clen;
	if (tlen > memlen)
	    tlen = memlen;
	for (x=clen; x >= z; x--)
	   pass[x+tlen] = pass[x];
	for (x=0; x < tlen; x++) 
	   pass[x+z] = Memory[x];
	clen += tlen;
	break;

        


      case '_':
        y = *rule++ - 1;
	if (y != len)
	    return(-1);
	break;
      case '<':
        y = *rule++ - 1; 
        if (clen < y)
          return (-1);
        break;
      case '>':
        y = *rule++ - 1; 
        if (clen > y)
          return (-1);
        break;

      case '!':
        c = *rule++;
	for (x=0; x < clen; x++)
	    if (pass[x] == c) return (-1);
        break;

      case '/':
        c = *rule++;
	for (x=0; x < clen; x++)
	   if (pass[x] == c) break;
        if (x >= clen )
          return (-1);
        break;

      case '(':
        c = *rule++;
        if (clen > 0 && pass[0] != c)
          return (-1);
        break;
      case ')':
        c = *rule++;
        if (clen > 0 && pass[clen - 1] != c)
          return (-1);
        break;


      case 'S':
        for (x = 0; x < clen; x++) {
          if (pass[x] == 'a' || pass[x] == 'A')
            pass[x] = 0xa;
        }
        break;

      case '#':
        goto app_exit;
        break;

      case ':':
      case ' ':
      case '\t':
        break;

      case 'l':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if (c >= 'A' && c <= 'Z')
            pass[x] = c ^ 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long)t & 15)  && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case 'u':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if (c >= 'a' && c <= 'z')
            pass[x] = c ^ 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));

	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case 'c':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'a' && c <= 'z')
              pass[x] = c - 0x20;
            z = 1;
            continue;
          }
          if (c >= 'A' && c <= 'Z')
            pass[x] = c + 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = pass[x];
	    if (c >= 'a' && c <= 'z') {
	        pass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case 'C':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'A' && c <= 'Z')
              pass[x] = c + 0x20;
            z = 1;
            continue;
          }
          if (c >= 'a' && c <= 'z')
            pass[x] = c - 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = pass[x];
	    if (c >= 'A' && c <= 'Z') {
	        pass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case 't':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
            pass[x] = c ^ 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    c128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    b128 = _mm_and_si128(b128,c128);
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case 'T':
        y = *rule++ - 1;
        c = pass[y];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
          pass[y] = c ^ 0x20;
        break;

      case 'r':
        for (x = 0; x < clen / 2; x++) {
          c = pass[x];
          pass[x] = pass[clen - x - 1];
          pass[clen - x - 1] = c;
        }
        break;

      case 'd':
        tlen = clen;
        if ((tlen + clen) > MAXLINE)
          tlen = MAXLINE - clen;
        if (tlen > 0) {
	  memcpy(pass+clen,pass,tlen);
	  clen += tlen;
	}
        break;

      case 'f':
        tlen = clen;
        if ((tlen + clen) > MAXLINE)
          tlen = MAXLINE - clen;
        if (tlen < 0)
          tlen = 0;
        for (x = 0; x < tlen; x++)
          pass[clen + tlen - x - 1] = pass[x];
        clen += tlen;
        break;

      case '{':
        if (clen > 0) {
          y = 1;
          while (*rule == '{' && y < clen) {
            y++;
            rule++;
          }
          for (x = 0; x < y; x++)
            pass[x + clen] = pass[x];
          for (; x < (clen + y); x++)
            pass[x - y] = pass[x];
        }
        break;

      case '}':
        if (clen > 0) {
          y = 1;
          while (*rule == '}' && y < clen) {
            y++;
            rule++;
          }
          for (x = clen - 1; x >= 0; x--)
            pass[x + y] = pass[x];
          for (x = 0; x < y; x++)
            pass[x] = pass[x + clen];
        }
        break;

      case '$':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in append at %s\n", orule);
          return (-3);
        }
	if ((clen+1) < MAXLINE) 
	  pass[clen++] = c;
        break;

      case '^':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in insert at %s\n", orule);
          return (-3);
        }
	if ((clen+1) < MAXLINE) {
	  memmove(pass+1,pass,clen);
	  pass[0] = c;
	  clen++;
	}
        break;

      case '[':
        if (clen > 0) {
          y = 1;
          while (*rule == '[' && y < clen) {
            y++;
            rule++;
          }
	  memmove(pass,pass+y,clen);
          clen -= y;
        }
        break;

      case ']':
        if (clen > 0) {
          y = 1;
          while (*rule == ']' && y < clen) {
            y++;
            rule++;
          }
          clen -= y;
        }
        break;

      case 'D':
        y = *rule++ - 1;
        if (y < clen) {
          for (x = y + 1; x < clen; x++)
            pass[x - 1] = pass[x];

          clen--;
        }
        break;

      case 'x':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y) {
          for (x = 0; x < z && ((y + x) < clen); x++) {
            pass[x] = pass[y + x];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case 'O':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y && (y + z) <= clen) {
          for (x = y; x < clen && (x + z) < clen; x++) {
            pass[x] = pass[x + z];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case 'i':
        y = *rule++ - 1;
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Invalid insert character in rule %s\n", orule);
          return (-3);
        }
        if (clen > y) {
	  if ((clen+1) < MAXLINE) {
	    for (x = clen; x >= y && x > 0; x--)
	      pass[x] = pass[x - 1];
	    clen++;
	    pass[y] = c;
	  }
        }
        break;
      case 'o':
        y = *rule++ - 1;
        c = *rule++;
	if (c == 0) {
	    fprintf(stderr,"Invalid character in o rule: %x\n",c);
	    return(-3);
	}
        if (y < clen)
          pass[y] = c;
	if (y == 0 && clen == 0) {
	   pass[0] = c; clen++;
	}
        break;
      case '\'':
        y = *rule++ - 1;
        if (y < clen)
          clen = y;
        break;

      case 'v':
	x = *rule++;
	c1 = *rule++;
	if (x <=0) {
	  fprintf(stderr,"Invalid count %d in rule: %c\n",x,c);
	  return(-3);
	}
        y = clen / x;
	s = &pass[clen-1];
	d = s + y;
        for (y = clen; y > 0; y--) {
	  if ((y%x) == 0) {
 	    *d-- = c1;
	    if (s == d) break;
	  }
	  *d-- = *s--;
        }
	clen += clen / x;
	pass[clen] = 0;
	break;
	
      case 's':
        c = *rule++;
        r = *rule++;
        if (!c || !r) {
          fprintf(stderr, "Invalid replace in rule: %c, %c\n", c,r);
          return (-3);
        }
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          if (pass[x] == c)
            pass[x] = r;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	    if (*t == c)
	        *t = r;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_cmpeq_epi8(d128,_mm_set1_epi8((char)c));
	    b128 = _mm_and_si128(a128,_mm_set1_epi8((char)(c^r)));
	    *p128++ = _mm_xor_si128(d128,b128);
	}
#endif
        break;

      case '@':
        c = *rule++;

        if (!c) {
          fprintf(stderr, "Invalid purge in rule %s\n", orule);
          return (-3);
        }
        d = pass;
        s = pass;
        for (x = 0; x < clen; x++) {
          if (*s != c)
            *d++ = *s;
          s++;
        }
        clen -= (s - d);
	if (clen < 0)
	  clen = 0;
        break;

      case 'z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((clen+y) > MAXLINE)
	    y = MAXLINE - clen;
          for (x = clen - 1; x > 0; x--)
            pass[x + y] = pass[x];
          for (x = 1; x <= y; x++)
            pass[x] = pass[0];
          clen += y;
        }
        break;

      case 'Z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((y + clen) > MAXLINE)
	    y = MAXLINE - clen;
          for (x = 0; x < y; x++)
            pass[x + clen] = pass[clen - 1];
          clen += y;
        }
        break;

      case 'q':
        tlen = clen * 2;
        if (tlen > MAXLINE)
          break;
        for (x = clen * 2; x > 0; x -= 2) {
          pass[x - 1] = pass[x / 2 - 1];
          pass[x - 2] = pass[x / 2 - 1];
        }
        clen += clen;
        break;

      case 'p':
        y = *rule++ - 1;
        if (clen > 0 && y > 0) {
          d = &pass[clen];
          z = y;
          tlen = clen;
          for (; y; y--) {
            if ((clen + tlen) > MAXLINE)
              break;
            for (x = 0; x < tlen; x++)
              *d++ = pass[x];
            clen += tlen;
          }
        }
        break;

      case 'k':
        if (clen >1) {
	   c = pass[0];
	   pass[0] = pass[1];
	   pass[1] = c;
	}
	break;

      case 'K':
        if (clen > 1) {
          c = pass[clen - 2];
          pass[clen - 2] = pass[clen - 1];
          pass[clen - 1] = c;
        }
        break;

      case '*':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (y < clen && z < clen) {
          c = pass[y];
          pass[y] = pass[z];
          pass[z] = c;
        }
        break;

      case 'L':
        y = *rule++ - 1;
        if (y < clen)
          pass[y] = pass[y] << 1;
        break;

      case 'R':
        y = *rule++ - 1;
        if (y < clen)
          pass[y] = pass[y] >> 1;
        break;

      case '+':
        y = *rule++ - 1;
        if (y < clen)
          pass[y]++;
        break;

      case '-':
        y = *rule++ - 1;
        if (y < clen)
          pass[y]--;
        break;

      case '.':
        y = *rule++ - 1;
        if (y < clen)
          pass[y] = pass[y + 1];
        break;

      case ',':
        y = *rule++ - 1;
        if (y < clen && y > 0)
          pass[y] = pass[y - 1];
        break;

      case 'y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > MAXLINE)
	     y = MAXLINE - clen;
          memmove(pass + y, pass, clen);
          clen += y;
        }
        break;
      
      case 'Y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > MAXLINE)
	    y = MAXLINE - clen;
          memmove(pass + clen, pass + (clen - y), y);
          clen += y;
        }
        break;

      case 'E':
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (c == ' ')
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            pass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            pass[x] = c ^ 0x20;
        }
        break;
      case 'e':
	c1 = *rule++;
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (c == c1)
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            pass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            pass[x] = c ^ 0x20;
        }
        break;
    }
  }
app_exit:
  if (clen < 0)
    return (-1);
  pass[clen] = 0;
  /* fprintf(stderr,"final rule=%s len=%d pass=%s\n",orule,clen,pass);  */
  if (len != clen || lfastcmp(line, pass, clen) != 0)
    return (clen);
  return (-2);
}

