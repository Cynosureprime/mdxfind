/*
 * hx.c - main driver for the hx hash expression language
 *
 * Usage:
 *   hx 'md5(pass . salt)'                     expression mode
 *   hx -f script.hx                           script mode
 *   hx -d 'sha1(md5(pass) . salt)'            dump AST (debug)
 *   hx -b 'sha1(md5(pass) . salt)'            dump bytecode (debug)
 *
 * Reads passwords from stdin (one per line) or uses -p for a single
 * password.  Salt provided via -s flag.
 */

/*
 * $Log: hx.c,v $
 * Revision 1.6  2026/09/25 16:27:35  dlr
 * Refuse an oversized $TESTVEC[] rather than shortening it. hx_expand_testvec clamped to HX_MAX_TESTVEC, which returns a well-formed digest for a password that was never presented and gives the caller no way to notice. It now names the requested size and the limit and exits 1. Pairs with hashpipe.c, where -X does the same and the bound is now the same number on both, so the 2 MB stated in hx.1 holds wherever a vector is read. The verify path in hashpipe keeps clamping on purpose, since a bulk ledger run should skip one oversized record rather than abort.
 *
 * Revision 1.5  2026/09/25 16:21:52  dlr
 * $TESTVEC[] was never supported in the standalone hx: it is a REPEAT COUNT and both decode paths treated it as a hex container. The comment stated the belief -- dollar TESTVEC bracket, same hex decoding, for large binary vectors -- and the code hex-decoded the text after the prefix until the first non-hex character. So a vector of 10,240 zero bytes read the pattern 00, stopped at the space, and hashed ONE NUL: md5 returned 93b885adfe0da089cdf634904fd59f71 where the true digest is 1276481102f218c981e0324180bafd9f. Well formed, no diagnostic, exit 0, and the notation exists precisely for vectors too large to check by eye. The -p path and the stdin path each carried their own copy of that loop, and the stdin one decoded in place into the line buffer, which cannot hold an expansion. There is now ONE hx_expand_testvec used by both. That is not tidiness: fixing only the -p copy first left the two paths disagreeing on the same input, which is the argument for a single implementation made concrete. The pattern is hex, the count decimal, the separator any run of non-digits, the closing bracket optional; hashpipe.c decode_testvec_password is the reference and the two must accept the same shapes. HX_MAX_TESTVEC is hoisted to file scope, since a capacity constant only one function can see is how two callers come to disagree about it. Verified against independently computed digests: 10,240 NUL bytes on -p, 51,200 on stdin, and a four-fold deadbeef pattern all match, and all four channels -- hx -p, hx stdin, hashpipe -X and hashpipe -c -- now return the same digest for the same string. $HEX[] decoding is unchanged and malformed vectors still fall through to literal text.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "hx_ast.h"
#include "hx_vm.h"

/* from flex/bison generated code */
extern int  yyparse(void);
extern FILE *yyin;
extern int  hx_line;

/* set by bison on successful parse */
extern hx_node *hx_parse_result;

/* flex buffer for parsing a string instead of a file */
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char *str);
extern void yy_delete_buffer(YY_BUFFER_STATE buf);

/* ---- Entry point for hashpipe integration ---- */

/*
 * Parse and compile an hx expression or script.
 * Returns a compiled program ready for hx_vm_init/hx_vm_run.
 * If script_file is non-NULL, reads from file; otherwise parses expr.
 */
/* Count of diagnostics emitted by the hx lexer and parser since it was
 * last reset. The lexer's catch-all rule PRINTS an unknown character and
 * then drops it, so an expression like md5(md5($p)) lexes as md5(md5(p)),
 * compiles, and yields a program that silently is not what was written.
 * Callers that must not accept that -- the user-defined type loader --
 * reset this before compiling and check it after. hx_compile_expr itself
 * deliberately does NOT consult it, so catalog behaviour is unchanged. */
int hx_diag_count = 0;

hx_program *hx_compile_expr(const char *expr, const char *script_file)
{
	hx_node *ast;
	hx_program *prog;

	hx_line = 1;

	if (script_file) {
		yyin = fopen(script_file, "r");
		if (!yyin) {
			perror(script_file);
			return NULL;
		}
		if (yyparse() != 0) {
			fclose(yyin);
			return NULL;
		}
		fclose(yyin);
	} else {
		char *buf = malloc(strlen(expr) + 2);
		YY_BUFFER_STATE bs;
		sprintf(buf, "%s\n", expr);
		bs = yy_scan_string(buf);
		free(buf);
		if (yyparse() != 0) {
			yy_delete_buffer(bs);
			return NULL;
		}
		yy_delete_buffer(bs);
	}

	if (!hx_parse_result)
		return NULL;

	ast = hx_parse_result;
	hx_parse_result = NULL;
	prog = hx_compile(ast);
	hx_free(ast);
	return prog;
}

#if defined(HX_STANDALONE) && !defined(HX_NO_MAIN)

static void usage(void)
{
	fprintf(stderr,
	    "hx - hash expression language (v0.1)\n"
	    "\n"
	    "Usage:\n"
	    "  hx [options] 'expression'\n"
	    "  hx [options] -f script.hx\n"
	    "\n"
	    "Options:\n"
	    "  -p pass     single password (otherwise reads stdin)\n"
	    "  -s salt     salt value\n"
	    "  -S salt2    second salt value\n"
	    "  -P pepper   pepper value\n"
	    "  -d          dump AST (debug)\n"
	    "  -b          dump bytecode (debug)\n"
	    "  -f file     read script from file\n"
	    "\n"
	    "Expression examples:\n"
	    "  md5(pass)                  simple MD5 hash\n"
	    "  sha1(md5(pass) . salt)     compound: SHA1 of MD5-hex + salt\n"
	    "  md5_bin(pass)              MD5 raw binary output\n"
	    "  md5^1000(pass)             iterate MD5 1000 times\n"
	    "  hex(md5_bin(salt . pass))  explicit hex encoding\n"
	    "\n"
	    "  -u user     user/userid value\n"
	    "\n"
	    "Built-in variables: pass, salt, salt2, pepper, user\n"
	    "Default encoding: lowercase hex.  _bin suffix for raw bytes.\n"
	);
	exit(1);
}

static hx_node *do_parse(const char *expr, const char *script_file)
{
	hx_line = 1;

	if (script_file) {
		yyin = fopen(script_file, "r");
		if (!yyin) {
			perror(script_file);
			exit(1);
		}
		if (yyparse() != 0) {
			fprintf(stderr, "hx: parse failed\n");
			exit(1);
		}
		fclose(yyin);
	} else {
		char *buf = malloc(strlen(expr) + 2);
		YY_BUFFER_STATE bs;
		sprintf(buf, "%s\n", expr);
		bs = yy_scan_string(buf);
		free(buf);
		if (yyparse() != 0) {
			fprintf(stderr, "hx: parse failed\n");
			exit(1);
		}
		yy_delete_buffer(bs);
	}

	if (!hx_parse_result) {
		fprintf(stderr, "hx: empty program\n");
		exit(1);
	}
	return hx_parse_result;
}

/* Hoisted to file scope from inside main: hx_expand_testvec below needs it,
 * and a capacity constant that only one function can see is how two callers
 * end up disagreeing about it. */
#define HX_MAX_TESTVEC (2 * 1024 * 1024)  /* 2MB max decoded */

/* Expand $TESTVEC[HH... x N] into a fresh buffer.
 *
 * Returns the expanded length and stores a malloc'd buffer in *out, or -1 if
 * the string is not a well-formed vector, in which case *out is untouched and
 * the caller should treat the text as literal.
 *
 * ONE implementation for both the -p and the stdin path. They each carried
 * their own copy, both of which hex-decoded until the first non-hex character
 * on the belief that $TESTVEC[ was a hex container like $HEX[. It is a REPEAT
 * COUNT: $TESTVEC[00 x 10240] means 10,240 zero bytes, and the old code
 * returned one, so md5 gave 93b885adfe0da089cdf634904fd59f71 where the true
 * digest is 1276481102f218c981e0324180bafd9f -- well formed, no diagnostic,
 * exit 0. Fixing one copy and not the other is how the two paths disagreed
 * for a while during this change, which is the argument for there being one.
 *
 * The pattern is hex, the count is decimal, the separator is any run of
 * non-digits, and the closing bracket is optional. hashpipe.c
 * decode_testvec_password is the reference implementation and the two must
 * accept the same shapes; it is static there, so this is a deliberate second
 * implementation. If either moves, move both. */
static int hx_expand_testvec(const char *src, int srclen, char **out)
{
#define HX_HEXDIG(c) ((c) >= '0' && (c) <= '9' ? (c) - '0' : \
                      (c) >= 'a' && (c) <= 'f' ? (c) - 'a' + 10 : \
                      (c) >= 'A' && (c) <= 'F' ? (c) - 'A' + 10 : -1)
	const unsigned char *h = (const unsigned char *)src + 9;
	const unsigned char *e = (const unsigned char *)src + srclen;
	unsigned char pat[256];
	int patlen = 0, i;
	unsigned long long count = 0, total;
	char *buf;

	if (srclen <= 9 || strncmp(src, "$TESTVEC[", 9) != 0) return -1;
	if (e > h && e[-1] == ']') e--;

	while (h + 1 < e && patlen < (int)sizeof pat) {
		int hi = HX_HEXDIG(h[0]);
		int lo = HX_HEXDIG(h[1]);
		if (hi < 0 || lo < 0) break;
		pat[patlen++] = (unsigned char)((hi << 4) | lo);
		h += 2;
	}
	while (h < e && (*h < '0' || *h > '9')) h++;
	while (h < e && *h >= '0' && *h <= '9') {
		if (count > 99999999ULL) return -1;   /* absurd; treat as literal */
		count = count * 10ULL + (unsigned long long)(*h - '0');
		h++;
	}
#undef HX_HEXDIG
	if (patlen <= 0 || count == 0) return -1;

	total = (unsigned long long)patlen * count;
	if (total > (unsigned long long)HX_MAX_TESTVEC) {
		/* Refuse rather than shorten.  A clamp here returns a
		 * well-formed digest for a password that was never presented,
		 * and the caller cannot tell: before this, the same 3,000,000
		 * byte vector gave one answer from hx, a different one from
		 * hashpipe -X, and no verify at all from hashpipe -c, because
		 * each clamped at its own buffer size. */
		fprintf(stderr,
		        "hx: $TESTVEC[] expands to %llu bytes, over the %d byte "
		        "limit; refusing to truncate it\n",
		        total, HX_MAX_TESTVEC);
		exit(1);
	}

	buf = malloc((size_t)total);
	if (buf == NULL) {
		fprintf(stderr, "hx: cannot allocate %llu bytes for the "
		        "$TESTVEC[] vector\n", total);
		exit(1);
	}
	for (i = 0; (unsigned long long)i < total; i++)
		buf[i] = (char)pat[i % patlen];
	*out = buf;
	return (int)total;
}

int main(int argc, char **argv)
{
	int dump_ast = 0, dump_bytecode = 0;
	const char *script_file = NULL;
	const char *expr = NULL;
	const char *password = NULL;
	const char *salt = "";
	const char *salt2 = "";
	const char *pepper = "";
	const char *user = "";
	int opt;
	hx_node *ast;
	hx_program *prog;
	hx_vm vm;

	while ((opt = getopt(argc, argv, "dbf:p:s:S:P:u:h")) != -1) {
		switch (opt) {
		case 'd':
			dump_ast = 1;
			break;
		case 'b':
			dump_bytecode = 1;
			break;
		case 'f':
			script_file = optarg;
			break;
		case 'p':
			password = optarg;
			break;
		case 's':
			salt = optarg;
			break;
		case 'S':
			salt2 = optarg;
			break;
		case 'P':
			pepper = optarg;
			break;
		case 'u':
			user = optarg;
			break;
		case 'h':
		default:
			usage();
		}
	}

	/* remaining arg is expression */
	if (optind < argc)
		expr = argv[optind];

	if (!expr && !script_file)
		usage();

	if (expr && script_file) {
		fprintf(stderr, "hx: specify expression or -f file, not both\n");
		exit(1);
	}

	/* ---- parse ---- */
	ast = do_parse(expr, script_file);

	if (dump_ast) {
		hx_dump(ast, 0);
		hx_free(ast);
		return 0;
	}

	/* ---- compile ---- */
	prog = hx_compile(ast);
	hx_free(ast);

	if (dump_bytecode) {
		hx_program_dump(prog);
		hx_program_free(prog);
		return 0;
	}

	/* ---- run ---- */
	hx_vm_init(&vm, prog);

	/*
	 * Decode $HEX[...] and $TESTVEC[...] input encoding.
	 * $HEX[ — hex-decode the following bytes until a non-hex
	 * character is encountered.  No closing ] required.
	 * $TESTVEC[ — same hex decoding, for large binary vectors.
	 * The decoded data replaces the original password.
	 * Decoding is done into arena memory, never on the stack.
	 */
	{
	/* heap-allocated line buffer for stdin reading.
	 * Must accommodate $TESTVEC[] up to 2MB decoded = 4MB hex + prefix.
	 * NEVER stack-allocated — passwords can be very large. */
	int linecap = HX_MAX_TESTVEC * 2 + 64;  /* room for hex + prefix */
	char *linebuf = malloc(linecap);
	char *tvbuf = NULL;   /* expanded $TESTVEC[] for the stdin path; a vector
	                       * grows, so it cannot share linebuf the way the
	                       * $HEX[] decode does */
	if (!linebuf) { perror("malloc"); exit(1); }

	if (password) {
		/* single password mode — decode $HEX[ if present */
		const char *pw = password;
		int pwlen = strlen(password);
		char *decoded = NULL;
		int dlen;

		if (pwlen > 5 && strncmp(pw, "$HEX[", 5) == 0) {
			decoded = malloc(pwlen);
			dlen = 0;
			const unsigned char *h = (const unsigned char *)pw + 5;
			while (*h) {
				int hi, lo;
				unsigned char c = *h;
				if ((c >= '0' && c <= '9'))      hi = c - '0';
				else if ((c >= 'a' && c <= 'f')) hi = c - 'a' + 10;
				else if ((c >= 'A' && c <= 'F')) hi = c - 'A' + 10;
				else break;  /* non-hex terminates (includes ]) */
				h++;
				c = *h;
				if ((c >= '0' && c <= '9'))      lo = c - '0';
				else if ((c >= 'a' && c <= 'f')) lo = c - 'a' + 10;
				else if ((c >= 'A' && c <= 'F')) lo = c - 'A' + 10;
				else { decoded[dlen++] = hi << 4; break; }
				h++;
				decoded[dlen++] = (hi << 4) | lo;
			}
			pw = decoded;
			pwlen = dlen;
		} else if (pwlen > 9 && strncmp(pw, "$TESTVEC[", 9) == 0) {
			int n = hx_expand_testvec(pw, pwlen, &decoded);
			if (n >= 0) {
				pw = decoded;
				pwlen = n;
			}
			/* Malformed: leave pw as the literal text, which is what
			 * every other unrecognised wrapper does here. */
		}

		{
		hx_val result = hx_vm_run(&vm,
		    pw, pwlen,
		    salt, strlen(salt),
		    salt2, strlen(salt2),
		    pepper, strlen(pepper),
		    user, strlen(user));

		if (!prog->has_emit) {
			if (result.data && result.len > 0)
				fwrite(result.data, 1, result.len, stdout);
			putchar('\n');
		}
		}
		if (decoded) free(decoded);
	} else {
		/* read passwords from stdin, one per line */
		while (fgets(linebuf, linecap, stdin)) {
			int len = strlen(linebuf);
			const char *pw;
			int pwlen;
			hx_val result;

			/* strip trailing newline */
			while (len > 0 && (linebuf[len-1] == '\n' ||
			                   linebuf[len-1] == '\r'))
				len--;
			linebuf[len] = '\0';

			/* decode $HEX[ or $TESTVEC[ */
			pw = linebuf;
			pwlen = len;

			if (len > 5 && strncmp(linebuf, "$HEX[", 5) == 0) {
				/* decode in-place (output <= input) */
				int dlen = 0;
				const unsigned char *h =
				    (const unsigned char *)linebuf + 5;
				char *d = linebuf;
				while (*h) {
					int hi, lo;
					unsigned char c = *h;
					if ((c >= '0' && c <= '9'))      hi = c - '0';
					else if ((c >= 'a' && c <= 'f')) hi = c - 'a' + 10;
					else if ((c >= 'A' && c <= 'F')) hi = c - 'A' + 10;
					else break;
					h++;
					c = *h;
					if ((c >= '0' && c <= '9'))      lo = c - '0';
					else if ((c >= 'a' && c <= 'f')) lo = c - 'a' + 10;
					else if ((c >= 'A' && c <= 'F')) lo = c - 'A' + 10;
					else { d[dlen++] = hi << 4; break; }
					h++;
					d[dlen++] = (hi << 4) | lo;
				}
				pw = linebuf;
				pwlen = dlen;
			} else if (len > 9 &&
			           strncmp(linebuf, "$TESTVEC[", 9) == 0) {
				/* Not decoded in place: the $HEX[] branch above can
				 * reuse linebuf because hex output is half its input,
				 * but a vector EXPANDS and will not fit. */
				char *tv = NULL;
				int n = hx_expand_testvec(linebuf, len, &tv);
				if (n >= 0) {
					free(tvbuf);
					tvbuf = tv;
					pw = tvbuf;
					pwlen = n;
				}
			}

			result = hx_vm_run(&vm,
			    pw, pwlen,
			    salt, strlen(salt),
			    salt2, strlen(salt2),
			    pepper, strlen(pepper),
			    user, strlen(user));

			if (!prog->has_emit) {
				if (result.data && result.len > 0)
					fwrite(result.data, 1, result.len, stdout);
				putchar('\n');
			}
		}
	}

	free(linebuf);
	free(tvbuf);
	}

	hx_vm_free(&vm);
	hx_program_free(prog);
	return 0;
}

#endif /* HX_STANDALONE && !HX_NO_MAIN */
