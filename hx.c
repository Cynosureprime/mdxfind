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
#define HX_MAX_TESTVEC (2 * 1024 * 1024)  /* 2MB max decoded */
	int linecap = HX_MAX_TESTVEC * 2 + 64;  /* room for hex + prefix */
	char *linebuf = malloc(linecap);
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
			decoded = malloc(pwlen > HX_MAX_TESTVEC ? HX_MAX_TESTVEC : pwlen);
			dlen = 0;
			const unsigned char *h = (const unsigned char *)pw + 9;
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
				else { decoded[dlen++] = hi << 4; break; }
				h++;
				decoded[dlen++] = (hi << 4) | lo;
			}
			pw = decoded;
			pwlen = dlen;
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
				int dlen = 0;
				const unsigned char *h =
				    (const unsigned char *)linebuf + 9;
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
	}

	hx_vm_free(&vm);
	hx_program_free(prog);
	return 0;
}

#endif /* HX_STANDALONE && !HX_NO_MAIN */
