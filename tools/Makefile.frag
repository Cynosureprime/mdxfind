# tools/Makefile.frag -- build-time tool object/link rules.
#
# Sub-phase 2a.2 (per project_hx_codegen_phase2_3_spec_2026-05-21.md
# D12.2.c Path A): tools/hx8_to_c is a build-time converter that
# reads ~/Documents/troff/mdxfind/hx.8 and emits codegen/hx_specs_data.c,
# a C-literal serialization of pre-compiled hx_program structs for
# every algorithm row.
#
# The tool links against the existing hx library ($(HX_OBJS), already
# defined in the top-level iMac Makefile). It does NOT ship with
# mdxfind; it runs once at `make` time on the build host.
#
# The generated codegen/hx_specs_data.c is itself a build artifact and
# MUST NOT be checked into RCS. Per the per-host Makefile design, on
# non-iMac hosts the file is rsync'd from the iMac (where the tool ran)
# rather than regenerated locally (which would require hx.8 + flex/bison
# regenerable hx.l/hx.y on the target host).
#
# 1.1
# Makefile.frag,v
# Revision 1.1  2026/05/21 23:22:54  dlr
# sub-phase 2a.2: tools Makefile fragment. Builds tools/hx8_to_c by recompiling the hx sources in HX_STANDALONE mode (with HX_NO_MAIN to gate out hx.c main). Generates codegen/hx_specs_data.c via the tool reading HX8_PATH (default ~/Documents/troff/mdxfind/hx.8). On non-iMac hosts the generated .c is rsync as a build artifact rather than regenerated locally (which would require flex/bison-regenerable hx.l/hx.y on the target).
#

# The tool compiles its sources in -DHX_STANDALONE mode (mirroring the
# existing `hx` standalone-CLI target), which gates out the heavy
# hashpipe-only crypt entries (bcrypt, scrypt, yescrypt, sm3crypt,
# gost12_512crypt etc.) that would otherwise drag in libstreebog,
# libyescrypt, libgost12, etc. Algorithms not in the standalone set
# compile-fail and route to hx_override_table[] -- exactly the same
# outcome as outliers. Phase 5 expansion can lift HX_STANDALONE and
# link hashpipe.o for full coverage.
#
# Recompiled from source (not linked against $(HX_OBJS)) because the
# project-wide $(HX_OBJS) lacks -DHX_STANDALONE and so omits the
# md5/sha/etc. entries the standalone build gates IN. Mirrors the
# existing `hx:` target structure.
tools/hx8_to_c: tools/hx8_to_c.c tools/hx_program_cmp.h hx_vm.h hx.c hx_ast.c \
                hx_compile.c hx_vm.c hx_func.c hx.tab.c hx.lex.c myprogress.c
	cc -DHX_STANDALONE -DHX_NO_MAIN -O3 -I. -I/opt/local/include \
	    -o tools/hx8_to_c \
	    tools/hx8_to_c.c hx.c hx_ast.c hx_compile.c hx_vm.c hx_func.c \
	    hx.tab.c hx.lex.c myprogress.c \
	    -L/opt/local/lib -lssl -lcrypto /opt/local/lib/libiconv.a

# codegen/maphashcat_data.h -- AUTO-GENERATED build artifact (Tier 2
# Feature 2). Extracts mdxfind.c's Maphashcat[] (hashcat-mode, mdxfind-eN)
# table into a tiny pure-data header so tools/hx_dedup_check can answer
# --hashcat-mode=N (and the eN->mode reverse annotation) WITHOUT linking
# mdxfind.c (which would drag the whole binary). Regenerated whenever
# mdxfind.c or the extractor changes. NOT checked into RCS.
codegen/maphashcat_data.h: mdxfind.c tools/gen_maphashcat.awk
	awk -f tools/gen_maphashcat.awk mdxfind.c > codegen/maphashcat_data.h.tmp \
	    && mv codegen/maphashcat_data.h.tmp codegen/maphashcat_data.h

# tools/hx_dedup_check -- standalone CLI that catches "this proposed hx
# expression already exists in the catalog under a different name." Per
# architect spec project_hx_dedup_check_spec_2026-05-26.md
# (Tier 1 + Tier 2 + Tier 3).
#
# Same link surface as tools/hx8_to_c (HX_STANDALONE + prescan stubs),
# plus codegen/hx_specs_data.o for the catalog the tool compares against.
# codegen/hx_specs_data.c is built earlier in the pipeline by tools/hx8_to_c
# itself; we don't list it as a prerequisite here because regenerating it
# every time the dedup tool is rebuilt would needlessly run hx8_to_c again.
#
# Tier 2 deps: tools/hx_program_cmp.h (shared comparator + Layer 2 canon)
# and codegen/maphashcat_data.h (the --hashcat-mode=N table).
#
# Tier 3 (Layer 3 test-vector equivalence) link-surface decision:
#   The architect spec (R1, §2.2) assumed Layer 3 would link hashpipe.o
#   and call hx_register_hashpipe_types() for "real" hash resolution. That
#   is NOT done here, for two reasons:
#     1. hx_register_hashpipe_types() is `static` in hashpipe.c, and
#        hashpipe.c carries its own unconditional main() -- linking it
#        into this leaf tool is impossible without editing hashpipe.c,
#        which the spec (§4) forbids in ALL tiers.
#     2. It is UNNECESSARY. The HX_STANDALONE build ALREADY links real
#        OpenSSL/iconv implementations of md5, md4, sha1, sha256, sha512,
#        hmac_sha1/256, the pbkdf2 family, siphash, murmur3, plus every
#        string transform (utf16le, zext16, hex, base64, xor, ...). Those
#        cover the Layer-3 G9 gate, the NTLM/NTLMH-class non-ASCII
#        distinction, and the broad md5/sha-family residual.
#   Consequence: the link line below is UNCHANGED from Tier 1/2 -- no
#   hashpipe.o, no new static libs (-lsph, -lmhash, etc.). Layer 3 runs
#   the real hx VM (hx_vm_run) but ONLY for expressions whose every callee
#   is a real (non-stub) function; long-tail crypt functions (bcrypt,
#   gost*, hmac_sha384, ...) that exist only as prescan stubs report
#   "Layer 3 UNAVAILABLE" rather than risk a false-equal verdict on empty
#   stub output. Lifting to full hashpipe coverage would require either
#   de-static-ing + main-gating hashpipe.c (out of scope) or a dedicated
#   hashpipe_hx_register.o TU; deferred until a real algorithm outside the
#   standalone set needs Layer-3 verification.
tools/hx_dedup_check: tools/hx_dedup_check.c tools/hx_program_cmp.h hx_vm.h \
                hx.c hx_ast.c hx_compile.c hx_vm.c hx_func.c hx.tab.c \
                hx.lex.c myprogress.c codegen/hx_specs_data.c \
                codegen/hx_spec_entry.h codegen/maphashcat_data.h
	cc -DHX_STANDALONE -DHX_NO_MAIN -O3 -I. -I/opt/local/include \
	    -o tools/hx_dedup_check \
	    tools/hx_dedup_check.c hx.c hx_ast.c hx_compile.c hx_vm.c hx_func.c \
	    hx.tab.c hx.lex.c myprogress.c codegen/hx_specs_data.c \
	    -L/opt/local/lib -lssl -lcrypto /opt/local/lib/libiconv.a

# Regenerate codegen/hx_specs_data.c from hx.8 via the build-time tool.
# Path to hx.8 is via HX8_PATH; defaults to ~/Documents/troff/mdxfind/hx.8
# on the iMac (matches user's troff source tree layout).
HX8_PATH ?= $(HOME)/Documents/troff/mdxfind/hx.8

codegen/hx_specs_data.c: tools/hx8_to_c $(HX8_PATH)
	./tools/hx8_to_c $(HX8_PATH) > codegen/hx_specs_data.c.tmp \
	    && mv codegen/hx_specs_data.c.tmp codegen/hx_specs_data.c
