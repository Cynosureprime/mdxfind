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
tools/hx8_to_c: tools/hx8_to_c.c hx_vm.h hx.c hx_ast.c hx_compile.c hx_vm.c \
                hx_func.c hx.tab.c hx.lex.c myprogress.c
	cc -DHX_STANDALONE -DHX_NO_MAIN -O3 -I. -I/opt/local/include \
	    -o tools/hx8_to_c \
	    tools/hx8_to_c.c hx.c hx_ast.c hx_compile.c hx_vm.c hx_func.c \
	    hx.tab.c hx.lex.c myprogress.c \
	    -L/opt/local/lib -lssl -lcrypto /opt/local/lib/libiconv.a

# Regenerate codegen/hx_specs_data.c from hx.8 via the build-time tool.
# Path to hx.8 is via HX8_PATH; defaults to ~/Documents/troff/mdxfind/hx.8
# on the iMac (matches user's troff source tree layout).
HX8_PATH ?= $(HOME)/Documents/troff/mdxfind/hx.8

codegen/hx_specs_data.c: tools/hx8_to_c $(HX8_PATH)
	./tools/hx8_to_c $(HX8_PATH) > codegen/hx_specs_data.c.tmp \
	    && mv codegen/hx_specs_data.c.tmp codegen/hx_specs_data.c
