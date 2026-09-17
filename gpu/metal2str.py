#!/usr/bin/env python3
"""Convert .metal kernel source to C string header for embedding.

$Revision: 1.2 $
$Log: metal2str.py,v $
Revision 1.2  2026/09/14 19:28:58  dlr
Same keyword split as cl2str.py, for the same reason and verified the same way: the generated header round-trips to the source byte-identically.

Revision 1.1  2026/05/12 13:35:01  dlr
Initial check-in: Phase 1 Metal port codegen. Sibling of gpu/cl2str.py; operates on .metal sources, emits C string headers for the JIT path. Default mode reads single .metal -> _str.h; --all batch-processes every metal_*.metal in the script dir. Mirrors cl2str.py shape exactly so future Metal codegen changes can fold in symmetrically.


Usage: python3 metal2str.py input.metal [output.h]
  output.h defaults to input_str.h (e.g., metal_common.metal -> metal_common_str.h)
  Variable name derived from output filename (e.g., metal_common_str)

  python3 metal2str.py --all
  Processes all metal_*.metal files in the script directory.

Sibling of gpu/cl2str.py; identical behavior, but operates on .metal source
files. The C string header it emits is consumed by the gpu_metal.m JIT
fallback path (MDXFIND_METAL_JIT=1) which concatenates _str.h contents into
a single NSString for [device newLibraryWithSource:options:error:]. The
metallib (default) path bypasses this and uses gpu/mdxfind_metallib.h.

Both paths embed identical Metal source; the only difference is whether the
Metal driver JITs at process start (JIT path) or loads pre-compiled AIR
linked into a metallib (offline path).
"""
import sys, os, glob

# RCS keywords, and why this exists.
#
# A hand-authored kernel source MUST carry the $Revision: 1.2 $ / $Log: metal2str.py,v $
# A hand-authored kernel source MUST carry the $Revision$ / Revision 1.2  2026/09/14 19:28:58  dlr
# A hand-authored kernel source MUST carry the $Revision$ / Same keyword split as cl2str.py, for the same reason and verified the same way: the generated header round-trips to the source byte-identically.
# A hand-authored kernel source MUST carry the $Revision$ / stanza, and
# this script copies the source verbatim into a C string literal.  RCS then
# expands those keywords INSIDE THE LITERAL when the generated header is
# checked in, and a multi-line $Log: metal2str.py,v $
# checked in, and a multi-line Revision 1.2  2026/09/14 19:28:58  dlr
# checked in, and a multi-line Same keyword split as cl2str.py, for the same reason and verified the same way: the generated header round-trips to the source byte-identically.
# checked in, and a multi-line expansion drops the closing quote off the
# end of the line -- so the header stops compiling with
# "expected ';' after top level declarator".
#
# The documented defence is to check the generated headers in with `-ko`.  That
# is still the rule, but it is a rule a human has to remember every time, and
# it failed on gpu_md5_rules32_str.h at revision 1.1 on 2026-09-14.
#
# This is the belt.  Each keyword is split across two adjacent C string
# literals: the C preprocessor concatenates them, so the JIT sees byte-for-byte
# the same kernel source it saw before -- while the pattern RCS matches,
# `` and friends, never appears contiguously in the generated file.  RCS
# then has nothing to expand whatever mode the file is in.
RCS_KEYWORDS = ('Author', 'Date', 'Header', 'Id', 'Locker', 'Log', 'Name',
                'RCSfile', 'Revision', 'Source', 'State')

def defang_rcs(escaped):
    """Split every RCS keyword so RCS cannot match it in the generated file.

    Operates on the ALREADY-ESCAPED text and emits `$Lo" "g` -- a literal
    break, not an escape -- because an escape would change what the JIT reads.
    Adjacent-literal concatenation makes the kernel source identical.
    """
    for kw in RCS_KEYWORDS:
        # Only the `$Keyword` form matters; RCS requires the leading `$`.
        escaped = escaped.replace('$' + kw, '$' + kw[:-1] + '" "' + kw[-1])
    return escaped


def convert(src, dst):
    with open(src, 'r') as f:
        lines = f.readlines()

    varname = os.path.basename(dst).replace('.h', '').replace('-', '_')

    with open(dst, 'w') as out:
        out.write("/* Auto-generated from %s -- do not edit */\n" % os.path.basename(src))
        out.write("static const char %s[] =\n" % varname)
        for line in lines:
            line = line.rstrip('\n')
            escaped = line.replace('\\', '\\\\').replace('"', '\\"')
            escaped = defang_rcs(escaped)
            out.write('    "%s\\n"\n' % escaped)
        out.write(";\n")

    print("%s -> %s (%d lines)" % (os.path.basename(src), os.path.basename(dst), len(lines)))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        for ml in sorted(glob.glob(os.path.join(script_dir, 'metal_*.metal'))):
            base = os.path.splitext(os.path.basename(ml))[0]
            hdr = os.path.join(script_dir, base + '_str.h')
            convert(ml, hdr)
        return

    src = sys.argv[1] if len(sys.argv) > 1 else "metal_common.metal"
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(src))[0]
        dst = base + '_str.h'

    if not os.path.isabs(src):
        src = os.path.join(script_dir, src)
    if not os.path.isabs(dst):
        dst = os.path.join(script_dir, dst)

    convert(src, dst)

if __name__ == '__main__':
    main()
