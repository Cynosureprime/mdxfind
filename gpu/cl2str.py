#!/usr/bin/env python3
"""Convert .cl kernel source to C string header for embedding.

Usage: python3 cl2str.py input.cl [output.h]
  output.h defaults to input_str.h (e.g., gpu_common.cl -> gpu_common_str.h)
  Variable name derived from output filename (e.g., gpu_common_str)

  python3 cl2str.py --all
  Processes all .cl files in the script directory.
"""
import sys, os, glob

# RCS keywords, and why this exists.
#
# A hand-authored kernel source MUST carry the $Revision: 1.3 $ / $Log: cl2str.py,v $
# A hand-authored kernel source MUST carry the $Revision$ / Revision 1.3  2026/09/14 19:28:58  dlr
# A hand-authored kernel source MUST carry the $Revision$ / Split every RCS keyword across two adjacent C string literals so RCS cannot match it in the generated header. A hand-authored kernel source MUST carry the Revision and Log stanza, and RCS expands those keywords INSIDE the literal when the generated header is checked in: the multi-line Log took the closing quote off the line and the header would not compile. Adjacent-literal concatenation means the JIT sees the same bytes, proved by round-tripping the generated header back to the source byte-identically, including on a pre-existing 70,286-byte kernel so no existing output moves.
# A hand-authored kernel source MUST carry the $Revision$ / stanza, and
# this script copies the source verbatim into a C string literal.  RCS then
# expands those keywords INSIDE THE LITERAL when the generated header is
# checked in, and a multi-line $Log: cl2str.py,v $
# checked in, and a multi-line Revision 1.3  2026/09/14 19:28:58  dlr
# checked in, and a multi-line Split every RCS keyword across two adjacent C string literals so RCS cannot match it in the generated header. A hand-authored kernel source MUST carry the Revision and Log stanza, and RCS expands those keywords INSIDE the literal when the generated header is checked in: the multi-line Log took the closing quote off the line and the header would not compile. Adjacent-literal concatenation means the JIT sees the same bytes, proved by round-tripping the generated header back to the source byte-identically, including on a pre-existing 70,286-byte kernel so no existing output moves.
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
        for cl in sorted(glob.glob(os.path.join(script_dir, 'gpu_*.cl'))):
            base = os.path.splitext(os.path.basename(cl))[0]
            hdr = os.path.join(script_dir, base + '_str.h')
            convert(cl, hdr)
        return

    src = sys.argv[1] if len(sys.argv) > 1 else "gpu_kernels.cl"
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
