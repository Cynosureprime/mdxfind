#!/usr/bin/env python3
"""Convert .metal kernel source to C string header for embedding.

$Revision: 1.1 $
$Log: metal2str.py,v $
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
