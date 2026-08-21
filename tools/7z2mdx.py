#!/usr/bin/env python3
#
# $Revision: $
#
# $Log: $
#
"""7z2mdx.py -- convert 7z2john output into mdxfind's 7ZIP (-m e1000) form.

Stock 7z2john emits the ENTIRE encrypted stream in the final field: 112 KB for
a small archive, 1.5 MB for the CMIYC challenge_2 set. mdxfind's MAXLINE is
40 KB, so such a line can never be read.

mdxfind's tier-0 check needs only the last TWO ciphertext blocks (32 bytes) --
it decrypts the final block in CBC using the preceding block as its IV and
requires the trailing `packedlen - unpackedlen` plaintext bytes to be zero.

But tier 2 (decompress + CRC32) needs the WHOLE stream, and that is what turns
a pad-check hit into a confirmed crack. So this only truncates a line that
would otherwise exceed mdxfind's MAXLINE, and warns when it does. A record
that fits is passed through untouched.

That truncation is why mdxfind can attack Deflate64 archives at all: it never
decompresses, which is exactly where hashcat and john silently fail.

Usage:
    7z2john.pl archive.7z | 7z2mdx.py > archive.mdx
    7z2mdx.py < stock.hash > truncated.hash
    7z2mdx.py file1.hash file2.hash > combined.hash

Leading "filename:" from john is stripped automatically.
Confirm a stage-1 hit out of band:  7zz t -p'<password>' archive.7z
"""
import sys

# mdxfind reads hash lines with gzgets(line, MAXLINE); anything longer arrives
# truncated, and its trailing bytes are then NOT the end of the AES stream.
MAXLINE = 40 * 1024


def convert(line):
    line = line.strip()
    if not line:
        return None
    # john prefixes "filename:"; mdxfind and hashcat both need it gone
    if not line.startswith('$7z$') and ':$7z$' in line:
        line = line[line.index(':$7z$') + 1:]
    if not line.startswith('$7z$'):
        return None
    f = line.split('$')
    # ['', '7z', type, log2, saltlen, salt, ivlen, iv, crc, packed, unpacked, data]
    if len(f) < 12:
        sys.stderr.write('skip: only %d fields: %.60s...\n' % (len(f), line))
        return None
    data = f[11]
    try:
        pad = int(f[9]) - int(f[10])
    except ValueError:
        sys.stderr.write('skip: unparsable packed/unpacked\n')
        return None
    if pad <= 0:
        # padsize 0 means no zero-padding exists to test, so stage 1 cannot
        # decide this archive. Emit it anyway -- mdxfind will refuse to report
        # it rather than produce a meaningless hit -- but say so loudly.
        sys.stderr.write('WARNING: padsize %d: stage 1 cannot verify this '
                         'archive; it needs full decrypt+CRC.\n' % pad)
    # Truncating destroys tier-2 verification (decompress + CRC32), which is
    # the only thing that can turn a pad-check hit into a confirmed crack. So
    # only truncate when the record genuinely will not fit, and say so.
    line_out = '$'.join(f)
    if len(line_out) < MAXLINE - 64:
        return line_out                # fits whole: keep every byte of evidence

    if len(data) < 64:
        sys.stderr.write(
            'skip: line needs truncating but its data field is already under '
            '32 bytes\n')
        return None
    if pad <= 0:
        sys.stderr.write(
            'skip: padsize %d and the line needs truncating to fit mdxfind\'s '
            '%d-byte limit; nothing would be left to verify with.\n'
            % (pad, MAXLINE))
        return None
    sys.stderr.write(
        'NOTE: truncating a %d-byte line to the final two blocks. This drops '
        'decompress+CRC verification, leaving %d bits of pad check; expect ~1 '
        'false hit per 2^%d candidates.\n' % (len(line_out), 8 * pad, 8 * pad))
    f[11] = data[-64:]                 # keep only the final two blocks
    return '$'.join(f)


def main(argv):
    srcs = argv[1:]
    streams = [open(a) for a in srcs] if srcs else [sys.stdin]
    n = 0
    for st in streams:
        for line in st:
            out = convert(line)
            if out:
                print(out)
                n += 1
    sys.stderr.write('%d archive(s) converted\n' % n)
    return 0 if n else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
