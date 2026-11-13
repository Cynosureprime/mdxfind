# Compressed Input

mdxfind reads compressed wordlists directly, with no need to decompress them
first. This page covers which formats work, where they work, and why a
compressed stream on a pipe is refused rather than read.

## What works, and where

The format is detected from the first bytes of the input (its magic number),
never from the file name or extension.

| Format | Typical extension | Wordlist named on the command line | Pipe or `stdin` | Hash file (`-f`, `-F`, `-J`, or hashes on stdin) |
|--------|------|:---:|:---:|:---:|
| plain text | — | yes | yes | yes |
| gzip | `.gz` | yes | yes | yes |
| xz | `.xz` | yes | **FATAL** | **FATAL** |
| bzip2 | `.bz2` | yes | **FATAL** | **FATAL** |
| Zstandard | `.zst` | yes | **FATAL** | **FATAL** |
| ZIP (first entry) | `.zip` | yes | **FATAL** | **FATAL** |
| 7z (one file, LZMA2) | `.7z` | yes | **FATAL** | **FATAL** |
| LZMA "alone" | `.lzma` | yes | **FATAL** | **FATAL** |

**FATAL** means mdxfind stops at once with exit status 1 and a message naming
the format and the command that will work, for example:

```
FATAL: mdxmain.c:755 xz-compressed input on a pipe, on stdin, or as a hash file (-f, -F, -J).
Only gzip is decompressed there. Name a compressed wordlist on the command line, or
decompress it first: xz -dc list.xz | mdxfind ...
```


## How to use it

Name the compressed wordlist as an argument. Any mix of formats can be given in
one run:

```bash
mdxfind -m e1 -f hashes.txt rockyou.txt.gz
mdxfind -m e1 -f hashes.txt dict1.txt.zst dict2.txt.bz2 big.found.7z plain.txt
```

To use a format that cannot be read from a pipe, decompress it into the pipe
and give `stdin` as the wordlist. mdxfind then reads plain text:

```bash
xz    -dc list.xz   | mdxfind -m e1 -f hashes.txt stdin
bzip2 -dc list.bz2  | mdxfind -m e1 -f hashes.txt stdin
zstd  -dc list.zst  | mdxfind -m e1 -f hashes.txt stdin
unzip -p  list.zip  | mdxfind -m e1 -f hashes.txt stdin
7zz x -so list.7z   | mdxfind -m e1 -f hashes.txt stdin
xz    -dc list.lzma | mdxfind -m e1 -f hashes.txt stdin
```

Hash files are the same: gzip is read directly; for any other format,
decompress first.

```bash
mdxfind -m e1 -f hashes.txt.gz wordlist.txt         # works
xz -dc hashes.txt.xz | mdxfind -m e1 wordlist.txt   # hashes on stdin, decompressed first
```

A named wordlist is preferable to a pipe where both are possible: the
background line counter (`-W auto`, or `MDXFIND_CACHE`) can count a named file
and cache the count for the progress ETA, but cannot count a pipe.

## Why a compressed pipe is refused

Two reasons, one about how the decoders work and one about what happens if
they are bypassed.

**How the decoders work.** gzip is decoded by zlib's `gzread`, which
recognises a gzip stream as it reads and decodes it on the fly from any source,
a pipe included. Every other format is detected after the first read and then
handled by its own library decoder, which **reopens the file by name** and
starts again from its first byte. A pipe has no name to reopen, and the bytes
already read cannot be read again. A hash file has a name, but its reader does
not record it, and some hash types read their file line by line without the
decoders at all, so a hash file is in the same position as a pipe. 7z adds a further
requirement: the archive's directory, which holds the dictionary size, the
decoded size and the CRC, is at the **end** of the file, and reaching it needs
a seek that a pipe cannot do.

**What happens if they are bypassed.** Without a decoder, the input is read as
plain text. Each "line" of the compressed stream is then hashed as a candidate
password: the run completes, exits 0, and finds nothing, which looks exactly
like a wordlist that held no matches. A hash file in the same position loads
no hashes. That is a wrong result that looks like a clean one, so mdxfind
refuses the input instead.

The check looks for the signature of each supported format at the start of the
input. For bzip2 it requires the block or end-of-stream magic after `BZh1` to
`BZh9`, so a plain-text wordlist or hash file whose first word begins with
`BZh` is read as plain text, on a pipe or as a file.

## Format details

**gzip.** Read anywhere, including concatenated gzip members.

**xz.** Read by liblzma's stream decoder, including concatenated streams.

**LZMA (`.lzma`).** The format that preceded xz, written by `xz --format=lzma`,
`lzma` and the LZMA SDK. It has no magic number: its 13-byte header is a
properties byte, a dictionary size and the decoded size (or all ones when the
size is unknown). mdxfind recognises it from those values, by the same rules
liblzma's own format detection uses, and additionally requires a dictionary
of at least 4 KB. A plain-text file cannot be mistaken for one, because a
valid header must contain zero bytes and text does not; a sweep of the
2,758,408 line starts of a real wordlist found no false match. The stream is
read with liblzma's `.lzma` decoder. A decode error, or a file that ends
before the stream does, is FATAL. Concatenated `.lzma` streams are not part
of the format, and only the first is read.

**bzip2.** Read by libbz2's streaming decoder.

**Zstandard.** Read by libzstd's streaming decoder.

**ZIP.** Only the **first entry** of the archive is read, using the method it
was stored with: stored, deflate, bzip2, LZMA or Zstandard. zip64 entries over
4 GB are read whole. For a multi-entry archive, extract the entries or pipe one
with `unzip -p list.zip entry.txt`. A deflate error, or deflate input that ends
early, is FATAL.

**7z.** Read when the archive holds **one file, in one block, compressed with
LZMA2 alone**, which is how large wordlists are distributed (the hashmob.net
found lists, for example). The block is decoded with liblzma's raw LZMA2
decoder in constant memory, at most the archive's dictionary (64 MB for
`LZMA2:26`), however large the wordlist. The decoded size and CRC32 recorded in
the archive are checked when the stream ends. Any other 7z is FATAL with the
reason: several files or blocks, a filter such as BCJ or Delta, encryption, a
method other than LZMA2 (LZMA, PPMd, BZip2, Deflate), or a compressed directory
(`-mhc`/`-mhe`). Use `7zz x -so list.7z | mdxfind ... stdin` for those.

Because the decoder streams, a CRC mismatch is found only when the whole stream
has been read, after every line has been hashed. The run then stops with exit
status 1. Treat its output as coming from a damaged wordlist.

This is unrelated to hash type `e1000`, which recovers the *password* of an
encrypted 7z archive; see [SEVENZIP.md](SEVENZIP.md).

**Speed.** Decompression runs on one thread, in the reader, so on a cheap hash
the decoder can become the limit. From slowest to fastest to decode: bzip2,
then xz, `.lzma` and 7z (all LZMA), then gzip, then Zstandard. On a 140 MB wordlist,
single-threaded decoding took about 5 s for bzip2, 2.4 s for LZMA2, 1 s for gzip
and 0.25 s for Zstandard. Where a wordlist is used repeatedly with cheap hashes,
Zstandard is the best choice of format.
