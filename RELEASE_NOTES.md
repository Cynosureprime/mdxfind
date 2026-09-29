# mdxfind v1.608: seven new hash types, and AMD GPUs that never initialised

Source: mdxfind.c 1.606 -> 1.608.

## New hash types (e1047 - e1053)

| index | name | construction |
|---|---|---|
| e1047 | `MD5-1xMD5pMD5SHA1p` | `md5(md5(pass) . md5(sha1(pass)))` |
| e1048 | `MD5-1xMD5pMD5SHA256p` | `md5(md5(pass) . md5(sha256(pass)))` |
| e1049 | `MD5-1xMD5pMD5SHA512p` | `md5(md5(pass) . md5(sha512(pass)))` |
| e1050 | `MD5SHA1revMD5` | `md5(sha1(rev(md5(pass))))` |
| e1051 | `MD5MD5RAWMD5PASS` | `md5(md5_bin(md5(pass) . pass))` |
| e1052 | `MD5MD5RAWMD5` | `md5(md5_bin(md5(pass)))` |
| e1053 | `MD5SHA1SHA1MD5MD5` | `md5(sha1(sha1(md5(md5(pass)))))` |

All seven are unsalted, catalogued in `hx.8`, and cleared the `hx_dedup_check`
gate before a number was assigned. Every vector was supplied externally and
reproduced with an independent implementation rather than by this code.

`e1051` and `e1052` consume the inner digest as hex and feed the outer `md5` the
raw sixteen bytes, which is what separates them from the hex-chained forms
already present. `e1050` reverses the thirty-two hex characters of the inner
digest, not its bytes.

`bench_rates.h` gains measured throughputs for e1030 through e1046. The seven
types above do not yet have one, so `-L` cannot decline them; that is the
permissive direction, and a rate can only make them harder to select.

## AMD GPUs that hung at initialisation now work

On AMD fglrx 1573.4, `clCreateCommandQueue` hung forever and no GPU work ever
started. The cause was not in the OpenCL path at all.

A `static __thread` array does not place a buffer on the thread stack. glibc
carves the static TLS block out of the same allocation and copies it into every
thread, including threads a `dlopen`ed driver creates for itself. mdxfind carried
6,881,648 bytes of `.tbss`, and the driver's helper thread could not start above
a threshold bisected to between 131,072 and 262,144 bytes.

The large rule scratch buffers now hold thread-local pointers allocated once per
thread. `.tbss` falls from 6,881,648 bytes to 41,256 on the Linux GPU build, and
a card that hung on every earlier build now initialises in two seconds and
cracks normally.

Rule processing is unchanged: `procrule -8` against `mdxfind -8` gives 10,946 of
10,947 before and after, the single difference being the known Turkish dotless-i
locale variant.
