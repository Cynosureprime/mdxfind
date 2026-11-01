# Email addresses as plaintexts: `-E`, `-a`, and the ordering that makes them work

An email address is not an opaque string. It has a local part, an `@`, and a
domain, and almost everything you want to do to it applies to the local part
only. A rule that uppercases "the password" should not uppercase `@GMAIL.COM`;
a digit mask should append to `jsmith`, not to `jsmith@gmail.com`.

Two options handle this, and they solve different problems:

| option | problem | what it does |
|---|---|---|
| `-E` | an address is **structured** | applies `-r`, `-n`, `-N` and `-k` to the local part, then puts the domain back |
| `-a` | Gmail ignores `.` in the local part | sweeps every dotting of the dotless local part |
| `-k` | the input has local parts but no domains | appends each domain in a file, making the domain an axis |

They are independent. Most work uses `-E`. `-a` is for the specific case where
the leak may hold any one of an account's many dotted spellings, and `-k` for
the case where the domains are yours to choose rather than given to you.

## `-E`: rules and masks act on the local part

`-E` splits each word at the **first** `@`, applies everything to the text
before it, and rejoins the domain onto every candidate afterwards.

```sh
mdxfind -h '^MD5$' -f hashes.txt -E -n '?d?d' addresses.txt
```

With `fred@gmail.com` in the wordlist that produces `fred00@gmail.com` through
`fred99@gmail.com`, not `fred@gmail.com00`.

Four things follow from the split:

- **A word with no `@` is unaffected.** A mixed wordlist needs no separation;
  plain words behave exactly as they would without `-E`.
- **The domain is preserved byte for byte.** It is never case-folded,
  normalized or re-encoded, including under `-8`, where only the local part is
  converted to UTF-32. A domain that is not valid UTF-8 survives unchanged
  rather than causing the candidate to be dropped.
- **Rule length predicates measure the local part.** `>N`, `<N`, `_N` and `'N`
  all test the text before the `@`, not the whole address.
- **A mask cannot grow a domain.** `-n` appends to each wordlist candidate, so
  under `-E` the `@` has to arrive in the word. To supply the domain instead of
  reading it, use `-k`, below.

`procrule -E` means the same thing, so a candidate stream generated there and
piped to mdxfind carries the same semantics.

## `-a`: the dot sweep

Gmail and googlemail ignore `.` in the local part. `f.r.ed@gmail.com`,
`fred@gmail.com` and `fr.ed@gmail.com` are one account, and a leak may store
any of them. `-a` normalizes the local part by removing its dots, then
generates **every** dotting of what remains.

The cost is exponential in the dotless length:

| dotless local | candidates per address |
|---|---|
| 6 | 64 |
| 10 | 1,024 |
| 14 | 16,384 |
| 18 | 262,144 |
| 22 | 4,194,304 |
| 24 and above | 16,777,216 (capped) |

A local part of L characters costs 2^L, because a dot may follow each of the L
characters. The sweep is **capped at 24 positions**: a longer local part is
swept to 24 and the rest of its positions are not tried. mdxfind reports how
many words hit that cap at the end of the run, because a partial sweep
otherwise looks exactly like a complete one.

`-a` with `-E` is accepted, and `-E` adds nothing: `-a` already splits at the
first `@` and preserves the domain.

## `-k`: the domain axis

`-k FILE` appends each line of FILE to every candidate. In email work that does
two quite different jobs, and which one you get depends on whether the input
already carries domains.

**Input with domains: the key lands after the domain.** `-E` rejoins the domain
before the key is appended, so `fred@gmail.com` with the key `1` gives
`fred@gmail.com1`. That follows from the construction order below, and it is
seldom what you want on an address.

**Input without domains: the key supplies the domain.** This is the more useful
case. Feed a list with no `@` in it, either ordinary words or the stripped left
hand side of an address list, and give `-k` a file of domains each written with
its leading `@`:

```sh
printf '@gmail.com\n@hotmail.co.uk\n@proton.me\n' > domains.txt
mdxfind -h '^MD5$' -f hashes.txt -k domains.txt locals.txt
```

Every local part is tried against every domain, so `fred` becomes
`fred@gmail.com`, `fred@hotmail.co.uk` and `fred@proton.me`. That is how to
explore a set of new or previously unused domains, without materializing the
cross product on disk first.

`-E` is unnecessary here, and rules still behave. The key is appended after the
rule has run, and with no `@` in the word there is nothing for a rule to
overreach onto: rule `u` over `fred` with the key `@proton.me` gives
`FRED@proton.me`, not `FRED@PROTON.ME`. A mask lands in the same place, before
the domain, so a single pass covers local parts x rules x masks x domains:

```sh
mdxfind -h '^MD5$' -f hashes.txt -r email-local.rule -n '?d?d' -k domains.txt locals.txt
```

Size that before running it, because every axis multiplies: 7.1 million local
parts against 20 domains with `?d?d` is 28.4 billion candidates.

**The GPU serves key lists of 1 to 16.** A longer list still gives the right
answer, but the run falls to the CPU with no device dispatch at all. Split a
large domain list and loop, which keeps every pass on the device:

```sh
split -l 16 domains.txt dom-
for d in dom-*; do
  mdxfind -h '^MD5$' -f hashes.txt -k "$d" locals.txt
done
```

## The order candidates are built in

For one word, with everything active:

```
[prepend mask] rule(local part) [dots] [append mask] domain [key]
```

Verified identical on the CPU and both GPU backends. Two consequences are worth
knowing because they bound what a single pass can reach:

- **The dots are swept before the mask is appended**, so a dot never lands
  after mask-generated text. With `ab@gmail.com`, `-a -n '?d'` produces
  `a.b5@gmail.com` but never `ab5.@gmail.com`.
- **The domain is appended after the mask**, so mask characters land inside the
  address, not after it.

## Prepare the list: `emailprep`

`emailprep` ships with mdxfind and `make all` builds it. Run it over the
address list before an `-a` run. **This is a correctness-and-throughput
precondition, not a tuning step.**

```sh
emailprep -d gmail.com leak.txt gmail-ordered.txt
mdxfind -h '^MD5$' -f hashes.txt -a gmail-ordered.txt
```

It does three things, in an order that cannot be rearranged:

1. **Normalize**: strip `.` from the local part, since the dotless spelling is
   the seed `-a` sweeps back out.
2. **Deduplicate**, which must follow normalization, because that is where the
   collapse happens. In one measured 10.2-million-plaintext corpus, 27,179
   distinct spellings of a single local part normalize to one seed. Undeduped,
   each sweeps its own 2^23 candidates.
3. **Order by dotless local-part length, shortest first**, which must follow
   deduplication.

### Why the ordering is not optional

`-a` batches GPU work by the **dotless local-part length**. Every change of
that key dispatches the batch early, so a list ordered any other way fragments
into tiny dispatches. Measured on a GTX 1080, 4 million words of the same two
widths:

| input | batches | wall | busy |
|---|---|---|---|
| ordered by dotless length | **246** | 2.30 s | 0.31 s |
| the same widths interleaved | **2,000,719** | 130.06 s | 87.00 s |

Same work, **44x** apart, one batch per two words in the bad case.

**Sorting by whole-line length looks right and is not.** Once any local part
carries a dot, line length and dotless length stop agreeing. In one real
single-domain list sorted by line length, 21.6% of locals in a single
line-length band carried a dot, and the dotless widths inside that band were
12, 11, 10, 9 and 8 interleaved, so the batch key changed roughly every fourth
word.

mdxfind says so rather than leaving you to infer it from the rate. The first
dotted local part reaching the GPU, and the first *decrease* in local-part
length, each produce one warning naming `emailprep`. An increase is normal at a
length boundary; a decrease cannot happen in a correctly ordered list. The
end-of-run summary counts every split and how many decreased.

## What the masks should be

Mask shares below are measured over **7,656,085 real local parts**, classifying
whatever follows the leading lowercase-alpha run. **53.2% have no suffix at
all**, so for those the bare address is the candidate and no mask is needed.

**The digit tail, 23.0% of local parts in six masks:**

| mask | share |
|---|---|
| `?d?d` | 7.11% |
| `?d?d?d?d` | 6.17% |
| `?d?d?d` | 5.84% |
| `?d` | 2.35% |
| `?d?d?d?d?d` | 0.89% |
| `?d?d?d?d?d?d` | 0.68% |

**A separator then digits, 1.81% in three masks** (`john_99`, `john.42`):
`?s?d?d` 0.83%, `?s?d?d?d?d` 0.57%, `?s?d?d?d` 0.41%.

**Not reachable by a mask:** the `?s?l?l?l?l?l` family, a separator plus a
second word (`first_last`, `first-last`), is 3.01% of local parts across widths
2–5. That wants a second wordlist, not a keyspace.

## `-8` and quoted rule operands

`-8` reads rules in UTF-32 and is what admits the `rules32` quoted operand.
Without it a quoted operand is refused: `procrule` emits nothing and mdxfind
reports `Unknown rule command` and `bad line N`. **Pass `-8` to both tools or
to neither**, because a rule file is only shared between them at the same
setting.

`-8` does not override the operator reading. `rules32` takes the operator
reading first and falls back to the quoted one only where that fails, so a
quoted operand means a string only when its content is not itself a valid
command sequence:

| rule | byte engine (no `-8`) | with `-8` |
|---|---|---|
| `^"abc"` | refused | `abcjsmith@gmail.com` |
| `$"12"` | refused | `jsmith12@gmail.com` |
| `^"the"` | `224a534d495448@gmail.com` | `224a534d495448@gmail.com` |

`^"the"` is five commands (`^"` `t` `h` `e` `"`) and compiles either way, so
it keeps that reading at both settings: `h` is hex-encode and `t` is
toggle-all, giving `"jsmith` → `"JSMITH` → `224a534d495448`. That output is
literal text, not an encoding. `-x` leaves it untouched, where it does strip
the `$HEX[]` from a genuinely encoded result.

**So quoting is not a way to force a literal.** Spell an ambiguous prefix out
instead: `^e^h^t` prepends in reverse and yields `the` + word.

## Recipes

A rule file for local parts, byte-form throughout so it loads at either `-8`
setting:

```
# trailing digits -- 26.8% of local parts
$1
$1 $2
$1 $9 $9 $0
$2 $0 $2 $4
# separators -- 11.5% carry _, 1.9% carry -
@_
s_-
s_.
@.
# word prefix: ^a^b^c prepends in reverse, so this yields "the" + word
^e^h^t
# case -- zero hits on a lowercased corpus; keep for lists that preserve it
c
u
T0
```

**Sweep the common digit tails:**

```sh
for m in '?d?d' '?d?d?d?d' '?d?d?d' '?d' '?d?d?d?d?d' '?d?d?d?d?d?d'; do
  mdxfind -h '^MD5$' -f hashes.txt -E -n "$m" addresses.txt
done
```

**Mixed classes, where a pure digit mask cannot reach:**

```sh
mdxfind -h '^MD5$' -f hashes.txt -E -n '?u?d'              addresses.txt
mdxfind -h '^MD5$' -f hashes.txt -E -n '?d?d?l?d'          addresses.txt
mdxfind -h '^MD5$' -f hashes.txt -E -n '?[0-9]?[0-9a-f]'   addresses.txt
```

**Rules x mask, the dot product.** Each rule reshapes the local part, then the
mask extends it. `mary_jones@hotmail.co.uk` with `@_` then `?d?d` gives
`maryjones88@hotmail.co.uk` in one pass:

```sh
mdxfind -h '^MD5$' -f hashes.txt -E -r email-local.rule -n '?d?d' addresses.txt
```

**Which rules earned their keep** (`-Z` reports per-rule hit counts):

```sh
mdxfind -h '^MD5$' -f hashes.txt -E -r email-local.rule -Z addresses.txt
```

**Generate the candidates instead of hashing them.** `procrule` never emits the
source word, so add it back:

```sh
{ cat addresses.txt; procrule -8 -E -r email-local.rule addresses.txt; } > cands.txt
```

**Measure a rule file against real local parts before spending GPU time.**
`procrule` on bare local parts is equivalent to `-E` on full addresses for
counting purposes:

```sh
procrule -8 -r email-local.rule -m real.user -s stats.txt bases.txt
```

## Sizing, and where to run it

A width-4 digit mask over 7.1 million bases is 71 billion candidates. That is a
GPU workload.

For `-a`, one measured pairing: a GeForce GTX 1080 sustains about 1.0x10^9
candidates/s, against about 5.8x10^7 for the same work on an AMD Ryzen 7 1800X
using all 16 threads. That is roughly 17x, and the ratio is flat across
local-part widths from 6 upward, so there is no width at which that CPU is the
better engine. Both figures are one host and one card; treat the ratio as the
transferable part, not the absolute numbers.

What costs you throughput is the batch key churning, which is what `emailprep`
ordering prevents. Local parts narrower than 5 characters stay on the CPU,
where the sweep is too small to be worth a dispatch at all.

`-G none` gives CPU-exact output and is the oracle when a GPU result looks
wrong.

## Rules of thumb

- Use `-E` for anything structural; it is almost always what you want.
- No domains in the input? Make them an axis with `-k`, one `@domain` per
  line, in groups of 16 to stay on the GPU.
- Run `emailprep` before `-a`. If mdxfind warns about a dotted local part or a
  decreasing length, the list is not ordered the way `-a` batches it.
- 53.2% of local parts have no suffix, so try the bare addresses first.
- `?d?d` and `?d?d?d?d` are the two best single masks; six digit masks cover
  23.0%.
- Keep rule operands in byte form unless you need `-8`, and if you need `-8`,
  pass it to `procrule` and `mdxfind` alike.
- Measure a rule file with `procrule -s` before committing GPU time to it.
