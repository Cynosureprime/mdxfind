# Salts and Usernames: how they reach mdxfind, and how often they are tried

A salted or user-keyed hash needs two things: the digest, and the per-hash value
that went into it. mdxfind will take that value from three different places, and
they do not behave the same way. Choosing the wrong one produces a run that
loads cleanly, reports a sensible-looking receipt, and finds nothing.

## The three ways in

| how | salts | usernames | reference count |
|---|---|---|---|
| alongside the hash | `hash:salt` via `-F` | `hash:user` via `-F` | **byte-exact** |
| a per-type list | `-S file` | `-U file` | counted (see below) |
| a global pool | `-s file` | `-u file` | effectively unlimited |

The per-type forms follow a preceding `-M`, and apply only to the types that
`-M` selected. The global forms apply to every selected type that can use them.
`-M` must come **before** `-S`, `-U`, `-F` and `-J` on the command line, because
the file is consumed at the moment the option is reached.

## Reference counts, and why a run can quietly under-report

Every salt and every userid carries a count. Each time it produces a hit the
count goes down, and at zero mdxfind stops trying it. That is what stops a
finished salt from being rehashed against every remaining candidate for the rest
of the run, and on an expensive type it is the difference between a run that
speeds up as it cracks and one that does not.

The count therefore has to be at least the number of hits that value will
produce. Where it comes from depends on how the value arrived:

- **Alongside the hash** it is exact, because the value is filed once per hash
  line. Nothing to think about.
- **From `-U`** it is the number of times the userid appears in the file. One
  occurrence, one hit.
- **From `-u` and `-s`** it is effectively unlimited, so the value never retires.

Measured, three hashes that all share one value:

```
                                       recovered
hash:salt via -F  (byte-exact)            3 of 3
-s  global pool                           3 of 3
-S  per-type list                         1 of 3
```

```
                                       recovered
hash:user via -F  (byte-exact)            3 of 3
-u  global pool                           3 of 3
-U  userid listed once                    1 of 3
-U  userid listed three times             3 of 3
```

**Under-counting loses hits silently.** There is no warning: the run completes,
the receipt is correct, and the missing hashes look exactly like passwords that
were not in the wordlist.

**Over-counting is harmless.** It only delays retirement. If you are unsure,
supply more.

### How many occurrences a userid needs in a `-U` file

Not one per hash — one per **hit**, and a hash can hit more than once. With `-i N`
each hash is tested at every depth from 1 to N, and each depth that matches is a
separate hit that spends one from the count.

```
occurrences = hashes for that userid  x  depths searched
```

Measured on an iterated user-keyed type with 110 hashes across 10 userids under
`-i 5`: listing each userid 11 times recovered 510 of 550, and listing each 55
times recovered all 550.

If that arithmetic is inconvenient, put the username next to its hash and use
`-F` instead, where the count is exact by construction and no arithmetic is
needed.

### `-S` does not count

`-S` deduplicates its file, so repeating a salt does not raise its count:
listing a shared salt once and listing it three times both recover 1 of 3. For a
salt used by more than one hash, supply `hash:salt` through `-F`.

## Which channel a hash file needs

Look at the hash text, not the algorithm:

- only `[0-9a-fA-F]` — use **`-f`**
- anything else, `$` or `:` or `/` or base64 — use **`-F`**

A structured hash read with `-f` loads nothing usable and the run still exits 0.
The `Options` column of `mdxfind -h` names what a type accepts: `f` plain hex,
`F` structured, `s` salts, `u` usernames, `j` peppers.

Always read the load receipt:

```
hashes.txt: 4213 hashes, 4213 salts, 0 users loaded
```

`0 salts` or `0 users` on a type that needs them is the whole diagnosis.

## Worked example: DCC2 (MSCACHE2, e918)

DCC2 is keyed on the username: it is
`pbkdf2_sha1(DCC1, utf16le(lower(user)), N, 16)`, where DCC1 itself folds the
username in. The username can arrive either way.

Inside the wrapper, where the count is exact:

```bash
mdxfind -M e918 -F dcc2.txt wordlist.txt
# $DCC2$10240#username#32-hex-digest
```

Or as bare digests with the usernames supplied separately:

```bash
mdxfind -M e918 -U users.txt -f digests.txt wordlist.txt
mdxfind -m e918 -u users.txt -f digests.txt wordlist.txt
```

With `-u` or `-U` the iteration count is not in the input, so the Windows default
of 10240 is used. A list whose hashes carry a different count must come in
through `$DCC2$`, which names it per hash; mdxfind honours whatever it finds
there.

Typical DCC2 lists hold one unique username per hash, which is the case where a
`-U` file listing each username once is already exact.

## Rules of thumb

1. If the per-hash value is in the file, keep it there and use `-F`. The count
   takes care of itself.
2. Use `-U` when the values live separately, and repeat each one once per hit it
   accounts for.
3. Use `-u` or `-s` when you are guessing at values rather than supplying known
   ones — a candidate pool should not retire.
4. Read the receipt before believing a zero.
