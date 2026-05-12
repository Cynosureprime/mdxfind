# Brute-Force Mode

Brute-force mode generates and tests every candidate in a mask keyspace without consulting a wordlist. There is no flag to enable it; activation is positional and is inferred from the shape of the command line. When active, mdxfind dispatches the full mask keyspace across every selected GPU (or, in the absence of a GPU, across CPU worker threads) and reports keyspace progress on the standard 15-second tick.

## Activation

Brute-force mode activates when **exactly one positional (non-option) argument is supplied** and `stat(2)` on that argument fails — i.e. there is no file by that name. The argument is then parsed as a mask pattern via `parse_mask_into()`. The check lives at `mdxfind.c:45605`.

The activation rules are:

| Command shape | Result |
|---------------|--------|
| `mdxfind ... '?l?l?l?l?l?l?l?l'` (one arg, no such file) | Brute force |
| `mdxfind ... wordlist.txt` (one arg, file exists) | Wordlist mode |
| `mdxfind ... wordlist1 wordlist2` (two or more args) | Wordlist mode |
| `mdxfind ... '?l?l?l' otherfile` (mask first, then a real file) | Wordlist mode (the mask is treated as a filename and fails to open) |

On activation mdxfind prints a single confirmation line on stderr before any progress ticks:

```
Brute force processing detected: 5429503678976 candidates from mask '?l?l?l?l?l?l?l?l?l'
```

The candidate count is `MaskTotal`, the cardinality of the parsed mask. It is also stored in `BruteForceTotal` and used as the denominator in every subsequent progress line. The job is dispatched with the flags `JOBFLAG_BRUTEFORCE | JOBFLAG_NUMBERS`.

If parsing the argument as a mask fails, mdxfind exits with `Could not parse '...' as a mask pattern, exiting`. There is no fall-through to wordlist mode in that case — a one-arg invocation that is neither a real file nor a valid mask is treated as a fatal error.

For full mask syntax (character classes `?l ?u ?d ?h ?H ?s ?a ?b`, custom sets `?[...]`, and literal mixing such as `password?d?d`), see [EXAMPLES.md](EXAMPLES.md).

## Single-GPU Dispatch

When one GPU is selected and a brute-force job is active, mdxfind builds the per-position mask tables, dispatches one job per algorithm type, and lets the kernel iterate the entire keyspace on-device. Per-batch chunk size is determined at startup by the GPU autotune, which runs binary-exponential timing probes on each algorithm family and picks the largest `max_items` value that still completes inside a 200–400ms target latency window. The window is chosen to keep the GPU saturated while leaving the OS watchdog clear and progress updates responsive.

The visible signal of autotune completion is a line of the form:

```
OpenCL GPU[0]: timed family 1: max_items=33554432 (287ms)
```

(One line per family per device. Family numbers correspond to algorithm groupings — MD5-class, SHA1-class, etc.) Once those are printed, dispatch starts and the standard `Brute ...` progress line takes over.

## Multi-GPU Dispatch

When more than one GPU is selected, mdxfind divides the mask keyspace among them in proportion to their measured hash rates, with a small shared tail buffer that absorbs thermal and rate variance during the run.

### Warm-probe (concurrent with mdxfind setup)

Immediately after the OpenCL kernels are compiled, mdxfind kicks off an asynchronous warm-probe — one pthread per device — that runs a short synthetic mask through each kernel and measures the sustained hash rate. The probe uses an internal dummy compact table and an internal `?l?l?l?l?l?l?l?l` mask, so it doesn't depend on (or interfere with) the user's hash file or mask, and it runs in parallel with the rest of mdxfind's startup work (mask parsing, BF detection, hash table upload). Each probe takes roughly 200–400 ms wall, in parallel across devices, so the total cost is the slowest device's probe time regardless of GPU count.

The probe writes the per-(device, family) rate into device state. Each device emits a one-line confirmation on stderr:

```
OpenCL GPU[0]: warm-probed family 1: max_items=2147483647 (237.4ms, 13525.9 Mh/s)
```

### Partition setup at BF activation

When the brute-force job is dispatched, `bf_partition_setup` waits for the warm-probes to finish (a no-op if they already completed concurrently with startup) and assigns each eligible device a contiguous range of the mask keyspace:

- `share_i = total × rate_i / Σ rate`
- `bounded_i = share_i × (1 - tail_frac)` with `tail_frac = 0.05`
- device `i` covers `[Σ_{j<i} bounded_j , Σ_{j≤i} bounded_j)`

The residual 5% (`[Σ bounded_i, total)`) is parked on the original atomic counter `_bf_cursor`. Each device drains its bounded range first, then falls through to atomic claims from the tail.

The mechanism is visible on stderr at activation. For an 8-character `?l` mask (208.8 × 10⁹ candidates) on a 4-eligible-GPU rig:

```
BF partition: 208827064576 candidates, 4/4 eligible GPUs, est 5.2s aggregate (family 1, 40499.7 Mh/s total)
BF partition: GPU[0] 13381.2 Mh/s -> [0, 65431230432) (65431230432 cands)
BF partition: GPU[1]  6659.4 Mh/s -> [65431230432, 97995479616) (32564249184 cands)
BF partition: GPU[2]  6933.6 Mh/s -> [97995479616, 131891562016) (33896082400 cands)
BF partition: GPU[3] 13525.9 Mh/s -> [131891562016, 198385711346) (66494149330 cands)
BF partition: tail pool [198385711346, 208827064576) = 10441353230 candidates
```

### Sub-second gate

For small jobs where multi-device coordination overhead would outweigh the parallelism win, the partition is skipped entirely. The threshold depends on the hash family — 1 second for fast hashes (MD5, SHA-1, SHA-256, MD4/NTLM, SHA-512, Keccak/SHA-3, etc.), 2 s for medium-iteration crypts (md5crypt, sha256crypt, sha512crypt, phpbb3), and 5 s for single-item-kernel KDFs (bcrypt, yescrypt). Below that threshold, brute force runs in single-cursor mode — whichever device dequeues first claims most of the work — and the gate is logged:

```
BF partition: est 0.59s below 1.0s threshold (family 1, fastest 13525.9 Mh/s) — single-cursor mode
```

### Devices excluded from the partition

A device may be excluded under three conditions:

- Its OpenCL platform was blacklisted at enumeration (some integrated GPUs and known-broken driver/device combinations).
- Its warm-probe didn't produce a measurable timing — only happens on very fast GPUs that complete the synthetic probe faster than the 1 ms reliability floor.
- The user explicitly omitted it via `-G N,M`.

Excluded devices still participate via the atomic tail cursor — they don't get a bounded share but they can claim from the residual 5%. The exclusion is logged:

```
BF partition: GPU[3] excluded (no rate)
```

### 5-GPU dispatch example

Reference data from a 5-GPU rig in `readme.md`, running mask `?l?l?l?l?l?l?l?l?l` (5.4 × 10¹² candidates):

| Configuration | GPUs | Rate |
|---------------|------|------|
| Single GPU    | RTX 4070 Ti | 11.1 GH/s |
| Dual GPU      | RTX 4070 Ti + RTX 3080 | 20.7 GH/s |
| All 5 GPUs    | RTX 4070 Ti + RTX 3080 + 2× RX 9070 XT + iGPU | 30.2 GH/s |

Speedup is sublinear because the AMD discrete and iGPU devices are slower per-device than the NVIDIA cards. The partition assigns shares proportional to measured rate, so each device finishes its bounded slice in roughly the same wall time and the tail cursor absorbs per-device variance from thermal throttle or background contention.

### OpenCL only

The partition layer is OpenCL-specific. Metal hosts run a single device by definition; the partition is a no-op there, and the underlying `gpujob_submit` path handles single-device dispatch directly.

## Sample Candidate in Progress

The `Brute ...` progress line carries a `'sample'` field — a concrete candidate string drawn from the current keyspace position via `mask_expand_into(progress_idx, ...)`. It is **the candidate at the completed-progress index, not the most recently tried candidate**. (The two are very close in practice but not identical, because a fast GPU may already be hashing values past the position that has been counted into `Tothash`.)

The sample is useful for two things:

1. **Sanity-checking the mask.** If you ran with `password?d?d` and the sample reads `aabcdwxyz`, your mask did not parse the way you expected.
2. **Estimating remaining work by inspection.** For a `?l*` mask, the sample's leading character tells you roughly which "decade" of the keyspace the run is in.

## `-G` Device Selection

The `-G` flag controls which GPUs participate, in both wordlist and brute-force modes. The forms are:

| Form | Meaning |
|------|---------|
| `-G auto` | Default. Use all GPUs the platform reports. |
| `-G none` | Disable GPU. CPU worker threads only. |
| `-G N`    | Use device index N (zero-based). |
| `-G N,M,...` | Use the listed device indices. |

Examples:

```bash
# All available GPUs (default)
mdxfind -m e1 -f hashes.txt '?l?l?l?l?l?l?l?l?l'

# CPU only (no GPU dispatch at all)
mdxfind -G none -m e1 -f hashes.txt '?l?l?l?l?l?l?l?l?l'

# Only the first NVIDIA card on a mixed system
mdxfind -G 0 -m e1 -f hashes.txt 'pass?d?d?d?d'

# Two specific GPUs (e.g. NVIDIA pair, skipping AMD iGPU)
mdxfind -G 2,3 -m e1 -f hashes.txt '?l?l?l?l?l?l?l?l'
```

`-G` is not "GPU on/off" in the usual sense — it is a device-selection mask. `-G none` is the only form that disables GPU dispatch entirely.

## Limitations

Mask cardinality compounds exponentially. A few reference points for printable-ASCII brute force:

| Mask | Candidates |
|------|-----------:|
| `?a?a?a?a?a?a` (6 chars)  | 7.4 × 10^11 |
| `?a?a?a?a?a?a?a` (7 chars) | 7.0 × 10^13 |
| `?a?a?a?a?a?a?a?a` (8 chars) | 6.6 × 10^15 |
| `?a?a?a?a?a?a?a?a?a` (9 chars) | 6.3 × 10^17 |

At 30 GH/s, an 8-character `?a` mask is roughly 60 hours of MD5; a 9-character `?a` mask is ~250 days. For SHA-class and slower algorithms multiply accordingly.

Practical guidance:

- **Use brute force only for short masks**, or for masks where most positions are literal and only a small tail is random (e.g. `Password20?d?d`).
- **For known-pattern attacks where most of the candidate is structured** — a wordlist plus appended digits, a wordlist with case toggles plus a year suffix — use a wordlist with `-r` rules and `-n` masks. The cross-product `(1 + N_rules) × M_masks × N_words` is almost always cheaper than enumerating every position from scratch. See [RULES.md](RULES.md).
- **CPU-only brute force is impractical for any non-trivial mask.** A modern x86 core does roughly 100–200 MH/s on MD5 in mdxfind. For a 7-character `?l` mask (~8 × 10^9 candidates) that is acceptable; for anything longer, GPU acceleration is required.

## Limitations and future direction

**Multi-op brute force.** When a single mdxfind invocation runs brute force across multiple hash families (e.g., `-h md5,sha1`), the partition is computed once at activation using the rate of the first selected op's family. Per-op repartitioning is possible but would re-stall every dispatch; the current single-shot partition gives correct results but slightly suboptimal balance when per-family rates differ significantly across devices. Most BF jobs are single-op in practice, so this limitation rarely matters.

**Hashcat-style `--increment`.** Iterating mask length (e.g., trying `?l?l`, then `?l?l?l`, then `?l?l?l?l`, etc., within one invocation) is not yet supported; the user runs mdxfind once per length. Designed but not implemented.

**Probe timing floor on extremely fast GPUs.** The warm-probe uses a 32-bit per-dispatch work-item count (capped at 2³¹ − 1). On GPUs that complete that workload faster than the 1 ms timing-reliability floor, the rescue multiplier dispatches the kernel back-to-back to accumulate measurable time; if even the rescue total stays under 1 ms, the device is excluded from the partition (it still participates via the tail cursor). Plumbing 64-bit work-item counts through the kernel-side `OCLParams` would lift this; the change is straightforward but not yet shipped.

## Source references

The brute-force path lives across two files. Symbolic references — line numbers shift with each release.

**`mdxfind.c`**

- BF detection / activation: the `argc == 1 && stat() != 0` branch in `main`. Sets `BruteForceMode = 1`, `BruteForceTotal = MaskTotal`, calls `gpu_opencl_set_mask` for the BF mask, then invokes `bf_partition_setup`.
- `bf_partition_setup(job_ops, n_ops, total_keyspace)` — collects the active-op set from `Dohash`, waits for warm-probes to finish, polls per-device `fam_rate_hps[]`, applies the per-op runtime threshold gate, computes proportional shares, calls `gpu_opencl_bf_set_partition` per eligible device, and primes the tail cursor via `gpu_opencl_bf_set_tail_start`.
- `bf_min_time_threshold_s(op)` — per-op runtime threshold below which partitioning is skipped.
- `gpujob_submit_bf` — OpenCL-only multi-GPU fan-out for the chokepoint's packed-format BF dispatches; byte-copies the packed buffer once per device so all devices begin dispatching in parallel.
- `JOBFLAG_NUMBERS | JOBFLAG_BRUTEFORCE` — the per-job flag pair that marks a dispatch as brute force.

**`gpu/gpu_opencl.c`**

- `gpu_opencl_warm_probe(dev_idx, op)` — synchronous timing probe; uses synthetic compact table + synthetic mask if the real ones haven't been uploaded yet, runs the same 200–400 ms doubling probe as the lazy autotune in `dispatch_batch`, includes the rescue-multiplier path for very fast GPUs whose probe completes under the 1 ms timing floor.
- `gpu_opencl_warm_probe_async(op)` / `gpu_opencl_warm_probe_wait()` — pthread-per-device async dispatcher and join. Called from mdxfind right after `gpu_opencl_compile_families()`, awaited at the top of `bf_partition_setup`.
- `gpu_opencl_fam_rate(dev_idx, fam)` — rate getter; returns 0 for an unprobed device.
- `gpu_opencl_bf_set_partition(dev_idx, start, end)` / `gpu_opencl_bf_set_tail_start(pos)` — partition installer and tail cursor primer.
- `_bf_cursor`, `_bf_active`, `gpu_opencl_bf_start()` — shared atomic counter and BF-mode flag, repurposed in Phase 6 as the post-partition tail pool.
- The `bf_mode` chunk-claim block in `gpu_opencl_dispatch_batch` — pulls from per-device `bf_range_pos` first, falls through to `_bf_cursor` once exhausted.

For the per-tick progress line emitted while BF is running, see `PROGRESS.md`.

## See Also

- [PROGRESS.md](PROGRESS.md) — full breakdown of the per-tick progress line, including the brute-force variant.
- [RULES.md](RULES.md) — the wordlist-plus-rules-plus-masks alternative, usually a better choice than full brute force.
- [EXAMPLES.md](EXAMPLES.md) — mask syntax in detail and worked examples.
