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

When more than one GPU is selected, all participating devices share a single atomic counter (`_bf_cursor`) over the mask keyspace. Each device claims a chunk of work at its own autotuned chunk size, hashes it, and atomically advances the cursor. Faster devices naturally claim more chunks; slower devices do not throttle the run.

The fan-out of the packed job to N devices is performed by the helper `gpujob_submit_bf` (`mdxfind.c:8733`), introduced in Phase 2b of the GPU dispatch consolidation. It byte-copies the chokepoint's packed buffer once per OpenCL device and submits all N copies, so each device begins dispatching in parallel rather than sequentially through a single submit queue. The helper is OpenCL-only; Metal hosts run a single device by definition and use the plain `gpujob_submit` path.

A 5-GPU example from `readme.md`:

| Configuration | GPUs | Rate | Time | Speedup |
|---------------|------|------|------|---------|
| Single GPU    | RTX 4070 Ti | 11.1 GH/s | 490s | 1.0x |
| Dual GPU      | RTX 4070 Ti + RTX 3080 | 20.7 GH/s | 185s | 1.9x |
| All 5 GPUs    | RTX 4070 Ti + RTX 3080 + 2x RX 9070 XT + iGPU | 30.2 GH/s | 185s | 2.7x |

Mask: `?l?l?l?l?l?l?l?l?l` (5.4 × 10^12 candidates). Speedup is sublinear because the AMD discrete and iGPU devices are slower per-device than the NVIDIA cards; the atomic-cursor scheduler load-balances by claim rate, not by static partition.

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

## Future Direction

A forthcoming release will replace the dynamic atomic-cursor scheduler with **per-device keyspace partitioning**, in which each GPU is given a fixed sub-range of the mask cardinality based on its autotune-derived hash rate, with a small tail buffer left on the shared cursor to absorb measurement variance. This is expected to reduce coordination overhead at the head of every chunk dispatch and to add a sub-second gate that bypasses multi-GPU coordination entirely for trivially small jobs. The work is design-only at present; nothing in the shipped binary changes today.

## Source References

- Activation: `mdxfind.c:45605` (the `argc == 1 && stat() != 0` branch).
- Activation message: `mdxfind.c:45628`.
- `BruteForceTotal = MaskTotal` assignment: `mdxfind.c:45631`.
- Job flag set: `mdxfind.c:45695` (`JOBFLAG_NUMBERS | JOBFLAG_BRUTEFORCE`).
- Progress line format: `mdxfind.c:36255`.
- Multi-GPU fan-out helper: `gpujob_submit_bf` at `mdxfind.c:8733`.
- OpenCL multi-GPU cursor priming: `gpu_opencl_bf_start()` at `mdxfind.c:45672`.

## See Also

- [PROGRESS.md](PROGRESS.md) — full breakdown of the per-tick progress line, including the brute-force variant.
- [RULES.md](RULES.md) — the wordlist-plus-rules-plus-masks alternative, usually a better choice than full brute force.
- [EXAMPLES.md](EXAMPLES.md) — mask syntax in detail and worked examples.
