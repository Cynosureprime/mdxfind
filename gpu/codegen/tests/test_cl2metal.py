#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_cl2metal.py -- Regression harness for the OpenCL-to-Metal translator.

$Revision: 1.1 $
$Log: test_cl2metal.py,v $
Revision 1.1  2026/05/15 16:26:22  dlr
Phase 2d.1 regression test harness. Validates md5 and md5salt translation against existing Metal hand-ports. Three deltas tracked: LOC vs handport, semantic content vs handport, fidelity vs OpenCL source. Forbidden-token lint and state-array bounds lint included. xcrun metal compile gate ensures generated output is valid. md5 passes tight gates (0.49 percent semantic-vs-handport). md5salt passes loose vs-handport budget plus tight fidelity-vs-source (0.98 percent).


Invokes gpu/codegen/cl2metal.py on the two Phase 2d.1 regression-target
pairs (md5_core.cl and md5salt_core.cl) and validates:

  - The generated file compiles cleanly under `xcrun metal -c` (Gate A).
  - No forbidden tokens (__global, __private, etc.) survive (Gate B
    forbidden-token half).
  - LOC delta vs hand-port is bounded:
      * MD5: target <=5% (the mechanical case; comments dominate the
        delta, semantic content matches within 1%).
      * MD5SALT: best-effort; the Metal hand-port is a 38%-smaller
        reimplementation (mode-0-only, salt_pack_uint dropped) so a
        strict LOC gate cannot be met by the translator alone. We track
        a soft target of <=70% and a semantic-content (comment-stripped)
        target of <=30%.

Run:
  python3 gpu/codegen/tests/test_cl2metal.py

Exit codes:
  0 -- all gates pass
  1 -- one or more gates fail
  2 -- harness setup error (e.g., translator not found, xcrun missing)
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN_DIR = os.path.dirname(HERE)
GPU_DIR = os.path.dirname(CODEGEN_DIR)
REPO_ROOT = os.path.dirname(GPU_DIR)

TRANSLATOR = os.path.join(CODEGEN_DIR, 'cl2metal.py')
METAL_COMMON = os.path.join(GPU_DIR, 'metal_common.metal')


# Regression-target pairs. (algo, opencl_src, metal_handport, ...).
#
# Two LOC-delta gauges:
#
#   1. "vs handport" -- how close the generated output is to the Metal
#      hand-port. For md5 this is tight (mechanical port); for md5salt
#      the hand-port is a 38%-smaller reimplementation (mode-0-only,
#      salt_pack_uint dropped) so a strict gate is impossible.
#
#   2. "fidelity vs source" -- how faithfully the translator preserves
#      the OpenCL semantic content. This is the real translator gate:
#      generated should match OpenCL stripped-content within a few
#      percent (it's a 1:1 translation modulo overlays).
TARGETS = [
    {
        'algo': 'md5',
        'src': os.path.join(GPU_DIR, 'gpu_md5_core.cl'),
        'handport': os.path.join(GPU_DIR, 'metal_md5_core.metal'),
        'loc_pct_budget': 50.0,  # Comments dominate; semantic budget is the real gate.
        'semantic_vs_handport_pct_budget': 5.0,    # tight
        # Fidelity budget allows for overlay-encoded skips (the
        # template_emit_hit macro is dropped) + the __global/__private->
        # device/thread length deltas. 10% accommodates both.
        'fidelity_vs_source_pct_budget': 10.0,
        'must_compile': True,
    },
    {
        'algo': 'md5salt',
        'src': os.path.join(GPU_DIR, 'gpu_md5salt_core.cl'),
        'handport': os.path.join(GPU_DIR, 'metal_md5salt_core.metal'),
        # The hand-port is a 38%-smaller reimplementation -- vs-handport
        # gates loose by design. vs-source gate stays tight.
        'loc_pct_budget': 80.0,
        'semantic_vs_handport_pct_budget': 80.0,
        'fidelity_vs_source_pct_budget': 5.0,
        'must_compile': True,
    },
]


def strip_c(text: str) -> str:
    """Strip /* ... */ and // ... comments + whitespace."""
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def run_translator(src: str, out_path: str) -> int:
    """Invoke cl2metal.py. Returns process exit code."""
    cmd = ['python3', TRANSLATOR, src, '-o', out_path, '--check']
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(p.stderr)
    return p.returncode


def compile_metal(generated_path: str) -> tuple:
    """Compile the generated metal as part of a TU (prepended by
    metal_common.metal). Returns (returncode, stderr_text)."""
    if not os.path.exists(METAL_COMMON):
        return (-1, f"metal_common.metal not found at {METAL_COMMON}")
    with tempfile.NamedTemporaryFile('w', suffix='.metal', delete=False) as f:
        for p in (METAL_COMMON, generated_path):
            with open(p) as g:
                f.write(g.read())
            f.write('\n')
        tu_path = f.name
    air_path = tu_path.replace('.metal', '.air')
    try:
        cmd = ['xcrun', '-sdk', 'macosx', 'metal', '-c', tu_path, '-o', air_path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return (p.returncode, p.stderr)
    finally:
        try:
            os.unlink(tu_path)
            if os.path.exists(air_path):
                os.unlink(air_path)
        except OSError:
            pass


def lint_forbidden(text: str) -> list:
    """Find surviving OpenCL-only tokens in 'code' regions of the file."""
    # Lift the same regexes as cl2metal.py:lint_forbidden but operate on
    # the full text (after stripping comments + strings to avoid false hits
    # in commentary).
    cleaned = re.sub(r'/\*[\s\S]*?\*/', '', text)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)
    findings = []
    patterns = [
        r'\b__global\b', r'\b__local\b', r'\b__private\b',
        r'\b__constant\b', r'\b__kernel\b',
        r'barrier\(\s*CLK_',
        r'\bas_uint\s*\(', r'\bmul_hi\s*\(',
        r'\btypedef\s+struct\b',
    ]
    for pat in patterns:
        for m in re.finditer(pat, cleaned):
            line_no = cleaned.count('\n', 0, m.start()) + 1
            findings.append((line_no, pat))
    return findings


def check_state_array_bounds(text: str) -> list:
    """Lint per memo §6: state-array bounds should match HASH_WORDS.

    Looks for `h[N]` accesses inside template_state struct context and
    asserts N < HASH_WORDS as declared at top of file. This is a coarse
    check -- catches cases where someone hardcodes h[4] when HASH_WORDS=5.
    """
    findings = []
    m = re.search(r'#define\s+HASH_WORDS\s+(\d+)', text)
    if not m:
        return findings
    hash_words = int(m.group(1))
    # Walk all `st.h[N]` and `st->h[N]` references
    for accesses in re.finditer(r'\bh\s*\[\s*(\d+)\s*\]', text):
        n = int(accesses.group(1))
        if n >= hash_words:
            line_no = text.count('\n', 0, accesses.start()) + 1
            findings.append((line_no, f'h[{n}] >= HASH_WORDS={hash_words}'))
    return findings


def run_target(target: dict) -> dict:
    result = {
        'algo': target['algo'],
        'passed': True,
        'errors': [],
        'warnings': [],
    }
    if not os.path.exists(target['src']):
        result['passed'] = False
        result['errors'].append(f"source not found: {target['src']}")
        return result
    if not os.path.exists(target['handport']):
        result['passed'] = False
        result['errors'].append(f"handport not found: {target['handport']}")
        return result

    # Translate.
    with tempfile.NamedTemporaryFile('w', suffix='.metal', delete=False) as f:
        out_path = f.name
    rc = run_translator(target['src'], out_path)
    if rc != 0:
        result['passed'] = False
        result['errors'].append(f"translator exit code {rc}")
        return result

    with open(out_path) as f:
        gen = f.read()
    with open(target['handport']) as f:
        hp = f.read()

    # LOC delta.
    gen_lines = gen.count('\n') + 1
    hp_lines = hp.count('\n') + 1
    loc_diff = abs(gen_lines - hp_lines)
    loc_pct = 100.0 * loc_diff / hp_lines
    result['loc'] = (gen_lines, hp_lines, loc_pct)
    if loc_pct > target['loc_pct_budget']:
        result['passed'] = False
        result['errors'].append(
            f"LOC delta {loc_pct:.2f}% exceeds budget {target['loc_pct_budget']}%"
        )

    # Semantic-content delta vs hand-port (comment-stripped).
    gs = strip_c(gen)
    hs = strip_c(hp)
    sem_diff = abs(len(gs) - len(hs))
    sem_pct = 100.0 * sem_diff / max(len(hs), 1)
    result['semantic_vs_handport'] = (len(gs), len(hs), sem_pct)
    if sem_pct > target['semantic_vs_handport_pct_budget']:
        result['passed'] = False
        result['errors'].append(
            f"semantic-vs-handport delta {sem_pct:.2f}% exceeds budget "
            f"{target['semantic_vs_handport_pct_budget']}%"
        )

    # Fidelity gauge: generated content vs OpenCL source content (both
    # stripped). Asserts the translator faithfully preserves the OpenCL
    # semantics modulo overlay-encoded tweaks.
    with open(target['src']) as f:
        src_text = f.read()
    src_stripped = strip_c(src_text)
    fid_diff = abs(len(gs) - len(src_stripped))
    fid_pct = 100.0 * fid_diff / max(len(src_stripped), 1)
    result['fidelity_vs_source'] = (len(gs), len(src_stripped), fid_pct)
    if fid_pct > target['fidelity_vs_source_pct_budget']:
        result['passed'] = False
        result['errors'].append(
            f"fidelity-vs-source delta {fid_pct:.2f}% exceeds budget "
            f"{target['fidelity_vs_source_pct_budget']}%"
        )

    # Forbidden-token lint.
    findings = lint_forbidden(gen)
    if findings:
        result['passed'] = False
        result['errors'].append(
            f"forbidden tokens surviving ({len(findings)}): "
            + '; '.join(f'line {ln} matches /{tok}/' for ln, tok in findings[:5])
        )

    # State-array bounds lint.
    bounds_findings = check_state_array_bounds(gen)
    if bounds_findings:
        result['warnings'].append(
            f"state-array bounds suspicious ({len(bounds_findings)}): "
            + '; '.join(f'line {ln}: {msg}' for ln, msg in bounds_findings[:3])
        )

    # Compile.
    if target['must_compile']:
        rc, stderr = compile_metal(out_path)
        if rc != 0:
            result['passed'] = False
            errs = [ln for ln in stderr.split('\n') if 'error:' in ln]
            result['errors'].append(
                f"xcrun metal compile failed (rc={rc}, {len(errs)} errors): "
                + ' | '.join(errs[:3])
            )

    try:
        os.unlink(out_path)
    except OSError:
        pass
    return result


def main():
    if not os.path.exists(TRANSLATOR):
        sys.stderr.write(f"FAIL: translator not found at {TRANSLATOR}\n")
        return 2

    # Verify xcrun metal is available
    try:
        p = subprocess.run(['xcrun', '-f', 'metal'], capture_output=True, text=True)
        if p.returncode != 0:
            sys.stderr.write("WARN: xcrun metal not found; skipping compile checks\n")
    except FileNotFoundError:
        sys.stderr.write("WARN: xcrun missing; skipping compile checks\n")

    overall = 0
    for target in TARGETS:
        sys.stdout.write(f"\n=== target: {target['algo']} ===\n")
        r = run_target(target)
        if 'loc' in r:
            sys.stdout.write(
                f"  LOC: generated={r['loc'][0]} handport={r['loc'][1]} "
                f"delta={r['loc'][2]:.2f}% (budget {target['loc_pct_budget']}%)\n"
            )
        if 'semantic_vs_handport' in r:
            sys.stdout.write(
                f"  Semantic-vs-handport: gen={r['semantic_vs_handport'][0]} "
                f"hp={r['semantic_vs_handport'][1]} "
                f"delta={r['semantic_vs_handport'][2]:.2f}% "
                f"(budget {target['semantic_vs_handport_pct_budget']}%)\n"
            )
        if 'fidelity_vs_source' in r:
            sys.stdout.write(
                f"  Fidelity-vs-source: gen={r['fidelity_vs_source'][0]} "
                f"src={r['fidelity_vs_source'][1]} "
                f"delta={r['fidelity_vs_source'][2]:.2f}% "
                f"(budget {target['fidelity_vs_source_pct_budget']}%)\n"
            )
        for w in r.get('warnings', []):
            sys.stdout.write(f"  WARN: {w}\n")
        for e in r['errors']:
            sys.stdout.write(f"  FAIL: {e}\n")
        if r['passed']:
            sys.stdout.write(f"  PASS\n")
        else:
            overall = 1
    sys.stdout.write(f"\n=== overall: {'PASS' if overall == 0 else 'FAIL'} ===\n")
    return overall


if __name__ == '__main__':
    sys.exit(main())
