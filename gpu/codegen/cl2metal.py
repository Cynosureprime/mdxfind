#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl2metal.py -- OpenCL-to-Metal codegen translator for mdxfind GPU cores.

$Revision: $
$Log: $

Translates gpu/gpu_<algo>_core.cl into gpu/metal_<algo>_core.metal using a
three-pass hybrid approach (Option C per project_metal_phase2d_arch.md):

  1. Tokenize-and-substitute: regex passes for 1:1 substitutions
     (__global -> device, __private -> thread, barrier -> threadgroup_barrier,
     as_uint -> as_type<uint>, mul_hi -> mulhi, etc.). Comments, strings,
     and preprocessor lines are protected.

  2. Structural rewriter: bracket-balanced parsing for function signatures
     and struct decls. Rewrites:
       typedef struct { ... } T;  -> struct T { ... };
       fn(template_state *st)     -> fn(thread template_state &st)
       fn(__global uchar *p)      -> fn(device const uchar *p) when read-only
       st->h[0]                   -> st.h[0]   (within the rewritten fn body)
       &st->h[i]                  -> st.h[i]   (md5_block expects scalars)

  3. Per-kernel overlay: reads cl2metal_overrides/<algo>.yaml for
     hand-curated tweaks (skip functions, override arg address space,
     line-range skips for content removal within a function).

Usage:
  python3 cl2metal.py gpu/gpu_md5_core.cl
  python3 cl2metal.py gpu/gpu_md5_core.cl -o gpu/metal_md5_core.metal.generated
  python3 cl2metal.py gpu/gpu_md5_core.cl --diff gpu/metal_md5_core.metal

The translator does NOT touch the existing hand-port .metal files. It writes
generated output to a side path (default: <input>.metal.generated next to
input, or as specified via -o).
"""

import argparse
import os
import re
import sys
from typing import List, Tuple, Optional

# -----------------------------------------------------------------------------
# Optional YAML support. Fall back to a tiny hand-written reader if PyYAML is
# unavailable (codegen tools per memo §3 use stdlib only as a default; PyYAML
# is preferred when present).
# -----------------------------------------------------------------------------
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


def _parse_yaml_minimal(text: str) -> dict:
    """Minimal YAML reader: supports the overlay subset we need.

    Format:
      algo: <name>
      skip_functions:
        - name1
        - name2
      skip_line_ranges:
        - start: 100
          end:   200
          reason: "modes 1-6 deferred"
      arg_address_space:
        function_name:
          arg_name: device
    """
    out = {}
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s or s.startswith('#'):
            i += 1
            continue
        # Top-level key
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$', line)
        if m:
            key = m.group(1)
            val = m.group(2).strip()
            if val and not val.startswith('#'):
                # scalar value
                out[key] = val.strip('"').strip("'")
                i += 1
                continue
            # block value -- read child block
            i += 1
            block = []
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip() or nxt.strip().startswith('#'):
                    i += 1
                    continue
                if re.match(r'^[a-zA-Z_]', nxt):
                    break
                block.append(nxt)
                i += 1
            if not block:
                out[key] = []
                continue
            # Try list vs dict
            first = block[0].lstrip()
            if first.startswith('-'):
                out[key] = _parse_list_block(block)
            else:
                out[key] = _parse_dict_block(block)
            continue
        i += 1
    return out


def _parse_list_block(block):
    """Parse a list block. Items begin with '-'. Items may be scalars,
    or compound (with sub-keys on subsequent lines)."""
    items = []
    i = 0
    while i < len(block):
        line = block[i]
        s = line.strip()
        if not s.startswith('-'):
            i += 1
            continue
        rest = s[1:].strip()
        if ':' in rest and not rest.startswith('"'):
            # compound: this is the first key of a dict item
            d = {}
            m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$', rest)
            if m:
                d[m.group(1)] = m.group(2).strip().strip('"').strip("'")
            # consume subsequent keys at the same sub-indent
            base_indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(block):
                nxt = block[i]
                if not nxt.strip():
                    i += 1
                    continue
                ind = len(nxt) - len(nxt.lstrip())
                if ind <= base_indent or nxt.lstrip().startswith('-'):
                    break
                mm = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$', nxt.strip())
                if mm:
                    d[mm.group(1)] = mm.group(2).strip().strip('"').strip("'")
                i += 1
            items.append(d)
        else:
            items.append(rest.strip('"').strip("'"))
            i += 1
    return items


def _parse_dict_block(block):
    """Parse a dict block of `key: value` lines (possibly nested)."""
    d = {}
    i = 0
    while i < len(block):
        line = block[i]
        s = line.strip()
        if not s or s.startswith('#'):
            i += 1
            continue
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$', s)
        if not m:
            i += 1
            continue
        key = m.group(1)
        val = m.group(2).strip()
        if val:
            d[key] = val.strip('"').strip("'")
            i += 1
            continue
        # nested dict
        base_indent = len(line) - len(line.lstrip())
        i += 1
        sub = []
        while i < len(block):
            nxt = block[i]
            if not nxt.strip():
                i += 1
                continue
            ind = len(nxt) - len(nxt.lstrip())
            if ind <= base_indent:
                break
            sub.append(nxt)
            i += 1
        if sub:
            d[key] = _parse_dict_block(sub)
        else:
            d[key] = {}
    return d


def load_overlay(path: str) -> dict:
    """Load an overlay YAML file. Returns {} if file doesn't exist."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        text = f.read()
    if _HAVE_YAML:
        return yaml.safe_load(text) or {}
    return _parse_yaml_minimal(text)


# -----------------------------------------------------------------------------
# Pass 1: Tokenize-and-substitute
#
# Strategy: split the source into a stream of (kind, text) tokens where
# kind is one of:
#   'code'   -- ordinary code (substitutions applied)
#   'comment' -- /* ... */ or // ... block (preserved verbatim, except
#                metal-specific header rewrites)
#   'string'  -- "..." or '...' literal (preserved verbatim)
#   'preproc' -- #... line (substitutions applied; cl#include not seen in
#                cores so this is mostly transparent)
# -----------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r'(?P<bcomment>/\*[\s\S]*?\*/)'
    r'|(?P<lcomment>//[^\n]*)'
    r'|(?P<string>"(?:\\.|[^"\\])*")'
    r'|(?P<char>\'(?:\\.|[^\'\\])*\')'
    r'|(?P<preproc>^[ \t]*#[^\n]*)',
    re.MULTILINE,
)


def tokenize(src: str):
    """Yield (kind, text) tokens covering the whole source. 'code' tokens
    are everything not matched by the special-kinds regex."""
    pos = 0
    for m in _TOKEN_RE.finditer(src):
        if m.start() > pos:
            yield ('code', src[pos:m.start()])
        for kind in ('bcomment', 'lcomment', 'string', 'char', 'preproc'):
            t = m.group(kind)
            if t is not None:
                yield (kind, t)
                break
        pos = m.end()
    if pos < len(src):
        yield ('code', src[pos:])


# 1:1 token substitutions applied to 'code' regions only.
# Order matters where one pattern is a substring of another.
_SUBSTITUTIONS = [
    # Address-space qualifiers.
    (re.compile(r'\b__global\b'),   'device'),
    (re.compile(r'\b__local\b'),    'threadgroup'),
    (re.compile(r'\b__private\b'),  'thread'),
    (re.compile(r'\b__constant\b'), 'constant'),
    (re.compile(r'\b__kernel\b'),   'kernel'),
    # Barriers.
    (re.compile(r'\bbarrier\(\s*CLK_LOCAL_MEM_FENCE\s*\)'),
        'threadgroup_barrier(mem_flags::mem_threadgroup)'),
    (re.compile(r'\bbarrier\(\s*CLK_GLOBAL_MEM_FENCE\s*\)'),
        'threadgroup_barrier(mem_flags::mem_device)'),
    # Casts / built-ins.
    (re.compile(r'\bas_uint\s*\('),    'as_type<uint>('),
    (re.compile(r'\bas_int\s*\('),     'as_type<int>('),
    (re.compile(r'\bas_ulong\s*\('),   'as_type<ulong>('),
    (re.compile(r'\bas_long\s*\('),    'as_type<long>('),
    (re.compile(r'\bas_float\s*\('),   'as_type<float>('),
    (re.compile(r'\bmul_hi\s*\('),     'mulhi('),
    (re.compile(r'\bconvert_uchar\s*\('),  '(uchar)('),
    (re.compile(r'\bconvert_uint\s*\('),   '(uint)('),
    (re.compile(r'\bconvert_int\s*\('),    '(int)('),
    (re.compile(r'\bconvert_ulong\s*\('),  '(ulong)('),
    # Phase 2d.5.3: hex_byte_* helpers in metal_common.metal carry the
    # mtl_ prefix to avoid future C-side collision (Pattern 2). The OpenCL
    # twin uses bare names. Rewrite call sites at translation time so
    # core .cl files can use hex_byte_* without manual overlay edits.
    # Affected algos: sha384* (template_iterate hex re-feed). sha512*
    # already inlines via sha512_to_hex_lc + does not call hex_byte_be64
    # directly; the rule below is harmless on cores that don't reference
    # them.
    (re.compile(r'\bhex_byte_be64\b'),     'mtl_hex_byte_be64'),
    (re.compile(r'\bhex_byte_lc\b'),       'mtl_hex_byte_lc'),
    (re.compile(r'\bhex_byte_uc\b'),       'mtl_hex_byte_uc'),
    # Phase 2d.7a: BLAKE2 IV / SIGMA constants in metal_common.metal carry
    # the MTL_ prefix per Pattern 2 (sibling to MTL_SHA256_K / MTL_SHA512_K).
    # The OpenCL twin uses bare names B2S_IV / B2S_SIGMA / B2B_IV / B2B_SIGMA;
    # rewrite at translation time so per-algo core .cl files (which
    # reference these constants in template_init / template_iterate)
    # compile against the metal_common.metal MTL_-prefixed names without
    # manual overlay edits. Harmless on cores that don't reference them.
    (re.compile(r'\bB2S_IV\b'),            'MTL_B2S_IV'),
    (re.compile(r'\bB2S_SIGMA\b'),         'MTL_B2S_SIGMA'),
    (re.compile(r'\bB2B_IV\b'),            'MTL_B2B_IV'),
    (re.compile(r'\bB2B_SIGMA\b'),         'MTL_B2B_SIGMA'),
    # Phase 2d.7b: Keccak/SHA-3 round constants in metal_common.metal carry
    # the MTL_ prefix per Pattern 2 (sibling to MTL_SHA256_K / MTL_SHA512_K
    # / MTL_B2{S,B}_IV+SIGMA). The OpenCL twin uses bare names KECCAK_RC /
    # KECCAK_ROTC; rewrite at translation time so the 8 per-algo Keccak /
    # SHA-3 core .cl files (which never reference these directly — only
    # via the keccakf1600 helper from gpu_common.cl) compile against the
    # MTL_-prefixed names without manual overlay edits. Harmless on cores
    # that don't reference them.
    (re.compile(r'\bKECCAK_RC\b'),         'MTL_KECCAK_RC'),
    (re.compile(r'\bKECCAK_ROTC\b'),       'MTL_KECCAK_ROTC'),
]


# Algorithm-specific call-site rewrites.
#
# The Metal twins of some OpenCL `*_block` helpers convert per-element
# state pointers into scalar-by-reference args (md5_block is the
# canonical example: `void md5_block(uint *h0, uint *h1, uint *h2, uint
# *h3, uint *M)` in OpenCL becomes `void md5_block(thread uint &h0,
# thread uint &h1, thread uint &h2, thread uint &h3, thread const uint
# *M)` in Metal). For these scalar-state helpers we MUST strip the
# leading `&` from call-site args to match the new reference signature.
#
# Other `*_block` helpers (sha1_block, sha256_block, ...) take a
# pointer to the whole state array in BOTH OpenCL and Metal (no
# per-element split). For these the call-site `&st.h[0]` must be
# preserved verbatim — stripping the `&` would pass a scalar uint
# where the helper expects a uint*.
#
# Phase 2d.3.1 (SHA-1 canary, 2026-05-15): introduce the split. Prior
# Phase 2d.1/2d.2 cl2metal.py revs lumped ALL `*_block` helpers into a
# single strip list, which was correct only because md5_block was the
# only helper seen by the md5 / md4 / md4utf16 / md5raw / md5passsalt /
# md5saltpass kernels.
_BLOCK_HELPERS_SCALAR_STATE = ('md5_block',)
_BLOCK_HELPERS_POINTER_STATE = ('sha1_block', 'sha256_block', 'sha512_block',
                                'rmd160_block', 'rmd320_block',
                                'b2s_compress', 'b2b_compress')

def rewrite_block_calls(code: str,
                        discovered_scalar_state_helpers: Optional[set] = None) -> str:
    """Strip leading `&` from the first N args of scalar-state `*_block`
    / `*_compress` calls. Scalar-state means: the Metal signature uses
    `thread uint &h0, thread uint &h1, ...` per state word (vs OpenCL's
    single pointer `uint *h` per state word).

    The set of scalar-state helpers is the UNION of:
      - the hardcoded `_BLOCK_HELPERS_SCALAR_STATE` (md5_block)
      - `discovered_scalar_state_helpers` (md4_compress and friends —
        discovered by rewrite_functions() noting which fn sigs got
        `_detect_scalar_ref_params` rewrites).

    Pointer-state helpers (sha1_block, sha256_block, ...) are left
    alone: their state pointer is a single arg in BOTH OpenCL and Metal,
    so `&st.h[0]` is the right call-site shape.

    Translation rule (scalar-state): for any `<helper>(&a, &b, &c, &d, X)`
    (5 args, first 4 preceded by `&`), drop the `&`s on the first 4.
    Generalised: drop `&` on every comma-separated arg in the call's arg list
    that's a pure identifier or `id->field` access (the `id.field`
    member-access shape after state-deref rewrite).
    """
    scalar_helpers = set(_BLOCK_HELPERS_SCALAR_STATE)
    if discovered_scalar_state_helpers:
        scalar_helpers.update(discovered_scalar_state_helpers)
    if not scalar_helpers:
        return code
    helpers_pat = '|'.join(re.escape(h) for h in sorted(scalar_helpers))
    # NOTE: no `\s*` between helper name and `(` — that avoids matching
    # prose like "md4_compress (foo bar)" inside `/* ... */` comments,
    # where the existing match-and-rebuild logic eats the inter-token
    # whitespace and corrupts comment readability. C source idiom for a
    # function call is `name(args)` with no space; if a code author writes
    # `name (args)` the translator will skip it (rare; live with it).
    pat = re.compile(r'\b(' + helpers_pat + r')\(')
    out = []
    pos = 0
    while True:
        m = pat.search(code, pos)
        if not m:
            out.append(code[pos:])
            break
        # Find the matching close paren.
        open_paren = m.end() - 1
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren < 0:
            out.append(code[pos:m.end()])
            pos = m.end()
            continue
        args_str = code[open_paren + 1:close_paren]
        args = _split_args(args_str)
        new_args = []
        for a in args:
            sa = a.strip()
            # Strip `& IDENT` or `& IDENT.FIELD` or `& IDENT[i]` etc.
            if sa.startswith('&'):
                rest = sa[1:].strip()
                # Only strip if the remainder is a plain lvalue
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*|\[[^\]]*\])*$', rest):
                    sa = rest
            new_args.append(sa)
        new_call = m.group(1) + '(' + ', '.join(new_args) + ')'
        out.append(code[pos:m.start()])
        out.append(new_call)
        pos = close_paren + 1
    return ''.join(out)


def substitute_pass(token_stream):
    """Apply 1:1 regex substitutions to 'code' tokens only."""
    out = []
    for kind, text in token_stream:
        if kind != 'code':
            out.append((kind, text))
            continue
        for pat, repl in _SUBSTITUTIONS:
            text = pat.sub(repl, text)
        out.append((kind, text))
    return out


# -----------------------------------------------------------------------------
# Pass 2: Structural rewriter
#
# After Pass 1 the text already uses `device`, `thread`, etc. We now rewrite
# function signatures and struct decls.
#
# Key transforms:
#   typedef struct { ... } T;  -> struct T { ... };
#   T *st                       -> thread T &st          (state-ptr by ref)
#   const uchar *block          -> thread const uchar *block (or device per overlay)
#   device const uchar *salt    -> already-qualified (Pass 1); keep as-is
#   void f(template_state *st)  -> void f(thread template_state &st)
#   st->h[0]                    -> st.h[0]               (within fn body)
#   &st->h[0]                   -> st.h[0]               (md5_block takes scalars)
#
# Implementation: walk the token stream's 'code' regions with bracket-depth
# tracking. When we see a function signature (return-type ident '(' ... ')')
# we rewrite the argument list, then identify pointer parameters to translate.
# We also track which fn args are state-ref so we can do st->h[i] -> st.h[i]
# inside the body.
# -----------------------------------------------------------------------------

# Functions whose state-ptr argument should be by-reference. Add to this list
# as new core algorithms surface state types.
_STATE_TYPES = {'template_state', 'template_pre_salt_state'}


def _is_likely_state_type(typename: str) -> bool:
    return typename in _STATE_TYPES


# Detect a typedef struct { ... } T; block in code. Returns (start, end, name,
# inner_body) or None.
def find_typedef_struct(code: str, start: int = 0) -> Optional[Tuple[int, int, str, str]]:
    pat = re.compile(r'typedef\s+struct\s*\{', re.MULTILINE)
    m = pat.search(code, start)
    if not m:
        return None
    brace_start = m.end() - 1  # position of {
    depth = 1
    i = brace_start + 1
    while i < len(code) and depth:
        c = code[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1
    if depth:
        return None
    # i is one past the closing brace. Now read whitespace + identifier + ;
    j = i
    while j < len(code) and code[j].isspace():
        j += 1
    nm = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', code[j:])
    if not nm:
        return None
    name = nm.group(0)
    k = j + len(name)
    while k < len(code) and code[k].isspace():
        k += 1
    if k >= len(code) or code[k] != ';':
        return None
    inner_body = code[brace_start + 1:i - 1]
    return (m.start(), k + 1, name, inner_body)


def rewrite_typedef_structs(code: str) -> str:
    """typedef struct { ... } T;  ->  struct T { ... };"""
    out = []
    pos = 0
    while True:
        f = find_typedef_struct(code, pos)
        if not f:
            out.append(code[pos:])
            break
        start, end, name, inner = f
        out.append(code[pos:start])
        out.append('struct ' + name + ' {' + inner + '};')
        pos = end
    return ''.join(out)


# Function signature parser. After Pass 1 substitutions, we see things like:
#   static inline void template_init(template_state *st) {
#   static inline void template_finalize(template_state *st,
#                                        const uchar *data, int len)
#
# We need to:
#   1. Find function definitions (return-type ident '(' args ')')
#   2. Rewrite args (state-ptr -> by-ref; addr-space inference)
#   3. Within the body, rewrite st->h[i] -> st.h[i]; &st->h[i] -> st.h[i]
#
# We use a coarse scanner: look for /^\s*static\s+inline/ followed by a function
# signature with balanced parens, then a body in braces.


_FN_HEAD_RE = re.compile(
    # Match function definitions at file scope. Two forms accepted:
    #   1. `static inline <ret> <name>(...)` — the canonical form for helpers
    #      authored as such in the OpenCL twin (md5/sha*/keccak/blake2/etc.).
    #   2. `<ret> <name>(...)` at the very start of a line (no leading
    #      whitespace) — for OpenCL cores that use plain `void` for helpers
    #      (streebog256/streebog512). Anchored strictly to newline+column-0
    #      so we don't false-match `return foo(` inside a function body.
    # Phase 2d.7c (2026-05-16): qualifier made optional + strict newline anchor
    # for the bare form. Phase 2d.7b and earlier required `static inline`.
    r'(?P<lead>(?:^|\n)(?:\s*static\s+inline\s+|(?=[A-Za-z_])))'
    r'(?P<ret>[a-zA-Z_][a-zA-Z0-9_\s\*]*?)\s+'
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
    re.MULTILINE,
)


def _find_matching_paren(code: str, open_idx: int) -> int:
    """Given index of '(' return index of matching ')' or -1."""
    depth = 1
    i = open_idx + 1
    while i < len(code) and depth:
        c = code[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0:
            return i
        i += 1
    return -1


def _find_matching_brace(code: str, open_idx: int) -> int:
    """Given index of '{' return index of matching '}' or -1."""
    depth = 1
    i = open_idx + 1
    while i < len(code) and depth:
        c = code[i]
        # skip strings, comments
        if code[i:i+2] == '/*':
            j = code.find('*/', i + 2)
            i = j + 2 if j >= 0 else len(code)
            continue
        if code[i:i+2] == '//':
            j = code.find('\n', i + 2)
            i = j + 1 if j >= 0 else len(code)
            continue
        if c == '"':
            j = i + 1
            while j < len(code):
                if code[j] == '\\':
                    j += 2
                    continue
                if code[j] == '"':
                    break
                j += 1
            i = j + 1
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        if depth == 0:
            return i
        i += 1
    return -1


def _split_args(arglist: str) -> List[str]:
    """Split a function arg list on top-level commas."""
    out = []
    depth = 0
    cur = []
    for c in arglist:
        if c == '(' or c == '[':
            depth += 1
        elif c == ')' or c == ']':
            depth -= 1
        if c == ',' and depth == 0:
            out.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if cur:
        out.append(''.join(cur).strip())
    return [a for a in out if a]


# Match an arg of the form "[qualifiers] type[ * ] [const] [* ...] name".
# We classify args into categories:
#   state-by-ptr: `template_state *st` or `const template_state *st`
#                  -> `thread template_state &st` (or `thread const template_state &st`)
#   addr-spaced pointer: `device const uchar *p`, already qualified -> keep
#   unqualified pointer: `const uchar *p`, `uchar *p`, `uint *p`
#                  -> `thread const uchar *p` (or per overlay -> device)
#   scalar / array : keep as-is
def _rewrite_arg(arg: str, fn_name: str, overlay_args: dict, body_text: str = '') -> Tuple[str, Optional[str]]:
    """Return (rewritten_arg, state_arg_name_or_None).

    state_arg_name_or_None: if this arg is a state-by-ptr (now ref), name of
    the C identifier used for the param (so we can rewrite st->h[i] in body).
    """
    # Tokenize a single arg
    arg = arg.strip()
    if not arg:
        return arg, None
    # Already metal-qualified (device/thread/threadgroup/constant) -> keep
    # but check for state-ptr-to-ref opportunity
    tokens = arg.split()
    if not tokens:
        return arg, None

    # Find the last identifier (parameter name) -- ignoring trailing [] or *
    # We match: ... <type> [*]+ name [array...]
    m = re.match(
        r'^(?P<pre>.*?)(?P<stars>\**)\s*'
        r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)'
        r'(?P<post>(?:\s*\[[^\]]*\])*)\s*$',
        arg, re.DOTALL)
    if not m:
        return arg, None
    pre = m.group('pre').strip()
    stars = m.group('stars')
    pname = m.group('name')
    post = m.group('post')

    if not stars:
        # Not a pointer -- scalar/array, keep as-is
        return arg, None

    # `pre` now holds the type with qualifiers. Examine.
    # Check overlay first.
    fn_overrides = overlay_args.get(fn_name, {}) if overlay_args else {}
    arg_override = fn_overrides.get(pname)

    # Identify the type token (last token of pre)
    pretoks = pre.split()
    # Detect existing address-space qualifier
    has_addr_space = any(q in pretoks for q in ('device', 'thread', 'threadgroup', 'constant'))
    has_const = 'const' in pretoks

    # Pull the type identifier (last non-qualifier token)
    qualifiers = {'device', 'thread', 'threadgroup', 'constant', 'const'}
    type_tok = None
    for t in reversed(pretoks):
        if t not in qualifiers:
            type_tok = t
            break
    if type_tok is None:
        return arg, None

    is_state = _is_likely_state_type(type_tok)

    if is_state and stars == '*':
        # state-by-ptr: rewrite to thread/const template_state &st
        if arg_override == 'thread const':
            new = 'thread const ' + type_tok + ' &' + pname + post
        elif arg_override == 'thread':
            new = 'thread ' + type_tok + ' &' + pname + post
        elif has_const:
            new = 'thread const ' + type_tok + ' &' + pname + post
        else:
            new = 'thread ' + type_tok + ' &' + pname + post
        return new, pname

    if has_addr_space:
        # Already qualified (e.g., post-substitution `device const uchar *p`).
        # Keep as-is.
        return arg, None

    # Unqualified pointer -- needs address space.
    if arg_override:
        new_qual = arg_override  # e.g., "device" or "device const" or "thread const"
    else:
        # Default inference: OpenCL unqualified pointer == __private (writable).
        # Translate to `thread`. If `const` is present in the source, preserve
        # it as `thread const`. We do NOT try to infer const-ness from body
        # use because OpenCL writers expect ptr-out params to be unqualified
        # and writable -- e.g., `uint *out_idx` is a pointer through which
        # the callee writes.
        if has_const:
            new_qual = 'thread const'
        else:
            new_qual = 'thread'

    # Strip old `const` from pretoks (we'll re-emit if needed)
    rest_pretoks = [t for t in pretoks if t != 'const']
    # Re-emit: <new_qual> <type-with-other-quals> <stars> <name><post>
    rest = ' '.join(rest_pretoks)
    new = (new_qual + ' ' + rest + ' ' + stars + pname + post).strip()
    # Normalize whitespace
    new = re.sub(r'\s+', ' ', new)
    # tidy up "*p" -> proper spacing
    new = re.sub(r'\s*\*\s*', ' *', new)
    new = re.sub(r'\*\s+', '*', new)
    new = re.sub(r'\s+\*', ' *', new)
    return new, None


_LOCAL_PTR_DECL_RE = re.compile(
    r'(?P<lead>(?:^|[\s;{}])\s*)'
    r'(?P<type>(?:const\s+)?(?:uchar|uint|int|ulong|long|short|ushort|char|float))'
    r'(?P<stars>\s*\*+)\s*'
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*'
    r'(?P<rest>(?:=[^;]*)?;)',
    re.MULTILINE,
)


def rewrite_local_ptr_decls(body: str) -> str:
    """Within a function body, add `thread` qualifier to unqualified pointer
    local var decls. Catches patterns like:
      uchar *mb = (uchar *)M;
      uint *p = ...
    Already-qualified decls (device/thread/threadgroup/constant) are skipped.
    Cast expressions are rewritten separately by rewrite_pointer_casts().
    """
    def _sub(m):
        lead = m.group('lead')
        # Check the context preceding the type for an existing addr-space qual.
        # We rely on the regex's `lead` group ending with whitespace -- it
        # cannot consume the address-space keyword, so any device/thread
        # immediately before `type` is the surrounding context.
        # Simplest check: peek at the tail of lead.
        tail = lead.rstrip()
        words = tail.rsplit(None, 1)
        if len(words) == 2 and words[1] in ('device', 'thread', 'threadgroup', 'constant'):
            return m.group(0)
        # Also skip "static const" / `const` already qualified types -- those
        # might be intended threadgroup_const etc. For our cores 'const' on a
        # local var is rare; we still apply thread.
        type_str = m.group('type')
        stars = m.group('stars')
        name = m.group('name')
        rest = m.group('rest')
        return (lead + 'thread ' + type_str + stars + ' ' + name + rest)
    return _LOCAL_PTR_DECL_RE.sub(_sub, body)


_PTR_CAST_RE = re.compile(
    r'\(\s*'
    r'(?P<type>(?:const\s+)?(?:uchar|uint|int|ulong|long|short|ushort|char|float))'
    r'\s*(?P<stars>\*+)\s*\)'
)


def rewrite_pointer_casts(code: str) -> str:
    """Rewrite `(uchar *)expr` -> `(thread uchar *)expr` in cast expressions.
    Apple Metal disallows unqualified pointer types in casts."""
    def _sub(m):
        type_str = m.group('type')
        stars = m.group('stars')
        return '(thread ' + type_str + ' ' + stars + ')'
    return _PTR_CAST_RE.sub(_sub, code)


def _rewrite_state_derefs(body: str, state_names: List[str]) -> str:
    """Within a function body, rewrite st->h[i] -> st.h[i] for state_names.

    Phase 2d.1: state-to-reference rewrites turn `template_state *st` into
    `thread template_state &st`. Pointer-deref `st->h[i]` becomes member-
    access `st.h[i]`. The leading `&` (if present) is PRESERVED — for
    md5_block (scalar-state) the `&` is stripped at the *call site* by
    rewrite_block_calls(); for sha1_block (pointer-state) the `&` must
    survive because the Metal helper signature still takes a uint*.
    Earlier revs unconditionally stripped `&st->h[i]` -> `st.h[i]`, which
    was correct only because md5_block was the sole `*_block` caller in
    the families ported through Phase 2d.2.5; Phase 2d.3.1 SHA-1 canary
    surfaced the need to keep `&`.
    """
    if not state_names:
        return body
    for n in state_names:
        # Strict order: rewrite `<n>->` to `<n>.`. Patterns like `&n->h[0]`
        # naturally become `&n.h[0]`, which rewrite_block_calls() then
        # strips for scalar-state helpers (md5_block) and leaves alone for
        # pointer-state helpers (sha1_block etc.).
        pat2 = re.compile(re.escape(n) + r'->')
        body = pat2.sub(n + '.', body)
    return body


def rewrite_macro_state_derefs(code: str) -> str:
    """Rewrite (st)->h[i] -> (st).h[i] patterns in #define lines.

    Macros that wrap EMIT_HIT_N (template_emit_hit, template_emit_hit_or_overflow)
    use `(st)->h[0]` to dereference a state-by-ptr arg. In Metal the macro
    callers pass a state-by-reference, so the macro body must use `(st).h[0]`.
    The simple textual replace below covers the macros in md5_core.cl and
    md5salt_core.cl. We are careful to NOT touch md5_block(...) call sites
    that pass `&st->h[0]` -- those have already been rewritten in fn-body
    pass to `st.h[0]`.
    """
    # Only transform within #define blocks (possibly multi-line via \).
    out = []
    i = 0
    lines = code.split('\n')
    in_macro = False
    while i < len(lines):
        ln = lines[i]
        stripped = ln.lstrip()
        if not in_macro and stripped.startswith('#define '):
            in_macro = True
        if in_macro:
            # Rewrite `(IDENT)->member` -> `(IDENT).member`
            ln = re.sub(r'\(([a-zA-Z_][a-zA-Z0-9_]*)\)\s*->\s*', r'(\1).', ln)
        # Track continuation
        if in_macro:
            if ln.rstrip().endswith('\\'):
                pass  # still in macro
            else:
                in_macro = False
        out.append(ln)
        i += 1
    return '\n'.join(out)


_PRIMITIVE_SCALAR_TYPES = ('uint', 'int', 'ulong', 'long',
                           'ushort', 'short', 'uchar', 'char',
                           'float', 'double')


def _detect_scalar_ref_params(args: List[str], body_str: str,
                               extra_scalar_types: Optional[set] = None) -> List[Tuple[int, str, str]]:
    """For each `<primitive> *NAME` arg, detect whether it's used as scalar-by-
    pointer (deref `*NAME` appears in body) vs array-pointer (`NAME[i]` only).

    Returns list of (arg_index, type_token, param_name) for args that are
    scalar-ref candidates. Used by rewrite_functions() to convert OpenCL's
    `uint *hx, ..., uint *M` compress-function signatures into Metal's
    `thread uint &hx, ..., thread const uint *M` style.

    Triggered for primitive-typed single-star pointer args (md4_compress,
    md4_compress_md4utf16 are the canonical Phase 2d.2 cases). Non-primitive
    or already-qualified args are left untouched -- those are handled by
    _rewrite_arg (state-by-ptr -> thread T &).

    Phase 2d.8b: `extra_scalar_types` (overlay-provided) is a set of typedef
    names that should be treated as scalar-ref candidates alongside the
    canonical primitives. SHACRYPT's `sc_counter_t *counter` (typedef'd to
    uint or ulong depending on HASH_WORDS width) is the canonical example
    -- the translator needs to convert it to `thread sc_counter_t &counter`
    so that `rewrite_block_calls` correctly strips the `&` from `&counter`
    at call sites (matching the `&bufpos` treatment for the sibling `int
    *bufpos` arg). Without this, the call-site `&` strip happens for ALL
    args of any function in `discovered_scalar_state_helpers` (the set is
    fn-name-keyed, not per-arg) and `&counter` gets stripped while
    `counter` parameter type stays `*counter`, producing the type-mismatch
    error `no known conversion from 'sc_counter_t' to 'sc_counter_t *'`.
    """
    out = []
    arg_re = re.compile(
        r'^\s*(?P<pre>(?:const\s+)?)'
        r'(?P<type>[a-zA-Z_][a-zA-Z0-9_]*)\s*'
        r'\*\s*'
        r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*$'
    )
    extra = set(extra_scalar_types or ())
    for idx, a in enumerate(args):
        m = arg_re.match(a)
        if not m:
            continue
        type_tok = m.group('type')
        if type_tok not in _PRIMITIVE_SCALAR_TYPES and type_tok not in extra:
            continue
        # Skip if pre already has device/thread/etc. (shouldn't, but defensive)
        pre = m.group('pre').strip()
        if pre and pre not in ('const',):
            continue
        pname = m.group('name')
        # Detect deref *name (not preceded by alpha/underscore so we don't
        # mistake multiplications) -- look for *NAME with word boundary.
        # Use negative-lookbehind to avoid matching identifiers ending in '*'.
        deref_pat = re.compile(r'(?<![a-zA-Z0-9_*])\*\s*' + re.escape(pname) + r'\b')
        if deref_pat.search(body_str):
            out.append((idx, type_tok, pname))
    return out


def _rewrite_scalar_ref_body(body: str, scalar_names: List[str]) -> str:
    """Rewrite *NAME -> NAME for each scalar-ref param name in body.

    Only matches *NAME with the leading-* being a unary deref, not
    multiplication. The deref-detection regex mirrors the one in
    _detect_scalar_ref_params.
    """
    for n in scalar_names:
        pat = re.compile(r'(?<![a-zA-Z0-9_*])\*\s*' + re.escape(n) + r'\b')
        body = pat.sub(n, body)
    return body


def rewrite_functions(code: str, overlay: dict,
                      discovered_scalar_state_helpers: Optional[set] = None) -> str:
    """Pass through code, rewriting function signatures and bodies.

    discovered_scalar_state_helpers: if provided, the set is populated with
    the names of functions whose signature was rewritten to use scalar-by-
    reference args (md4_compress, md4_compress_md4utf16, etc.). These are
    used by rewrite_block_calls() to know which call sites need the
    leading `&` stripped from their args. Phase 2d.3.1 SHA-1 canary: this
    discovery channel lets the translator distinguish scalar-state helpers
    (md5_block, md4_compress) from pointer-state helpers (sha1_block,
    sha256_block) without a hardcoded list.
    """
    overlay_args = overlay.get('arg_address_space', {}) if overlay else {}
    # Phase 2d.8b: SHACRYPT overlay provides `extra_scalar_ref_types` -- a
    # list of typedef names (e.g. `sc_counter_t`) that the translator should
    # treat as scalar-ref candidates. See _detect_scalar_ref_params docstring
    # for the trap this avoids.
    extra_scalar_types = set(overlay.get('extra_scalar_ref_types', []) or []) if overlay else set()
    out = []
    pos = 0
    while True:
        m = _FN_HEAD_RE.search(code, pos)
        if not m:
            out.append(code[pos:])
            break
        # Verify this is a real fn def: must have body { ... } after the args.
        open_paren = m.end() - 1
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren < 0:
            out.append(code[pos:m.end()])
            pos = m.end()
            continue
        # Look for opening brace
        rest_start = close_paren + 1
        j = rest_start
        while j < len(code) and code[j].isspace():
            j += 1
        if j >= len(code) or code[j] != '{':
            # Maybe a declaration without body, skip
            out.append(code[pos:rest_start])
            pos = rest_start
            continue
        body_open = j
        body_close = _find_matching_brace(code, body_open)
        if body_close < 0:
            out.append(code[pos:])
            break

        fn_name = m.group('name')
        ret_type = m.group('ret').strip()
        lead = m.group('lead')
        args_str = code[open_paren + 1:close_paren]
        body_str = code[body_open + 1:body_close]

        # Rewrite each arg
        args = _split_args(args_str)

        # Pre-pass: detect primitive scalar-ref params (uint *hx where body
        # uses *hx). These are compress-function signatures like md4_compress.
        # We pre-rewrite the affected args (replacing them in the args list
        # in place) so _rewrite_arg doesn't downgrade them to `thread uint *`.
        scalar_refs = _detect_scalar_ref_params(args, body_str, extra_scalar_types)
        scalar_ref_names = []
        for idx, type_tok, pname in scalar_refs:
            args[idx] = 'thread ' + type_tok + ' &' + pname
            scalar_ref_names.append(pname)
        # Record discovered scalar-state helpers for rewrite_block_calls.
        if scalar_ref_names and discovered_scalar_state_helpers is not None:
            discovered_scalar_state_helpers.add(fn_name)

        state_names = []
        new_args = []
        for a in args:
            ra, sn = _rewrite_arg(a, fn_name, overlay_args, body_str)
            new_args.append(ra)
            if sn:
                state_names.append(sn)

        # Rewrite body: st->h[i] -> st.h[i]; &st->h[i] -> st.h[i]
        new_body = _rewrite_state_derefs(body_str, state_names)
        # Rewrite *NAME -> NAME for primitive scalar-ref params.
        new_body = _rewrite_scalar_ref_body(new_body, scalar_ref_names)
        # Rewrite local-var pointer decls + cast expressions in body.
        new_body = rewrite_local_ptr_decls(new_body)
        new_body = rewrite_pointer_casts(new_body)

        # Format the new signature.
        # Preserve the original arglist's line-break style: if original had
        # newlines, join with `,\n<indent>`; otherwise `, `.
        if '\n' in args_str:
            # Compute indent from the first line after the open paren.
            # Use leading whitespace of the second arg's original line.
            sep = ',\n' + ' ' * 32  # 32-space indent matches the hand-port style
            joined = sep.join(new_args)
        else:
            joined = ', '.join(new_args)

        new_sig = lead + ret_type + ' ' + fn_name + '(' + joined + ')'
        out.append(code[pos:m.start()])
        out.append(new_sig)
        out.append('\n{')
        out.append(new_body)
        out.append('}')
        pos = body_close + 1
    return ''.join(out)


# -----------------------------------------------------------------------------
# Pass 3: Per-kernel overlay -- function skips and line-range skips.
# -----------------------------------------------------------------------------

def apply_overlay_skip_functions(code: str, skip_names: list) -> str:
    """Remove entire static inline function definitions named in skip_names."""
    if not skip_names:
        return code
    out = []
    pos = 0
    while True:
        m = _FN_HEAD_RE.search(code, pos)
        if not m:
            out.append(code[pos:])
            break
        open_paren = m.end() - 1
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren < 0:
            out.append(code[pos:m.end()])
            pos = m.end()
            continue
        rest_start = close_paren + 1
        j = rest_start
        while j < len(code) and code[j].isspace():
            j += 1
        if j >= len(code) or code[j] != '{':
            out.append(code[pos:rest_start])
            pos = rest_start
            continue
        body_open = j
        body_close = _find_matching_brace(code, body_open)
        if body_close < 0:
            out.append(code[pos:])
            break
        fn_name = m.group('name')
        if fn_name in skip_names:
            # Skip this function. Preserve content before fn start, omit the
            # function (and any preceding doc comment is implicitly retained,
            # but we walk back over whitespace+comment so the omission is
            # cleaner).
            # Find a sensible cut point: backtrack over /* ... */ comments
            # immediately preceding `m.start()`.
            cut = m.start()
            # Look at code[pos:cut]: trim trailing blank-line + preceding
            # comment block.
            chunk = code[pos:cut]
            # Walk back over trailing whitespace
            trimmed = chunk.rstrip()
            # If it ends with */, walk back to matching /*
            if trimmed.endswith('*/'):
                idx = trimmed.rfind('/*')
                if idx >= 0:
                    trimmed = trimmed[:idx].rstrip()
            out.append(trimmed + '\n')
            pos = body_close + 1
            continue
        # Keep this function intact
        out.append(code[pos:body_close + 1])
        pos = body_close + 1
    return ''.join(out)


def apply_overlay_skip_typedef_structs(code: str, skip_names: list) -> str:
    """Remove typedef struct ... <name>; or struct <name> { ... }; blocks."""
    if not skip_names:
        return code
    out = []
    pos = 0
    while True:
        # First try typedef struct
        f = find_typedef_struct(code, pos)
        # And bare struct decl
        m_struct = re.search(r'(?m)^[ \t]*struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{', code[pos:])
        next_typedef = (f[0] if f else -1)
        next_struct = (m_struct.start() + pos if m_struct else -1)
        # Pick earliest
        candidates = []
        if next_typedef >= 0:
            candidates.append(('typedef', next_typedef, f))
        if next_struct >= 0:
            candidates.append(('struct', next_struct, m_struct))
        if not candidates:
            out.append(code[pos:])
            break
        candidates.sort(key=lambda x: x[1])
        kind, start, info = candidates[0]
        if kind == 'typedef':
            tstart, tend, name, _inner = info
            if name in skip_names:
                out.append(code[pos:tstart].rstrip() + '\n')
                pos = tend
                continue
            out.append(code[pos:tend])
            pos = tend
        else:
            # bare struct
            mm = info
            name = mm.group(1)
            brace_open = pos + mm.end() - 1
            brace_close = _find_matching_brace(code, brace_open)
            if brace_close < 0:
                out.append(code[pos:])
                break
            # walk to ;
            k = brace_close + 1
            while k < len(code) and code[k].isspace():
                k += 1
            if k < len(code) and code[k] == ';':
                end = k + 1
            else:
                end = brace_close + 1
            if name in skip_names:
                out.append(code[pos:pos + mm.start()].rstrip() + '\n')
                pos = end
                continue
            out.append(code[pos:end])
            pos = end
    return ''.join(out)


def apply_dual_addr_space_helpers(code: str, helpers: list) -> str:
    """Phase 2d.7b: emit a SECOND overload of each listed helper function
    with `data` arg promoted to `device const uchar *` (preserves the
    existing `thread const uchar *` overload). Metal's overload resolution
    picks the right one at each call site based on the address space of
    the passed pointer.

    `helpers` is a list of dicts, each:
        - name: function name to duplicate
        - arg: name of the pointer arg to flip (default: 'data')

    The matched function is identified by its already-translated
    `thread const <type> *<arg>` signature. The duplicate is appended
    immediately after the matched function with `device const` in place
    of `thread const`. Bodies are textually identical.

    Caught Phase 2d.7b Keccak/SHA-3 absorb helpers (keccak_absorb_full /
    keccak_absorb_pad called from both template_iterate hex[] thread
    buffer AND template_finalize device-const buf_scratch_pool slice).
    """
    if not helpers:
        return code
    # Normalise helper entries to (name, arg) tuples
    normalized = []
    for h in helpers:
        if isinstance(h, dict):
            n = h.get('name')
            a = h.get('arg', 'data')
            if n:
                normalized.append((n, a))
        else:
            normalized.append((str(h), 'data'))
    out = []
    pos = 0
    while True:
        m = _FN_HEAD_RE.search(code, pos)
        if not m:
            out.append(code[pos:])
            break
        open_paren = m.end() - 1
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren < 0:
            out.append(code[pos:m.end()])
            pos = m.end()
            continue
        rest_start = close_paren + 1
        j = rest_start
        while j < len(code) and code[j].isspace():
            j += 1
        if j >= len(code) or code[j] != '{':
            out.append(code[pos:rest_start])
            pos = rest_start
            continue
        body_open = j
        body_close = _find_matching_brace(code, body_open)
        if body_close < 0:
            out.append(code[pos:])
            break
        fn_name = m.group('name')
        match = next(((n, a) for (n, a) in normalized if n == fn_name), None)
        if match is None:
            # Keep this function intact and move on
            out.append(code[pos:body_close + 1])
            pos = body_close + 1
            continue
        # Emit original function + a duplicate with `thread const` -> `device const`
        # on the named arg. We rewrite the duplicate's signature only; body
        # text is preserved verbatim.
        _name, arg_name = match
        original_text = code[m.start():body_close + 1]
        # Locate the `thread const ... *<arg_name>` pattern in the signature.
        # Args may span multiple lines; restrict the substitution to the
        # arg-list slice.
        sig_open_paren_local = open_paren - m.start()
        sig_close_paren_local = close_paren - m.start()
        sig_args = original_text[sig_open_paren_local + 1:sig_close_paren_local]
        # Regex: `thread const <type> *<arg_name>` (handles whitespace
        # variations; allows the type token).
        dup_args = re.sub(
            r'thread\s+const\s+([A-Za-z_][A-Za-z_0-9]*)\s*\*\s*' +
            re.escape(arg_name) + r'\b',
            r'device const \1 *' + arg_name,
            sig_args, count=1)
        if dup_args == sig_args:
            # No replacement made — original sig didn't have `thread const ... *<arg>`.
            # Keep original intact, skip duplication (loud at smoke time).
            out.append(code[pos:body_close + 1])
            pos = body_close + 1
            continue
        dup_text = (original_text[:sig_open_paren_local + 1]
                    + dup_args
                    + original_text[sig_close_paren_local:])
        out.append(code[pos:body_close + 1])
        # Insert a brief comment + the duplicate immediately after.
        out.append('\n\n/* Phase 2d.7b: device-const overload (mirror of '
                   'thread-const above). Selected by Metal\'s overload '
                   'resolution when called from template_finalize\'s hot path '
                   'with a device-const slice of buf_scratch_pool. */\n')
        out.append(dup_text)
        pos = body_close + 1
    return ''.join(out)


def apply_overlay_skip_line_ranges(code: str, ranges: list) -> str:
    """Remove arbitrary line ranges from code (1-indexed). Used for content
    deletion within a function (e.g., dropping modes 1-6 from
    template_finalize)."""
    if not ranges:
        return code
    lines = code.split('\n')
    skip = [False] * (len(lines) + 1)
    for r in ranges:
        if isinstance(r, dict):
            s = int(r.get('start', 0))
            e = int(r.get('end', 0))
        else:
            # tolerate "start-end" string
            parts = str(r).split('-')
            s = int(parts[0]); e = int(parts[1]) if len(parts) > 1 else s
        for i in range(max(1, s), min(len(lines), e) + 1):
            skip[i] = True
    out_lines = []
    for i, ln in enumerate(lines, start=1):
        if skip[i]:
            continue
        out_lines.append(ln)
    return '\n'.join(out_lines)


# -----------------------------------------------------------------------------
# Header rewrite: replace RCS keywords in the leading /* ... */ block with the
# metal-flavoured equivalents. Specifically:
#   $Revision: <X> $   ->  $Revision: $
#   $Log: gpu_<name>_core.cl,v $   ->  $Log: metal_<name>_core.metal,v $
#   <RCS log expansion content>    ->  stripped (becomes "Initial check-in" on
#                                       first ci -l of the new file)
# -----------------------------------------------------------------------------

def rewrite_header(code: str, src_basename: str, dst_basename: str,
                   src_path_full: str) -> str:
    """Rewrite the top /* ... */ comment block to mirror the hand-port
    convention.
    """
    # Find first /* ... */ block at top
    m = re.match(r'\A\s*(/\*[\s\S]*?\*/)\s*\n', code)
    if not m:
        # No header to rewrite. Prepend a fresh stanza.
        stanza = _make_fresh_stanza(src_basename, dst_basename, src_path_full)
        return stanza + '\n' + code
    # Build a fresh stanza referencing the source file's revision.
    src_rev = _get_rcs_head_rev(src_path_full) or '?'
    stanza = _make_fresh_stanza(src_basename, dst_basename, src_path_full,
                                src_rev=src_rev)
    # Strip the matched first block
    return stanza + '\n' + code[m.end():]


def _get_rcs_head_rev(src_path_full: str) -> Optional[str]:
    """Grep the source file for `$Revision: N.M $` and return the rev string.
    No RCS shell invocation -- the file's expanded keyword is enough."""
    try:
        with open(src_path_full, 'r') as f:
            for line in f:
                m = re.search(r'\$Revision:\s*([0-9.]+)\s*\$', line)
                if m:
                    return m.group(1)
                if not line.startswith('/*') and not line.startswith(' *') and not line.strip().startswith('*'):
                    break
    except OSError:
        return None
    return None


def _make_fresh_stanza(src_basename: str, dst_basename: str, src_path_full: str,
                       src_rev: str = '?') -> str:
    # NOTE: build the RCS keyword markers at runtime via string concatenation
    # so that this source file (cl2metal.py) does NOT contain literal RCS
    # keyword tokens that RCS would expand at ci -l time. The generated
    # .metal output IS supposed to contain these keywords (so that the metal
    # file, if ci -l'd in a future Phase 2d.2+, gets its own history).
    dollar = '$'
    rev_marker = dollar + 'Revision: ' + dollar
    log_marker = dollar + 'Log: ' + dollar
    return (
        "/*\n"
        " * " + rev_marker + "\n"
        " * " + log_marker + "\n"
        " *\n"
        " * Auto-generated by gpu/codegen/cl2metal.py from\n"
        " *   " + src_basename + " (RCS rev " + str(src_rev) + ").\n"
        " *\n"
        " * Do NOT edit this file directly -- edit the OpenCL twin and\n"
        " * re-run cl2metal.py. The translator is intentionally lossless on\n"
        " * mechanical transforms; hand-curated Apple-specific tweaks live\n"
        " * in gpu/codegen/cl2metal_overrides/<algo>.yaml.\n"
        " *\n"
        " * Mirrors the byte-exact OpenCL hash chain; address-space port:\n"
        " *   __global  -> device\n"
        " *   __private -> thread\n"
        " *   barrier   -> threadgroup_barrier\n"
        " *   typedef struct ... T;  ->  struct T { ... };\n"
        " *   T *st     -> thread T &st\n"
        " *\n"
        " * Patterns 1/3 enforced: every pointer arg is address-space\n"
        " * qualified; every helper is static inline.\n"
        " */\n"
    )


# -----------------------------------------------------------------------------
# Post-pass: forbidden-token lint.
# -----------------------------------------------------------------------------

FORBIDDEN_TOKENS = [
    r'\b__global\b',
    r'\b__local\b',
    r'\b__private\b',
    r'\b__constant\b',
    r'\b__kernel\b',
    r'barrier\(\s*CLK_',
    r'\bas_uint\s*\(',
    r'\bmul_hi\s*\(',
    r'\btypedef\s+struct\b',
]


def lint_forbidden(code: str) -> List[Tuple[int, str]]:
    """Return [(line_no, token)] for each surviving forbidden token in 'code'
    regions (we ignore comments and strings)."""
    findings = []
    for tk_kind, tk_text in tokenize(code):
        if tk_kind != 'code':
            continue
        for pat in FORBIDDEN_TOKENS:
            for m in re.finditer(pat, tk_text):
                # Find line number in code where m starts
                idx = code.find(tk_text) + m.start()
                line_no = code.count('\n', 0, idx) + 1
                findings.append((line_no, pat))
    return findings


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def translate(src_path: str, overlay_path: Optional[str] = None) -> str:
    with open(src_path, 'r') as f:
        src = f.read()

    overlay = load_overlay(overlay_path) if overlay_path else {}

    # Pass 0: line-range skips applied to RAW source (line numbers reference
    # the input file).
    line_ranges = []
    skip_lines_entries = overlay.get('skip_line_ranges') or []
    for entry in skip_lines_entries:
        line_ranges.append(entry)
    if line_ranges:
        src = apply_overlay_skip_line_ranges(src, line_ranges)

    # Pass 0b: skip entire functions / typedefs / structs
    skip_blocks = overlay.get('skip_blocks') or []
    skip_names = []
    for b in skip_blocks:
        if isinstance(b, dict):
            n = b.get('identifier')
            if n:
                skip_names.append(n)
        else:
            skip_names.append(str(b))
    skip_functions = overlay.get('skip_functions') or []
    for n in skip_functions:
        skip_names.append(n if isinstance(n, str) else n.get('identifier'))

    if skip_names:
        src = apply_overlay_skip_functions(src, skip_names)
        src = apply_overlay_skip_typedef_structs(src, skip_names)

    # Pass 1: tokenize + substitute
    tokens = list(tokenize(src))
    tokens = substitute_pass(tokens)
    code = ''.join(t[1] for t in tokens)

    # Pass 2: structural rewrites
    code = rewrite_typedef_structs(code)
    # Phase 2d.3.1: discover scalar-state helpers via fn-sig rewrite pass.
    # Populated by rewrite_functions() when _detect_scalar_ref_params rewrites
    # a fn signature; consumed by rewrite_block_calls() to know which call
    # sites need their leading `&` stripped (md5_block, md4_compress) vs
    # left intact (sha1_block, sha256_block — pointer-state).
    discovered_scalar_state_helpers = set()
    code = rewrite_functions(code, overlay, discovered_scalar_state_helpers)
    code = rewrite_macro_state_derefs(code)
    code = rewrite_block_calls(code, discovered_scalar_state_helpers)

    # Phase 2d.7b: dual address-space overload emission. For helpers called
    # from BOTH template_iterate (thread-state path) AND template_finalize
    # (device-const buf_scratch_pool slice) with the same pointer arg,
    # Metal forbids generic-address-space pointers in fn signatures (no
    # __generic). Solution: emit TWO function defs sharing the same body,
    # one with `thread const` on the named arg and one with `device const`.
    # Metal's overload resolution picks the right one at each call site by
    # the arg's address space. Pattern documented in
    # feedback_metal_dual_addrspace_overload.md (caught Phase 2d.7a Blake2;
    # applied Phase 2d.7b Keccak/SHA-3 absorb helpers).
    dual_helpers = overlay.get('dual_addr_space_helpers') or []
    if dual_helpers:
        code = apply_dual_addr_space_helpers(code, dual_helpers)

    # Pass 3: header rewrite
    src_basename = os.path.basename(src_path)
    algo = overlay.get('algo')
    if algo:
        dst_basename = f"metal_{algo}_core.metal"
    else:
        # Derive: gpu_md5_core.cl -> metal_md5_core.metal
        m = re.match(r'gpu_(.+)_core\.cl$', src_basename)
        if m:
            dst_basename = f"metal_{m.group(1)}_core.metal"
        else:
            dst_basename = src_basename.replace('.cl', '.metal')

    code = rewrite_header(code, src_basename, dst_basename, src_path)

    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Path to gpu/gpu_<algo>_core.cl')
    ap.add_argument('-o', '--output', help='Output path (default: <input>.metal.generated)')
    ap.add_argument('--overlay', help='Path to overlay YAML '
                    '(default: gpu/codegen/cl2metal_overrides/<algo>.yaml)')
    ap.add_argument('--diff', help='If given, diff translator output against this file')
    ap.add_argument('--check', action='store_true',
                    help='Run forbidden-token lint and exit nonzero if any survive')
    args = ap.parse_args(argv)

    src_path = args.input
    src_basename = os.path.basename(src_path)
    m = re.match(r'gpu_(.+)_core\.cl$', src_basename)
    algo = m.group(1) if m else None

    overlay_path = args.overlay
    if not overlay_path and algo:
        default_overlay = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'cl2metal_overrides', f'{algo}.yaml')
        if os.path.exists(default_overlay):
            overlay_path = default_overlay

    output = translate(src_path, overlay_path)

    out_path = args.output
    if not out_path:
        out_path = src_path + '.metal.generated'

    with open(out_path, 'w') as f:
        f.write(output)
    sys.stderr.write(f"cl2metal: wrote {out_path} ({len(output)} bytes, {output.count(chr(10))} lines)\n")
    if overlay_path:
        sys.stderr.write(f"cl2metal: overlay {overlay_path}\n")

    rc = 0
    if args.check:
        findings = lint_forbidden(output)
        if findings:
            sys.stderr.write(f"cl2metal: FORBIDDEN tokens survived:\n")
            for ln, tok in findings[:20]:
                sys.stderr.write(f"  line {ln}: matches /{tok}/\n")
            rc = 2

    if args.diff:
        with open(args.diff, 'r') as f:
            target = f.read()
        delta = _loc_delta(output, target)
        sys.stderr.write(f"cl2metal: LOC delta vs {args.diff}: "
                         f"generated={delta['generated']} target={delta['target']} "
                         f"abs_diff={delta['abs_diff']} pct={delta['pct']:.2f}%\n")

    return rc


def _loc_delta(generated: str, target: str) -> dict:
    g = generated.count('\n') + 1
    t = target.count('\n') + 1
    ad = abs(g - t)
    pct = 100.0 * ad / max(t, 1)
    return {'generated': g, 'target': t, 'abs_diff': ad, 'pct': pct}


if __name__ == '__main__':
    sys.exit(main())
