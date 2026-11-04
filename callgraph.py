#!/usr/bin/env python3
"""A call graph: many functions, calls between them, and the tail call.

Studio, 2026-11-04. Yesterday (calls.py) one self-recursive function proved the
two-stacks point: control can ride C's call stack, the data stack in $sp cannot.
Every jal there targeted the same entry, so "resolve an address to a C function"
was trivial. This file makes it non-trivial.

Functions are discovered the way the simplest static recompilers do it: the
program entry plus every jal target is a function entry, and a function runs
from its entry to the next one. Each becomes its own C function over one shared
global register file.

The test is the tail call. An optimising compiler does not end is_even with
`jal is_odd; jr $ra`. It branches straight into is_odd's body and lets is_odd's
own `jr $ra` return to is_even's caller. On the machine that is just a branch;
there was never a wall between the two functions. In C there is: a `goto` cannot
leave the function it is written in.

Three programs, one reference. The interpreter knows nothing about functions,
only addresses, so it is the independent record for all of them.
"""
import subprocess

from mips import (addiu, addu, beq, disasm, jal, jr, lw, nop, pretty, slt, sw)

SENTINEL = 0xFFFFFFFF
STACK_TOP = 0x8000
MEMSZ = 65536


# ---- a two-pass assembler with labels -------------------------------------
def assemble(prog):
    """prog: list of strings (labels, ending in ':') and tuples (op, *args).
    Branch/jal targets may be label names. Returns (words, labels)."""
    labels, pc = {}, 0
    for item in prog:
        if isinstance(item, str):
            labels[item.rstrip(":")] = pc
        else:
            pc += 4
    words, pc = [], 0
    for item in prog:
        if isinstance(item, str):
            continue
        op, *a = item
        if op in ("beq", "bne"):
            off = (labels[a[2]] - (pc + 4)) // 4
            words.append(beq(a[0], a[1], off) if op == "beq" else
                         __import__("mips").bne(a[0], a[1], off))
        elif op == "jal":
            words.append(jal(labels[a[0]]))
        else:
            words.append({"addiu": addiu, "addu": addu, "slt": slt, "lw": lw,
                          "sw": sw, "jr": jr, "nop": nop}[op](*a))
        pc += 4
    return words, labels


def b(label):                      # unconditional branch, the tail-call shape
    return ("beq", "zero", "zero", label)


# main(n) = is_even(n) + 2*is_odd(n). Calling BOTH from main is what makes is_odd
# a discovered function entry, which is what puts a wall between it and is_even.
MAIN = [
    "main:",
    ("addiu", "sp", "sp", -12),
    ("sw", "ra", 8, "sp"), ("sw", "s0", 4, "sp"), ("sw", "a0", 0, "sp"),
    ("jal", "is_even"), ("nop",),
    ("addu", "s0", "v0", "zero"),          # s0 = is_even(n)
    ("lw", "a0", 0, "sp"),
    ("jal", "is_odd"), ("nop",),
    ("addu", "v0", "v0", "v0"),            # 2*is_odd(n)
    ("addu", "v0", "v0", "s0"),
    ("lw", "ra", 8, "sp"), ("lw", "s0", 4, "sp"),
    ("addiu", "sp", "sp", 12),
    ("jr", "ra"), ("nop",),
]

# Version A: ordinary calls. Each function saves $ra, calls the other, returns.
CALLS = MAIN + [
    "is_even:",
    ("beq", "a0", "zero", "even_base"), ("nop",),
    ("addiu", "sp", "sp", -4), ("sw", "ra", 0, "sp"),
    ("addiu", "a0", "a0", -1),
    ("jal", "is_odd"), ("nop",),
    ("lw", "ra", 0, "sp"), ("addiu", "sp", "sp", 4),
    ("jr", "ra"), ("nop",),
    "even_base:",
    ("addiu", "v0", "zero", 1), ("jr", "ra"), ("nop",),
    "is_odd:",
    ("beq", "a0", "zero", "odd_base"), ("nop",),
    ("addiu", "sp", "sp", -4), ("sw", "ra", 0, "sp"),
    ("addiu", "a0", "a0", -1),
    ("jal", "is_even"), ("nop",),
    ("lw", "ra", 0, "sp"), ("addiu", "sp", "sp", 4),
    ("jr", "ra"), ("nop",),
    "odd_base:",
    ("addiu", "v0", "zero", 0), ("jr", "ra"), ("nop",),
]

# Version B: tail calls. No frames, no saved $ra: each function branches into
# the other, and whichever one hits its base case returns to the ORIGINAL caller.
TAIL = MAIN + [
    "is_even:",
    ("beq", "a0", "zero", "even_base"), ("nop",),
    ("addiu", "a0", "a0", -1),
    b("is_odd"), ("nop",),                 # tail call: a branch, not a jal
    "even_base:",
    ("addiu", "v0", "zero", 1), ("jr", "ra"), ("nop",),
    "is_odd:",
    ("beq", "a0", "zero", "odd_base"), ("nop",),
    ("addiu", "a0", "a0", -1),
    b("is_even"), ("nop",),                # and back
    "odd_base:",
    ("addiu", "v0", "zero", 0), ("jr", "ra"), ("nop",),
]


# ---- the reference: addresses only, no functions --------------------------
def _be_lw(m, a):
    return (m[a] << 24) | (m[a+1] << 16) | (m[a+2] << 8) | m[a+3]


def _be_sw(m, a, v):
    v &= 0xFFFFFFFF
    m[a:a+4] = bytes(((v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255))


def _exec(words, reg, mem, pc):
    m, o, _ = disasm(words[pc // 4], pc)

    def wr(d, v):
        if d:
            reg[d] = v & 0xFFFFFFFF
    if m == "nop": return
    if m == "addu":   wr(o[0], reg[o[1]] + reg[o[2]])
    elif m == "addiu": wr(o[0], reg[o[1]] + o[2])
    elif m == "slt":  wr(o[0], int((reg[o[1]] ^ 0x80000000) < (reg[o[2]] ^ 0x80000000)))
    elif m == "lw":   wr(o[0], _be_lw(mem, reg[o[2]] + o[1]))
    elif m == "sw":   _be_sw(mem, reg[o[2]] + o[1], reg[o[0]])
    else: raise ValueError(m)


def interp(words, a0, max_steps=50_000_000):
    reg, mem = [0] * 32, bytearray(MEMSZ)
    reg[4], reg[29], reg[31] = a0, STACK_TOP, SENTINEL
    pc, lowest_sp = 0, STACK_TOP
    for _ in range(max_steps):
        if pc == SENTINEL:
            return reg[2], STACK_TOP - lowest_sp
        m, o, tgt = disasm(words[pc // 4], pc)
        if m == "jal":
            reg[31] = pc + 8; _exec(words, reg, mem, pc + 4); pc = tgt
        elif m == "jr":
            _exec(words, reg, mem, pc + 4); pc = reg[o[0]]
        elif m in ("beq", "bne"):
            take = (reg[o[0]] == reg[o[1]]) == (m == "beq")
            _exec(words, reg, mem, pc + 4); pc = tgt if take else pc + 8
        else:
            _exec(words, reg, mem, pc); pc += 4
        lowest_sp = min(lowest_sp, reg[29])
    raise RuntimeError("did not halt")


# ---- the recompiler: one C function per discovered entry ------------------
def functions(words):
    """Entry 0 plus every jal target. A function runs to the next entry."""
    entries = {0} | {disasm(w, i * 4)[2] for i, w in enumerate(words)
                     if disasm(w, i * 4)[0] == "jal"}
    es = sorted(entries)
    return [(e, (es[i + 1] if i + 1 < len(es) else len(words) * 4)) for i, e in enumerate(es)]


def _stmt(w, pc):
    m, o, _ = disasm(w, pc)
    return {"nop": lambda: "/* nop */",
            "addu": lambda: f"W({o[0]}, r[{o[1]}] + r[{o[2]}]);",
            "addiu": lambda: f"W({o[0]}, r[{o[1]}] + {o[2]});",
            "slt": lambda: f"W({o[0]}, (int32_t)r[{o[1]}] < (int32_t)r[{o[2]}]);",
            "lw": lambda: f"W({o[0]}, LW(r[{o[2]}] + {o[1]}));",
            "sw": lambda: f"SW(r[{o[2]}] + {o[1]}, r[{o[0]}]);"}[m]()


def emit(words, tailcalls=True):
    """tailcalls=False is the naive emitter: every branch becomes a goto, even
    one whose target lies in a different C function."""
    fns = functions(words)
    entry_of = {e for e, _ in fns}
    out = [f"static void fn_{e:04x}(void);" for e, _ in fns]
    for start, end in fns:
        labs = set()
        for pc in range(start, end, 4):
            m, _, t = disasm(words[pc // 4], pc)
            if m in ("beq", "bne") and (not tailcalls or start <= t < end):
                labs.add(t)
        out.append(f"static void fn_{start:04x}(void) {{\n  uint32_t *r = RF, cond; (void)cond;")
        pc = start
        while pc < end:
            w = words[pc // 4]
            m, o, t = disasm(w, pc)
            if pc in labs:
                out.append(f"L_{pc:04x}:")
            if m in ("beq", "bne"):
                c = f"r[{o[0]}] {'==' if m == 'beq' else '!='} r[{o[1]}]"
                out.append(f"  cond = ({c});   /* {pretty(w, pc)} */")
                out.append(f"  {_stmt(words[pc//4 + 1], pc + 4)}   /* delay slot */")
                if start <= t < end or not tailcalls:
                    out.append(f"  if (cond) goto L_{t:04x};")
                elif t in entry_of:
                    # A branch into another function's entry is a tail call:
                    # call it, then return, so its `jr $ra` reaches OUR caller.
                    out.append(f"  if (cond) {{ fn_{t:04x}(); return; }}   /* tail call */")
                else:
                    raise ValueError(f"branch at {pc:#x} into the middle of another function ({t:#x})")
                pc += 8
                continue
            if m == "jal":
                out.append(f"  {_stmt(words[pc//4 + 1], pc + 4)}   /* delay slot */")
                out.append(f"  fn_{t:04x}();   /* {pretty(w, pc)} */")
                pc += 8
                continue
            if m == "jr":
                out.append(f"  {_stmt(words[pc//4 + 1], pc + 4)}   /* delay slot */")
                out.append("  return;")
                pc += 8
                continue
            out.append(f"  {_stmt(w, pc)}".ljust(40) + f"/* {pretty(w, pc)} */")
            pc += 4
        out.append("}")
    return "\n".join(out)


def regions(words):
    """Group functions that branch into each other. A jal target is where a
    CALL lands; it is not a wall. Any branch from one discovered function into
    another (a tail call, or a jump into a shared epilogue) welds the two into
    one region, which becomes ONE C function, so the branch can stay a goto."""
    fns = functions(words)
    parent = list(range(len(fns)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def owner(addr):
        return next(i for i, (s, e) in enumerate(fns) if s <= addr < e)
    for i, (s, e) in enumerate(fns):
        for pc in range(s, e, 4):
            m, _, t = disasm(words[pc // 4], pc)
            if m in ("beq", "bne") and not (s <= t < e):
                parent[find(i)] = find(owner(t))
    groups = {}
    for i, f in enumerate(fns):
        groups.setdefault(find(i), []).append(f)
    return [sorted(g) for g in groups.values()], fns


def emit_regions(words):
    """One C function per region, with a switch for its entry points. A tail
    branch is a goto inside the region, so it costs no C stack at any -O level."""
    groups, fns = regions(words)
    out = []
    for k, g in enumerate(groups):
        out.append(f"static void region_{k}(int entry);")
        for idx, (s, _) in enumerate(g):
            out.append(f"static void fn_{s:04x}(void) {{ region_{k}({idx}); }}")
    for k, g in enumerate(groups):
        labs = {s for s, _ in g}
        for s, e in g:
            for pc in range(s, e, 4):
                m, _, t = disasm(words[pc // 4], pc)
                if m in ("beq", "bne"):
                    labs.add(t)
        out.append(f"static void region_{k}(int entry) {{\n  uint32_t *r = RF, cond; (void)cond;")
        out.append("  switch (entry) {" + " ".join(f"case {i}: goto L_{s:04x};" for i, (s, _) in enumerate(g)) + " }")
        for s, e in g:
            pc = s
            while pc < e:
                w = words[pc // 4]
                m, o, t = disasm(w, pc)
                if pc in labs:
                    out.append(f"L_{pc:04x}:")
                if m in ("beq", "bne"):
                    c = f"r[{o[0]}] {'==' if m == 'beq' else '!='} r[{o[1]}]"
                    out.append(f"  cond = ({c});   /* {pretty(w, pc)} */")
                    out.append(f"  {_stmt(words[pc//4 + 1], pc + 4)}   /* delay slot */")
                    out.append(f"  if (cond) goto L_{t:04x};")
                    pc += 8
                    continue
                if m in ("jal", "jr"):
                    out.append(f"  {_stmt(words[pc//4 + 1], pc + 4)}   /* delay slot */")
                    out.append(f"  fn_{t:04x}();   /* {pretty(w, pc)} */" if m == "jal" else "  return;")
                    pc += 8
                    continue
                out.append(f"  {_stmt(w, pc)}".ljust(40) + f"/* {pretty(w, pc)} */")
                pc += 4
        out.append("}")
    return "\n".join(out)


PRELUDE = r"""#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W(d, v) do { uint32_t _v = (uint32_t)(v); if ((d) != 0) r[(d)] = _v; } while (0)
static uint8_t mem[65536];
static uint32_t RF[32];
#define LW(a) ((uint32_t)mem[(a)]<<24 | (uint32_t)mem[(a)+1]<<16 | (uint32_t)mem[(a)+2]<<8 | (uint32_t)mem[(a)+3])
#define SW(a,v) do { uint32_t _a=(a), _v=(uint32_t)(v); mem[_a]=_v>>24; mem[_a+1]=_v>>16; mem[_a+2]=_v>>8; mem[_a+3]=_v; } while (0)
"""
MAIN_C = r"""
int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    memset(mem, 0, sizeof mem); memset(RF, 0, sizeof RF);
    RF[29] = 0x8000; RF[4] = (uint32_t)strtoul(argv[i], 0, 10);
    fn_0000();
    printf("%u\n", RF[2]);
  }
  return 0;
}
"""


def build(words, name, tailcalls=True, opt="-O2"):
    src = PRELUDE + emit(words, tailcalls) + MAIN_C
    open(f"{name}.c", "w").write(src)
    p = subprocess.run(["gcc", opt, "-o", name, f"{name}.c"], capture_output=True, text=True)
    return p.returncode == 0, p.stderr


def run(name, ns, timeout=60):
    p = subprocess.run([f"./{name}", *map(str, ns)], capture_output=True, text=True, timeout=timeout)
    return [int(x) for x in p.stdout.split()], p.returncode


if __name__ == "__main__":
    want = lambda n: 1 if n % 2 == 0 else 2
    ns = list(range(0, 31))
    for label, prog in (("calls", CALLS), ("tail", TAIL)):
        words, labels = assemble(prog)
        print(f"\n== {label}: {len(words)} words, functions at",
              ", ".join(f"{e:#x}" for e, _ in functions(words)))
        ref = [interp(words, n)[0] for n in ns]
        assert ref == [want(n) for n in ns], "interpreter disagrees with the definition"
        print("   interp == definition on n=0..30")
        for tc in (False, True):
            ok, err = build(words, f"cg_{label}_{'tc' if tc else 'naive'}", tailcalls=tc)
            tag = "tail-call emitter" if tc else "naive emitter   "
            if not ok:
                first = next((l for l in err.splitlines() if "error" in l), err.strip()[:200])
                print(f"   {tag}: DOES NOT COMPILE -> {first.split('error:')[-1].strip()}")
                continue
            out, rc = run(f"cg_{label}_{'tc' if tc else 'naive'}", ns)
            print(f"   {tag}: compiles, {'matches interp' if out == ref else 'MISMATCH ' + str(out)}")
        if label == "tail":
            # The tail-call emitter is correct only if gcc turns `fn(); return;`
            # back into a jump. Measured: -O0/-O1 overflow the 8 MB C stack at
            # n=1e6 (the MIPS program needs 12 bytes for any n); -O2 survives
            # via sibling calls OR inlining, and dies with both switched off.
            big = 1_000_000
            for opt in ("-O0", "-O2"):
                build(words, "cg_tail_big", tailcalls=True, opt=opt)
                out, rc = run("cg_tail_big", [big])
                print(f"   tail-call emitter {opt}, n=1e6: {'ok' if out == [1] else f'exit {rc} (stack overflow)'}")
            open("cg_tail_reg.c", "w").write(PRELUDE + emit_regions(words) + MAIN_C)
            for opt in ("-O0", "-O2"):
                subprocess.run(["gcc", opt, "-o", "cg_tail_reg", "cg_tail_reg.c"], check=True)
                small, _ = run("cg_tail_reg", ns)
                out, rc = run("cg_tail_reg", [big])
                print(f"   region emitter    {opt}: n=0..30 {'matches' if small == ref else 'MISMATCH'}, "
                      f"n=1e6 {'ok' if out == [1] else f'exit {rc}'}")
