#!/usr/bin/env python3
"""Real function calls: jal / jr $ra, and the two-stacks finding, MEASURED.

Studio, 2026-11-03. Yesterday I added the jal DECODER and reasoned my way to a
claim I had not yet tested: that recompiling a MIPS call as a C call forces you
to thread the stack pointer, because the MIPS stack (the one $sp walks through
memory) and the C call stack are two different stacks. This file turns the claim
into a measurement.

The experiment is a single self-recursive function, Fibonacci, assembled by hand
from mips.py, run three ways:

  1. interp_calls  -- the faithful reference. ONE register file, ONE memory,
                      shared across every call. jal writes $ra; jr $ra reads it.
                      The outermost return address is a SENTINEL; when jr lands
                      on it, the program has returned to the caller that does not
                      exist, i.e. it is done. This is how a real emulator handles
                      the top of the stack, and it is the second independent
                      record every result below is checked against.

  2. emit_calls(mode="global")  -- the CORRECT recompiler. One global register
                      file RF[32] shared by every C invocation, so $sp (RF[29])
                      is a single value that marches downward as the C recursion
                      nests. jal -> C call; jr $ra -> C return. Correctness comes
                      entirely from the program's own save/restore discipline
                      operating on that one shared $sp -- exactly as on hardware,
                      where there is also only one register file.

  3. emit_calls(mode="local")   -- the NAIVE recompiler, the mistake that looks
                      natural: give each C function its own local uint32_t
                      r[32]. Then $sp is reset to the same top on entry to every
                      invocation, so every frame lands at the same memory
                      addresses. The recursion's own bookkeeping -- the saved
                      $ra, the saved $s0, the reloaded n -- collides. Kept so the
                      difference is measured, not asserted.

Both recompilers share the same big-endian memory; the ONLY thing that differs
is whether the register file (and so $sp) is one object or one-per-call. That is
the whole experiment: a single-variable contrast.
"""
import subprocess
import sys

from mips import (RNAME, addiu, addu, beq, disasm, jal, jr, lw, nop, pretty, r,
                  slt, sw)

SENTINEL = 0xFFFFFFFF   # a return address that means "returned past the top"
STACK_TOP = 0x8000      # $sp starts here and grows downward; code lives near 0


# ---- the program: recursive Fibonacci -------------------------------------
# fib(n): if n < 2 return n; else return fib(n-1) + fib(n-2).
# It saves $ra (so a nested call does not lose the return), saves $s0 and uses
# it to carry fib(n-1) across the SECOND call (a callee-saved register, which is
# exactly the thing a collided stack destroys), and reloads n from the stack
# after the first call. Every one of those three leans on $sp being one value.
#
#  addr  instruction                     meaning
#  0x00  addiu $sp,$sp,-12                allocate a 3-word frame
#  0x04  sw    $ra,8($sp)                 save return address
#  0x08  sw    $s0,4($sp)                 save callee-saved $s0
#  0x0c  sw    $a0,0($sp)                 save n
#  0x10  addiu $t0,$zero,2                $t0 = 2
#  0x14  slt   $t1,$a0,$t0                $t1 = (n < 2)   (signed; n is small)
#  0x18  beq   $t1,$zero,0x2c             if !(n<2) goto RECURSE
#  0x1c  nop                              (delay slot)
#  0x20  addu  $v0,$a0,$zero              base case: $v0 = n
#  0x24  beq   $zero,$zero,0x50           goto DONE (unconditional)
#  0x28  nop
# RECURSE (0x2c):
#  0x2c  addiu $a0,$a0,-1                 $a0 = n-1
#  0x30  jal   fib (0x00)                 $v0 = fib(n-1)
#  0x34  nop
#  0x38  addu  $s0,$v0,$zero              $s0 = fib(n-1)  (must survive next call)
#  0x3c  lw    $a0,0($sp)                 reload n
#  0x40  addiu $a0,$a0,-2                 $a0 = n-2
#  0x44  jal   fib (0x00)                 $v0 = fib(n-2)
#  0x48  nop
#  0x4c  addu  $v0,$v0,$s0                $v0 = fib(n-2) + fib(n-1)
# DONE (0x50):
#  0x50  lw    $ra,8($sp)                 restore
#  0x54  lw    $s0,4($sp)
#  0x58  addiu $sp,$sp,12                 pop the frame
#  0x5c  jr    $ra
#  0x60  nop

def fib_program():
    SP, RA, S0, A0, T0, T1, V0, Z = 29, 31, 16, 4, 8, 9, 2, 0

    def off(frm_idx, to_addr):
        # beq/bne offset is measured in instructions from the delay slot.
        return (to_addr - ((frm_idx * 4) + 4)) // 4

    return [
        addiu("sp", "sp", -12),          # 0x00
        sw("ra", 8, "sp"),               # 0x04
        sw("s0", 4, "sp"),               # 0x08
        sw("a0", 0, "sp"),               # 0x0c
        addiu("t0", "zero", 2),          # 0x10
        slt("t1", "a0", "t0"),           # 0x14
        beq("t1", "zero", off(6, 0x2c)), # 0x18 -> RECURSE
        nop(),                           # 0x1c
        addu("v0", "a0", "zero"),        # 0x20  base: v0 = n
        beq("zero", "zero", off(9, 0x50)),  # 0x24 -> DONE
        nop(),                           # 0x28
        addiu("a0", "a0", -1),           # 0x2c  a0 = n-1
        jal(0x00),                       # 0x30  fib(n-1)
        nop(),                           # 0x34
        addu("s0", "v0", "zero"),        # 0x38  s0 = fib(n-1)
        lw("a0", 0, "sp"),               # 0x3c  reload n
        addiu("a0", "a0", -2),           # 0x40  a0 = n-2
        jal(0x00),                       # 0x44  fib(n-2)
        nop(),                           # 0x48
        addu("v0", "v0", "s0"),          # 0x4c  v0 = fib(n-2)+fib(n-1)
        lw("ra", 8, "sp"),               # 0x50  DONE: restore
        lw("s0", 4, "sp"),               # 0x54
        addiu("sp", "sp", 12),           # 0x58  pop
        jr("ra"),                        # 0x5c
        nop(),                           # 0x60
    ]


# ---- the faithful interpreter (one register file, one memory) --------------
MEMSZ = 65536


def _be_lw(mem, a):
    return (mem[a] << 24) | (mem[a + 1] << 16) | (mem[a + 2] << 8) | mem[a + 3]


def _be_sw(mem, a, v):
    v &= 0xFFFFFFFF
    mem[a], mem[a + 1], mem[a + 2], mem[a + 3] = \
        (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF


def _exec(words, reg, mem, base, pc):
    """Execute one non-control instruction at pc (also used for delay slots)."""
    w = words[(pc - base) // 4]
    m, o, _ = disasm(w, pc)

    def wr(d, v):
        if d != 0:
            reg[d] = v & 0xFFFFFFFF

    if m == "nop":   return
    if m == "addu":  wr(o[0], reg[o[1]] + reg[o[2]])
    elif m == "subu":  wr(o[0], reg[o[1]] - reg[o[2]])
    elif m == "addiu": wr(o[0], reg[o[1]] + o[2])
    elif m == "slt":   wr(o[0], int((reg[o[1]] ^ 0x80000000) < (reg[o[2]] ^ 0x80000000)))
    elif m == "lw":    wr(o[0], _be_lw(mem, reg[o[2]] + o[1]))
    elif m == "sw":    _be_sw(mem, reg[o[2]] + o[1], reg[o[0]])
    else: raise ValueError(f"_exec cannot run {m}")


def interp_calls(words, a0, base=0):
    """Reference. jal saves $ra and jumps; jr jumps to its register; landing on
    the SENTINEL return address means the top-level call has returned. One
    register file and one memory, shared by every (conceptual) call."""
    reg = [0] * 32
    mem = bytearray(MEMSZ)
    reg[4] = a0
    reg[29] = STACK_TOP
    reg[31] = SENTINEL
    pc, steps = base, 0
    while steps < 2_000_000:
        steps += 1
        if pc == SENTINEL:
            return reg[2]
        w = words[(pc - base) // 4]
        m, o, tgt = disasm(w, pc)
        if m == "jal":
            reg[31] = pc + 8          # return to the instruction after the delay slot
            _exec(words, reg, mem, base, pc + 4)   # delay slot runs first
            pc = tgt
            continue
        if m == "jr":
            _exec(words, reg, mem, base, pc + 4)   # delay slot
            pc = reg[o[0]]            # $ra or any register; sentinel-halt at the top
            continue
        if m in ("beq", "bne"):
            take = (reg[o[0]] == reg[o[1]]) if m == "beq" else (reg[o[0]] != reg[o[1]])
            _exec(words, reg, mem, base, pc + 4)
            pc = tgt if take else pc + 8
            continue
        _exec(words, reg, mem, base, pc)
        pc = pc + 4
    raise RuntimeError("interpreter did not halt")


# ---- the two recompilers ---------------------------------------------------
def _stmt(w, pc):
    m, o, _ = disasm(w, pc)
    if m == "nop":   return "/* nop */"
    if m == "addu":  return f"W({o[0]}, r[{o[1]}] + r[{o[2]}]);"
    if m == "subu":  return f"W({o[0]}, r[{o[1]}] - r[{o[2]}]);"
    if m == "addiu": return f"W({o[0]}, r[{o[1]}] + {o[2]});"
    if m == "slt":   return f"W({o[0]}, (int32_t)r[{o[1]}] < (int32_t)r[{o[2]}]);"
    if m == "lw":    return f"W({o[0]}, LW(r[{o[2]}] + {o[1]}));"
    if m == "sw":    return f"SW(r[{o[2]}] + {o[1]}, r[{o[0]}]);"
    raise ValueError(f"no statement form for {m}")


def _branch_labels(words, base):
    labs = set()
    for i, w in enumerate(words):
        m, _, t = disasm(w, base + i * 4)
        if m in ("beq", "bne") and t is not None:
            labs.add(t)
    return labs


def emit_calls(words, base=0, mode="global"):
    """mode='global': one shared register file, jal->C call, jr->C return. The
    correct recompiler. mode='local': a fresh r[32] per call, so $sp resets every
    invocation and frames collide. The naive one, kept to measure the gap."""
    labs = _branch_labels(words, base)
    call = "fib"
    if mode == "global":
        sig = f"static void {call}(void) {{"
        head = ["  uint32_t *r = RF;", "  uint32_t cond; (void)cond;"]
        do_call = f"  {call}();"
        do_ret = "  return;"
    else:
        sig = f"static uint32_t {call}(uint32_t a0) {{"
        head = ["  uint32_t r[32] = {0};",
                "  r[4] = a0;",
                "  r[29] = STACK_TOP;   /* each call sets up its OWN stack -- the bug */",
                "  uint32_t cond; (void)cond;"]
        do_call = f"  r[2] = {call}(r[4]);"
        do_ret = "  return r[2];"

    L = [sig] + head
    i = 0
    while i < len(words):
        pc = base + i * 4
        w = words[i]
        if pc in labs:
            L.append(f"L_{pc:04x}:")
        m, o, tgt = disasm(w, pc)
        if m in ("beq", "bne"):
            op = "==" if m == "beq" else "!="
            L.append(f"  cond = (r[{o[0]}] {op} r[{o[1]}]);   /* {pretty(w, pc)} */")
            L.append(f"  {_stmt(words[i+1], pc+4)}   /* delay slot */")
            L.append(f"  if (cond) goto L_{tgt:04x};")
            i += 2
            continue
        if m == "jal":
            L.append(f"  {_stmt(words[i+1], pc+4)}   /* delay slot */")
            L.append(f"{do_call}   /* {pretty(w, pc)} */")
            i += 2
            continue
        if m == "jr":
            L.append(f"  {_stmt(words[i+1], pc+4)}   /* delay slot */")
            L.append(f"{do_ret}   /* {pretty(w, pc)} */")
            i += 2
            continue
        L.append(f"  {_stmt(w, pc)}".ljust(44) + f"/* {pretty(w, pc)} */")
        i += 1
    L.append("}")
    return "\n".join(L)


PRELUDE = """#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W(d, v) do { uint32_t _v = (uint32_t)(v); if ((d) != 0) r[(d)] = _v; } while (0)
#define MEMSZ 65536
#define STACK_TOP 0x8000u
static uint8_t mem[MEMSZ];
#define LW(a)   ((uint32_t)mem[(a)]<<24 | (uint32_t)mem[(a)+1]<<16 \\
                 | (uint32_t)mem[(a)+2]<<8 | (uint32_t)mem[(a)+3])
#define SW(a,v) do { uint32_t _a=(a), _v=(uint32_t)(v); \\
    mem[_a]=_v>>24; mem[_a+1]=_v>>16; mem[_a+2]=_v>>8; mem[_a+3]=_v; } while (0)
"""


def c_source(words, mode):
    body = emit_calls(words, mode=mode)
    if mode == "global":
        body = "static uint32_t RF[32];\n\n" + body   # declared before fib() uses it
        glue = """int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        memset(mem, 0, MEMSZ);
        memset(RF, 0, sizeof RF);
        RF[29] = STACK_TOP;
        RF[4] = (uint32_t)strtoul(argv[i], 0, 10);
        fib();
        printf("%u\\n", RF[2]);
    }
    return 0;
}
"""
    else:
        glue = """int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        memset(mem, 0, MEMSZ);
        uint32_t n = (uint32_t)strtoul(argv[i], 0, 10);
        printf("%u\\n", fib(n));
    }
    return 0;
}
"""
    return PRELUDE + "\n" + body + "\n\n" + glue


def _pyfib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _compile_run(mode, ns, path):
    with open(f"{path}.c", "w") as f:
        f.write(c_source(fib_program(), mode))
    subprocess.run(["gcc", "-O2", "-o", path, f"{path}.c"], check=True)
    out = subprocess.run([f"./{path}", *[str(n) for n in ns]],
                         capture_output=True, text=True, timeout=10)
    return [int(x) for x in out.stdout.split()]


def main():
    words = fib_program()
    ns = list(range(0, 16))
    truth = [_pyfib(n) for n in ns]
    interp = [interp_calls(words, n) for n in ns]
    ok = _compile_run("global", ns, "fib_ok")
    try:
        naive = _compile_run("local", ns, "fib_naive")
    except subprocess.TimeoutExpired:
        naive = None

    print(f"{'n':>2} {'fib(n)':>8} {'interp':>8} {'ok':>8} {'naive':>12}")
    for i, n in enumerate(ns):
        nv = str(naive[i]) if naive else "TIMEOUT"
        flag = "" if (interp[i] == truth[i] == ok[i]) else "  <-- MISMATCH"
        nflag = "" if (naive and naive[i] == truth[i]) else " WRONG"
        print(f"{n:>2} {truth[i]:>8} {interp[i]:>8} {ok[i]:>8} {nv:>12}{nflag}{flag}")

    assert interp == truth, "interpreter disagrees with the definition of fib"
    assert ok == truth, "the correct recompiler disagrees with the interpreter"
    print("\ninterp == ok == fib(0..15): the shared-$sp recompiler is faithful.")
    if naive:
        nwrong = sum(1 for i in range(len(ns)) if naive[i] != truth[i])
        first = next((ns[i] for i in range(len(ns)) if naive[i] != truth[i]), None)
        print(f"naive: wrong on {nwrong}/{len(ns)} inputs; first wrong at n={first}.")


if __name__ == "__main__":
    main()
