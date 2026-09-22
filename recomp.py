#!/usr/bin/env python3
"""Static recompilation: MIPS machine code -> C, with real control flow.

The distinction that makes this worth an hour, and the thing I did not
understand yesterday:

An INTERPRETER is a loop with a switch in it. Every instruction costs a fetch,
a decode and a dispatch, forever, at runtime.

A STATIC RECOMPILER does the fetch and decode ONCE, ahead of time, and emits a
C statement per instruction. Branches become `goto` to real C labels, so the
host CPU's own branch predictor and the C compiler's optimiser see the
program's actual shape. There is no dispatch left at runtime. That is why the
Ogre Battle 64 and Amiga Unix people can get native speed out of a binary they
have no source for.

The catch, and it is the whole lesson: MIPS has BRANCH DELAY SLOTS. The
instruction after a branch executes whether or not the branch is taken —
the pipeline had already fetched it, and the architecture chose to expose that
rather than hide it. So

    beq  $a0, $t0, done
    addiu $a0, $a0, 100     <- this runs EITHER WAY

cannot be recompiled as `if (cond) goto done;` followed by the addiu. The
branch condition must be evaluated with the register values as they are AT THE
BRANCH, then the delay slot runs, then control transfers. Get the order wrong
and you have a recompiler that is correct on every program whose delay slots
happen to be nops — which is most of them, which is why this bug would survive
a long way.
"""
import subprocess
import sys

from mips import (RNAME, addiu, addu, andi, beq, bne, disasm, jr, nop, pretty,
                  r, slt, sll, srl, subu)


def targets(words, base=0):
    """Pass 1: every branch target becomes a C label."""
    out = set()
    for i, w in enumerate(words):
        _, _, t = disasm(w, base + i * 4)
        if t is not None:
            out.add(t)
    return out


def stmt(w, pc):
    """One instruction as a C expression statement (no control flow)."""
    m, o, _ = disasm(w, pc)
    if m == "nop":
        return "/* nop */"
    if m == "addu":  return f"W({o[0]}, r[{o[1]}] + r[{o[2]}]);"
    if m == "subu":  return f"W({o[0]}, r[{o[1]}] - r[{o[2]}]);"
    if m == "sll":   return f"W({o[0]}, r[{o[1]}] << {o[2]});"
    if m == "srl":   return f"W({o[0]}, r[{o[1]}] >> {o[2]});"
    if m == "slt":   return f"W({o[0]}, (int32_t)r[{o[1]}] < (int32_t)r[{o[2]}]);"
    if m == "addiu": return f"W({o[0]}, r[{o[1]}] + {o[2]});"
    if m == "andi":  return f"W({o[0]}, r[{o[1]}] & 0x{o[2]:x}u);"
    if m == "ori":   return f"W({o[0]}, r[{o[1]}] | 0x{o[2]:x}u);"
    raise ValueError(f"no statement form for {m}")


def emit(words, base=0, name="run", delay_slots=True):
    """Pass 2: emit C. delay_slots=False produces the NAIVE version, kept so
    the difference can be measured rather than asserted."""
    labs = targets(words, base)
    L = [f"static uint32_t {name}(uint32_t a0) {{",
         "  uint32_t r[32] = {0};",
         "  r[4] = a0;                 /* $a0 */",
         "  uint32_t cond;  (void)cond;"]
    i = 0
    while i < len(words):
        pc = base + i * 4
        w = words[i]
        if pc in labs:
            L.append(f"L_{pc:04x}:")
        m, o, tgt = disasm(w, pc)

        if m in ("beq", "bne"):
            op = "==" if m == "beq" else "!="
            c = f"r[{o[0]}] {op} r[{o[1]}]"
            if delay_slots and i + 1 < len(words):
                # Condition FIRST, using pre-delay-slot register values.
                L.append(f"  cond = ({c});           /* {pretty(w, pc)} */")
                L.append(f"  {stmt(words[i+1], pc+4)}   /* delay slot */")
                L.append(f"  if (cond) goto L_{tgt:04x};")
                i += 2
                continue
            L.append(f"  if ({c}) goto L_{tgt:04x};   /* {pretty(w, pc)} */")
            i += 1
            continue

        if m == "jr":
            L.append(f"  return r[2];               /* {pretty(w, pc)} -> $v0 */")
            # the jr delay slot still executes, before the return
            if delay_slots and i + 1 < len(words):
                L.insert(len(L) - 1, f"  {stmt(words[i+1], pc+4)}   /* delay slot */")
                i += 2
                continue
            i += 1
            continue

        L.append(f"  {stmt(w, pc)}".ljust(40) + f"/* {pretty(w, pc)} */")
        i += 1

    L += ["  return r[2];", "}"]
    return "\n".join(L)


PRELUDE = """#include <stdint.h>
#include <stdio.h>
/* $zero is hardwired to 0: writes to it are discarded, not stored. */
#define W(d, v) do { uint32_t _v = (uint32_t)(v); if ((d) != 0) r[(d)] = _v; } while (0)
"""


def interp(words, a0, base=0, delay_slots=True):
    """Reference interpreter — the independent second record. Deliberately
    written as a fetch/decode/dispatch loop, i.e. the thing recompilation
    replaces."""
    reg = [0] * 32
    reg[4] = a0
    pc, steps = base, 0
    while steps < 100000:
        steps += 1
        w = words[(pc - base) // 4]
        m, o, tgt = disasm(w, pc)
        nxt = pc + 4

        def wr(d, v):
            if d != 0:
                reg[d] = v & 0xFFFFFFFF

        if m == "jr":
            # $ra is a return; any other register is an INDIRECT JUMP and the
            # target is whatever that register holds. Treating both as "return"
            # was my bug on 2026-09-21, and it was in the interpreter too — so
            # when indirect.py first ran, the "ground truth" reference agreed
            # with the broken recompiler and I nearly read that as confirmation.
            # Two records that share a defect are one record.
            if delay_slots:
                run_one(words, reg, base, pc + 4)
            if o[0] == 31:
                return reg[2]
            pc = reg[o[0]]
            continue
        if m in ("beq", "bne"):
            take = (reg[o[0]] == reg[o[1]]) if m == "beq" else (reg[o[0]] != reg[o[1]])
            if delay_slots:
                run_one(words, reg, base, pc + 4)
                nxt = pc + 8
            pc = tgt if take else nxt
            continue
        run_one(words, reg, base, pc)
        pc = nxt
    raise RuntimeError("interpreter did not halt")


def run_one(words, reg, base, pc):
    w = words[(pc - base) // 4]
    m, o, _ = disasm(w, pc)

    def wr(d, v):
        if d != 0:
            reg[d] = v & 0xFFFFFFFF

    if m == "nop":   return
    if m == "addu":  wr(o[0], reg[o[1]] + reg[o[2]])
    elif m == "subu":  wr(o[0], reg[o[1]] - reg[o[2]])
    elif m == "sll":   wr(o[0], reg[o[1]] << o[2])
    elif m == "srl":   wr(o[0], reg[o[1]] >> o[2])
    elif m == "slt":   wr(o[0], int((reg[o[1]] ^ 0x80000000) < (reg[o[2]] ^ 0x80000000)))
    elif m == "addiu": wr(o[0], reg[o[1]] + o[2])
    elif m == "andi":  wr(o[0], reg[o[1]] & o[2])
    elif m == "ori":   wr(o[0], reg[o[1]] | o[2])
    else: raise ValueError(f"run_one cannot execute {m}")
