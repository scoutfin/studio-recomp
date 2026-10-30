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

from mips import (RNAME, addiu, addu, andi, beq, bne, disasm, jr, lb, lbu, lw,
                  nop, pretty, r, sb, slt, sll, srl, subu, sw)


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
    # Memory, added 2026-10-30. disasm gives loads/stores as (rt, offset, rs);
    # the effective address is r[rs] + offset. Stores write memory, not a
    # register, so they do not use the W() zero-guard. The endianness lives
    # entirely in the LW/SW/LB macros in the prelude, in one place, so the
    # recompiler emits the same byte order the interpreter reads.
    if m == "lw":    return f"W({o[0]}, LW(r[{o[2]}] + {o[1]}));"
    if m == "sw":    return f"SW(r[{o[2]}] + {o[1]}, r[{o[0]}]);"
    if m == "lb":    return f"W({o[0]}, LB(r[{o[2]}] + {o[1]}));"
    if m == "lbu":   return f"W({o[0]}, LBU(r[{o[2]}] + {o[1]}));"
    if m == "sb":    return f"SB(r[{o[2]}] + {o[1]}, r[{o[0]}]);"
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

/* Memory is a flat byte array. MIPS I is BIG-ENDIAN: the byte at the lowest
 * address is the most significant. Every load and store names the byte order
 * explicitly here, so it does NOT depend on the host's endianness. That is the
 * whole point: this file is compiled and run on a little-endian x86 host, and
 * the emitted program must still behave like a big-endian machine. */
#define MEMSZ 65536
static uint8_t mem[MEMSZ];
#define LW(a)   ((uint32_t)mem[(a)]<<24 | (uint32_t)mem[(a)+1]<<16 \\
                 | (uint32_t)mem[(a)+2]<<8 | (uint32_t)mem[(a)+3])
#define SW(a,v) do { uint32_t _a=(a), _v=(uint32_t)(v); \\
    mem[_a]=_v>>24; mem[_a+1]=_v>>16; mem[_a+2]=_v>>8; mem[_a+3]=_v; } while (0)
#define LB(a)   ((uint32_t)(int32_t)(int8_t)mem[(a)])   /* sign-extend */
#define LBU(a)  ((uint32_t)mem[(a)])                    /* zero-extend */
#define SB(a,v) (mem[(a)] = (uint8_t)(v))
"""


MEMSZ = 65536


def _be_lw(mem, a):
    return (mem[a] << 24) | (mem[a + 1] << 16) | (mem[a + 2] << 8) | mem[a + 3]


def _be_sw(mem, a, v):
    v &= 0xFFFFFFFF
    mem[a] = (v >> 24) & 0xFF
    mem[a + 1] = (v >> 16) & 0xFF
    mem[a + 2] = (v >> 8) & 0xFF
    mem[a + 3] = v & 0xFF


def interp(words, a0, base=0, delay_slots=True, mem=None):
    """Reference interpreter — the independent second record. Deliberately
    written as a fetch/decode/dispatch loop, i.e. the thing recompilation
    replaces. Memory is a big-endian byte array, matching the prelude macros;
    it is the SECOND independent record of the byte order, which is the only
    kind of check that works here."""
    reg = [0] * 32
    reg[4] = a0
    if mem is None:
        mem = bytearray(MEMSZ)
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
                run_one(words, reg, mem, base, pc + 4)
            if o[0] == 31:
                return reg[2]
            pc = reg[o[0]]
            continue
        if m in ("beq", "bne"):
            take = (reg[o[0]] == reg[o[1]]) if m == "beq" else (reg[o[0]] != reg[o[1]])
            if delay_slots:
                run_one(words, reg, mem, base, pc + 4)
                nxt = pc + 8
            pc = tgt if take else nxt
            continue
        run_one(words, reg, mem, base, pc)
        pc = nxt
    raise RuntimeError("interpreter did not halt")


def run_one(words, reg, mem, base, pc):
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
    # Memory: effective address is reg[rs] + offset, with rs = o[2], off = o[1].
    # Big-endian, matching the prelude macros byte for byte.
    elif m == "lw":    wr(o[0], _be_lw(mem, reg[o[2]] + o[1]))
    elif m == "sw":    _be_sw(mem, reg[o[2]] + o[1], reg[o[0]])
    elif m == "lb":
        b = mem[reg[o[2]] + o[1]]
        wr(o[0], b | 0xFFFFFF00 if b & 0x80 else b)   # sign-extend
    elif m == "lbu":   wr(o[0], mem[reg[o[2]] + o[1]])
    elif m == "sb":    mem[reg[o[2]] + o[1]] = reg[o[0]] & 0xFF
    else: raise ValueError(f"run_one cannot execute {m}")
