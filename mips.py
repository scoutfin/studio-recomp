#!/usr/bin/env python3
"""A tiny MIPS I assembler, written to learn the encoding by doing it.

Studio, 2026-09-21. Yesterday I wrote up a project that recompiled 807 MIPS
functions of a 1999 N64 game into native C, and a separate group reading 1990
Amiga Unix kernels out of binaries with no source. I admire both and could not
have explained how the first one works. So: build the smallest version that
actually runs.

MIPS I is a fixed 32-bit encoding in three shapes:

    R-type  | op 6 | rs 5 | rt 5 | rd 5 | shamt 5 | funct 6 |   register ops
    I-type  | op 6 | rs 5 | rt 5 |        imm 16           |   immediates, branches, loads
    J-type  | op 6 |             target 26                 |   jumps

Every instruction is the same width and aligned, which is the property that
makes static recompilation possible at all: you can walk the text section
linearly and know you are always on an instruction boundary. On x86 you cannot
— variable-length encoding means you must know where a function starts to know
what its bytes mean, and a jump into the middle of an instruction is a legal
thing for a program to do.
"""

REG = {"zero": 0, "at": 1, "v0": 2, "v1": 3, "a0": 4, "a1": 5, "a2": 6, "a3": 7,
       "t0": 8, "t1": 9, "t2": 10, "t3": 11, "t4": 12, "t5": 13, "t6": 14,
       "t7": 15, "s0": 16, "s1": 17, "s2": 18, "s3": 19, "s4": 20, "s5": 21,
       "s6": 22, "s7": 23, "t8": 24, "t9": 25, "gp": 28, "sp": 29, "fp": 30,
       "ra": 31}
RNAME = {v: k for k, v in REG.items()}


def r(name):
    return REG[name.lstrip("$")]


def R(rs, rt, rd, funct, shamt=0):
    # shamt occupies bits [10:6], funct [5:0]. Writing `shamt << 5` instead of
    # `<< 6` — my first version — puts the shift amount's low bit on top of
    # funct's high bit, so `srl $a0,$a0,1` (funct 0x02) silently assembled as
    # funct 0x22, which is `sub`. Caught in seconds because the disassembler
    # decodes the encoder's own output and refused to recognise it. Two
    # independent records of one fact, which is the only check that ever works.
    return (0 << 26) | (r(rs) << 21) | (r(rt) << 16) | (r(rd) << 11) \
        | (shamt << 6) | funct


def I(op, rs, rt, imm):
    return (op << 26) | (r(rs) << 21) | (r(rt) << 16) | (imm & 0xFFFF)


# The subset this project needs. Each returns a 32-bit word.
def addu(rd, rs, rt):   return R(rs, rt, rd, 0x21)
def subu(rd, rs, rt):   return R(rs, rt, rd, 0x23)
def sll(rd, rt, sh):    return R("zero", rt, rd, 0x00, sh)
def srl(rd, rt, sh):    return R("zero", rt, rd, 0x02, sh)
def slt(rd, rs, rt):    return R(rs, rt, rd, 0x2A)
def jr(rs):             return R(rs, "zero", "zero", 0x08)
def addiu(rt, rs, imm): return I(0x09, rs, rt, imm)
def andi(rt, rs, imm):  return I(0x0C, rs, rt, imm)
def ori(rt, rs, imm):   return I(0x0D, rs, rt, imm)
def beq(rs, rt, off):   return I(0x04, rs, rt, off)
def bne(rs, rt, off):   return I(0x05, rs, rt, off)
def nop():              return 0x00000000


def disasm(w, pc=0):
    """Decode one word. Returns (mnemonic, operands, branch_target_or_None)."""
    op = w >> 26
    rs, rt, rd = (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
    sh, fn = (w >> 6) & 31, w & 63
    imm = w & 0xFFFF
    simm = imm - 0x10000 if imm & 0x8000 else imm
    if w == 0:
        return ("nop", (), None)
    if op == 0:
        if fn == 0x21: return ("addu", (rd, rs, rt), None)
        if fn == 0x23: return ("subu", (rd, rs, rt), None)
        if fn == 0x00: return ("sll", (rd, rt, sh), None)
        if fn == 0x02: return ("srl", (rd, rt, sh), None)
        if fn == 0x2A: return ("slt", (rd, rs, rt), None)
        if fn == 0x08: return ("jr", (rs,), None)
        raise ValueError(f"unknown R funct 0x{fn:02x} in 0x{w:08x}")
    # Branch targets are PC-relative in INSTRUCTIONS, measured from the
    # instruction AFTER the branch — because the delay slot has already been
    # committed to by the time the branch resolves.
    if op == 0x04: return ("beq", (rs, rt, simm), pc + 4 + simm * 4)
    if op == 0x05: return ("bne", (rs, rt, simm), pc + 4 + simm * 4)
    if op == 0x09: return ("addiu", (rt, rs, simm), None)
    if op == 0x0C: return ("andi", (rt, rs, imm), None)
    if op == 0x0D: return ("ori", (rt, rs, imm), None)
    raise ValueError(f"unknown opcode 0x{op:02x} in 0x{w:08x}")


def pretty(w, pc=0):
    m, ops, tgt = disasm(w, pc)
    if m == "nop":
        return "nop"
    if m in ("beq", "bne"):
        return f"{m} ${RNAME[ops[0]]}, ${RNAME[ops[1]]}, 0x{tgt:04x}"
    if m == "jr":
        return f"jr ${RNAME[ops[0]]}"
    if m in ("sll", "srl"):
        return f"{m} ${RNAME[ops[0]]}, ${RNAME[ops[1]]}, {ops[2]}"
    if m in ("addiu", "andi", "ori"):
        return f"{m} ${RNAME[ops[0]]}, ${RNAME[ops[1]]}, {ops[2]}"
    return f"{m} ${RNAME[ops[0]]}, ${RNAME[ops[1]]}, ${RNAME[ops[2]]}"
