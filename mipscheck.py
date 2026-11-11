#!/usr/bin/env python3
"""Diff my MIPS encoder against GNU as, instruction by instruction.

Studio, 2026-11-11. mips.py has been checked against itself since September: the
disassembler decodes the encoder's output and the two agree. That caught a real
bug on day one (a shamt shifted by 5 instead of 6) and it has caught nothing
since, which is the expected behaviour of a check whose two records share an
author. A shared misreading of the manual agrees perfectly.

binutils has a MIPS target. So: write the same instruction twice, once as a call
into mips.py and once as a line of assembly, and diff the 32-bit words.

    python3 mipscheck.py

Needs binutils-mips-linux-gnu (mips-linux-gnu-as, mips-linux-gnu-objdump).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from addrpair import split_addiu, split_ori
from mips import (addiu, addu, andi, beq, bne, jal, jalr, jr, lb, lbu, lui, lw,
                  nop, ori, sb, sll, slt, srl, subu, sw)

AS = "mips-linux-gnu-as"
OBJDUMP = "mips-linux-gnu-objdump"

# (assembly text, the word my encoder produces). The assembly is written the way
# GNU as wants it; the call is written the way mips.py wants it. Nothing is shared
# between the two sides except my belief that they mean the same thing.
CASES = [
    ("addu  $v0, $a0, $a1",      addu("v0", "a0", "a1")),
    ("subu  $t0, $t1, $t2",      subu("t0", "t1", "t2")),
    ("addu  $zero, $zero, $zero", addu("zero", "zero", "zero")),
    ("slt   $t3, $a0, $a1",      slt("t3", "a0", "a1")),
    ("sll   $s0, $s1, 0",        sll("s0", "s1", 0)),
    ("sll   $s0, $s1, 1",        sll("s0", "s1", 1)),
    ("sll   $s0, $s1, 31",       sll("s0", "s1", 31)),
    ("srl   $s0, $s1, 1",        srl("s0", "s1", 1)),
    ("srl   $ra, $gp, 16",       srl("ra", "gp", 16)),
    ("addiu $sp, $sp, -12",      addiu("sp", "sp", -12)),
    ("addiu $sp, $sp, 12",       addiu("sp", "sp", 12)),
    ("addiu $t0, $zero, 32767",  addiu("t0", "zero", 32767)),
    ("addiu $t0, $zero, -32768", addiu("t0", "zero", -32768)),
    ("addiu $t9, $t9, -32740",   addiu("t9", "t9", -32740)),
    ("andi  $t1, $t2, 0xffff",   andi("t1", "t2", 0xFFFF)),
    ("ori   $t1, $t2, 0x801c",   ori("t1", "t2", 0x801C)),
    ("lui   $t9, 0x41",          lui("t9", 0x41)),
    ("lui   $at, 0xffff",        lui("at", 0xFFFF)),
    ("lw    $ra, 8($sp)",        lw("ra", 8, "sp")),
    ("lw    $ra, -8($sp)",       lw("ra", -8, "sp")),
    ("sw    $s0, 4($sp)",        sw("s0", 4, "sp")),
    ("lb    $a0, 3($t0)",        lb("a0", 3, "t0")),
    ("lbu   $a0, 3($t0)",        lbu("a0", 3, "t0")),
    ("sb    $a0, -1($t0)",       sb("a0", -1, "t0")),
    ("jr    $ra",                jr("ra")),
    ("jr    $t9",                jr("t9")),
    ("jalr  $t9",                jalr("t9")),
    ("jalr  $t8, $t9",           jalr("t9", "t8")),
    ("nop",                      nop()),
]

# Branches and jumps need a label, so their expected word depends on where the
# instruction sits. Each entry is (asm, offset_of_instruction, target_address).
# The target is 8 bytes in: one branch plus one delay slot.
REL = [
    ("beq   $a0, $a1, L", 0, 8, lambda: beq("a0", "a1", 1)),
    ("bne   $a0, $zero, L", 0, 8, lambda: bne("a0", "zero", 1)),
    ("beq   $zero, $zero, L", 0, 8, lambda: beq("zero", "zero", 1)),
    ("jal   L", 0, 8, lambda: jal(8)),
]


def assemble_lines(lines):
    """Run GNU as over one instruction per line and return the words it made."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "t.s")
        obj = os.path.join(d, "t.o")
        with open(src, "w") as f:
            f.write("\t.set noreorder\n\t.set noat\n\t.text\n")
            f.write("".join(f"\t{l}\n" for l in lines))
        p = subprocess.run([AS, "-EB", "-mips1", "-o", obj, src],
                           capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr.strip())
        d2 = subprocess.run([OBJDUMP, "-d", "--section=.text", obj],
                            capture_output=True, text=True, check=True).stdout
    words = []
    for line in d2.splitlines():
        m = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]{8})\s", line)
        if m:
            words.append(int(m.group(1), 16))
    return words


def main():
    for tool in (AS, OBJDUMP):
        if not shutil.which(tool):
            print(f"need {tool} (apt install binutils-mips-linux-gnu)")
            return 2

    print("=== every instruction mips.py can encode, against GNU as ===")
    theirs = assemble_lines([a for a, _ in CASES])
    assert len(theirs) == len(CASES), (len(theirs), len(CASES))
    bad = 0
    for (asm, mine), their in zip(CASES, theirs):
        ok = mine == their
        bad += not ok
        if not ok:
            print(f"  MISMATCH  {asm:26} mine 0x{mine:08x}  as 0x{their:08x}"
                  f"  xor 0x{mine ^ their:08x}")
    print(f"  {len(CASES) - bad}/{len(CASES)} identical"
          + ("" if bad else "  (every word, bit for bit)"))

    print("\n=== branches and jal, where the word depends on the address ===")
    for asm, _, _, make in REL:
        their = assemble_lines([asm, "nop", "L:", "nop"])[0]
        mine = make()
        flag = "ok" if mine == their else f"MISMATCH as 0x{their:08x}"
        bad += mine != their
        print(f"  {asm:22} mine 0x{mine:08x}  {flag}")

    print("\n=== `la` expanded by GNU as, against split_addiu / split_ori ===")
    # This is the one that matters for addrpair.py: the %hi() carry. GNU as owns
    # the convention; my split is a reimplementation of it from the manual.
    for pad in (0x7F00, 0x8000, 0x8100, 0x1F000):
        their = assemble_lines(["la $t9, h", f".space {pad:#x}", "h:", "nop"])
        addr = 8 + pad                        # la is two words, then the padding
        hi, lo = split_addiu(addr)
        mine = [lui("t9", hi), addiu("t9", "t9", lo)]
        got_hi, got_lo = (their[0] & 0xFFFF), (their[1] & 0xFFFF)
        print(f"  h at {addr:#07x}: as -> lui 0x{got_hi:04x}, addiu 0x{got_lo:04x}"
              f"   mine -> lui 0x{hi:04x}, addiu 0x{lo:04x}"
              f"   {'same' if mine == their[:2] else 'DIFFER'}")
        bad += mine != their[:2]
    # %hi()/%lo() written out, which is the definition rather than the expansion.
    for addr in (0x00400044, 0x0040801C, 0x0041FFFF, 0x00408000):
        their = assemble_lines([f"lui $t9, %hi({addr:#x})",
                                f"addiu $t9, $t9, %lo({addr:#x})"])
        hi, lo = split_addiu(addr)
        same = their == [lui("t9", hi), addiu("t9", "t9", lo)]
        print(f"  %hi/%lo({addr:#010x}): as hi=0x{their[0] & 0xFFFF:04x}"
              f" lo=0x{their[1] & 0xFFFF:04x}   mine hi=0x{hi:04x} lo=0x{lo:04x}"
              f"   {'same' if same else 'DIFFER'}")
        bad += not same
        # ori takes the low half with no carry, so the SAME address needs a
        # different high half depending on which instruction completes the pair.
        ohi, olo = split_ori(addr)
        if (hi, lo & 0xFFFF) != (ohi, olo):
            print(f"      and with ori: hi=0x{ohi:04x} lo=0x{olo:04x}"
                  f"  <- different high half for the same address")

    print(f"\n{'ALL AGREE' if not bad else str(bad) + ' MISMATCHES'}")
    return 1 if bad else 0


if __name__ == "__main__" and "--mutate" not in sys.argv:
    sys.exit(main())


# ---- commissioning: make it go red -----------------------------------------
# A differential test that has only ever passed is not shown to be a test. Each
# mutation below is a plausible encoding error; the harness asserts the edit
# landed, that the named case goes red, and that nothing else does.
#
# ⚠️ THE HARNESS ITSELF HAD THIS BUG, 2026-11-11. The first batched run scored
# two of three mutations against UNMUTATED code, and it looked like a result
# rather than a failure because mutation 3 printed mutation 1's mismatches.
# CPython validates a .pyc against (source mtime IN SECONDS, source size). A
# one-character mutation preserves the size, and a revert within the same second
# preserves the second, so the restored file is byte-indistinguishable from the
# cached mutant. Measured: pyc header mtime 1794421739 size 8350, restored file
# mtime 1794421739 size 8350, test still red; `touch` +2s with no content change,
# green. A mutation harness is the WORST case for that cache by construction,
# because small-edit/run/revert/repeat is exactly the pattern it cannot see.
# Hence sys.dont_write_bytecode and the cleared cache below.
MUTATIONS = [
    ("mips.py", "| (shamt << 6) | funct", "| (shamt << 5) | funct",
     "shamt into funct's high bit", ("sll", "srl")),
    ("mips.py", "def ori(rt, rs, imm):   return I(0x0D, rs, rt, imm)",
     "def ori(rt, rs, imm):   return I(0x09, rs, rt, imm)",
     "ori with addiu's opcode", ("ori",)),
    ("mips.py", "def J(op, addr):\n", "def J(op, addr):\n    addr = addr\n",
     "no-op edit, control: nothing should go red", ()),
    ("addrpair.py", "hi = ((a >> 16) + (1 if lo & 0x8000 else 0)) & 0xFFFF",
     "hi = (a >> 16) & 0xFFFF", "%hi carry dropped", ("DIFFER",)),
]


def mutate():
    import shutil as sh
    rc = 0
    for path, old, new, label, expect in MUTATIONS:
        src = open(path).read()
        if old not in src:
            print(f"  {label:36} CANNOT APPLY (pattern not in {path})")
            rc = 1
            continue
        sh.copy(path, path + ".orig")
        open(path, "w").write(src.replace(old, new, 1))
        assert new in open(path).read(), label        # the edit landed
        for d in ("__pycache__",):
            sh.rmtree(d, ignore_errors=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([sys.executable, "-B", __file__], capture_output=True,
                           text=True, env=env)
        sh.move(path + ".orig", path)
        out = p.stdout
        red = [l.strip() for l in out.splitlines()
               if "MISMATCH" in l or "DIFFER" in l]
        if not expect:
            ok = p.returncode == 0 and not red
        else:
            ok = p.returncode == 1 and red and all(
                any(e in l for l in red) for e in expect)
        print(f"  {label:36} {'caught' if ok else 'NOT CAUGHT'}"
              f"  ({len(red)} line(s) red)")
        rc |= not ok
    for d in ("__pycache__",):
        shutil.rmtree(d, ignore_errors=True) if os.path.isdir(d) else None
    return rc


if __name__ == "__main__" and "--mutate" in sys.argv:
    print("=== mutation: each of these must make the test go red ===")
    sys.exit(mutate())
