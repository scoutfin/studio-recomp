#!/usr/bin/env python3
"""Loads, stores, and the byte order that a word-addressed shortcut would hide.

Studio, 2026-10-30. Until today both halves of this project had no memory at
all, only registers. A word load or store moves all four bytes together, so you
can fake it with a host-native word array and never notice which end is which.
The moment you store a word and read back ONE of its bytes, byte order becomes
observable, and MIPS I is big-endian: the byte at the lowest address is the most
significant.

The trap this is built to show is the project's oldest lesson in a new place.
If I model memory the lazy way, as a host word array, in BOTH the interpreter
and the recompiler, the two agree with each other and are both wrong on a
little-endian host. Two records that share a defect are one record. The honest
model is a flat byte array with the byte order written out explicitly, which is
what recomp.py now emits and interp now reads.
"""
from mips import jr, lbu, nop, ori, pretty, sll, sw
from recomp import PRELUDE, emit, interp
from build import build_and_run

BASE = 0x100  # where the test program keeps its one word of scratch

# Build 0x12345678 in $t0, store it at 0($a0), then load a single byte back.
# $a0 is the scratch base on entry. Returns the loaded byte in $v0.
def endian_prog(load_off):
    return [
        ori("t0", "zero", 0x1234),   # t0 = 0x00001234
        sll("t0", "t0", 16),         # t0 = 0x12340000
        ori("t0", "t0", 0x5678),     # t0 = 0x12345678
        sw("t0", 0, "a0"),           # BE store: [a0..a0+3] = 12 34 56 78
        lbu("v0", load_off, "a0"),   # read one byte back
        jr("ra"),
        nop(),
    ]


def host_word_shortcut(words, a0):
    """The lazy model I did NOT use: memory as a dict of host-native words,
    sub-word loads extracted in the HOST'S byte order. Written here only to show
    that it is self-consistent and still wrong, so that two implementations
    sharing it would agree and mislead."""
    reg = [0] * 32
    reg[4] = a0
    words_mem = {}
    for w in words:
        m, o, _ = __import__("mips").disasm(w, 0)
        if m == "ori":   reg[o[0]] = reg[o[1]] | o[2]
        elif m == "sll": reg[o[0]] = (reg[o[1]] << o[2]) & 0xFFFFFFFF
        elif m == "sw":  words_mem[(reg[o[2]] + o[1]) // 4] = reg[o[0]]
        elif m == "lbu":
            a = reg[o[2]] + o[1]
            word = words_mem[a // 4]
            reg[o[0]] = (word >> (8 * (a % 4))) & 0xFF   # little-endian: byte 0 = low byte
        elif m == "jr":  return reg[2]
        elif m in ("nop",): pass
    return reg[2]


if __name__ == "__main__":
    print("  --- the endianness probe: store 0x12345678, read one byte ---")
    for w in endian_prog(0):
        print(f"    {w:08x}   {pretty(w)}")

    print("\n  byte  | interp (BE) | recompiled C (BE) | host-word shortcut")
    print("  ------+-------------+-------------------+-------------------")
    rows = []
    for off in (0, 1, 2, 3):
        prog = endian_prog(off)
        bi = interp(prog, BASE)
        bc = build_and_run(prog, f"mem{off}", [BASE])[0]
        bs = host_word_shortcut(prog, BASE)
        rows.append((off, bi, bc, bs))
        print(f"   +{off}   |    0x{bi:02x}     |       0x{bc:02x}        |       0x{bs:02x}")

    true_be = {0: 0x12, 1: 0x34, 2: 0x56, 3: 0x78}
    agree_honest = all(bi == bc == true_be[off] for off, bi, bc, _ in rows)
    shortcut_wrong = any(bs != true_be[off] for off, _, _, bs in rows)
    shortcut_self_consistent = True  # it round-trips its own stores by construction

    print()
    print(f"    interp and recompiler agree AND match big-endian truth: {agree_honest}")
    print(f"    the host-word shortcut disagrees with truth somewhere : {shortcut_wrong}")
    print()
    if agree_honest and shortcut_wrong:
        print("    The honest byte-array model is right in both records. The shortcut")
        print("    is internally consistent — it reads back whatever it wrote — yet on")
        print("    this little-endian host it reports the bytes reversed. Had I used it")
        print("    in BOTH the interpreter and the recompiler, they would have agreed")
        print("    with each other and been wrong together. Byte order is the kind of")
        print("    fact a shared shortcut erases without leaving a disagreement behind.")

    # A plain word round-trip, to show lw/sw themselves are endianness-invariant
    # (all four bytes move together) — which is exactly why a word-only project
    # never had to decide.
    from mips import lw
    wprog = [ori("t0", "zero", 0x1234), sll("t0", "t0", 16), ori("t0", "t0", 0x5678),
             sw("t0", 0, "a0"), lw("v0", 0, "a0"), jr("ra"), nop()]
    wi = interp(wprog, BASE)
    wc = build_and_run(wprog, "memword", [BASE])[0]
    print(f"\n    word round-trip (sw then lw): interp=0x{wi:08x} recompiled=0x{wc:08x} "
          f"-> {'match' if wi == wc == 0x12345678 else 'MISMATCH'}")
