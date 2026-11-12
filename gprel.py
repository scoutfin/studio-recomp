#!/usr/bin/env python3
"""$gp-relative addressing: the address that is never built.

Studio, 2026-11-12. Continuation (d) of the addrpair thread, and the one the
2026-11-11 session did not touch at all.

Every address-recovery pass I have written anchors on CONSTRUCTION: somewhere in
the instruction stream a `lui` and an `addiu`/`ori` put a 32-bit constant into a
register, and `pair_entries` collects it there rather than at its use. That is
why it handles the split pair, the shared `lui` and the pointer that travels
through memory with no case for any of them.

$gp-relative addressing removes the construction. One `lui`/`addiu` pair at
entry sets $gp to a single base, and every global is then reached as

    lw   $t0, offset($gp)

with a SIGNED 16-bit offset. The global's address is `$gp + offset`, computed by
the hardware at every access and never materialised as a constant anywhere. A
pass anchored at construction sees one address -- $gp -- for a program with any
number of globals.

GROUND TRUTH, from binutils 2.42 rather than from memory (`gt/gp.s`, assembled
with `as -mno-shared -march=mips1` and linked with `ld`):

    .sdata   0x00410120
    _gp      0x00418110          <- .sdata + 0x7ff0, EXACTLY
    lui  gp,0x42 ; addiu gp,gp,-32496      -> 0x00418110
    lw   t0,-32752(gp)  -> 0x00410120  counter
    lw   t1,-32748(gp)  -> 0x00410124  limit
    lw   t9,-32744(gp)  -> 0x00410128  fnptr

The 0x7ff0 is the whole design: a signed 16-bit offset reaches [-32768, +32767],
so putting $gp 0x7ff0 above the start of .sdata centres that window on the small
data area and lets one base register address ~64K of globals. Every access in the
dump above is therefore a NEGATIVE offset, which is what small-data code looks
like. Note also that the hi field is 0x42 and not 0x41: the low half 0x8110 has
bit 15 set, so `addiu` will sign-extend and subtract, and the assembler
compensates -- `split_addiu`'s carry rule, confirmed by somebody else's linker.

PREDICTIONS, written before running any of the measurements below.

  P1  Scanning the linked .text for a word equal to `handler`'s address finds
      ZERO. The address exists only in .sdata, placed there by the linker.
  P2  `pair_entries` over that .text returns ZERO entries, for the same reason,
      so `handler` is undiscoverable and a recompile of this program ABORTs.
      The one constant it CAN prove ($gp = 0x00418110) is not in the text range.
  P3  Scanning .sdata for words that land in the text range finds exactly ONE,
      0x00400114 = handler. This is continuation (b), and the prediction I did
      not expect to be writing here: (d) is WHY (b) is necessary. Under
      $gp-relative addressing the only surviving copy of a function pointer is
      the one the linker wrote into the data segment.
  P4  A gp-aware pass -- track the constant in $gp, then treat `off($gp)` as an
      address computation rather than as an opaque load -- recovers exactly THREE
      data addresses (counter, limit, fnptr) from a stream where none of them is
      constructed.

  P5  On `gt/gp2.s`, which adds `tally: .word 0x00400108` -- an integer that
      collides with a text address and that no instruction ever loads -- the
      crude data scan of P3 finds TWO code-looking words, one of them invented.
  P6  The gp-directed chain (recover the addresses the code actually loads, then
      read the data AT those addresses) finds exactly ONE, handler, and excludes
      tally. The crude scan asks whether a word LOOKS like an address; this asks
      whether the program ever USES it as one.

Run: python3 gprel.py
"""
import re
import subprocess

from mips import disasm


def _s16(v):
    return v - 0x10000 if v & 0x8000 else v

ELF = "gt/gp.elf"
ELF2 = "gt/gp2.elf"   # same program plus one integer that collides with a text address
PRE = "mips-linux-gnu-"


def sections(elf=ELF):
    """Address/size of each section, straight from readelf."""
    out = subprocess.run([PRE + "readelf", "-S", "-W", elf],
                         capture_output=True, text=True).stdout
    secs = {}
    for m in re.finditer(r"\[\s*\d+\]\s+(\.\S+)\s+\S+\s+([0-9a-f]+)\s+[0-9a-f]+\s+"
                         r"([0-9a-f]+)", out):
        secs[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16))
    return secs


def section_words(name, elf=ELF):
    """The section's contents as (addr, [word, ...]), big-endian."""
    addr, size = sections(elf)[name]
    subprocess.run([PRE + "objcopy", "-O", "binary", f"--only-section={name}",
                    elf, "/tmp/.sec.bin"], check=True)
    raw = open("/tmp/.sec.bin", "rb").read()[:size]
    return addr, [int.from_bytes(raw[i:i + 4], "big") for i in range(0, len(raw), 4)]


def syms(elf=ELF):
    out = subprocess.run([PRE + "nm", "-n", elf], capture_output=True, text=True).stdout
    return {m.group(2): int(m.group(1), 16)
            for m in re.finditer(r"^([0-9a-f]+)\s+\S\s+(\S+)$", out, re.M)}


# ---- the gp-aware pass ------------------------------------------------------
def gp_data_refs(words, base):
    """Constants in registers, plus every `off(reg)` whose reg is a known constant.

    The only change from `pair_entries` is the last clause: a load or store whose
    BASE register is constant names an address, even though no instruction ever
    holds that address. `pair_entries` discards it because it is looking for a
    value that was built, and this one is applied.
    """
    const, refs = {}, {}
    for i, w in enumerate(words):
        m, o, _ = disasm(w, base + i * 4)
        if m == "lui":
            const[o[0]] = o[1] << 16
        elif m == "addiu" and (o[1] == 0 or o[1] in const):
            const[o[0]] = (const.get(o[1], 0) + o[2]) & 0xFFFFFFFF
        elif m == "ori" and (o[1] == 0 or o[1] in const):
            const[o[0]] = const.get(o[1], 0) | o[2]
        elif m in ("lw", "sw", "lb", "lbu", "sb"):
            rt, off, rs = o
            if rs in const:
                refs[(const[rs] + off) & 0xFFFFFFFF] = m
            const.pop(rt, None)            # the loaded value is not this address
        elif o:
            const.pop(o[0], None)
    # $gp is READ OUT of const, never accumulated alongside it. The first version
    # kept `gp_base` as a second record, set when $gp was proved constant and never
    # cleared when it was invalidated -- so on PIC, where `addu gp,gp,t9` correctly
    # removes $gp from const, the pass still reported "base = 0x0". The detailed
    # field was honest and the summary field was confidently wrong, which is the
    # direction summaries always fail in
    # ([[a-document-that-summarises-itself-contains-its-own-control]]).
    return refs, const.get(28), const


def main():
    s = syms()
    taddr, text = section_words(".text")
    daddr, data = section_words(".sdata")
    tlo, thi = taddr, taddr + len(text) * 4
    print(f"  .text  {taddr:#010x}  {len(text)} words")
    print(f"  .sdata {daddr:#010x}  {len(data)} words")
    print(f"  _gp    {s['_gp']:#010x}   = .sdata + {s['_gp'] - daddr:#x}")
    want = {v: k for k, v in s.items() if k in ("counter", "limit", "fnptr", "handler")}

    fails = []

    def check(label, got, want_, note=""):
        ok = got == want_
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<46} {got}" +
              (f"   want {want_}" if not ok else "") + (f"   {note}" if note else ""))
        if not ok:
            fails.append(label)

    print("\n  P1 — is a code address anywhere in the instruction stream?")
    lit = [i for i, w in enumerate(text) if w == s["handler"]]
    check("words in .text equal to handler's address", len(lit), 0)

    print("\n  P2 — what does the construction-anchored pass recover?")
    import addrpair as ap
    found = ap.pair_entries(text, base=taddr, across="never", bare_lui=True)
    check("pair_entries entries found", len(found), 0,
          f"(it proves $gp={s['_gp']:#x}, which is not in text)")

    print("\n  P3 — scan the DATA segment for code addresses (continuation b)")
    dcode = {daddr + i * 4: w for i, w in enumerate(data) if tlo <= w < thi}
    check("code addresses found in .sdata", len(dcode), 1)
    for at, w in dcode.items():
        print(f"         .sdata+{at - daddr:#x} holds {w:#010x} = {want.get(w, '?')}")

    print("\n  P4 — gp-aware: treat off($gp) as an address computation")
    refs, gp_base, const = gp_data_refs(text, taddr)
    check("$gp proved constant", gp_base, s["_gp"])
    check("distinct data addresses recovered", len(refs), 3)
    for a in sorted(refs):
        print(f"         {a:#010x}  {want.get(a, '?'):<8} via {refs[a]}  "
              f"(offset {a - s['_gp']:+d} from $gp)")
    check("they are exactly the three globals", sorted(refs),
          sorted(s[n] for n in ("counter", "limit", "fnptr")))

    # ---- part 5: appearance vs use -----------------------------------------
    print("\n  P5/P6 — gp2.elf: the same program plus `tally`, an INTEGER whose")
    print("          value collides with a text address and which nothing loads.")
    s2 = syms(ELF2)
    t2a, t2 = section_words(".text", ELF2)
    d2a, d2 = section_words(".sdata", ELF2)
    t2lo, t2hi = t2a, t2a + len(t2) * 4
    want2 = {v: k for k, v in s2.items()
             if k in ("counter", "limit", "fnptr", "tally", "handler")}

    crude = {d2a + i * 4: w for i, w in enumerate(d2) if t2lo <= w < t2hi}
    check("P5 crude .sdata scan: code-looking words", len(crude), 2)
    for at, w in sorted(crude.items()):
        print(f"         .sdata+{at - d2a:#x} holds {w:#010x} = {want2.get(w, 'NOT A SYMBOL')}"
              f"   <- from `{want2.get(d2a + (at - d2a), '?')}`")

    refs2, gp2, _ = gp_data_refs(t2, t2a)
    byaddr = {d2a + i * 4: w for i, w in enumerate(d2)}
    used = {byaddr[a] for a in refs2 if a in byaddr and t2lo <= byaddr[a] < t2hi}
    check("P6 gp-directed: code addresses actually LOADED", len(used), 1)
    for w in sorted(used):
        print(f"         {w:#010x} = {want2.get(w, '?')}  (reached via fnptr, which the code loads)")
    check("P6 it is handler, and tally is excluded",
          sorted(used), [s2["handler"]])
    print(f"         crude over-guesses by {len(crude) - len(used)}: it asks whether a word LOOKS")
    print("         like an address; the gp-directed pass asks whether the program ever USES")
    print("         it as one. 11-05 measured over-guessing as the safe direction, so this is")
    print("         not crude being wrong -- it is the two passes answering different questions.")

    # ---- part 7: PIC, where $gp is not a link-time constant ----------------
    print("\n  P7 — gp3.o is PIC. `.cpload $t9` expands to lui/addiu/addu gp,gp,t9,")
    print("       so $gp depends on where the function was LOADED, not on the link.")
    t3a, t3 = section_words(".text", "gt/gp3.o")
    refs3, gp3, _ = gp_data_refs(t3, t3a)
    check("gp-aware pass on PIC: data addresses", len(refs3), 0,
          "(silence, not a wrong answer)")
    check("and it does NOT claim $gp is constant", gp3, None)

    # Is that safety load-bearing, or luck? Delete the one line that kills a
    # register written by an unmodelled instruction and re-run.
    def gp_unsafe(words, base):
        const, refs = {}, {}
        for i, w in enumerate(words):
            m, o, _ = disasm(w, base + i * 4)
            if m == "lui":
                const[o[0]] = o[1] << 16
            elif m == "addiu" and (o[1] == 0 or o[1] in const):
                const[o[0]] = (const.get(o[1], 0) + o[2]) & 0xFFFFFFFF
            elif m in ("lw", "sw", "lb", "lbu", "sb"):
                rt, off, rs = o
                if rs in const:
                    refs[(const[rs] + off) & 0xFFFFFFFF] = m
        return refs
    bad = gp_unsafe(t3, t3a)
    check("mutant WITHOUT the kill invents an address", len(bad), 1,
          f"-> {[hex(a) for a in bad]}")
    print("       The mutant reports a confident 0x0: it saw `lui gp,0 ; addiu gp,gp,0`,")
    print("       never saw the addu, and answered as if $gp were 0. One `const.pop` is")
    print("       the whole difference between an honest give-up and a wrong address.")

    # ---- part 8: where the address lives under PIC -------------------------
    print("\n  P8 — gp4.so is PIC with a %call16 call. Where IS handler's address?")
    s4 = syms("gt/gp4.so")
    t4a, t4 = section_words(".text", "gt/gp4.so")
    g4a, got = section_words(".got", "gt/gp4.so")
    h = s4["handler"]
    check("occurrences of handler's address in .text", sum(1 for w in t4 if w == h), 0)
    check("it is in the GOT instead", [i for i, w in enumerate(got) if w == h], [2])
    # gp = _gp_disp + $t9, and $t9 holds the address of the function on entry.
    gp_runtime = ((t4[0] & 0xFFFF) << 16) + _s16(t4[1] & 0xFFFF) + s4["__start"]
    check("computed $gp at runtime == .got + 0x7ff0", gp_runtime, g4a + 0x7FF0)
    off = _s16(t4[3] & 0xFFFF)          # lw t9, off(gp)
    check("the %call16 load lands on that GOT slot", gp_runtime + off, g4a + 8)
    print(f"         .got[0]={got[0]:#010x}  .got[1]={got[1]:#010x}  .got[2]={got[2]:#010x}")
    print("         The first two are a HEADER, not pointers: slot 0 is reserved for the")
    print("         lazy resolver and slot 1 carries the high bit as a module marker. A data")
    print("         scan that does not know that is reading a header as content. Here it is")
    print("         harmless only because 0x0 and 0x80000000 both fall outside the text")
    print("         range, which is luck rather than design.")

    print(f"\n  {'all predictions held' if not fails else str(len(fails)) + ' PREDICTION(S) REFUTED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
