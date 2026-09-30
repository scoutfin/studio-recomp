#!/usr/bin/env python3
"""Does each test program's control flow match what its comment claims?

Written 2026-09-30, an hour after finding that `absint.GUARDED` does not. Its
comment says "if NOT less, branch away to a default" and its `beq` offset lands
on 0x1c — the `jr` instruction itself. There is no default. The guard-failed
path falls onto the dispatch with the register never assigned, so the honest
answer for that program is TOP, and the "exactly four handlers" I published on
2026-09-24 was my straight-line analyser being unable to follow a branch.

One fixture diverged from its own comment, so I have no reason to trust the
others. This is the mechanical half of that audit — the half I can automate:

  1. does every branch target land inside the program?
  2. does any branch target land ON a `jr`?  <- the bug I found
  3. is any instruction unreachable from the entry point?
  4. is the delay slot after each branch/jr accounted for?

What it cannot check is whether the prose above a program describes what the
program does. That still needs reading, and the point of automating (1)-(4) is
to make the reading shorter rather than to replace it.
"""
import sys
from mips import disasm

BRANCHES = ("beq", "bne")


def audit(name, words, base=0, verbose=False):
    n = len(words)
    end = base + n * 4
    findings = []
    jr_pcs, dispatch_pcs, branch_edges = set(), set(), []

    rows = []
    for i, w in enumerate(words):
        pc = base + i * 4
        m, o, tgt = disasm(w, pc)
        rows.append((pc, m, o, tgt))
        if m == "jr":
            jr_pcs.add(pc)
            if o and o[0] != 31:                      # 31 = $ra, i.e. a return
                dispatch_pcs.add(pc)
        if m in BRANCHES:
            branch_edges.append((pc, tgt))

    # 1 + 2: branch targets
    for pc, tgt in branch_edges:
        if tgt is None:
            findings.append(("?", f"branch at {pc:#04x} has no computed target"))
            continue
        if not (base <= tgt < end):
            findings.append(("OUT", f"branch at {pc:#04x} targets {tgt:#04x}, "
                                    f"outside [{base:#04x},{end:#04x})"))
        elif tgt in dispatch_pcs:
            findings.append(("JR", f"branch at {pc:#04x} targets {tgt:#04x}, "
                                   f"which IS an indirect `jr` — the guard-failed "
                                   f"path lands on the dispatch"))
        elif tgt in jr_pcs:
            # `jr $ra` is a RETURN, and "if done, return" is the most ordinary
            # control flow there is. My first version flagged this and called
            # build.COLLATZ suspicious — a program validated by three
            # implementations agreeing on 500 inputs. The check was wrong, not
            # the fixture. Only a jr on a COMPUTED register is a dispatch.
            pass
        elif tgt == pc + 4:
            findings.append(("NOP", f"branch at {pc:#04x} targets its own delay slot"))

    # 3: reachability, following both edges and stopping at jr
    seen, work = set(), [base]
    while work:
        pc = work.pop()
        if pc in seen or not (base <= pc < end):
            continue
        seen.add(pc)
        idx = (pc - base) // 4
        m, o, tgt = rows[idx][1], rows[idx][2], rows[idx][3]
        if m == "jr":
            continue                                  # leaves the program
        if m in BRANCHES:
            if tgt is not None:
                work.append(tgt)
            work.append(pc + 4)                       # the delay slot
            work.append(pc + 8)                       # fall-through past it
        else:
            work.append(pc + 4)
    dead = [pc for pc, m, o, t in rows if pc not in seen]
    if dispatch_pcs:
        # An indirect `jr` can go anywhere this analysis has not resolved, so
        # static reachability is not available and "unreachable" is not a claim
        # I can make. My first run reported all 16 handler instructions of
        # indirect.JUMPTABLE as dead — every one reached through the dispatch
        # the fixture exists to demonstrate. 16 of 23 findings were this: a 70%
        # false-positive rate, which made the report useless as a report.
        if dead:
            findings.append(("n/a", f"{len(dead)} instruction(s) not statically "
                                    f"reachable, but this program dispatches "
                                    f"indirectly — reachability is undecidable "
                                    f"here and I am not calling them dead"))
    else:
        for pc in dead:
            findings.append(("DEAD", f"{pc:#04x} ({rows[(pc-base)//4][1]}) is "
                                     f"unreachable from the entry point"))

    # 4: delay slots
    for pc, tgt in branch_edges:
        idx = (pc - base) // 4 + 1
        if idx >= n:
            findings.append(("TAIL", f"branch at {pc:#04x} is the last instruction "
                                     f"— its delay slot is off the end"))
        elif rows[idx][1] != "nop":
            # INFO, not a defect. build.DELAY_PROBE exists precisely to probe
            # delay-slot semantics, so a filled slot is its purpose rather than
            # its bug. Worth surfacing because it changes what the program
            # means; not worth calling wrong.
            findings.append(("info", f"delay slot at {pc+4:#04x} is "
                                     f"`{rows[idx][1]}`, not a nop — executes on "
                                     f"BOTH paths (deliberate in a delay probe)"))

    if verbose or findings:
        print(f"\n  {name}  ({n} instructions, {base:#04x}–{end:#04x})")
        if verbose:
            for pc, m, o, t in rows:
                mark = " <-- jr" if m == "jr" else (f" -> {t:#04x}" if t is not None else "")
                print(f"      {pc:#04x}  {m:<6} {o}{mark}")
        for kind, msg in findings:
            print(f"      [{kind:4}] {msg}")
    if not findings:
        print(f"  clean   {name}")
    return findings


def main():
    verbose = "--verbose" in sys.argv
    import absint, cfg, indirect, build
    fixtures = [
        ("absint.UNGUARDED",   absint.UNGUARDED),
        ("absint.GUARDED",     absint.GUARDED),
        ("absint.LOADED",      absint.LOADED),
        ("cfg.COUNTER_LOOP",   cfg.COUNTER_LOOP),
        ("cfg.GUARDED_DEFAULT", cfg.GUARDED_DEFAULT),
        ("indirect.JUMPTABLE", indirect.JUMPTABLE),
        ("build.COLLATZ",      build.COLLATZ),
        ("build.DELAY_PROBE",  build.DELAY_PROBE),
    ]
    total = 0
    for name, words in fixtures:
        total += len(audit(name, words, 0, verbose))
    print(f"\n  {total} finding(s) across {len(fixtures)} fixtures.")
    print("  A finding is not automatically a bug — `absint.GUARDED` is kept")
    print("  deliberately broken as a regression test. It IS a place where the")
    print("  program and its comment need to be read against each other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
