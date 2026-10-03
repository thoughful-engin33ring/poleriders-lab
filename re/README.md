# Reverse-engineering helpers

- `dis.txt` — full disassembly of the ORIGINAL binary (`Sportsfriends.orig`), objdump with symbols.
  Regenerate: `objdump -d --no-show-raw-insn --macho "<game>/Sportsfriends.orig" > re/dis.txt`
- `dz.py START_HEX COUNT` — print COUNT instructions from address START with every RIP-relative
  literal resolved to its float/double value (e.g. `python3 re/dz.py 10013b8c8 200`).
- Symbols: `nm -C Sportsfriends.orig | grep PoleRiders::` (the binary is unstripped C++).
- Find a function: `grep -n "^__ZN10PoleRiders9PlayState6UpdateEv:" re/dis.txt`
- Binary is PIE x86_64; file offset = vmaddr - 0x100000000 for __TEXT. Runs under Rosetta.
