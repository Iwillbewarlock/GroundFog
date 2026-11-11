"""List direct calls inside a Skyrim AE function, naming targets by Address Library ID.

Usage: python -I addrlib_calls.py <versionlib.bin> <SkyrimSE.exe> <function_id> [max_bytes]
Read-only: opens both files for reading, prints to stdout.
"""
from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path

import capstone
import pefile

log = logging.getLogger("addrlib_calls")

FORMAT_VERSION_AE = 2


class Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        if self._pos + size > len(self._data):
            raise ValueError(f"address library truncated at byte {self._pos}")
        (value,) = struct.unpack_from(fmt, self._data, self._pos)
        self._pos += size
        return value

    def skip(self, count: int) -> None:
        self._pos += count


def load_id_map(path: Path) -> dict[int, int]:
    """Port of CommonLibSSE-NG REL::IDDatabase::unpack_file (include/REL/ID.h:337-415)."""
    r = Reader(path.read_bytes())
    fmt = r.read("<i")
    if fmt != FORMAT_VERSION_AE:
        raise ValueError(f"{path}: format {fmt}, expected {FORMAT_VERSION_AE} (AE address library)")
    r.skip(16)  # version[4]
    r.skip(r.read("<i"))  # name
    ptr_size = r.read("<i")
    count = r.read("<i")
    id_reads = {2: "<B", 3: "<B", 4: "<H", 5: "<H"}
    off_reads = {2: "<B", 3: "<B", 4: "<H", 5: "<H"}
    id2off: dict[int, int] = {}
    prev_id = prev_off = 0
    for _ in range(count):
        t = r.read("<B")
        lo, hi = t & 0xF, t >> 4
        if lo == 0:
            id_ = r.read("<Q")
        elif lo == 1:
            id_ = prev_id + 1
        elif lo in (2, 4):
            id_ = prev_id + r.read(id_reads[lo])
        elif lo in (3, 5):
            id_ = prev_id - r.read(id_reads[lo])
        elif lo == 6:
            id_ = r.read("<H")
        elif lo == 7:
            id_ = r.read("<I")
        else:
            raise ValueError(f"unhandled id type {lo}")
        tmp = prev_off // ptr_size if hi & 8 else prev_off
        h = hi & 7
        if h == 0:
            off = r.read("<Q")
        elif h == 1:
            off = tmp + 1
        elif h in (2, 4):
            off = tmp + r.read(off_reads[h])
        elif h in (3, 5):
            off = tmp - r.read(off_reads[h])
        elif h == 6:
            off = r.read("<H")
        else:
            off = r.read("<I")
        if hi & 8:
            off *= ptr_size
        id2off[id_] = off
        prev_id, prev_off = id_, off
    return id2off


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("addrlib", type=Path)
    ap.add_argument("exe", type=Path)
    ap.add_argument("function_id", type=int)
    ap.add_argument("max_bytes", type=lambda s: int(s, 0), nargs="?", default=0x1200)
    a = ap.parse_args()

    id2off = load_id_map(a.addrlib)
    off2id = {v: k for k, v in id2off.items()}
    if a.function_id not in id2off:
        log.error("ID %d not in %s", a.function_id, a.addrlib)
        return 1
    pe = pefile.PE(str(a.exe), fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    rva = id2off[a.function_id]
    code = pe.get_data(rva, a.max_bytes)
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    log.info("function %d at RVA 0x%X", a.function_id, rva)
    for ins in md.disasm(code, base + rva):
        rel = ins.address - base - rva
        if ins.mnemonic == "call" and ins.op_str.startswith("0x"):
            target = int(ins.op_str, 16) - base
            log.info("+0x%04X call %-8s (RVA 0x%X)", rel, off2id.get(target, "?"), target)
        elif ins.mnemonic == "int3" and rel > 0x20:
            log.info("+0x%04X int3 (function end?)", rel)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
