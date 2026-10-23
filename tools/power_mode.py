"""Get or set the Windows power mode (power slider overlay) and verify it.

GUIDs from Microsoft Learn, "Customize the Windows performance power slider":
  efficiency  961cc777-2547-4f9d-8174-7d86181b8a7a  (Better Battery / Best power efficiency)
  balanced    00000000-0000-0000-0000-000000000000
  performance ded574b5-45a0-4f42-8737-46345c09c238  (Best Performance)

  python tools/power_mode.py            # print the effective mode
  python tools/power_mode.py performance
"""
import ctypes
import sys
import uuid

MODES = {
    "efficiency": "961cc777-2547-4f9d-8174-7d86181b8a7a",
    "balanced": "00000000-0000-0000-0000-000000000000",
    "performance": "ded574b5-45a0-4f42-8737-46345c09c238",
}
NAMES = {v: k for k, v in MODES.items()}


class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, s):
        g = cls()
        ctypes.memmove(ctypes.addressof(g), uuid.UUID(s).bytes_le, 16)
        return g

    def __str__(self):
        return str(uuid.UUID(bytes_le=ctypes.string_at(ctypes.addressof(self), 16)))


_powrprof = ctypes.WinDLL("powrprof")


def get_mode():
    g = GUID()
    if _powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(g)) != 0:
        raise OSError("PowerGetEffectiveOverlayScheme failed")
    return NAMES.get(str(g), str(g))


def set_mode(name):
    rc = _powrprof.PowerSetActiveOverlayScheme(GUID.parse(MODES[name]))
    if rc != 0:
        raise OSError(f"PowerSetActiveOverlayScheme failed ({rc})")
    if get_mode() != name:
        raise OSError(f"mode did not change: effective mode is {get_mode()}")
    return name


def main(argv):
    if argv and argv[0] not in MODES:
        print(f"usage: power_mode.py [{'|'.join(MODES)}]", file=sys.stderr)
        return 2
    if argv:
        set_mode(argv[0])
    print(get_mode())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
