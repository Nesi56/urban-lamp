"""Start PhaseScope:  ``python -m phasescope [--source sim|opencv|thorlabs] [--device N]``"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phasescope", description=__doc__)
    parser.add_argument("--source", choices=("sim", "opencv", "thorlabs"), default="sim",
                        help="camera source to open at start-up (default: simulated interferometer)")
    parser.add_argument("--device", default="",
                        help="OpenCV camera index or Thorlabs serial number (default: first found)")
    parser.add_argument("--no-gl", action="store_true",
                        help="use matplotlib instead of OpenGL for the 3D view")
    parser.add_argument("--no-autostart", action="store_true",
                        help="do not connect to the camera automatically")
    parser.add_argument("--list-thorlabs", action="store_true",
                        help="print the serial numbers of connected Thorlabs cameras and exit")
    args = parser.parse_args(argv)

    if args.list_thorlabs:
        from .camera import ThorlabsCamera
        for serial in ThorlabsCamera.list_cameras():
            print(serial)
        return 0

    import pyqtgraph as pg
    from .gui import MainWindow

    app = pg.mkQApp("PhaseScope")
    win = MainWindow(source=args.source, device=args.device, use_gl=not args.no_gl,
                     autostart=not args.no_autostart)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
