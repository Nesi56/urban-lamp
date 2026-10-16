"""Double-click / ``python run_phasescope.py`` launcher (same options as ``python -m phasescope``)."""

import sys

from phasescope.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
