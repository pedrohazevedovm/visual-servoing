import os
import sys

_octhed_dir = os.path.dirname(os.path.abspath(__file__))
if _octhed_dir not in sys.path:
    sys.path.insert(0, _octhed_dir)
