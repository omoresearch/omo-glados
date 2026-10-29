#!/usr/bin/env python3
"""Debug: print uv ranges (atlas px) of selected meshes of an asset.

    python3 tools/uv_debug.py <asset> <mesh-name-substring> ...
"""
import json
import os
import sys

from PIL import Image

import numpy as np

ASSETS = os.environ.get('GLADOS_ASSETS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets'))
d = os.path.join(ASSETS, sys.argv[1])
a = json.load(open(os.path.join(d, 'asset.json')))
AW, AH = Image.open(os.path.join(d, a.get('atlas') or 'atlas.png')).size
for m in a['meshes']:
    if any(k in m['name'] for k in sys.argv[2:]):
        uv = np.array(m['uv']).reshape(-1, 2) * [AW, AH]
        V = np.array(m['verts']).reshape(-1, 3)
        print(m['name'], len(uv), 'uv px min', uv.min(0).round(1), 'max', uv.max(0).round(1))
        print('  verts min', V.min(0).round(3), 'max', V.max(0).round(3))
        print('  first uvs', uv[:12].round(1).tolist())
