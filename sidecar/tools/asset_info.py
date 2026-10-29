#!/usr/bin/env python3
"""Print per-mesh world extents/stats of an asset (assets/<name>/asset.json)."""
import json
import os
import sys

import numpy as np

ASSETS = os.environ.get('GLADOS_ASSETS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets'))
name = sys.argv[1] if len(sys.argv) > 1 else 'procedural'
a = json.load(open(os.path.join(ASSETS, name, 'asset.json')))
bod = {b['name']: b for b in a['bodies']}
for b in a['bodies']:
    print(b)
print('eye', a['eye'], 'voice', a['voice'], 'tris', a['triangles'])
allv = []
for mm in a['meshes']:
    V = np.array(mm['verts']).reshape(-1, 3)
    if not mm.get('skin'):
        V = V + np.array(bod[mm['body']]['rest'][:3])
    allv.append(V)
    print(f"{mm['name']:24s} {mm['body']:6s} tag={mm.get('tag')} n={len(mm['tris'])//3:5d} "
          f"min {np.round(V.min(0), 2)} max {np.round(V.max(0), 2)} {mm.get('objects')}")
allv = np.vstack(allv)
print('ALL min', np.round(allv.min(0), 2), 'max', np.round(allv.max(0), 2))
