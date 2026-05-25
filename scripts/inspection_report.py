"""Inspection PDF report builder for Group Gamma.

Consumes data from the RobotCommander node and writes a PDF summary:
  - Ring counting (requester + total + per-color breakdown)
  - Barrel inspection (requester + per-barrel color/orientation/spill, photo on spill)
  - Anomaly detection (requester + cells visited + per-tile photos for anomalies)

Requires `pip install reportlab`.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table,
)


GROUP_NAME = 'Gamma'
ANOMALIES_DIR = '/home/gamma/colcon_ws/anomalies'
ANOMALY_RESULTS_PATH = '/home/gamma/colcon_ws/tile_anomaly_results.json'


def parse_face_class(cls: str | None) -> dict:
    """Split a face class such as 'alice_she_her_engineer' into {name, gender, role}."""
    if not cls:
        return {'name': 'unknown', 'gender': 'unknown', 'role': 'unknown', 'raw': str(cls or '')}
    parts = [p for p in cls.strip().lower().split('_') if p]
    gender = 'unknown'
    p_start = p_end = -1
    for i in range(len(parts) - 1):
        if parts[i] == 'she' and parts[i + 1] == 'her':
            gender = 'woman'
            p_start, p_end = i, i + 1
            break
        if parts[i] == 'he' and parts[i + 1] == 'him':
            gender = 'man'
            p_start, p_end = i, i + 1
            break
    if p_start >= 0:
        name = '_'.join(parts[:p_start]) or 'unknown'
        role = '_'.join(parts[p_end + 1:]) or 'unknown'
    elif len(parts) >= 2:
        name, role = parts[0], '_'.join(parts[1:])
    else:
        name = parts[0] if parts else 'unknown'
        role = 'unknown'
    return {'name': name, 'gender': gender, 'role': role, 'raw': cls}


def color_name(rgb01: tuple[float, float, float]) -> str:
    r, g, b = rgb01
    bgr = np.uint8([[[int(b * 255), int(g * 255), int(r * 255)]]])
    h, s, v = (int(x) for x in cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].tolist())
    if v < 50:
        return 'black'
    if v > 200 and s < 30:
        return 'white'
    if s < 40:
        return 'gray'
    if h < 10 or h >= 170:
        return 'red'
    if h < 22:
        return 'orange'
    if h < 35:
        return 'yellow'
    if h < 85:
        return 'green'
    if h < 130:
        return 'blue'
    if h < 160:
        return 'purple'
    return 'pink'


def _requesters_for(face_requests: list[dict], tasks: set[str]) -> list[dict]:
    return [fr['identity'] for fr in face_requests if fr.get('task') in tasks]


def _format_requester(identities: list[dict]) -> str:
    if not identities:
        return 'no requester'
    return '; '.join(
        f"{i['name']} ({i['gender']}, {i['role']})" for i in identities
    )


def _load_anomaly_json() -> dict:
    try:
        with open(ANOMALY_RESULTS_PATH) as f:
            data = json.load(f)
        return {
            'total':     data.get('tiles_inspected', 0),
            'entries':   data.get('results', []),
            'anomalous': sum(1 for r in data.get('results', []) if r.get('anomaly')),
        }
    except Exception:
        return {'total': 0, 'entries': [], 'anomalous': 0}


def _ring_color_breakdown(rings: list[tuple]) -> Counter:
    return Counter(color_name((r[2], r[3], r[4])) for r in rings if len(r) >= 5)


def _match_spills_to_horizontal(horizontals, spills, snapshots: dict,
                                 max_dist: float = 0.8) -> list[dict]:
    out, used = [], set()
    # cyl_horiz entries are (cx, cy, gx, gy, r, g, b) — see robot_commander._cyl_horiz_cb.
    # Legacy (cx, cy, r, g, b) tuples are still tolerated for older recordings.
    for b in horizontals:
        bx, by = b[0], b[1]
        if len(b) >= 7:
            col = (b[4], b[5], b[6])
        elif len(b) >= 5:
            col = (b[2], b[3], b[4])
        else:
            col = (0.0, 0.0, 0.0)
        best_i, best_d = -1, max_dist
        for i, s in enumerate(spills):
            if i in used:
                continue
            d = math.hypot(s[0] - bx, s[1] - by)
            if d < best_d:
                best_d, best_i = d, i
        snap = snapshots.get(best_i) if best_i >= 0 else None
        if best_i >= 0:
            used.add(best_i)
        out.append({
            'x': bx, 'y': by, 'color': col, 'orientation': 'horizontal',
            'spill': best_i >= 0, 'snapshot': snap,
        })
    return out


def generate_report(rc, output_path: str) -> None:
    """Render the inspection report. `rc` is the RobotCommander node."""
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    h1 = styles['Heading1']
    body = styles['BodyText']

    elements = [
        Paragraph(f'Inspection Report — Group {GROUP_NAME}', title_style),
        Paragraph(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', body),
        Spacer(1, 0.3 * inch),
    ]

    face_requests = getattr(rc, 'face_requests', [])

    # ── Ring counting ───────────────────────────────────────────────────────
    elements.append(Paragraph('Ring Counting', h1))
    ring_reqs = _requesters_for(face_requests, {'RINGS'})
    elements.append(Paragraph(f'Requested by: {_format_requester(ring_reqs)}', body))
    rings = rc._confirmed.get('ring', [])
    elements.append(Paragraph(f'Total rings detected: {len(rings)}', body))
    if rings:
        for name, n in _ring_color_breakdown(rings).most_common():
            elements.append(Paragraph(f'&nbsp;&nbsp;{name}: {n}', body))
    elements.append(Spacer(1, 0.2 * inch))

    # ── Barrel inspection ───────────────────────────────────────────────────
    elements.append(Paragraph('Barrel Inspection', h1))
    barrel_reqs = _requesters_for(face_requests, {'BARRELS'})
    elements.append(Paragraph(f'Requested by: {_format_requester(barrel_reqs)}', body))

    horizontals = rc._confirmed.get('cyl_horiz', [])
    verticals = rc._confirmed.get('cyl_vert', [])
    spills = rc._confirmed.get('spill', [])
    snapshots = getattr(rc, 'spill_snapshots', {})

    h_barrels = _match_spills_to_horizontal(horizontals, spills, snapshots)
    v_barrels = [{
        'x': b[0], 'y': b[1],
        'color': (b[2], b[3], b[4]) if len(b) >= 5 else (0.0, 0.0, 0.0),
        'orientation': 'vertical', 'spill': False, 'snapshot': None,
    } for b in verticals]
    all_barrels = v_barrels + h_barrels
    elements.append(Paragraph(f'Total barrels inspected: {len(all_barrels)}', body))

    for i, b in enumerate(all_barrels, 1):
        elements.append(Paragraph(
            f'Barrel {i}: color={color_name(b["color"])}, '
            f'orientation={b["orientation"]}, '
            f'spill={"yes" if b["spill"] else "no"}',
            body,
        ))
        if b['spill'] and b['snapshot'] and os.path.exists(b['snapshot']):
            elements.append(RLImage(b['snapshot'], width=4 * inch, height=3 * inch))
            elements.append(Spacer(1, 0.1 * inch))
    elements.append(Spacer(1, 0.2 * inch))

    # ── Anomaly detection ───────────────────────────────────────────────────
    elements.append(Paragraph('Anomaly Detection', h1))
    anom_reqs = _requesters_for(face_requests, {'ANOMALY_RED', 'ANOMALY_GREEN'})
    elements.append(Paragraph(f'Requested by: {_format_requester(anom_reqs)}', body))

    cells = set(getattr(rc, 'anomaly_cells_visited', []))
    if {'red', 'green'} <= cells:
        cell_str = 'both red and green cells'
    elif cells == {'red'}:
        cell_str = 'red cell'
    elif cells == {'green'}:
        cell_str = 'green cell'
    else:
        cell_str = 'none'
    elements.append(Paragraph(f'Cells approached: {cell_str}', body))

    anom = _load_anomaly_json()
    elements.append(Paragraph(f'Total tiles inspected: {anom["total"]}', body))
    elements.append(Paragraph(f'Anomalies found: {anom["anomalous"]}', body))

    for entry in anom['entries']:
        idx = entry['index']
        normal_path = os.path.join(ANOMALIES_DIR, f'tile_{idx:03d}_normal.png')
        heatmap_path = os.path.join(ANOMALIES_DIR, f'tile_{idx:03d}.png')
        mask_path = os.path.join(ANOMALIES_DIR, f'tile_{idx:03d}_mask.png')
        anomaly_flag = entry.get('anomaly')
        if anomaly_flag is True:
            verdict = 'ANOMALY'
        elif anomaly_flag is False:
            verdict = 'OK'
        else:
            verdict = 'uninspected'
        score = entry.get('score')
        score_str = f'{score:.3f}' if isinstance(score, (int, float)) else 'n/a'
        elements.append(Paragraph(f'Tile #{idx}: {verdict} (score={score_str})', body))
        row = []
        for p in (normal_path, heatmap_path, mask_path):
            if os.path.exists(p):
                row.append(RLImage(p, width=2.0 * inch, height=1.5 * inch))
        if row:
            elements.append(Table([row], colWidths=[2.1 * inch] * len(row)))
        elements.append(Spacer(1, 0.1 * inch))

    SimpleDocTemplate(output_path, pagesize=A4).build(elements)
