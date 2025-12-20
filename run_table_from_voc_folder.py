"""
Copyright (c) Docugami, Inc. All rights reserved.

Created by Louise Naud on 10/7/25
Description:
Usage      :
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse, json, sys, xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
BBox = Tuple[float, float, float, float]  # (x1,y1,x2,y2)

# -------------------- Geometry helpers --------------------
def norm_class(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")

def inter_rect(a: BBox, b: BBox) -> Optional[BBox]:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def area(b: BBox) -> float:
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def overlap_ratio(a: BBox, b: BBox, mode: str = "iof_a") -> float:
    inter = inter_rect(a, b)
    if inter is None:
        return 0.0
    ai, bi = area(a), area(b)
    ia = area(inter)
    if mode == "iof_a":
        return ia / ai if ai > 0 else 0.0
    if mode == "iof_b":
        return ia / bi if bi > 0 else 0.0
    if mode == "iou":
        u = ai + bi - ia
        return ia / u if u > 0 else 0.0
    raise ValueError("bad mode")

def center(b: BBox) -> Tuple[float, float]:
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

def v_overlap_ratio(a: BBox, b: BBox) -> float:
    top = max(a[1], b[1]); bot = min(a[3], b[3])
    inter = max(0.0, bot - top)
    ha = max(0.0, a[3]-a[1]); hb = max(0.0, b[3]-b[1])
    denom = min(ha, hb) if min(ha, hb) > 0 else (ha + hb)
    return inter / denom if denom > 0 else 0.0

def h_overlap_ratio(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0]); right = min(a[2], b[2])
    inter = max(0.0, right - left)
    wa = max(0.0, a[2]-a[0]); wb = max(0.0, b[2]-b[0])
    denom = min(wa, wb) if min(wa, wb) > 0 else (wa + wb)
    return inter / denom if denom > 0 else 0.0

def expand_bbox(b: BBox, px: float = 2.0, rel: float = 0.02) -> BBox:
    w = max(0.0, b[2]-b[0]); h = max(0.0, b[3]-b[1])
    dx = px + rel * w
    dy = px + rel * h
    return (b[0]-dx, b[1]-dy, b[2]+dx, b[3]+dy)

# -------------------- Data structures --------------------
@dataclass
class VocObject:
    cls: str
    bbox: BBox

@dataclass
class TextChunk:
    text: str
    bbox: BBox

@dataclass
class Cell:
    r: int
    c: int
    bbox: BBox
    rowspan: int = 1
    colspan: int = 1
    text_chunks: List[TextChunk] = None

    def key(self): return (self.r, self.c)

# -------------------- Parsing --------------------
def parse_voc(xml_path: str) -> Dict[str, List[VocObject]]:
    tree = ET.parse(xml_path); root = tree.getroot()
    objects: List[VocObject] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        bnd = obj.find("bndbox");
        if bnd is None: continue
        x1 = float(bnd.findtext("xmin")); y1 = float(bnd.findtext("ymin"))
        x2 = float(bnd.findtext("xmax")); y2 = float(bnd.findtext("ymax"))
        objects.append(VocObject(norm_class(name), (x1,y1,x2,y2)))
    out = {"table": [], "table row": [], "table column": [], "table spanning cell": [], "spanning cell": [], "cell": []}
    for o in objects:
        if o.cls in out: out[o.cls].append(o)
        elif "spanning" in o.cls and "cell" in o.cls: out["table spanning cell"].append(o)
        elif o.cls in ("row",): out["table row"].append(o)
        elif o.cls in ("column","col"): out["table column"].append(o)
        elif o.cls == "table cell" or o.cls.endswith("cell"): out["cell"].append(o)
    return out

def parse_text_json(json_path: str) -> List[TextChunk]:
    with open(json_path, "r", encoding="utf-8") as f: items = json.load(f)
    chunks: List[TextChunk] = []
    for it in items:
        text = it.get("text","")
        bb = it.get("bbox") or it.get("box") or it.get("bounds")
        if not text or not bb or len(bb)!=4: continue
        x1,y1,x2,y2 = map(float, bb)
        chunks.append(TextChunk(text=text, bbox=(x1,y1,x2,y2)))
    return chunks

# -------------------- Grid building --------------------
def sort_rows_cols(rows: List[VocObject], cols: List[VocObject]):
    rows_sorted = sorted(rows, key=lambda o: (o.bbox[1], o.bbox[3]-o.bbox[1]))
    cols_sorted = sorted(cols, key=lambda o: (o.bbox[0], o.bbox[2]-o.bbox[0]))
    return rows_sorted, cols_sorted

def build_base_cells(rows: List[VocObject], cols: List[VocObject],
                     gap_px: float = 2.0, gap_rel: float = 0.01) -> List[Cell]:
    cells: List[Cell] = []
    exp_rows = [expand_bbox(r.bbox, px=gap_px, rel=gap_rel) for r in rows]
    exp_cols = [expand_bbox(c.bbox, px=gap_px, rel=gap_rel) for c in cols]
    for ri, rbox in enumerate(exp_rows):
        for ci, cbox in enumerate(exp_cols):
            inter = inter_rect(rbox, cbox)
            if inter:
                cells.append(Cell(r=ri, c=ci, bbox=inter, rowspan=1, colspan=1, text_chunks=[]))
    return cells

def map_spanner_to_grid(sp_bbox: BBox, rows: List[VocObject], cols: List[VocObject],
                        v_thresh: float = 0.5, h_thresh: float = 0.5) -> Optional[Tuple[int,int,int,int]]:
    r_idxs = [i for i, r in enumerate(rows) if v_overlap_ratio(sp_bbox, r.bbox) >= v_thresh]
    c_idxs = [j for j, c in enumerate(cols) if h_overlap_ratio(sp_bbox, c.bbox) >= h_thresh]
    if not r_idxs or not c_idxs: return None
    return min(r_idxs), max(r_idxs), min(c_idxs), max(c_idxs)

def apply_spanners(base_cells: List[Cell], rows: List[VocObject], cols: List[VocObject],
                   spanners: List[VocObject]):
    """
    Returns:
      ordered_cells: list[Cell] with anchors updated
      span_groups: dict[(r0,c0)] -> set of (r,c) covered INCLUDING the anchor
      suppressed_positions: set[(r,c)] of all covered non-anchor cells (used by renderer)
    """
    cell_map: Dict[Tuple[int,int], Cell] = {(cell.r, cell.c): cell for cell in base_cells}
    anchors: List[Cell] = []
    span_groups: Dict[Tuple[int,int], set] = {}
    suppressed_positions: set[Tuple[int,int]] = set()

    for sp in spanners:
        mapped = map_spanner_to_grid(expand_bbox(sp.bbox, px=2.0, rel=0.01), rows, cols)
        if not mapped: continue
        r0, r1, c0, c1 = mapped

        anchor = cell_map.get((r0, c0))
        if not anchor:
            # synthesize if a base cell is missing
            rbox = rows[r0].bbox; rbox2 = rows[r1].bbox
            cbox = cols[c0].bbox; cbox2 = cols[c1].bbox
            synth = (min(rbox[0], rbox2[0], sp.bbox[0], cbox[0]),
                     min(rbox[1], cbox[1], sp.bbox[1]),
                     max(rbox2[2], cbox2[2], sp.bbox[2]),
                     max(rbox2[3], cbox2[3], sp.bbox[3]))
            anchor = Cell(r=r0, c=c0, bbox=synth, rowspan=r1-r0+1, colspan=c1-c0+1, text_chunks=[])
            cell_map[(r0, c0)] = anchor
        else:
            anchor.rowspan = r1 - r0 + 1
            anchor.colspan = c1 - c0 + 1

        anchors.append(anchor)

        covered = set()
        for rr in range(r0, r1+1):
            for cc in range(c0, c1+1):
                covered.add((rr, cc))
                if (rr, cc) != (r0, c0):
                    suppressed_positions.add((rr, cc))
        span_groups[(r0, c0)] = covered

    # keep all base cells (we'll fold text first; renderer will skip suppressed)
    final_cells: Dict[Tuple[int,int], Cell] = {(c.r, c.c): c for c in base_cells}
    for a in anchors: final_cells[(a.r, a.c)] = a
    ordered_cells = sorted(final_cells.values(), key=lambda ce: (ce.r, ce.c))
    return ordered_cells, span_groups, suppressed_positions

# -------------------- Text assignment & folding --------------------
def reading_order_sort(chunks: List[TextChunk]) -> List[TextChunk]:
    if not chunks: return []
    heights = sorted((c.bbox[3]-c.bbox[1]) for c in chunks)
    med_h = heights[len(heights)//2] if heights else 10.0
    tol = max(6.0, 0.6 * med_h)
    def line_key(c: TextChunk):
        cx, cy = center(c.bbox)
        return (int(round(cy / tol)), cx)
    return sorted(chunks, key=line_key)

def chunks_to_text(chunks: List[TextChunk]) -> str:
    ordered = reading_order_sort(chunks)
    parts = []
    for t in ordered:
        s = (t.text or "").strip()
        if s: parts.append(s)
    return " ".join(parts)

def assign_text_to_cells(cells: List[Cell], chunks: List[TextChunk]) -> None:
    if not cells or not chunks: return
    cmap: Dict[Tuple[int,int], Cell] = {(c.r, c.c): c for c in cells}
    # Stage 1: max IoF(text, expanded cell)
    expanded: Dict[Tuple[int,int], BBox] = {(c.r,c.c): expand_bbox(c.bbox, px=3.0, rel=0.02) for c in cells}
    assigned = set()
    for idx, ch in enumerate(chunks):
        best_rc, best_sc = None, -1.0
        for ce in cells:
            sc = overlap_ratio(ch.bbox, expanded[(ce.r, ce.c)], mode="iof_a")
            if sc > best_sc:
                best_sc, best_rc = sc, (ce.r, ce.c)
        if best_rc and best_sc >= 0.15:
            ce = cmap[best_rc]
            ce.text_chunks = (ce.text_chunks or []) + [ch]
            assigned.add(idx)
    # Stage 2: row-band + nearest column fallback
    rows_n = max(c.r for c in cells) + 1
    cols_n = max(c.c for c in cells) + 1
    row_edges = []
    for r in range(rows_n):
        ys = [cmap[(r,c)].bbox for c in range(cols_n) if (r,c) in cmap]
        if not ys: row_edges.append((float("inf"), float("-inf"))); continue
        row_edges.append((min(b[1] for b in ys), max(b[3] for b in ys)))
    col_edges = []
    for c in range(cols_n):
        xs = [cmap[(r,c)].bbox for r in range(rows_n) if (r,c) in cmap]
        if not xs: col_edges.append((float("inf"), float("-inf"))); continue
        col_edges.append((min(b[0] for b in xs), max(b[2] for b in xs)))
    def which_row(y: float, tol: float = 6.0):
        for i,(y1,y2) in enumerate(row_edges):
            if (y1 - tol) <= y <= (y2 + tol): return i
        return None
    def nearest_col(x: float):
        best, bid = 1e18, None
        for j,(x1,x2) in enumerate(col_edges):
            d = 0.0 if (x1 <= x <= x2) else min(abs(x-x1), abs(x-x2))
            if d < best: best, bid = d, j
        return bid
    for idx, ch in enumerate(chunks):
        if idx in assigned: continue
        cx, cy = center(ch.bbox)
        r = which_row(cy); c = nearest_col(cx)
        if r is None or c is None or (r,c) not in cmap: continue
        ce = cmap[(r,c)]
        ce.text_chunks = (ce.text_chunks or []) + [ch]
        assigned.add(idx)

def fold_spanning_text(cells: List[Cell], span_groups: Dict[Tuple[int,int], set]) -> None:
    if not span_groups: return
    cmap: Dict[Tuple[int,int], Cell] = {(c.r, c.c): c for c in cells}
    for anchor_rc, covered in span_groups.items():
        collected = []
        for rc in covered:
            ce = cmap.get(rc)
            if ce and ce.text_chunks:
                collected.extend(ce.text_chunks)
        if not collected: continue
        ordered = reading_order_sort(collected)
        anc = cmap.get(anchor_rc)
        if anc: anc.text_chunks = ordered
        for rc in covered:
            if rc == anchor_rc: continue
            ce = cmap.get(rc)
            if ce: ce.text_chunks = []  # clear covered cells' chunks

# -------------------- HTML rendering (with suppression) --------------------
def render_html(rows_n: int, cols_n: int, cells: List[Cell],
                suppressed_positions: Optional[set] = None) -> str:
    """
    suppressed_positions contains ALL (r,c) that are covered by any span **except** the anchor.
    We skip rendering those, across all affected rows, so rowspan/colspan produce no empty TDs.
    """
    suppressed_positions = suppressed_positions or set()
    cmap: Dict[Tuple[int,int], Cell] = {(c.r, c.c): c for c in cells}

    lines = []
    lines.append("<table>")
    for r in range(rows_n):
        lines.append("  <tr>")
        c = 0
        while c < cols_n:
            rc = (r, c)
            # skip anything covered by a span (not an anchor)
            if rc in suppressed_positions:
                c += 1
                continue
            cell = cmap.get(rc)
            if cell is None:
                # no cell at this position (could be artifact) -> empty skip
                c += 1
                continue
            attrs = []
            if cell.rowspan and cell.rowspan > 1: attrs.append(f'rowspan="{cell.rowspan}"')
            if cell.colspan and cell.colspan > 1: attrs.append(f'colspan="{cell.colspan}"')
            txt = chunks_to_text(cell.text_chunks or [])
            txt = txt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            lines.append(f"    <td{attr_str}>{txt}</td>")
            # Advance by colspan on this row. Rows below are handled by suppressed_positions.
            c += max(1, cell.colspan)
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Generate HTML table from Pascal VOC + text chunks.")
    ap.add_argument("--image_path", help="Path to table crop image (not read; for reference)")
    ap.add_argument("--xml_path", help="Pascal VOC XML with table/row/column/(spanning cell)")
    ap.add_argument("--json_path", help="JSON with [{'text':..., 'bbox':[x1,y1,x2,y2]}, ...]")
    ap.add_argument("--out", default=None, help="Write HTML to this file (prints to stdout otherwise)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(args.xml_path, "*.xml")))

    for xml_file in xml_files:
        fn = Path(xml_file).stem
        json_path = os.path.join(args.json_path, f"{fn}_words.json")
        if not os.path.exists(json_path):
            print(f"Warning: No JSON found for {json_path}. Skipping.", file=sys.stderr)
            continue

        voc = parse_voc(xml_file)
        tables = voc.get("table", []) or []
        table_box = None
        if tables:
            tables.sort(key=lambda o: area(o.bbox), reverse=True)
            table_box = tables[0].bbox

        rows = voc.get("table row", []) or []
        cols = voc.get("table column", []) or []
        spanners = (voc.get("table spanning cell", []) or []) + (voc.get("spanning cell", []) or [])

        if not rows or not cols:
            print("Error: need at least one 'table row' and one 'table column' in XML.", file=sys.stderr)
            continue

        rows, cols = sort_rows_cols(rows, cols)
        base_cells = build_base_cells(rows, cols)

        # Build span metadata BEFORE assigning text (so covered cells exist for folding)
        cells, span_groups, suppressed_positions = apply_spanners(base_cells, rows, cols, spanners)

        chunks = parse_text_json(json_path)
        if table_box is not None:
            # keep chunks whose centers fall inside the main table
            chunks = [ch for ch in chunks if (table_box[0] <= center(ch.bbox)[0] <= table_box[2] and
                                              table_box[1] <= center(ch.bbox)[1] <= table_box[3])]

        assign_text_to_cells(cells, chunks)
        fold_spanning_text(cells, span_groups)

        html = render_html(rows_n=len(rows), cols_n=len(cols), cells=cells,
                           suppressed_positions=suppressed_positions)

        if args.out:
            out_path = os.path.join(args.out, f"{fn}.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Wrote HTML to {out_path}")
        else:
            print(html)

if __name__ == "__main__":
    main()