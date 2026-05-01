"""Generate datasets of (instance, baseline, query) triples for the four
emergent-property archetypes:

    cluster        — facility density (count of opened sites in a region)
    coverage_gap   — interior pocket of stranded voters (renamed from doughnut)
    contiguity     — non-contiguous catchments (count metric)
    shape_niceness — ugly catchment shapes (mean NPI)

Each pair carries BOTH a `vague_text` (no entity reference) and a
`precise_text` (names the offending site / region) variant. run_dataset.py
picks one based on a CLI flag and runs both modalities against it.

Output structure:

    <output_dir>/
        index.json                      summary + list of pair records
        pairs/00/instance.pkl
        pairs/00/baseline_solution.pkl
        pairs/00/query_metadata.json    archetype meta + vague_text + precise_text
        pairs/00/views/baseline_text_only.png
        ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =========================================================================
# JSON / region helpers
# =========================================================================
def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _region_label(center: Tuple[float, float],
                    bounds=(0.0, 0.0, 10.0, 10.0)) -> str:
    """Map a centre to a coarse region phrase ('eastern north', etc.)."""
    x, y = center
    river_y = (bounds[1] + bounds[3]) / 2
    ns = "north" if y > river_y else "south"
    if x < (bounds[0] + bounds[2]) / 3:
        ew = "western"
    elif x > 2 * (bounds[0] + bounds[2]) / 3:
        ew = "eastern"
    else:
        ew = "central"
    return f"{ew} {ns}"


def _format_template(template: str, **kwargs) -> str:
    """Format a template, ignoring placeholders not provided."""
    class _SafeFmt(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(_SafeFmt(**kwargs))


def _archetype_render_polygon(center, radius=1.2, n=24):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([center[0] + radius * np.cos(ang),
                          center[1] + radius * np.sin(ang)])
    return np.vstack([p, p[0:1]])


# =========================================================================
# Text templates per archetype × {vague, precise}
# =========================================================================
# VAGUE templates: stakeholder-style, no specific entity reference. The
# agent must localise the issue from the rendered map (multimodal) or
# from probing structured tools (tools-only).
#
# PRECISE templates: name the offending site index or region. The agent
# still has to reason about the fix, but identification is given.

CLUSTER_VAGUE_TEMPLATES: List[str] = [
    ("There's a cluster of polling places packed too tightly somewhere "
     "in the county. Multiple sites within a small radius is redundant "
     "when one or two would suffice. Spread them out by closing some, "
     "freeing budget for sparser areas."),
    ("Looking at the solution, I see an over-concentration of polling "
     "sites — several bunched together in a small area while other "
     "regions are sparser. Re-balance by reducing the cluster."),
    ("The current solution clusters too many sites in a small area. "
     "It looks redundant — please thin the cluster out."),
    ("There's an unnecessary concentration of opened sites somewhere on "
     "the map. Multiple sites within walking distance of each other is "
     "wasteful. Close one or two."),
    ("I see way too many polling locations bunched together. They're so "
     "close that closing one or two wouldn't really hurt coverage."),
]

CLUSTER_PRECISE_TEMPLATES: List[str] = [
    ("Polling places {site_list} are clustered too tightly together — "
     "I count {n_sites} sites within roughly {radius:.1f} km of each "
     "other. Close one or two of them and let the optimiser "
     "redistribute coverage, but be mindful about creating clusters elsewhere."),
    ("Sites {site_list} form a tight cluster of {n_sites} polling "
     "places in a small area. That's redundant — please thin it out, but be mindful about creating clusters elsewhere."),
    ("There's an over-concentration of polling places: sites "
     "{site_list} are within walking distance of each other. Close "
     "some so we free budget for sparser areas, but be mindful about creating clusters elsewhere."),
    ("Polling sites {site_list} are clustered together. That's "
     "wasteful — please reduce the cluster, but be mindful about creating clusters elsewhere."),
    ("I see {n_sites} polling places ({site_list}) bunched together. "
     "Spread them out by closing one or two, but be mindful about creating clusters elsewhere."),
]

COVERAGE_GAP_VAGUE_TEMPLATES: List[str] = [
    ("Looking at the solution, there's a noticeable hole in coverage "
     "somewhere — voters in a small interior region are stuck with "
     "much longer distances than their neighbours. Find that gap and "
     "fix it."),
    ("I see a coverage gap on the map. There's a small zone where "
     "voters have to travel substantially farther to reach a polling "
     "place compared to nearby areas. Address it."),
    ("There's a clear hole in the polling coverage. Voters in a small "
     "region are stranded — there's no nearby polling place even "
     "though the surrounding precincts are reasonably served. Fix it."),
    ("The solution has a pocket where voters are routed unusually far. "
     "The surrounding precincts have short trips, but this small "
     "interior region is anomalously underserved. Fix the worst of it."),
    ("There's a cluster of underserved precincts surrounded by reasonably-"
     "served areas — a coverage gap. Bring their access in line with "
     "the rest of the county."),
]

COVERAGE_GAP_PRECISE_TEMPLATES: List[str] = [
    ("Voters in precincts {affected_short} have anomalously bad polling "
     "access — their nearest opened site is much farther than for the "
     "surrounding precincts. Fix the coverage gap there, but be mindful about creating coverage gaps elsewhere."),
    ("Precincts {affected_short} are stranded — no polling place within "
     "reasonable distance. Bring their access in line with the "
     "surrounding area, but be mindful about creating coverage gaps elsewhere."),
    ("Affected precincts: {affected_short}. They have nearest-site "
     "distances much higher than their neighbours. Open a candidate "
     "near them or reassign, but be mindful about creating coverage gaps elsewhere."),
    ("There's a coverage gap affecting precincts {affected_short}. "
     "Their nearest polling place is much farther than their "
     "neighbours' — fix it, but be mindful about creating coverage gaps elsewhere."),
    ("Voters in precincts {affected_short} are anomalously underserved. "
     "Improve their access, but be mindful about creating coverage gaps elsewhere."),
]

CONTIGUITY_VAGUE_TEMPLATES: List[str] = [
    ("Looking at the colored service areas, at least one polling place "
     "is serving voters from visibly separated pockets — its catchment "
     "splits into disjoint clumps. Each polling place should serve one "
     "contiguous region. Fix every site whose service area is split."),
    ("I see polling places whose service areas aren't connected — the "
     "same color appears in two or more disjoint patches on the map, "
     "separated by other polling places' catchments. Address every "
     "split-catchment."),
    ("Some opened sites cover patches of precincts that aren't joined "
     "to each other. Each polling place's catchment should be a single "
     "connected region. Reassign so no site has a fragmented service "
     "area."),
    ("The map shows polling places whose voters come from disjoint "
     "pockets — same colour on two separate patches with a gap between. "
     "Break up any such split-catchment."),
    ("There are visible service-area splits: opened sites whose "
     "catchments appear in disconnected regions. Eliminate them."),
]

CONTIGUITY_PRECISE_TEMPLATES: List[str] = [
    ("Polling place {worst_site}'s service area is split into "
     "disconnected pockets — its catchment isn't a single contiguous "
     "region. Fix the split, but be mindful about causing this issue elsewhere."),
    ("The catchment of polling place {worst_site} is non-contiguous: "
     "its assigned precincts come in disjoint clumps. Reassign so the "
     "service area is one connected piece, but be mindful about causing this issue elsewhere."),
    ("Polling place {worst_site} has a fragmented service area — voters "
     "in separated pockets are all routed to it. Fix the topology, but be mindful about causing this issue elsewhere."),
    ("Site {worst_site}'s catchment looks split on the map — same color "
     "appearing in two disconnected patches. Repair the contiguity, but be mindful about causing this issue elsewhere."),
    ("Polling place {worst_site} is serving voters from at least two "
     "disjoint regions. That can't be right — make its catchment "
     "contiguous, but be mindful about causing this issue elsewhere."),
]

SHAPE_NICENESS_VAGUE_TEMPLATES: List[str] = [
    ("Some of these service regions look weird on the map — too "
     "elongated, oddly twisted, or with strange thin tails. Make them "
     "more compact and reasonable in shape."),
    ("Looking at the colored catchments, several are visibly bowtied "
     "or stretched out. Service areas should look more like compact "
     "blobs. Improve the shape."),
    ("The shapes of the service regions on the map are ugly — long, "
     "thin, jagged. Tidy them up so each catchment is more rounded."),
    ("Several polling places have catchments that look bizarre — "
     "stringy or with awkward protrusions. Reshape so the service "
     "areas are visually compact."),
    ("The catchments on the map have odd, elongated shapes. Real-world "
     "service areas should look reasonably compact. Address the worst "
     "offenders."),
]

SHAPE_NICENESS_PRECISE_TEMPLATES: List[str] = [
    ("Polling place {worst_site}'s catchment is unusually elongated — "
     "it has a long thin tail or protrusion that makes the service "
     "area look bad. Make it more compact, but be mindful about causing this issue elsewhere."),
    ("The catchment of polling place {worst_site} has a weird shape — "
     "stretched and jagged rather than a nice compact region. Reshape it, but be mindful about causing this issue elsewhere."),
    ("Site {worst_site} is serving an awkwardly-shaped catchment. The "
     "service area should be more circular / compact, not elongated, but be mindful about causing this issue elsewhere."),
    ("Polling place {worst_site}'s service region has an ugly shape on "
     "the map — long and thin with awkward bumps. Tidy it up, but be mindful about causing this issue elsewhere."),
    ("The catchment around polling place {worst_site} is bowtied / "
     "stretched. Make it visually compact, but be mindful about causing this issue elsewhere."),
]


# =========================================================================
# Per-archetype template formatting + meta extraction
# =========================================================================
def _build_query_texts(
    archetype: str,
    instance,
    metadata: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[str, str, int, int]:
    """Pick a (vague, precise) template pair for the archetype and format
    them with the instance-specific placeholders. Returns
    (vague_text, precise_text, vague_template_idx, precise_template_idx)."""
    if archetype == "cluster":
        cx, cy = metadata["cluster_center"]
        center = (float(cx), float(cy))
        affected_sites = metadata.get("affected_sites") or []
        # Format the cluster's site indices as a readable list.
        if affected_sites:
            site_list = ", ".join(str(int(j)) for j in affected_sites)
        else:
            site_list = "in that area"
        kwargs = {
            "region": _region_label(center),
            "site_list": site_list,
            "n_sites": int(metadata.get("cluster_size", len(affected_sites))),
            "radius": float(metadata.get("cluster_radius", 0.0)),
        }
        vt = list(CLUSTER_VAGUE_TEMPLATES)
        pt = list(CLUSTER_PRECISE_TEMPLATES)
    elif archetype == "coverage_gap":
        cx, cy = metadata["coverage_gap_center"]
        center = (float(cx), float(cy))
        affected = metadata.get("affected_precincts", [])
        # Show up to 5 indices in the precise text; "..." if more.
        if not affected:
            affected_short = "in that area"
        elif len(affected) <= 5:
            affected_short = ", ".join(str(int(i)) for i in affected)
        else:
            affected_short = (", ".join(str(int(i)) for i in affected[:5])
                              + ", ...")
        kwargs = {
            "region": _region_label(center),
            "affected_short": affected_short,
        }
        vt = list(COVERAGE_GAP_VAGUE_TEMPLATES)
        pt = list(COVERAGE_GAP_PRECISE_TEMPLATES)
    elif archetype == "contiguity":
        worst = int(metadata.get("worst_culprit_site",
                                    metadata.get("culprits",
                                                  [{"site": 0}])[0]["site"]))
        ctr = metadata.get("smallest_component_centroid", (5.0, 5.0))
        center = (float(ctr[0]), float(ctr[1]))
        kwargs = {
            "region": _region_label(center),
            "worst_site": worst,
        }
        vt = list(CONTIGUITY_VAGUE_TEMPLATES)
        pt = list(CONTIGUITY_PRECISE_TEMPLATES)
    elif archetype == "shape_niceness":
        worst = int(metadata.get("worst_catchment_site", 0))
        # Use the worst catchment's centroid as the "region anchor".
        per = metadata.get("per_catchment_npi", {})
        # per is keyed by str(site_idx); get the worst's components.
        # We don't have the centroid stored directly — compute from the
        # instance.
        try:
            assigned = instance.precinct_centroids[
                np.where(np.argmax(
                    np.array([1 if (k == str(worst)) else 0
                              for k in per.keys()]),
                    axis=0
                ) == 0)
            ]
            center = (5.0, 5.0)
        except Exception:
            center = (5.0, 5.0)
        kwargs = {
            "region": _region_label(center),
            "worst_site": worst,
        }
        vt = list(SHAPE_NICENESS_VAGUE_TEMPLATES)
        pt = list(SHAPE_NICENESS_PRECISE_TEMPLATES)
    else:
        raise ValueError(f"Unknown archetype: {archetype}")

    vidx = int(rng.integers(len(vt)))
    pidx = int(rng.integers(len(pt)))
    vague = _format_template(vt[vidx], **kwargs)
    precise = _format_template(pt[pidx], **kwargs)
    return vague, precise, vidx, pidx


# =========================================================================
# Main dataset generator
# =========================================================================
ARCHETYPE_NAMES = ["cluster", "coverage_gap", "contiguity", "shape_niceness"]


def generate_archetype_dataset(
    archetype: str,
    n_pairs: int = 25,
    output_dir: str = "out/dataset",
    seed_max: int = 15000,
    sampling_seed: int = 42,
    render_baselines: bool = True,
    verbose: bool = True,
    archetype_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate `n_pairs` pairs for one archetype.

    For coverage_gap, the generator does its own internal sweep over
    candidate centres; we simply call generate_coverage_gap_instance with
    incrementing seeds and let its inner loop handle the geometry sweep.

    Each accepted pair gets BOTH a vague_text and a precise_text written
    to query_metadata.json so the runner can serve either.
    """
    from generation import (
        generate_cluster_instance, generate_coverage_gap_instance,
        generate_contiguity_instance, generate_shape_niceness_instance,
    )
    try:
        from rendering import render_view, render_all_pair_views
    except Exception:
        render_view = None
        render_all_pair_views = None

    GEN_FNS = {
        "cluster": generate_cluster_instance,
        "coverage_gap": generate_coverage_gap_instance,
        "contiguity": generate_contiguity_instance,
        "shape_niceness": generate_shape_niceness_instance,
    }
    if archetype not in GEN_FNS:
        raise ValueError(
            f"Unknown archetype '{archetype}'. "
            f"Valid: {list(GEN_FNS)}"
        )
    gen_fn = GEN_FNS[archetype]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_root = out / "pairs"
    pairs_root.mkdir(exist_ok=True)

    rng = np.random.default_rng(sampling_seed)
    archetype_kwargs = dict(archetype_kwargs or {})

    pair_records: List[Dict[str, Any]] = []
    rejection_log: List[Dict[str, Any]] = []
    total_attempts = 0

    seed = 1
    while len(pair_records) < n_pairs and seed < seed_max:
        total_attempts += 1
        try:
            inst, sol, meta = gen_fn(
                base_seed=seed, max_attempts=1, verbose=False,
                **archetype_kwargs,
            )
        except RuntimeError as e:
            rejection_log.append({"seed": seed, "reason": str(e)[:200]})
            seed += 1
            continue

        # Build vague + precise text variants.
        vague, precise, vidx, pidx = _build_query_texts(
            archetype, inst, meta, rng)
        pair_idx = len(pair_records)
        meta_full = dict(meta)
        meta_full["vague_text"] = vague
        meta_full["precise_text"] = precise
        meta_full["vague_template_idx"] = vidx
        meta_full["precise_template_idx"] = pidx
        meta_full["pair_id"] = f"{pair_idx:02d}"

        pair_dir = pairs_root / f"{pair_idx:02d}"
        pair_dir.mkdir(exist_ok=True)
        inst.save(str(pair_dir / "instance.pkl"))
        sol.save(str(pair_dir / "baseline_solution.pkl"))
        with open(pair_dir / "query_metadata.json", "w") as f:
            json.dump(_make_json_safe(meta_full), f, indent=2)

        if render_baselines and render_all_pair_views is not None:
            views_dir = pair_dir / "views"
            views_dir.mkdir(exist_ok=True)
            anno_poly = _annotation_polygon(archetype, meta)
            render_all_pair_views(
                inst, sol, views_dir,
                annotation_polygon=anno_poly,
                title_prefix=f"Pair {pair_idx:02d} ({archetype}) — ",
            )

        record = {
            "pair_id": f"{pair_idx:02d}",
            "pair_dir": str(pair_dir.relative_to(out)),
            "archetype": archetype,
            "base_seed": meta.get("base_seed", seed),
            "vague_template_idx": vidx,
            "precise_template_idx": pidx,
        }
        pair_records.append(record)
        if verbose:
            print(f"  pair {pair_idx:02d}: seed={seed} archetype={archetype} "
                   f"vague_t={vidx} precise_t={pidx}")
        seed += 1

    if len(pair_records) < n_pairs and verbose:
        print(f"\nWARNING: only generated {len(pair_records)} of {n_pairs} "
               f"for {archetype}. ({total_attempts} seeds attempted, "
               f"{len(rejection_log)} rejected)")

    index = {
        "n_pairs": len(pair_records),
        "n_pairs_requested": n_pairs,
        "archetype": archetype,
        "sampling_seed": sampling_seed,
        "total_attempts": total_attempts,
        "n_rejected": len(rejection_log),
        "archetype_kwargs": archetype_kwargs,
        "pairs": pair_records,
    }
    with open(out / "index.json", "w") as f:
        json.dump(_make_json_safe(index), f, indent=2)
    if verbose:
        print(f"\nWrote {out / 'index.json'} with {len(pair_records)} pairs.")
    return index


def _annotation_polygon(archetype: str,
                          meta: Dict[str, Any]) -> Optional[List[np.ndarray]]:
    """Return an overlay polygon list for the precise/annotated condition,
    or None if the archetype has no natural anchor polygon."""
    if archetype == "cluster":
        cx, cy = meta["cluster_center"]
        rr = float(meta["cluster_radius"])
        return [_archetype_render_polygon((cx, cy), rr)]
    if archetype == "coverage_gap":
        cx, cy = meta["coverage_gap_center"]
        rr = float(meta["coverage_gap_radius"])
        return [_archetype_render_polygon((cx, cy), rr)]
    # contiguity and shape_niceness: no clean anchor polygon — the worst
    # site index is referenced in precise text instead.
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archetype", default="cluster", choices=ARCHETYPE_NAMES,
        help="Which archetype to generate.",
    )
    parser.add_argument("--n_pairs", type=int, default=25)
    parser.add_argument("--output_dir", default=None,
                        help="Default: out/<archetype>_dataset.")
    parser.add_argument("--seed_max", type=int, default=15000)
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--no_render", action="store_true",
                        help="Skip per-pair PNG rendering.")
    args = parser.parse_args()

    output_dir = args.output_dir or f"out/{args.archetype}_dataset"
    print(f"Generating {args.n_pairs} {args.archetype} pairs into "
           f"{output_dir}/ ...")
    index = generate_archetype_dataset(
        archetype=args.archetype,
        n_pairs=args.n_pairs,
        output_dir=output_dir,
        seed_max=args.seed_max,
        sampling_seed=args.sampling_seed,
        render_baselines=not args.no_render,
        verbose=True,
    )
    print(f"\nDone: {index['n_pairs']} pairs.")
    if index["n_pairs"] < args.n_pairs:
        sys.exit(1)


if __name__ == "__main__":
    main()
