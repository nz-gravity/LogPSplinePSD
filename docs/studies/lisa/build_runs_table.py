"""Build an HTML table of LISA run summaries with PSD thumbnails.

Usage
-----
    .venv/bin/python docs/studies/lisa/build_runs_table.py

This scans ``docs/studies/lisa/runs/`` for saved ``compact_run_summary.json``
and ``psd_matrix.png`` artifacts, selects one representative seed directory per
run condition (preferring ``seed_0``), and writes a standalone HTML report.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
README_PATH = RUNS_DIR / "README.md"
DEFAULT_OUTPUT = RUNS_DIR / "index.html"


@dataclass(frozen=True)
class ReadmeEntry:
    label: str
    order: int
    section: str
    values: dict[str, str]


def _clean_md(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("`", "")
    return cleaned


def _split_md_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise ValueError(f"Not a markdown table row: {line!r}")
    return [_clean_md(cell) for cell in stripped[1:-1].split("|")]


def _is_separator_row(line: str) -> bool:
    stripped = line.strip().replace("|", "").replace(" ", "")
    return bool(stripped) and all(ch in "-:" for ch in stripped)


def parse_readme_tables(readme_path: Path) -> dict[str, ReadmeEntry]:
    entries: dict[str, ReadmeEntry] = {}
    if not readme_path.exists():
        return entries

    lines = readme_path.read_text(encoding="utf-8").splitlines()
    section = ""
    order = 0
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("## "):
            section = line[3:].strip()

        if (
            line.strip().startswith("|")
            and idx + 1 < len(lines)
            and _is_separator_row(lines[idx + 1])
        ):
            headers = _split_md_row(line)
            idx += 2
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                cells = _split_md_row(lines[idx])
                if len(cells) == len(headers):
                    values = dict(zip(headers, cells, strict=True))
                    label = values.get("Run", "").strip()
                    if label:
                        order += 1
                        entries[label] = ReadmeEntry(
                            label=label,
                            order=order,
                            section=section,
                            values=values,
                        )
                idx += 1
            continue
        idx += 1

    description_re = re.compile(r"^- \*\*run_([A-Za-z]+)\*\*: (.+)$")
    for line in lines:
        match = description_re.match(line.strip())
        if not match:
            continue
        label, description = match.groups()
        if label in entries and "Description" not in entries[label].values:
            entries[label].values["Description"] = _clean_md(description)
    return entries


def _extract_run_label(dirname: str) -> str:
    body = dirname[4:] if dirname.startswith("run_") else dirname
    match = re.match(r"([A-Za-z]+)(?:_|$)", body)
    return match.group(1) if match else body


def _parse_seed_num(seed_name: str) -> int:
    try:
        return int(seed_name.replace("seed_", ""))
    except ValueError:
        return 10**9


def _choose_representative_seed(seed_dirs: list[Path]) -> Path:
    preferred = sorted(seed_dirs, key=lambda path: _parse_seed_num(path.name))
    for seed_dir in preferred:
        if seed_dir.name == "seed_0":
            return seed_dir
    return preferred[0]


def _parse_run_slug_component(run_slug: str, prefix: str) -> str | None:
    pattern = re.compile(rf"(?:^|_){re.escape(prefix)}([^_]+)")
    match = pattern.search(run_slug)
    return match.group(1) if match else None


def _format_number(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if number == 0.0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e3:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _alpha_beta_from_readme_or_path(
    readme_entry: ReadmeEntry | None, run_dir_name: str
) -> str:
    if readme_entry is not None:
        for key in ("α_δ / β_δ", "α_δ/β_δ"):
            value = readme_entry.values.get(key)
            if value:
                return value.replace("/", " / ")
        note = readme_entry.values.get("Notes", "")
        match = re.search(
            r"alpha\s*=\s*([0-9.]+)\s*,\s*beta\s*=\s*([0-9.]+)",
            note,
            flags=re.IGNORECASE,
        )
        if match:
            return f"{match.group(1)} / {match.group(2)}"
    match = re.search(r"_a([0-9.]+)_b([0-9.]+)", run_dir_name)
    if match:
        return f"{match.group(1)} / {match.group(2)}"
    return "3.0 / 3.0"


def _null_excision_from_readme_or_name(
    readme_entry: ReadmeEntry | None, run_dir_name: str
) -> str:
    if readme_entry is not None:
        for key in ("Excision", "Null excision"):
            value = readme_entry.values.get(key)
            if value:
                return value
    lowered = run_dir_name.lower()
    if "no_excision" in lowered:
        return "none"
    if "wide_excise" in lowered:
        return "wide"
    if "hw3mhz" in lowered:
        return "hw±3mHz"
    if "nonull" in lowered:
        return "outside band"
    return ""


def _floor_from_readme_or_name(
    readme_entry: ReadmeEntry | None, run_dir_name: str
) -> str:
    if readme_entry is not None:
        value = readme_entry.values.get("Floor")
        if value:
            return value
    lowered = run_dir_name.lower()
    if "perfreq" in lowered:
        return "per-freq"
    if "floor_excise" in lowered:
        return "global"
    return ""


def _find_plot(seed_dir: Path) -> Path | None:
    primary = seed_dir / "psd_matrix.png"
    if primary.exists():
        return primary
    candidates = sorted(seed_dir.glob("psd_matrix*.png"))
    if candidates:
        return candidates[0]
    return None


def _condition_key(config_dir: Path) -> tuple[str, str]:
    return (config_dir.parent.name, config_dir.name)


def collect_rows(
    runs_dir: Path, readme_entries: dict[str, ReadmeEntry]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config_dirs = sorted(
        path
        for path in runs_dir.glob("run_*/*")
        if path.is_dir() and path.parent.name != "diagnostics"
    )
    for config_dir in config_dirs:
        seed_dirs = sorted(
            path for path in config_dir.glob("seed_*") if path.is_dir()
        )
        if not seed_dirs:
            continue
        seed_dir = _choose_representative_seed(seed_dirs)
        summary_path = seed_dir / "compact_run_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)

        run_dir_name = config_dir.parent.name
        label = _extract_run_label(run_dir_name)
        readme_entry = readme_entries.get(label)
        plot_path = _find_plot(seed_dir)
        run_slug = str(summary.get("run_slug", config_dir.name))
        nc_value = _parse_run_slug_component(run_slug, "nc")

        notes = ""
        if readme_entry is not None:
            notes = (
                readme_entry.values.get("Notes")
                or readme_entry.values.get("Hypothesis")
                or readme_entry.values.get("Description", "")
            )

        row = {
            "run_label": label,
            "run_dir": run_dir_name,
            "condition_dir": config_dir.name,
            "seed": summary.get("seed", _parse_seed_num(seed_dir.name)),
            "seed_count": len(seed_dirs),
            "duration_days": summary.get("duration_days"),
            "Nc": nc_value,
            "K": summary.get("K"),
            "diff_order": summary.get("diff_order"),
            "knot_method": summary.get("knot_method", ""),
            "alpha_beta": _alpha_beta_from_readme_or_path(
                readme_entry, run_dir_name
            ),
            "null_excision": _null_excision_from_readme_or_name(
                readme_entry, run_dir_name
            ),
            "floor": _floor_from_readme_or_name(readme_entry, run_dir_name),
            "ess_median": summary.get("ess_median"),
            "riae_matrix": summary.get("riae_matrix"),
            "coherence_riae": summary.get("coherence_riae"),
            "coverage": summary.get("coverage"),
            "ciw_psd_diag_median": summary.get("ciw_psd_diag_median"),
            "n_divergences": summary.get("n_divergences"),
            "rhat_max": summary.get("rhat_max"),
            "notes": notes,
            "plot_relpath": (
                plot_path.relative_to(runs_dir).as_posix() if plot_path else ""
            ),
            "seed_dir_relpath": seed_dir.relative_to(runs_dir).as_posix(),
            "readme_section": readme_entry.section if readme_entry else "",
            "sort_order": readme_entry.order if readme_entry else 10_000,
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            int(row["sort_order"]),
            str(row["run_label"]),
            float(row["duration_days"] or 0),
            str(row["run_dir"]),
            str(row["condition_dir"]),
        )
    )
    return rows


def _render_thumb_cell(plot_relpath: str, run_label: str) -> str:
    if not plot_relpath:
        return '<span class="muted">missing</span>'
    safe_src = escape(plot_relpath, quote=True)
    safe_alt = escape(f"PSD matrix for run {run_label}", quote=True)
    return (
        '<button class="thumb-button" '
        f'data-full="{safe_src}" data-alt="{safe_alt}" '
        f'aria-label="Enlarge PSD matrix for run {escape(run_label, quote=True)}">'
        f'<img class="thumb" src="{safe_src}" alt="{safe_alt}" loading="lazy"></button>'
    )


def build_html(rows: list[dict[str, Any]]) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    total_rows = len(rows)
    preview_rows = "\n".join(
        f"""
        <tr data-search="{escape(' '.join(str(value) for value in row.values()).lower(), quote=True)}">
          <td class="sticky-col run-col">{escape(str(row['run_label']))}</td>
          <td class="sticky-col thumb-col">{_render_thumb_cell(row['plot_relpath'], str(row['run_label']))}</td>
          <td>{escape(str(row['duration_days']))}</td>
          <td>{escape(str(row['seed']))}</td>
          <td>{escape(str(row['seed_count']))}</td>
          <td>{escape(str(row['Nc']))}</td>
          <td>{escape(str(row['K']))}</td>
          <td>{escape(str(row['diff_order']))}</td>
          <td>{escape(str(row['knot_method']))}</td>
          <td>{escape(str(row['alpha_beta']))}</td>
          <td>{escape(str(row['null_excision']))}</td>
          <td>{escape(str(row['floor']))}</td>
          <td>{escape(_format_number(row['ess_median']))}</td>
          <td>{escape(_format_number(row['riae_matrix']))}</td>
          <td>{escape(_format_number(row['coherence_riae']))}</td>
          <td>{escape(_format_number(row['coverage']))}</td>
          <td>{escape(_format_number(row['ciw_psd_diag_median']))}</td>
          <td>{escape(_format_number(row['n_divergences'], digits=0))}</td>
          <td>{escape(_format_number(row['rhat_max']))}</td>
          <td class="notes">{escape(str(row['notes']))}</td>
          <td class="path-cell"><code>{escape(str(row['seed_dir_relpath']))}</code></td>
        </tr>
        """.strip()
        for row in rows
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LISA Runs PSD Table</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf7;
      --panel-2: #f8f3e7;
      --grid: #d8ccaf;
      --text: #1d261f;
      --muted: #5a655f;
      --accent: #2f6d62;
      --accent-2: #b35c44;
      --shadow: rgba(19, 29, 23, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(179, 92, 68, 0.14), transparent 26rem),
        radial-gradient(circle at top right, rgba(47, 109, 98, 0.12), transparent 24rem),
        linear-gradient(180deg, #f8f4eb 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(98vw, 1800px);
      margin: 0 auto;
      padding: 2rem 1.2rem 3rem;
    }}
    .hero {{
      display: grid;
      gap: 0.8rem;
      margin-bottom: 1rem;
      padding: 1.2rem 1.4rem;
      border: 1px solid rgba(47, 109, 98, 0.18);
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(255, 253, 247, 0.95), rgba(247, 241, 225, 0.92));
      box-shadow: 0 18px 40px var(--shadow);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.8rem, 2.2vw, 2.8rem);
      line-height: 1.05;
      letter-spacing: 0.01em;
    }}
    .subhead, .meta {{
      margin: 0;
      color: var(--muted);
      font-size: 0.98rem;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.8rem;
      align-items: center;
      margin: 1rem 0 1.2rem;
    }}
    .controls label {{
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .controls input {{
      width: min(28rem, 100%);
      padding: 0.7rem 0.9rem;
      border: 1px solid var(--grid);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.82);
      font: inherit;
      color: inherit;
    }}
    .controls .hint {{
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid rgba(47, 109, 98, 0.18);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 18px 40px var(--shadow);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1550px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: var(--panel-2);
      border-bottom: 1px solid var(--grid);
      color: var(--muted);
      text-align: left;
      font-size: 0.83rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }}
    th, td {{
      padding: 0.75rem 0.8rem;
      border-bottom: 1px solid rgba(216, 204, 175, 0.75);
      vertical-align: top;
      font-size: 0.92rem;
    }}
    tbody tr:nth-child(even) td {{
      background: rgba(248, 243, 231, 0.42);
    }}
    tbody tr:hover td {{
      background: rgba(47, 109, 98, 0.08);
    }}
    .sticky-col {{
      position: sticky;
      z-index: 2;
      background: inherit;
    }}
    .run-col {{
      left: 0;
      min-width: 4.5rem;
      font-weight: 700;
    }}
    .thumb-col {{
      left: 4.5rem;
      min-width: 8.5rem;
    }}
    .thumb-button {{
      padding: 0;
      border: 0;
      background: transparent;
      cursor: zoom-in;
    }}
    .thumb {{
      display: block;
      width: 120px;
      height: 78px;
      object-fit: cover;
      border: 1px solid rgba(47, 109, 98, 0.18);
      border-radius: 10px;
      box-shadow: 0 10px 18px rgba(19, 29, 23, 0.12);
      background: #fff;
    }}
    .notes {{
      min-width: 22rem;
      max-width: 32rem;
      line-height: 1.35;
    }}
    .path-cell {{
      min-width: 24rem;
    }}
    code {{
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 0.8rem;
      color: var(--accent-2);
    }}
    .muted {{
      color: var(--muted);
    }}
    .lightbox {{
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      padding: 2rem;
      background: rgba(20, 28, 22, 0.84);
      z-index: 20;
    }}
    .lightbox.open {{
      display: grid;
    }}
    .lightbox img {{
      max-width: min(92vw, 1500px);
      max-height: 88vh;
      border-radius: 14px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.4);
      background: white;
    }}
    .lightbox button {{
      position: absolute;
      top: 1rem;
      right: 1rem;
      border: 0;
      border-radius: 999px;
      padding: 0.65rem 0.95rem;
      background: rgba(255, 255, 255, 0.92);
      color: #17211b;
      font: inherit;
      cursor: pointer;
    }}
    @media (max-width: 900px) {{
      .page {{
        padding-inline: 0.6rem;
      }}
      .hero {{
        padding: 1rem;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>LISA Run Table</h1>
      <p class="subhead">Representative PSD matrix thumbnail per run condition, using saved artifacts under <code>docs/studies/lisa/runs/</code>.</p>
      <p class="meta">Generated {escape(generated)}. Rows: {total_rows}. Representative seed prefers <code>seed_0</code> when present.</p>
    </section>

    <div class="controls">
      <label for="table-filter">Filter rows</label>
      <input id="table-filter" type="search" placeholder="Try x, uniform, 30.0, per-freq, none, 8192">
      <span class="hint">Search matches any visible value, including notes and paths.</span>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="sticky-col run-col">Run</th>
            <th class="sticky-col thumb-col">PSD</th>
            <th>Days</th>
            <th>Seed</th>
            <th>Seeds</th>
            <th>Nc</th>
            <th>K</th>
            <th>d</th>
            <th>Knots</th>
            <th>&alpha;/&beta;</th>
            <th>Null Excision</th>
            <th>Floor</th>
            <th>Median ESS</th>
            <th>RIAE</th>
            <th>Coh RIAE</th>
            <th>Coverage</th>
            <th>CI Width (diag med)</th>
            <th>Divs</th>
            <th>R-hat</th>
            <th>Notes</th>
            <th>Seed Dir</th>
          </tr>
        </thead>
        <tbody>
          {preview_rows}
        </tbody>
      </table>
    </div>
  </div>

  <div class="lightbox" id="lightbox" aria-hidden="true">
    <button type="button" id="lightbox-close">Close</button>
    <img id="lightbox-image" alt="">
  </div>

  <script>
    const filterInput = document.getElementById("table-filter");
    const rows = Array.from(document.querySelectorAll("tbody tr"));
    filterInput.addEventListener("input", () => {{
      const query = filterInput.value.trim().toLowerCase();
      for (const row of rows) {{
        const haystack = row.dataset.search || "";
        row.style.display = haystack.includes(query) ? "" : "none";
      }}
    }});

    const lightbox = document.getElementById("lightbox");
    const lightboxImage = document.getElementById("lightbox-image");
    const closeButton = document.getElementById("lightbox-close");
    for (const button of document.querySelectorAll(".thumb-button")) {{
      button.addEventListener("click", () => {{
        lightboxImage.src = button.dataset.full;
        lightboxImage.alt = button.dataset.alt || "";
        lightbox.classList.add("open");
        lightbox.setAttribute("aria-hidden", "false");
      }});
    }}
    function closeLightbox() {{
      lightbox.classList.remove("open");
      lightbox.setAttribute("aria-hidden", "true");
      lightboxImage.removeAttribute("src");
    }}
    closeButton.addEventListener("click", closeLightbox);
    lightbox.addEventListener("click", (event) => {{
      if (event.target === lightbox) {{
        closeLightbox();
      }}
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape" && lightbox.classList.contains("open")) {{
        closeLightbox();
      }}
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing run_* folders.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=README_PATH,
        help="README used to enrich run metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML file.",
    )
    args = parser.parse_args()

    readme_entries = parse_readme_tables(args.readme)
    rows = collect_rows(args.runs_dir, readme_entries)
    html = build_html(rows)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
