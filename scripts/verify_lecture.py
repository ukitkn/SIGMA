"""Reject failed or incomplete LecTrace builds before publishing the site."""

import json
from pathlib import Path


def main():
    trace_path = Path("_site/traces/01_sigma_execution.json")
    trace = json.loads(trace_path.read_text())
    if trace["metadata"]["error"] is not None:
        raise RuntimeError(trace["metadata"]["error"])
    renderings = [r for step in trace["steps"] for r in step["renderings"]]
    markdown = "\n".join(r.get("data") or "" for r in renderings if r["type"] == "markdown")
    headings = [line for line in markdown.splitlines() if line.startswith("## State ")]
    if len(headings) != 5:
        raise AssertionError(f"Expected five selected execution states, got {headings!r}")
    images = sum(r["type"] == "image" for r in renderings)
    if images != 7:
        raise AssertionError(f"Expected the overview, five states and results figure, got {images}")
    if not all(value in markdown for value in ["0.4901", "0.4204", "0.6775", "0.7319"]):
        raise AssertionError("The conclusion is missing report-table evidence.")
    if "0:00–2:20 | Healthcare problem and motivation" in markdown:
        raise AssertionError("Long presenter notes must remain outside the webpage.")
    print(f"Verified: {len(trace['steps'])} selected steps, five states, seven figures, no execution error.")


if __name__ == "__main__":
    main()
