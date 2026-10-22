"""Reject failed or incomplete LecTrace builds before publishing the site."""

import csv
import json
from pathlib import Path


def verify_report_tables(evidence):
    reference = json.loads((evidence / "report_reference.json").read_text())
    for name, expected in reference["tables"].items():
        with (evidence / f"{name}.csv").open(newline="") as handle:
            actual = list(csv.DictReader(handle))
        if actual != expected:
            raise AssertionError(f"{name}.csv disagrees with the reviewed report reference.")
    benchmark = reference["tables"]["benchmark"]
    for method, expected in reference["benchmark_mean_ARI"].items():
        mean = sum(float(row[method]) for row in benchmark) / len(benchmark)
        if round(mean, 4) != expected:
            raise AssertionError(f"Incorrect benchmark mean for {method}: {mean}")
    return reference


def main():
    reference = verify_report_tables(Path("lectures/evidence"))
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
    if not all(value in markdown for value in ["0.4901", "0.4204", "0.7328", "0.6595", "0.8618", "0.7630"]):
        raise AssertionError("The conclusion is missing report-table evidence.")
    if "0:00–2:20 | Healthcare problem and motivation" in markdown:
        raise AssertionError("Long presenter notes must remain outside the webpage.")
    required = ["31,296", "36,601", "1,665", "0.780", "0.765", "−0.286", "0.85",
                "0.6775", "4 of 7", "6 of 7", "U=1−P", "F=−∇U",
                "anatomical", "root", "smoothing", "What we learned",
                reference["software_3_assistance"]]
    for value in required:
        if value.casefold() not in markdown.casefold():
            raise AssertionError(f"The trace is missing current report content: {value}")
    fields = [step["env"]["field"]["contents"] for step in trace["steps"]
              if "field" in step["env"]]
    if not fields:
        raise AssertionError("The state-5 trace is missing the field metadata.")
    field = fields[-1]
    for key, expected in [("moving_average_window", 31), ("bandwidth", 80),
                          ("neighbors", 60)]:
        if field[key]["contents"] != expected:
            raise AssertionError(f"Incorrect teaching field setting: {key}")
    shape = [entry["contents"] for entry in field["grid_shape"]["contents"]]
    if shape != [80, 80]:
        raise AssertionError(f"Incorrect field grid: {shape}")
    source = Path("lectures/01_sigma_execution.py").read_text()
    if "_trajectory(adata" in source or "demo.rooted_pseudotime()" not in source:
        raise AssertionError("The trace must use the report's state-guided root construction.")
    print(f"Verified: {len(trace['steps'])} selected steps, five states, seven figures, "
          "all three report tables, current methods and disclosure, no execution error.")


if __name__ == "__main__":
    main()
