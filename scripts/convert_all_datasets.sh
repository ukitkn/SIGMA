#!/usr/bin/env bash
set -euo pipefail

for rda_file in data/ST-colon1.rda data/ST-colon2.rda data/ST-colon3.rda data/ST-colon4.rda \
                data/ST-liver1.rda data/ST-liver2.rda data/ST-liver3.rda data/ST-liver4.rda; do
  output_file="data/processed/$(basename "${rda_file%.rda}").h5ad"
  sigma convert-rda --rda-input "$rda_file" --h5ad-output "$output_file"
done
