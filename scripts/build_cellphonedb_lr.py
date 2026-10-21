from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


def read_table(source: Path, name: str) -> pd.DataFrame:
    if source.is_dir():
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"CellPhoneDB table not found: {path}")
        return pd.read_csv(path)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            matches = [entry for entry in archive.namelist() if entry.endswith("/" + name) or entry == name]
            if not matches:
                raise FileNotFoundError(f"CellPhoneDB table '{name}' is missing from {source}.")
            with archive.open(matches[0]) as handle:
                return pd.read_csv(handle)
    raise ValueError("The database source must be a CellPhoneDB directory or zip file.")


def make_pairs(source: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    interactions = read_table(source, "interaction_table.csv")
    multidata = read_table(source, "multidata_table.csv")
    composition = read_table(source, "complex_composition_table.csv")
    proteins = read_table(source, "protein_table.csv")
    genes = read_table(source, "gene_table.csv")

    protein_genes = genes.merge(
        proteins[["id_protein", "protein_multidata_id"]],
        left_on="protein_id",
        right_on="id_protein",
        how="inner",
    )
    protein_symbols = (
        protein_genes.dropna(subset=["hgnc_symbol"])
        .groupby("protein_multidata_id")["hgnc_symbol"]
        .apply(lambda values: tuple(sorted(set(str(value).upper() for value in values))))
        .to_dict()
    )
    complex_proteins = composition.merge(
        proteins[["protein_multidata_id"]],
        on="protein_multidata_id",
        how="inner",
    )
    complex_genes = complex_proteins.merge(
        protein_genes[["protein_multidata_id", "hgnc_symbol"]],
        on="protein_multidata_id",
        how="inner",
    )
    complex_symbols = (
        complex_genes.dropna(subset=["hgnc_symbol"])
        .groupby("complex_multidata_id")["hgnc_symbol"]
        .apply(lambda values: tuple(sorted(set(str(value).upper() for value in values))))
        .to_dict()
    )
    entity_rows = multidata.set_index("id_multidata")
    entity_genes: dict[int, tuple[str, ...]] = {}
    entity_receptor: dict[int, bool] = {}
    for entity_id, row in entity_rows.iterrows():
        is_complex = _as_bool(row.get("is_complex", False))
        mapped = complex_symbols.get(int(entity_id), ()) if is_complex else protein_symbols.get(int(entity_id), ())
        if mapped:
            entity_genes[int(entity_id)] = mapped
        entity_receptor[int(entity_id)] = _as_bool(row.get("receptor", False))

    records = []
    dropped = 0
    for row in interactions.itertuples(index=False):
        left_id = int(row.multidata_1_id)
        right_id = int(row.multidata_2_id)
        left_is_receptor = entity_receptor.get(left_id, False)
        right_is_receptor = entity_receptor.get(right_id, False)
        if left_is_receptor == right_is_receptor:
            dropped += 1
            continue
        ligand_id, receptor_id = (right_id, left_id) if left_is_receptor else (left_id, right_id)
        ligand_genes = entity_genes.get(ligand_id, ())
        receptor_genes = entity_genes.get(receptor_id, ())
        if not ligand_genes or not receptor_genes:
            dropped += 1
            continue
        records.append({
            "ligand": "&".join(ligand_genes),
            "receptor": "&".join(receptor_genes),
            "interaction_id": str(row.id_cp_interaction),
            "source": str(row.source),
            "classification": str(row.classification),
        })

    result = pd.DataFrame(records).drop_duplicates(subset=["ligand", "receptor"])
    if result.empty:
        raise ValueError("No directed ligand-receptor pairs could be resolved from this database.")
    metadata = {
        "database_source": source.name,
        "source_repository": "https://github.com/ventolab/cellphonedb-data",
        "database_bundle_version": "not recorded in the supplied archive",
        "interaction_rows": int(len(interactions)),
        "directed_gene_pairs": int(len(result)),
        "excluded_ambiguous_or_unmapped_interactions": int(dropped),
        "complex_encoding": "Subunits are joined with '&' and scored by the minimum subunit abundance.",
        "direction_rule": "The CellPhoneDB multidata receptor flag determines ligand/receptor orientation.",
    }
    return result, metadata


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gene-level ligand-receptor table from CellPhoneDB data.")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pairs, metadata = make_pairs(args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(pairs)} ligand-receptor pairs to {args.output}")
    print(f"Saved provenance to {metadata_path}")


if __name__ == "__main__":
    main()
