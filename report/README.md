# Report (LaTeX)

This folder contains a NeurIPS-style report for the repository.

## Files

- `main.tex` – main paper draft
- `references.bib` – bibliography

## Notes

- `main.tex` expects the NeurIPS style file `neurips_2024.sty` to be available to LaTeX.
  - If you don't have it locally, either copy it into this folder or comment out:
    - `\usepackage[final]{neurips_2024}`
    and compile as a standard `article`.

- Figures are referenced from `../results/...` so the images will show up after you run the project scripts.

## Suggested workflow

1. Run the project scripts to generate `results/*/*.png` and `metrics_*.csv`.
2. Fill the metric tables in `main.tex` by copying values from the CSV files.
3. Compile `main.tex`.
