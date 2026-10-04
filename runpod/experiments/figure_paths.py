"""File locations for manuscript figures, alternate formats and validation records."""

from pathlib import Path


def export_path(figure_root: Path, stem: str, suffix: str) -> Path:
    """Return the canonical output path, creating its containing directory."""
    if suffix == '.pdf':
        category = ('supplementary' if stem.startswith('FigS') else
                    'tables' if stem.startswith('table') else 'main')
        parent = figure_root / category
    elif suffix in {'.png', '.svg', '.tiff'}:
        parent = figure_root / 'formats' / suffix[1:]
    else:
        raise ValueError(f'Unsupported figure format: {suffix}')
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f'{stem}{suffix}'


def validation_path(figure_root: Path, filename: str) -> Path:
    """Keep figure validation outputs alongside the repository documentation."""
    parent = figure_root.parent / 'docs' / 'figure_validation'
    parent.mkdir(parents=True, exist_ok=True)
    return parent / filename
