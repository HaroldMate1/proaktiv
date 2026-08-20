"""Kinase-domain sequence windowing.

The submitted pipeline passed full-length sequences to ESM2 with
``max_length=1024, truncation=True``. ALK is 1620 aa and every curated ALK
mutation lies past residue 1024, so all 3,245 ALK records encoded to the
identical wild-type sequence and the model could not distinguish an ALK variant
from ALK wild type.

This module replaces truncation with an explicit, biologically defined window
around the UniProt protein-kinase domain. The window is defined in canonical
(wild-type) coordinates and transferred to variant sequences by preserving the
untouched prefix and suffix, which keeps indel-bearing variants in register.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "kinases.yml"


class SequenceWindowError(ValueError):
    """Raised when a sequence cannot be windowed without losing variant content."""


@dataclass(frozen=True)
class KinaseWindow:
    """A canonical-coordinate window on one kinase."""

    name: str
    uniprot: str
    target_pref_name: str
    length: int
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive

    @property
    def prefix_len(self) -> int:
        return self.start - 1

    @property
    def suffix_len(self) -> int:
        return self.length - self.end

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def covers(self, position: int) -> bool:
        return self.start <= position <= self.end


def load_windows(config_path: Path | str = CONFIG_PATH) -> dict[str, KinaseWindow]:
    """Build the windows declared in ``configs/kinases.yml``.

    The kinase-domain annotation is widened by ``flank`` on both sides and
    clamped to the sequence, then checked against the encoder token budget.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    max_residues = config["max_tokens"] - 2  # ESM2 adds <cls> and <eos>

    windows: dict[str, KinaseWindow] = {}
    for name, spec in config["targets"].items():
        domain_start, domain_end = spec["kinase_domain"]
        flank = spec["flank"]
        length = spec["length"]
        start = max(1, domain_start - flank)
        end = min(length, domain_end + flank)
        window = KinaseWindow(
            name=name,
            uniprot=spec["uniprot"],
            target_pref_name=spec["target_pref_name"],
            length=length,
            start=start,
            end=end,
        )
        if window.size > max_residues:
            raise SequenceWindowError(
                f"{name} window is {window.size} residues, over the "
                f"{max_residues}-residue encoder budget"
            )
        windows[name] = window
    return windows


def window_by_target_name(
    windows: dict[str, KinaseWindow],
) -> dict[str, KinaseWindow]:
    """Index windows by the ChEMBL ``target_pref_name`` used in the raw data."""
    return {w.target_pref_name: w for w in windows.values()}


def apply_window(window: KinaseWindow, sequence: str, wild_type: str) -> str:
    """Cut ``sequence`` down to ``window``, keeping indel-bearing variants in register.

    The window is defined on wild-type coordinates. A variant sequence may differ
    in length, so the slice is anchored on the residue counts outside the window
    rather than on absolute indices: the prefix before the window and the suffix
    after it must be identical to wild type, which is exactly the condition that
    every indel falls inside the window.

    Raises:
        SequenceWindowError: if the variant differs from wild type outside the
            window, meaning the window would drop or misalign variant content.
    """
    if len(wild_type) != window.length:
        raise SequenceWindowError(
            f"{window.name} wild type is {len(wild_type)} aa, "
            f"expected {window.length} aa"
        )

    prefix, suffix = window.prefix_len, window.suffix_len
    if sequence[:prefix] != wild_type[:prefix]:
        raise SequenceWindowError(
            f"{window.name} variant differs from wild type before residue "
            f"{window.start}; window would misalign the sequence"
        )
    tail_start = len(sequence) - suffix
    if tail_start <= prefix:
        raise SequenceWindowError(
            f"{window.name} variant is too short to contain the window"
        )
    if suffix and sequence[tail_start:] != wild_type[window.end :]:
        raise SequenceWindowError(
            f"{window.name} variant differs from wild type after residue "
            f"{window.end}; window would drop variant content"
        )
    return sequence[prefix:tail_start]


def mutation_is_encoded(window: KinaseWindow, positions: list[int]) -> bool:
    """True if every mutated position survives the window."""
    return all(window.covers(p) for p in positions)
