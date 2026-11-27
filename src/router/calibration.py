"""Learn how often the cheap model is right, so we can predict accuracy without labels.

Every cheap-model field falls into one of four cells:

    (flagged by a check?  yes/no)  x  (complex document?  yes/no)

On a small random calibration sample we also run the expensive model and
count how often the cheap answer agrees with it. Each cell's accuracy is
smoothed with a Beta prior, so a cell with three samples does not claim 100%.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .normalize import values_equal
from .schemas import FIELDS

# Starting beliefs before any data is seen.
PRIORS = {
    (False, False): 0.95,  # clean field, simple document
    (False, True): 0.85,  # clean field, complex document
    (True, False): 0.45,  # flagged field, simple document
    (True, True): 0.35,  # flagged field, complex document
}


def cell_name(flagged: bool, complex_doc: bool) -> str:
    return f"{'flagged' if flagged else 'clean'}/{'complex' if complex_doc else 'simple'}"


@dataclass
class Calibration:
    cell_accuracy: dict[str, float] = field(default_factory=dict)
    cell_counts: dict[str, list[int]] = field(default_factory=dict)  # [agree, total]
    expensive_accuracy: float = 0.97
    reference: str = "expensive_model"
    complexity_threshold: float = 0.0
    sample_size: int = 0

    def p_correct(self, flagged: bool, complex_doc: bool) -> float:
        return self.cell_accuracy.get(cell_name(flagged, complex_doc), PRIORS[(flagged, complex_doc)])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def fit(samples: list[dict], *, prior_strength: float, expensive_accuracy: float,
        complexity_threshold: float) -> Calibration:
    """samples: [{cheap_values, reference_values, flags, complex_doc}]"""
    counts = {cell: [0, 0] for cell in PRIORS}
    for s in samples:
        for spec in FIELDS:
            cell = (bool(s["flags"].get(spec.name)), s["complex_doc"])
            counts[cell][1] += 1
            counts[cell][0] += values_equal(
                spec, s["cheap_values"].get(spec.name), s["reference_values"].get(spec.name)
            )
    cal = Calibration(expensive_accuracy=expensive_accuracy, sample_size=len(samples),
                      complexity_threshold=complexity_threshold)
    for cell, (c, n) in counts.items():
        a = PRIORS[cell] * prior_strength
        agreement = (c + a) / (n + prior_strength)
        # The reference is itself only ~97% right, so agreement cannot prove more than
        # that: P(cheap correct) ~= P(agree) * P(reference correct). Conservative on purpose.
        cal.cell_accuracy[cell_name(*cell)] = round(agreement * expensive_accuracy, 4)
        cal.cell_counts[cell_name(*cell)] = [c, n]
    return cal
