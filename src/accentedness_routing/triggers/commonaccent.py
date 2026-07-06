"""CommonAccent-style trigger: 1 - max_softmax from accent classifier.

For our implementation, we approximate this using the per-accent mean WER
as a proxy for accent difficulty. This avoids needing a separate accent
classifier model while testing the same routing concept.
"""

from __future__ import annotations

from accentedness_routing.triggers.base import RoutingTrigger


class ArgmaxAccentTrigger(RoutingTrigger):
    """Accent-based trigger: score = per-accent mean WER of the default model.

    This is the 'argmax classifier' baseline from the plan — it routes based
    on which accent group the utterance belongs to, using mean group WER as
    the escalation score.
    """

    def __init__(self, accent_map: dict[str, str], accent_mean_wers: dict[str, float]):
        """
        Args:
            accent_map: utterance_id → accent label
            accent_mean_wers: accent label → mean default-model WER for that accent
        """
        self._accent_map = accent_map
        self._accent_wers = accent_mean_wers

        # Normalize to [0, 1]
        vals = list(accent_mean_wers.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx - mn > 1e-8 else 1.0
        self._normed = {acc: (w - mn) / rng for acc, w in accent_mean_wers.items()}

    @property
    def name(self) -> str:
        return "argmax_accent"

    def score(self, utterance_id: str) -> float:
        accent = self._accent_map[utterance_id]
        return self._normed[accent]
