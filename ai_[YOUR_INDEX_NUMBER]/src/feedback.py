"""
Feedback Module – Part G Novel Feature: User Feedback Loop
----------------------------------------------------------
Implements a lightweight feedback mechanism for iterative retrieval improvement.

How it works:
  1. After each response, the Streamlit UI shows 👍 / 👎 buttons.
  2. The user's rating is recorded against the chunk IDs that were retrieved.
  3. On the next query, the Retriever applies a cumulative score adjustment:
       • 👍 → +BOOST_DELTA (up to MAX_BOOST maximum)
       • 👎 → −BOOST_DELTA (down to −MAX_BOOST minimum)
  4. Adjustments persist to logs/feedback.json across sessions.

This is an evidence-based, interpretable alternative to full RLHF – every
score change is traceable to a specific user interaction, which makes it
auditable and appropriate for an academic demo context.

Design trade-offs noted:
  • Simple cumulative rule avoids the need for re-training or re-indexing.
  • MAX_BOOST cap (0.20) prevents a single chunk from dominating all queries.
  • Adjustments are chunk-level, not query-level, so they generalise across
    related questions that pull the same chunk.

Author : [YOUR_FULL_NAME]  ([YOUR_INDEX_NUMBER])
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.json")

BOOST_DELTA = 0.05     # score adjustment per feedback event
MAX_BOOST   = 0.20     # maximum cumulative adjustment (positive or negative)


class FeedbackStore:
    """Persistent store for chunk-level score adjustments."""

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._boosts: dict[str, float] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict[str, float]:
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read feedback file; starting fresh.")
        return {}

    def _persist(self) -> None:
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._boosts, fh, indent=2, ensure_ascii=False)

    # ── Public interface ──────────────────────────────────────────────────────

    def record(self, chunk_ids: list[str], positive: bool) -> None:
        """
        Record user feedback for a list of retrieved chunk IDs.

        Args:
            chunk_ids : IDs of the chunks that were shown to the user
            positive  : True = helpful (boost), False = not helpful (penalise)
        """
        delta = BOOST_DELTA if positive else -BOOST_DELTA
        for cid in chunk_ids:
            current = self._boosts.get(cid, 0.0)
            updated = max(-MAX_BOOST, min(MAX_BOOST, current + delta))
            self._boosts[cid] = round(updated, 4)

        self._persist()
        direction = "👍 boosted" if positive else "👎 penalised"
        logger.info("Feedback %s: %d chunks updated", direction, len(chunk_ids))

    def get_boosts(self) -> dict[str, float]:
        """Return a snapshot of the current boost table (used by Retriever)."""
        return dict(self._boosts)

    def get_stats(self) -> dict:
        """Summary statistics for display in the sidebar."""
        if not self._boosts:
            return {"total": 0, "positive": 0, "negative": 0, "neutral": 0}
        pos = sum(1 for v in self._boosts.values() if v > 0)
        neg = sum(1 for v in self._boosts.values() if v < 0)
        return {
            "total":    len(self._boosts),
            "positive": pos,
            "negative": neg,
            "neutral":  len(self._boosts) - pos - neg,
        }

    def reset(self) -> None:
        """Clear all feedback (useful for fresh experiments)."""
        self._boosts = {}
        self._persist()
        logger.info("Feedback store reset.")
