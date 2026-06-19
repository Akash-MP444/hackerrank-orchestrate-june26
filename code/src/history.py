"""
User-history lookup (Stage 1 reference data).

Kept as its own module — separate from the generic CSV/ingestion helpers in utils.py
— because ARCHITECTURE_AND_STRATEGY.md §5.3 and §1.5 single out user history as
structurally special: it is the one reference data source that must NEVER be allowed
to influence `claim_status` directly. Keeping its loading/lookup code isolated here
makes that boundary easy to audit — decision_engine.py imports only the narrow
`UserHistoryRecord` it needs and is the only place that reads `has_risk_signal`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.schemas import UserHistoryRecord
from src.utils import read_csv_rows

logger = logging.getLogger("evidence_review")


class UserHistoryIndex:
    """An in-memory lookup table over user_history.csv."""

    def __init__(self, records: dict[str, UserHistoryRecord]):
        self._records = records

    @classmethod
    def load(cls, path: str | Path) -> "UserHistoryIndex":
        records: dict[str, UserHistoryRecord] = {}
        for row in read_csv_rows(path):
            record = UserHistoryRecord(**row)
            records[record.user_id] = record
        logger.info("Loaded user history for %d users from %s", len(records), path)
        return cls(records)

    def get(self, user_id: str) -> UserHistoryRecord:
        """Returns the user's history record, or a conservative synthetic default
        for a user_id that isn't in user_history.csv at all.

        Defaulting to "no risk signal" (rather than e.g. treating an unknown user as
        automatically risky) matches how a brand-new user is described elsewhere in
        user_history.csv itself (history_flags='none', history_summary mentions "new
        user with no prior claim history") — an absent record is just the most extreme
        case of that, not evidence of risk on its own.
        """
        record = self._records.get(user_id)
        if record is not None:
            return record
        logger.info("No history record for user_id=%s; using a no-history default.", user_id)
        return UserHistoryRecord(
            user_id=user_id,
            past_claim_count=0,
            accept_claim=0,
            manual_review_claim=0,
            rejected_claim=0,
            last_90_days_claim_count=0,
            history_flags=[],
            history_summary="No prior claim history available for this user.",
        )

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._records