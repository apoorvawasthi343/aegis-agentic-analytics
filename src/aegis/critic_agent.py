"""Deterministic Critic Agent for AEGIS.

Reviews BaselineVsEngineeredComparison reports and decides whether
engineered features should be accepted, rejected, or flagged for review.
Uses only deterministic logic — no LLM calls in this version.
"""

from __future__ import annotations

from src.aegis.schemas import (
    BaselineVsEngineeredComparison,
    CriticReport,
)


class CriticAgent:
    """Deterministic critic that evaluates feature engineering results.

    Analyzes a BaselineVsEngineeredComparison and produces a CriticReport
    with a decision (accept/reject/review) based on whether the engineered
    features actually improved model performance.
    """

    def review(
        self,
        comparison: BaselineVsEngineeredComparison,
    ) -> CriticReport:
        """Review a feature-engineering comparison.

        Args:
            comparison: The BaselineVsEngineeredComparison to evaluate.

        Returns:
            A CriticReport with a decision and supporting reasons.
        """
        reasons: list[str] = []
        accepted: list[str] = []
        rejected: list[str] = []
        leakage_warning = False

        # Determine which metrics improved and declined
        accuracy_improved = comparison.accuracy_change > 0
        f1_improved = comparison.f1_change > 0
        roc_auc_improved = (
            comparison.roc_auc_change is not None
            and comparison.roc_auc_change > 0
        )
        roc_auc_declined = (
            comparison.roc_auc_change is not None
            and comparison.roc_auc_change < 0
        )
        any_improved = accuracy_improved or f1_improved or roc_auc_improved

        # --- Decision logic ---

        # Rule 4: No created features → reject (checked first)
        if not comparison.features_created:
            rejected.extend(comparison.features_skipped)
            reasons.append("No engineered features were successfully created.")
            decision = "reject"

        # Rule 1: Both primary metrics improve and no meaningful metric declines → accept
        elif accuracy_improved and f1_improved and not roc_auc_declined:
            accepted = list(comparison.features_created)
            rejected = list(comparison.features_skipped)
            reasons.append(
                f"Model performance improved: accuracy Δ{comparison.accuracy_change:+.4f}, "
                f"F1 Δ{comparison.f1_change:+.4f}."
            )
            if roc_auc_improved:
                reasons.append(
                    f"ROC-AUC Δ{comparison.roc_auc_change:+.4f}."
                )
            decision = "accept"

        # Rule 2: No metric improvement → reject
        elif not any_improved:
            rejected = list(comparison.features_created)
            rejected.extend(comparison.features_skipped)
            reasons.append(
                f"No performance improvement: accuracy Δ{comparison.accuracy_change:+.4f}, "
                f"F1 Δ{comparison.f1_change:+.4f}."
            )
            if roc_auc_improved:
                reasons.append(
                    f"ROC-AUC Δ{comparison.roc_auc_change:+.4f} — no meaningful change."
                )
            decision = "reject"

        # Rule 3: Some metrics improve while others decline → review
        else:
            accepted = list(comparison.features_created)
            rejected = list(comparison.features_skipped)
            mixed_parts: list[str] = []
            if accuracy_improved and not f1_improved:
                mixed_parts.append(
                    f"accuracy improved by {comparison.accuracy_change:+.4f} "
                    f"but F1 declined by {comparison.f1_change:+.4f}"
                )
            elif f1_improved and not accuracy_improved:
                mixed_parts.append(
                    f"F1 improved by {comparison.f1_change:+.4f} "
                    f"but accuracy declined by {comparison.accuracy_change:+.4f}"
                )
            if roc_auc_declined:
                mixed_parts.append(
                    f"ROC-AUC declined by {comparison.roc_auc_change:+.4f}"
                )
            if mixed_parts:
                reasons.append(
                    "Mixed results: " + "; ".join(mixed_parts) + "."
                )
            reasons.append(
                "Engineered features show some benefit but not uniformly positive."
            )
            decision = "review"

        # Build summary
        summary_parts = [
            f"Critic reviewed {len(comparison.features_created)} engineered feature(s).",
            f"Decision: {decision.upper()}.",
        ]
        if reasons:
            summary_parts.append("Key reasons: " + "; ".join(reasons[:3]))

        summary = " ".join(summary_parts)

        return CriticReport(
            decision=decision,
            accepted_features=accepted,
            rejected_features=rejected,
            reasons=reasons,
            performance_improved=comparison.improved,
            leakage_warning=leakage_warning,
            summary=summary,
        )