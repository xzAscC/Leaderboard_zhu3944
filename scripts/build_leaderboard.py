"""
scripts/build_leaderboard.py
============================
Reads all JSON files from submissions/ and writes leaderboard.md.

Scores are independently verified from raw predictions against the
ground truth labels — self-reported scores in the JSON are not trusted.

The baseline score is read from submissions/baseline.json — no manual
configuration needed.

Run by GitHub Actions on every push to submissions/.

Environment variables:
  GROUND_TRUTH : path to public test ground truth CSV
"""

import os
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.metrics import f1_score


SUBMISSIONS_DIR = Path("submissions")
LEADERBOARD_OUT = Path("readme.md")
BASELINE_JSON   = SUBMISSIONS_DIR / "baseline.json"
EMOTIONS        = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EMOTION_TO_IDX  = {e: i for i, e in enumerate(EMOTIONS)}


def load_ground_truth(gt_path):
    df = pd.read_csv(gt_path).sort_values("clip_id").reset_index(drop=True)
    return df["emotion"].map(EMOTION_TO_IDX).tolist()


def load_submissions():
    rows = []
    for json_file in sorted(SUBMISSIONS_DIR.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            if "team_name" not in data or "predictions" not in data:
                print(f"  !!  {json_file.name}: missing required fields")
                continue
            rows.append(data)
            print(f"  OK  {json_file.name}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  !!  {json_file.name}: {e}")
    return rows


def verify(data, ground_truth):
    preds = data["predictions"]
    if len(preds) != len(ground_truth):
        raise ValueError(
            f"Prediction count mismatch: got {len(preds)}, "
            f"expected {len(ground_truth)}"
        )
    weighted_f1  = f1_score(ground_truth, preds, average="weighted")
    per_class_f1 = f1_score(ground_truth, preds, average=None,
                             labels=list(range(len(EMOTIONS))))
    return weighted_f1, {
        name: round(float(score), 4)
        for name, score in zip(EMOTIONS, per_class_f1)
    }


def get_baseline_f1(all_rows):
    """Read the verified baseline F1 from the verified rows list."""
    for row in all_rows:
        if row["team_name"].lower() == "baseline":
            return row["weighted_f1"]
    raise FileNotFoundError(
        "baseline.json not found in submissions/. "
        "Run test.py with --team_name baseline and push the JSON first."
    )


def build_markdown(rows, baseline_f1, generated_at):
    # Exclude baseline from the ranked table — it's the reference, not a competitor
    ranked = sorted(
        [r for r in rows if r["team_name"].lower() != "baseline"],
        key=lambda r: r["weighted_f1"],
        reverse=True,
    )

    lines = []
    lines.append("# Leaderboard — Speech Emotion Recognition Challenge")
    lines.append("")
    lines.append(f"> Last updated: **{generated_at}**  ")
    lines.append(f"> Baseline weighted F1: **{baseline_f1:.4f}**  ")
    lines.append(f"> Metric: weighted F1-score on the public test set  ")
    lines.append(f"> Scores are verified server-side from submitted predictions.")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not ranked:
        lines.append("No team submissions yet.")
        lines.append("")
    else:
        # Rankings table
        lines.append("| Rank | Team | Weighted F1 | vs. Baseline | Last submitted |")
        lines.append("|------|------|-------------|--------------|----------------|")

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}

        for i, row in enumerate(ranked, start=1):
            rank      = f"{medal.get(i, '')} {i}"
            team      = row["team_name"]
            wf1       = row["weighted_f1"]
            delta     = wf1 - baseline_f1
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            beats     = "✅" if delta > 0 else "❌"
            submitted = row.get("submitted_at", "—")
            try:
                dt        = datetime.fromisoformat(submitted)
                submitted = dt.strftime("%b %d, %H:%M UTC")
            except Exception:
                pass
            lines.append(
                f"| {rank} | {team} | {wf1:.4f} | {beats} {delta_str} | {submitted} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

        # Per-class F1 breakdown
        lines.append("## Per-class F1 breakdown")
        lines.append("")
        lines.append(
            "| Team | " + " | ".join(e.capitalize() for e in EMOTIONS) + " |"
        )
        lines.append("|------|" + "|".join(["------"] * len(EMOTIONS)) + "|")

        for row in ranked:
            team   = row["team_name"]
            pcf1   = row["per_class_f1"]
            scores = " | ".join(f"{pcf1.get(e, 0.0):.4f}" for e in EMOTIONS)
            lines.append(f"| {team} | {scores} |")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*This leaderboard updates automatically when teams push a "
        "new submission JSON to `submissions/`. "
        "Scores are verified server-side.*"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    gt_path = os.environ.get("GROUND_TRUTH",
                              "ground_truth/public_test_labels.csv")

    print(f"Ground truth : {gt_path}")

    ground_truth = load_ground_truth(gt_path)
    print(f"Ground truth clips: {len(ground_truth)}")

    print("\nLoading and verifying submissions...")
    raw_rows = load_submissions()
    print(f"Found {len(raw_rows)} submission(s)")

    # Verify all submissions
    rows = []
    for data in raw_rows:
        try:
            verified_f1, verified_pcf1 = verify(data, ground_truth)
            rows.append({
                "team_name":   data["team_name"],
                "weighted_f1": round(verified_f1, 4),
                "per_class_f1": verified_pcf1,
                "submitted_at": data.get("submitted_at", "—"),
            })
            print(f"  verified  {data['team_name']}: F1={verified_f1:.4f}")
        except ValueError as e:
            print(f"  failed    {data['team_name']}: {e}")

    # Read baseline F1 from verified rows
    baseline_f1 = get_baseline_f1(rows)
    print(f"\nBaseline F1  : {baseline_f1:.4f}")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md           = build_markdown(rows, baseline_f1, generated_at)

    LEADERBOARD_OUT.write_text(md)
    print(f"Leaderboard written to {LEADERBOARD_OUT}")


if __name__ == "__main__":
    main()