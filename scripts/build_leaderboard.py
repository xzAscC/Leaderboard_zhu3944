"""
scripts/build_leaderboard.py
============================
Reads all CSV files from submissions/ and writes README.md.

Each CSV has two columns: clip_id, predicted_emotion
Weighted F1 is computed server-side against the ground truth —
no self-reported scores.

Run by GitHub Actions on every push to submissions/.

Environment variables:
  GROUND_TRUTH : path to public test ground truth CSV
"""

import os
import csv
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.metrics import f1_score


SUBMISSIONS_DIR = Path("submissions")
LEADERBOARD_OUT = Path("README.md")
EMOTIONS        = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EMOTION_TO_IDX  = {e: i for i, e in enumerate(EMOTIONS)}


def load_ground_truth(gt_path):
    df = pd.read_csv(gt_path).sort_values("clip_id").reset_index(drop=True)
    return df.set_index("clip_id")["emotion"].to_dict()


def load_submissions():
    rows = []
    for csv_file in sorted(SUBMISSIONS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if not {"clip_id", "predicted_emotion"}.issubset(df.columns):
                print(f"  !!  {csv_file.name}: missing required columns")
                continue
            rows.append({
                "team_name": csv_file.stem.replace("_", " "),
                "filename":  csv_file,
                "df":        df,
                "submitted_at": datetime.fromtimestamp(
                    csv_file.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            })
            print(f"  OK  {csv_file.name}")
        except Exception as e:
            print(f"  !!  {csv_file.name}: {e}")
    return rows


def verify(data, ground_truth):
    df   = data["df"].sort_values("clip_id").reset_index(drop=True)
    true = [EMOTION_TO_IDX[ground_truth[c]] for c in df["clip_id"]
            if c in ground_truth]
    pred = [EMOTION_TO_IDX.get(e, -1)
            for c, e in zip(df["clip_id"], df["predicted_emotion"])
            if c in ground_truth]

    if len(true) != len(pred) or len(true) == 0:
        raise ValueError(f"Prediction count mismatch or no matching clip IDs")

    return f1_score(true, pred, average="weighted")


def get_baseline_f1(rows):
    for row in rows:
        if row["team_name"].lower() == "baseline":
            return row["weighted_f1"]
    raise FileNotFoundError(
        "baseline.csv not found in submissions/. "
        "Run test.py with --team_name baseline and push the CSV first."
    )


def build_markdown(rows, baseline_f1, generated_at):
    ranked = sorted(
        rows,
        key=lambda r: r["weighted_f1"],
        reverse=True,
    )

    lines = []
    lines.append("# Leaderboard — Speech Emotion Recognition Challenge")
    lines.append("")
    lines.append(f"> Last updated: **{generated_at}**  ")
    lines.append(f"> Baseline weighted F1: **{baseline_f1:.4f}**  ")
    lines.append(f"> Metric: weighted F1-score on the public test set")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not ranked:
        lines.append("No team submissions yet.")
        lines.append("")
    else:
        lines.append("| Team Name | Weighted F1 | Date and Time |")
        lines.append("|-----------|-------------|---------------|")

        for row in ranked:
            team      = row["team_name"]
            wf1       = row["weighted_f1"]
            submitted = row.get("submitted_at", "—")
            try:
                dt        = datetime.fromisoformat(submitted)
                submitted = dt.strftime("%b %d, %Y  %H:%M UTC")
            except Exception:
                pass
            lines.append(f"| {team} | {wf1:.4f} | {submitted} |")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*This leaderboard updates automatically when teams push a "
        "new submission CSV to `submissions/`.*"
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

    rows = []
    for data in raw_rows:
        try:
            wf1 = verify(data, ground_truth)
            rows.append({
                "team_name":   data["team_name"],
                "weighted_f1": round(wf1, 4),
                "submitted_at": data["submitted_at"],
            })
            print(f"  verified  {data['team_name']}: F1={wf1:.4f}")
        except Exception as e:
            print(f"  failed    {data['team_name']}: {e}")

    baseline_f1  = get_baseline_f1(rows)
    print(f"\nBaseline F1  : {baseline_f1:.4f}")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md           = build_markdown(rows, baseline_f1, generated_at)

    LEADERBOARD_OUT.write_text(md)
    print(f"Leaderboard written to {LEADERBOARD_OUT}")


if __name__ == "__main__":
    main()