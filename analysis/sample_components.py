"""Recombine baseline joint counts into shared and unique sample components."""
from pathlib import Path
import argparse
import csv

def calculate(path):
    rows = [(int(r["profile"]), int(r["recorded_type"]), int(r["count"]))
            for r in csv.DictReader(path.open()) if r["scheme"] == "baseline"]
    result = []
    for d, name in enumerate(["Neglect", "Emotional abuse", "Physical abuse", "Sexual abuse"]):
        for component in ["shared", "score_only", "type_only"]:
            n = m = 0
            for profile, type_, count in rows:
                score, type_match = bool(profile & (1 << d)), type_ == d
                key = "shared" if score and type_match else "score_only" if score else "type_only" if type_match else None
                if key == component:
                    n += count
                    m += count * (profile.bit_count() >= 2)
            result.append(dict(domain=name, component=component, n=n, multidomain=m,
                               multidomain_percent=100*m/n if n else ""))
    return result

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "composition_ci_outputs/joint_counts.csv")
    parser.add_argument("--output", type=Path, default=root / "composition_ci_outputs/sample_components.csv")
    args = parser.parse_args()
    result = calculate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} aggregate rows to {args.output}")
