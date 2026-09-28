"""Paired whole-record bootstrap of sample-composition differences.

Export anonymous joint cell counts once with --raw-zips; subsequent runs use
only those counts. Multinomial cell resampling is equivalent to resampling
whole records, preserving overlap of the two sample definitions.
"""
from pathlib import Path
import argparse, json, zipfile, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/composition_ci_outputs'
KO = ['방임', '정서학대', '신체학대', '성학대']
EN = ['Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse']
SCHEMES = {'baseline': [4, 5, 5, 5], 'common6': [6]*4, 'common7': [7]*4}
B, SEED = 9999, 20260914

def export(zips):
    scores, labels, hashes = [], [], {}
    for path in map(Path, zips):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if not name.endswith('.json'):
                    continue
                obj = json.loads(z.read(name).decode('utf-8-sig'))
                value = obj['info']['학대의심'].strip().replace('(', '').replace(')', '')
                labels.append(KO.index(value) if value in KO else 4)
                found = {x['항목']: x['점수'] for s in obj['list']
                         if s.get('문항') == '학대여부' for x in s['list']
                         if x.get('항목') in KO}
                scores.append([found[d] for d in KO])
    scores, labels = np.array(scores), np.array(labels)
    assert scores.shape == (3236, 4)
    rows = []
    for scheme, cut in SCHEMES.items():
        y = scores >= cut
        profile = (y * (2**np.arange(4))).sum(axis=1)
        counts = np.bincount(profile*5 + labels, minlength=80)
        for cell, count in enumerate(counts):
            rows.append(dict(scheme=scheme, profile=cell//5, recorded_type=cell%5, count=int(count)))
        # Independently check cell totals against direct record masks.
        for d in range(4):
            assert sum(counts[c] for c in range(80) if (c//5) & (1<<d)) == y[:, d].sum()
            assert sum(counts[c] for c in range(80) if c%5 == d) == (labels == d).sum()
    pd.DataFrame(rows).to_csv(OUT/'joint_counts.csv', index=False)
    (OUT/'source_hashes.json').write_text(json.dumps(hashes, indent=2))

def analyze():
    joint = pd.read_csv(OUT/'joint_counts.csv')
    rows = []
    for scheme in SCHEMES:
        a = joint[joint.scheme == scheme]
        counts = a['count'].to_numpy()
        assert counts.sum() == 3236
        # Same seed per scheme; within each scheme every estimate uses the
        # same complete-record bootstrap sample. No cross-scheme test is made.
        draws = np.random.default_rng(SEED).multinomial(3236, counts/3236, size=B)
        multi = np.array([int(p).bit_count() >= 2 for p in a.profile])
        for d, domain in enumerate(EN):
            score = (a.profile.to_numpy() & (1<<d)) != 0
            typ = a.recorded_type.to_numpy() == d
            sn, tn = counts[score].sum(), counts[typ].sum()
            sm, tm = counts[score & multi].sum(), counts[typ & multi].sum()
            ds, dt = draws[:, score].sum(axis=1), draws[:, typ].sum(axis=1)
            assert (ds > 0).all() and (dt > 0).all()
            delta = 100*(draws[:, score & multi].sum(axis=1)/ds - draws[:, typ & multi].sum(axis=1)/dt)
            lo, hi = np.quantile(delta, [.025, .975])
            rows.append(dict(scheme=scheme, domain=domain, score_n=sn, type_n=tn,
                             shared=counts[score & typ].sum(), score_multi=sm, type_multi=tm,
                             score_multi_percent=100*sm/sn, type_multi_percent=100*tm/tn,
                             difference_pp=100*(sm/sn-tm/tn), ci_lower_pp=lo, ci_upper_pp=hi))
    result = pd.DataFrame(rows)
    prior = pd.read_csv(ROOT/'analysis/contract_outputs/threshold_composition.csv')
    for col in ['score_n', 'type_n', 'shared', 'score_multi', 'type_multi', 'difference_pp']:
        assert np.allclose(result[col], prior[col]), col
    result.to_csv(OUT/'sample_composition_ci.csv', index=False)
    (OUT/'metadata.json').write_text(json.dumps(dict(replicates=B, seed=SEED,
        method='whole-record paired multinomial bootstrap; percentile 95% CI',
        population_claim=False, all_denominators_positive=True), indent=2))
    print(result.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-zips', nargs=2)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.raw_zips:
        export(args.raw_zips)
    analyze()
