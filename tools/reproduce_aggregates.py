"""Regenerate paper statistics without restricted records or reconstructed people."""
from pathlib import Path
import contextlib
import io
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AN=ROOT/'analysis'
OUT=ROOT/'generated/aggregate_reproduction'
sys.path.insert(0,str(AN))
import final_contract_audit as uniform
import sample_composition_uncertainty as paired
import cooccurrence_benchmarks as cooccurrence
import reproduce_representation_audit as representation
from sample_components import calculate


def compare(actual,expected):
    a=pd.read_csv(actual);b=pd.read_csv(expected)
    assert list(a.columns)==list(b.columns),(actual.name,'columns')
    assert a.shape==b.shape,(actual.name,'shape')
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]):
            np.testing.assert_allclose(a[col],b[col],rtol=1e-9,atol=1e-8,err_msg=f'{actual.name}:{col}')
        else:assert a[col].fillna('').tolist()==b[col].fillna('').tolist(),(actual.name,col)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        uniform.OUT=OUT/'contract';uniform.OUT.mkdir(exist_ok=True)
        uniform.uniform()
        compare(uniform.OUT/'uniform_choice.csv',AN/'contract_outputs/uniform_choice.csv')
        paired.OUT=OUT/'paired';paired.OUT.mkdir(exist_ok=True)
        shutil.copy2(AN/'composition_ci_outputs/joint_counts.csv',paired.OUT/'joint_counts.csv')
        paired.analyze()
        compare(paired.OUT/'sample_composition_ci.csv',AN/'composition_ci_outputs/sample_composition_ci.csv')
        cooccurrence.OUT=OUT/'cooccurrence';cooccurrence.main()
        for name in ('cooccurrence_share_ci','cooccurrence_components','benchmark_contrasts',
                     'type_defined_cooccurrence_load','score_only_recorded_types'):
            compare(cooccurrence.OUT/f'{name}.csv',AN/'benchmark_outputs'/f'{name}.csv')
        pd.DataFrame(calculate(AN/'composition_ci_outputs/joint_counts.csv')).to_csv(OUT/'sample_components.csv',index=False)
        compare(OUT/'sample_components.csv',AN/'composition_ci_outputs/sample_components.csv')
        representation.run(AN/'sensitivity_outputs/combination_record_matrix.csv',OUT/'representation')
        for name in ('audit_pairwise_comparisons','audit_domain_estimates','audit_record_estimates'):
            compare(OUT/'representation'/f'{name}.csv',AN/'sensitivity_outputs'/f'{name}.csv')
    (OUT/'execution.log').write_text(log.getvalue())
    report=dict(status='passed',aggregate_csv_comparisons=11,restricted_records_read=False,
                reconstructed_participant_rows=False,paired_bootstrap_replicates=9999,
                paired_bootstrap_seed=20260914,representation_bootstrap_seed=20260907,
                full_raw_data_models_rerun=False)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
