#!/usr/bin/env python3
"""Audit source missingness, score support and agreement; ordinal choice sensitivity.

Reads authorized ZIPs locally. Exports aggregate counts only, never source rows,
identifiers, comments, or counseling text. Existing numerical-model outputs remain
unchanged for comparison. Highest-score membership uses order, not score distances.
"""
import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from statsmodels.discrete.conditional_models import ConditionalLogit
import selection_audit as old


def ordered_design(scores, available, kind):
    top = (scores == np.where(available, scores, -1).max(axis=1, keepdims=True))[:, :, None].astype(float)
    effects = np.broadcast_to(np.eye(4)[:, [0, 2, 3]], (len(scores), 4, 3))
    return top if kind == 'order' else effects if kind == 'domain' else np.concatenate([top, effects], axis=2)


def separation(x, available, y):
    diffs = np.concatenate([x[i, y[i]]-x[i, np.flatnonzero(available[i])]
                            for i in range(len(y))])
    diffs = np.unique(diffs, axis=0)
    # A nonzero improving direction with no negative margin diagnoses separation.
    r = linprog(-diffs.sum(axis=0), A_ub=-diffs, b_ub=np.zeros(len(diffs)),
                bounds=[(-1, 1)]*x.shape[-1], method='highs')
    return dict(separating_direction=bool(r.success and -r.fun > 1e-7),
                contrast_rank=int(np.linalg.matrix_rank(diffs)))


def models(scores, labels, available, out, reps=999, seed=20260907):
    result = {}
    for kind in ('order', 'domain', 'combined'):
        x = ordered_design(scores, available, kind)
        b, nll, cov, prob = old.fit(x, available, labels)
        cv = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, labels):
            bt, *_ = old.fit(x[tr], available[tr], labels[tr])
            loss, _, _, pr = old.likelihood(bt, x[te], available[te], labels[te])
            best = np.isclose(pr, pr.max(axis=1, keepdims=True), atol=1e-12, rtol=0)
            credit = best[np.arange(len(te)), labels[te]] / best.sum(axis=1)
            cv.append((float(loss), float(credit.sum())))
        result[kind] = dict(n=len(labels), beta=b.tolist(), log_likelihood=-float(nll),
                            aic=float(2*nll+2*len(b)), cv_log_loss=sum(v[0] for v in cv)/len(labels),
                            cv_accuracy=sum(v[1] for v in cv)/len(labels))
    x = ordered_design(scores, available, 'combined')
    b = np.array(result['combined']['beta'])
    mask = available.ravel()
    endog = np.zeros_like(available, dtype=int); endog[np.arange(len(labels)), labels] = 1
    independent = ConditionalLogit(endog.ravel()[mask], x.reshape(-1,4)[mask],
            groups=np.repeat(np.arange(len(labels)),4)[mask])
    opt = minimize(lambda b:-independent.loglike(b),np.zeros(4),jac=lambda b:-independent.score(b),
                   method='BFGS',options={'gtol':1e-7,'maxiter':200})
    assert np.max(np.abs(opt.x-b)) < 3e-6
    assert abs(independent.loglike(opt.x)-result['combined']['log_likelihood']) < 1e-8
    # Reproduce and diagnose the previously reported numerical-model failures.
    numeric = old.design(scores, 'combined')
    numeric_beta, *_ = old.fit(numeric, available, labels)
    rng = np.random.default_rng(seed); boots=[]; failures=[]; old_failures=[]
    for k in range(reps):
        ix = rng.integers(0,len(labels),len(labels))
        for tag,xx,bb,destination in [('order',x,b,failures),('numeric',numeric,numeric_beta,old_failures)]:
            try:
                fitted, *_ = old.fit(xx[ix], available[ix], labels[ix], initial=bb)
                if tag=='order':boots.append(fitted)
            except (RuntimeError, np.linalg.LinAlgError):
                destination.append(dict(replicate=k+1, **separation(xx[ix],available[ix],labels[ix]),
                                       recorded_domain_counts=np.bincount(labels[ix],minlength=4).tolist()))
    if len(boots)<.95*reps:raise RuntimeError('Too many failed ordered fits')
    ci=np.quantile(boots,[.025,.975],axis=0)
    terms=['Highest-score domain vs lower','Neglect vs emotional','Physical vs emotional','Sexual vs emotional']
    coefs=[dict(term=t,beta=float(b[j]),odds_ratio=float(np.exp(b[j])),
                ci_low=float(np.exp(ci[0,j])),ci_high=float(np.exp(ci[1,j]))) for j,t in enumerate(terms)]
    pd.DataFrame(coefs).to_csv(out/'ordered_choice_coefficients.csv',index=False)
    result.update(coefficients=coefs,bootstrap_successful=len(boots),bootstrap_requested=reps,
                  bootstrap_failures=failures,previous_numeric_bootstrap_failures=old_failures,
                  independent_statsmodels_check=True,seed=seed)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-zip',type=Path,required=True);p.add_argument('--validation-zip',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();out=a.out_dir;out.mkdir(parents=True,exist_ok=True)
    scores=[];labels=[];status=[];splits=[];dates=[];evaluation_dates=[];crisis=[];raw_counts=Counter();hashes={};names=[]
    for split,path in [('train',a.training_zip),('validation',a.validation_zip)]:
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if not name.endswith('.json') or name.startswith('__MACOSX'):continue
                names.append(Path(name).name)
                o=json.loads(z.read(name).decode('utf-8-sig')); info=o['info'];v=info.get('학대의심')
                st='missing_key' if '학대의심' not in info else 'null' if v is None else 'blank' if not str(v).strip() else 'entered'
                label=str(v).strip().replace('(','').replace(')','') if st=='entered' else None
                if st=='entered' and label not in (*old.DOMAINS,'해당 없음','해당없음','없음'):raise ValueError('Unknown label')
                status.append(st);raw_counts[str(v) if st=='entered' else st]+=1
                labels.append(old.DOMAINS.index(label) if label in old.DOMAINS else 4)
                found={i['항목']:i['점수'] for section in o['list'] if section.get('문항')=='학대여부' for i in section['list'] if i.get('항목') in old.DOMAINS}
                assert set(found)==set(old.DOMAINS)
                assert all(isinstance(v,int) and not isinstance(v,bool) and 0<=v<=10 for v in found.values())
                scores.append([found[d] for d in old.DOMAINS]);splits.append(split)
                dates.append(str(info.get('상담일자','')));evaluation_dates.append(str(info.get('평가일시','')))
                crisis.append(str(info.get('위기단계','')))
    assert len(set(names))==len(names)
    order=np.argsort(names)
    scores=np.asarray(scores)[order];labels=np.asarray(labels)[order];status=np.asarray(status)[order];splits=np.asarray(splits)[order]
    available=old.verify_corpus(scores,labels,Path(__file__).parent/'sensitivity_outputs/combination_record_matrix.csv')
    pos=available.any(axis=1);named=labels<4;match=named & available[np.arange(len(labels)),np.minimum(labels,3)]
    assert len(labels)==3236 and match.sum()==1331
    support=pd.DataFrame({'score':range(11),**{d:[int((scores[:,j]==v).sum()) for v in range(11)] for j,d in enumerate(old.ENGLISH)}})
    support.to_csv(out/'score_support.csv',index=False)
    rows=[]
    for scope,mask in [('all',np.ones(len(labels),bool)),('train',splits=='train'),('validation',splits=='validation')]:
        for st in ['missing_key','null','blank','explicit_none','named_type']:
            select=mask & ((status==st) if st in ['missing_key','null','blank'] else (status=='entered') & ((labels==4) if st=='explicit_none' else named))
            rows.append(dict(scope=scope,status=st,n=int(select.sum()),score_positive=int((select&pos).sum())))
    pd.DataFrame(rows).to_csv(out/'raw_category_missingness.csv',index=False)
    # Complete-field re-estimation, with missing excluded instead of recoded.
    clean=status=='entered'; sensitivity=[]
    for scope,mask in [('legacy_blank_as_none',np.ones(len(labels),bool)),('observed_category_only',clean)]:
        sensitivity.append(dict(scope=scope,metric='RQ1',numerator=int((match&mask).sum()),denominator=int((pos&mask).sum())))
        for j,d in enumerate(old.ENGLISH):
            m=mask&available[:,j];sensitivity.append(dict(scope=scope,metric=d,numerator=int((m&(labels!=j)).sum()),denominator=int(m.sum())))
    pd.DataFrame(sensitivity).to_csv(out/'missingness_sensitivity.csv',index=False)
    cells=np.array([(named&pos).sum(),(named&~pos).sum(),(~named&pos).sum(),(~named&~pos).sum()])
    def kapp(c):
        n=c.sum(axis=-1);po=(c[...,0]+c[...,3])/n
        pe=((c[...,0]+c[...,1])*(c[...,0]+c[...,2])+(c[...,2]+c[...,3])*(c[...,1]+c[...,3]))/n**2
        return (po-pe)/(1-pe)
    rng=np.random.default_rng(20260909);boot=rng.multinomial(len(labels),cells/len(labels),size=9999)
    kap=float(kapp(cells));assert abs(kap-cohen_kappa_score(named,pos))<1e-12
    agreement=dict(cells=cells.tolist(),order=['both_positive','type_only','score_only','both_negative'],
        kappa=kap,kappa_ci=np.quantile(kapp(boot),[.025,.975]).tolist(),bootstrap_replicates=9999,seed=20260909,
        presence_agreement=old.rate(int(cells[0]+cells[3]),len(labels)),conditional_match=old.rate(int(match.sum()),int(cells[0])))
    # Threshold changes are supported by actual values, including rare intermediary scores.
    threshold=[]
    for cutoff in range(3,8):
        for j,d in enumerate(old.ENGLISH):
            m=scores[:,j]>=cutoff;threshold.append(dict(cutoff=cutoff,domain=d,n=int(m.sum()),not_same=int((m&(labels!=j)).sum())))
    pd.DataFrame(threshold).to_csv(out/'observed_thresholds.csv',index=False)
    eligible=(available.sum(axis=1)>=2)&match
    model=models(scores[eligible],labels[eligible],available[eligible],out)
    # All-domain alternative sensitivity for the same order-based predictor.
    all_named=(available.sum(axis=1)>=2)&named;av4=np.ones((all_named.sum(),4),bool)
    x4=ordered_design(scores[all_named],av4,'combined');b4,nll4,cov4,_=old.fit(x4,av4,labels[all_named])
    se4=np.sqrt(np.diag(cov4));model['all_four_alternatives']=dict(n=int(all_named.sum()),beta=b4.tolist(),
         odds_ratio=np.exp(b4).tolist(),ci_low=np.exp(b4-1.96*se4).tolist(),ci_high=np.exp(b4+1.96*se4).tolist())
    def date_summary(values):
        parsed=pd.to_datetime(pd.Series(values),errors='coerce')
        return dict(min=str(parsed.min().date()),max=str(parsed.max().date()),missing_or_invalid=int(parsed.isna().sum()),
                    by_year={str(k):int(v) for k,v in parsed.dt.year.value_counts().sort_index().items()})
    result=dict(source_zip_sha256=hashes,n=len(labels),raw_category_counts=dict(raw_counts),missing_category_count=int((~clean).sum()),
        raw_score_missing_count=0,source_joint_table_verified=True,counseling_dates=date_summary(dates),evaluation_dates=date_summary(evaluation_dates),
        agreement=agreement,ordered_models=model,privacy='Aggregate outputs only; no identifiers, narrative, or row-level data.')
    (out/'measurement_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='ordered_models'},ensure_ascii=False,indent=2))
    print(json.dumps(model,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
