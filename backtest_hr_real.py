from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

from train_hr_real import (
    read_sources, build_pa, make_features, fit_bundle, pred, exposure,
    game_eval, metrics, ece, topk
)

def wilson(k, n, z=1.96):
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [max(0.0, ctr-half), min(1.0, ctr+half)]

def daily_topk(g, k):
    rows=[]
    for day,d in g.groupby('game_date'):
        q=d.nlargest(k,'p_game')
        rows.append({
            'game_date': str(pd.Timestamp(day).date()),
            'k': k,
            'any_hr': int(q.is_hr.max()>0),
            'hits': int(q.is_hr.sum()),
            'picks': int(len(q)),
            'pick_precision': float(q.is_hr.mean()) if len(q) else float('nan'),
            'mean_model_p': float(q.p_game.mean()) if len(q) else float('nan'),
        })
    z=pd.DataFrame(rows)
    hits=int(z.any_hr.sum()) if len(z) else 0
    return z, {
        f'top_{k}_days_with_hit': float(z.any_hr.mean()) if len(z) else None,
        f'top_{k}_pick_precision': float(z.hits.sum()/z.picks.sum()) if len(z) else None,
        f'top_{k}_days': int(len(z)),
        f'top_{k}_days_hit_count': hits,
        f'top_{k}_days_with_hit_95ci': wilson(hits, len(z)),
        f'top_{k}_total_hits': int(z.hits.sum()) if len(z) else 0,
        f'top_{k}_total_picks': int(z.picks.sum()) if len(z) else 0,
    }

def calibration_table(y,p,bins=10):
    d=pd.DataFrame({'y':np.asarray(y,dtype=int),'p':np.asarray(p,dtype=float)})
    d['bin']=pd.cut(d.p, bins=np.linspace(0,1,bins+1), include_lowest=True, right=True)
    out=d.groupby('bin',observed=True).agg(n=('y','size'),mean_pred=('p','mean'),actual_rate=('y','mean')).reset_index()
    out['abs_gap']=(out.mean_pred-out.actual_rate).abs()
    out['bin']=out['bin'].astype(str)
    return out

def fold_eval(feat, pa, year):
    start=pd.Timestamp(f'{year}-01-01')
    end=pd.Timestamp(f'{year}-12-31')
    train=feat[feat.game_date < start].copy()
    test=feat[feat.game_date.between(start,end)].copy()
    if train.game_date.nunique() < 60 or test.empty:
        raise RuntimeError(f'fold {year} insufficient rows/dates')
    bundle=fit_bundle(train)
    ppa=pred(bundle,test)
    exp_train=exposure(pa[pa.game_date < start])
    g=game_eval(test,ppa,exp_train)

    baseline_pa=test['h_hr_rate_40'].fillna(train.is_hr.mean()).clip(1e-6,.5).to_numpy()
    gb=game_eval(test,baseline_pa,exp_train)

    top5df,top5m=daily_topk(g,5)
    top10df,top10m=daily_topk(g,10)

    model_game=metrics(g.is_hr.to_numpy(),g.p_game.to_numpy())
    base_game=metrics(gb.is_hr.to_numpy(),gb.p_game.to_numpy())
    report={
      'year':year,
      'train_start':str(train.game_date.min().date()),
      'train_end':str(train.game_date.max().date()),
      'test_start':str(test.game_date.min().date()),
      'test_end':str(test.game_date.max().date()),
      'train_pa_rows':int(len(train)),
      'test_pa_rows':int(len(test)),
      'test_game_hitter_rows':int(len(g)),
      'test_hr_rate_pa':float(test.is_hr.mean()),
      'test_hr_rate_game':float(g.is_hr.mean()),
      'always_no_hr_game_accuracy':float(1-g.is_hr.mean()),
      'calibration_split_date':bundle[4],
      'pa_metrics':metrics(test.is_hr.to_numpy(),ppa),
      'game_metrics':model_game,
      'game_ece_10bin':ece(g.is_hr.to_numpy(),g.p_game.to_numpy()),
      'baseline_40game_game_metrics':base_game,
      'delta_vs_baseline':{
        'log_loss':float(model_game['log_loss']-base_game['log_loss']),
        'brier':float(model_game['brier']-base_game['brier']),
        'pr_auc':float(model_game['pr_auc']-base_game['pr_auc']),
        'roc_auc':float(model_game['roc_auc']-base_game['roc_auc']),
      },
      'top5':top5m,
      'top10':top10m,
    }
    cal=calibration_table(g.is_hr.to_numpy(),g.p_game.to_numpy(),10)
    return report, top5df, top10df, cal

def pooled_summary(folds):
    out={}
    for key in ['log_loss','brier','pr_auc','roc_auc']:
        vals=np.array([f['game_metrics'][key] for f in folds],float)
        weights=np.array([f['test_game_hitter_rows'] for f in folds],float)
        out[key+'_weighted']=float(np.average(vals,weights=weights))
    for k in [5,10]:
        hit=sum(f[f'top{k}'][f'top_{k}_days_hit_count'] for f in folds)
        days=sum(f[f'top{k}'][f'top_{k}_days'] for f in folds)
        total_hits=sum(f[f'top{k}'][f'top_{k}_total_hits'] for f in folds)
        total_picks=sum(f[f'top{k}'][f'top_{k}_total_picks'] for f in folds)
        out[f'top_{k}_days_with_hit']=hit/days
        out[f'top_{k}_days_with_hit_95ci']=wilson(hit,days)
        out[f'top_{k}_pick_precision']=total_hits/total_picks
        out[f'top_{k}_days']=days
        out[f'top_{k}_days_hit_count']=hit
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-dir',default='source')
    ap.add_argument('--outdir',default='backtest/annual-2024-2026')
    ap.add_argument('--start',default='2023-03-30')
    ap.add_argument('--end',default='2026-09-17')
    a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)

    pitches=read_sources(a.source_dir,a.start,a.end)
    pa=build_pa(pitches); del pitches
    feat=make_features(pa)
    print('feature rows',len(feat),'dates',feat.game_date.min(),feat.game_date.max(),flush=True)

    fold_reports=[]; daily=[]; cal=[]
    for year in [2024,2025,2026]:
        print('FOLD',year,flush=True)
        rep,t5,t10,ct=fold_eval(feat,pa,year)
        fold_reports.append(rep)
        t5['year']=year; t10['year']=year
        daily += [t5,t10]
        ct['year']=year; cal.append(ct)
        print(json.dumps(rep,indent=2),flush=True)

    final={
      'design':'expanding-window annual out-of-sample backtest; each test season trained only on prior seasons; all rolling features shifted by one game',
      'source_start':str(pa.game_date.min().date()),
      'source_end':str(pa.game_date.max().date()),
      'regular_season_only':True,
      'folds':fold_reports,
      'pooled':pooled_summary(fold_reports),
    }
    (out/'annual_backtest_report.json').write_text(json.dumps(final,indent=2))
    pd.concat(daily,ignore_index=True).to_csv(out/'daily_topk_backtest.csv',index=False)
    pd.concat(cal,ignore_index=True).to_csv(out/'calibration_bins.csv',index=False)
    print('FINAL',json.dumps(final,indent=2),flush=True)

if __name__=='__main__':
    main()
