from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

SEED=20260918
HITTER_WINDOWS=(10,40); PITCHER_WINDOWS=(10,40); SPLIT_WINDOW=30
NEEDED=['game_date','game_pk','at_bat_number','pitch_number','batter','pitcher','events','stand','p_throws','home_team','away_team','inning_topbot','launch_speed','launch_speed_angle','launch_angle','game_type']
BASE=['lineup_slot','is_home','platoon_adv','batter_L','batter_R','batter_S','pitcher_R']
def metric_names(prefix, wins):
    out=[]
    for w in wins: out += [f'{prefix}_pa_{w}',f'{prefix}_hr_rate_{w}',f'{prefix}_barrel_rate_{w}',f'{prefix}_hard_rate_{w}',f'{prefix}_ev_{w}',f'{prefix}_sweet_rate_{w}']
    return out
FEATURES=BASE+metric_names('h',HITTER_WINDOWS)+[f'h_split_pa_{SPLIT_WINDOW}',f'h_split_hr_rate_{SPLIT_WINDOW}',f'h_split_barrel_rate_{SPLIT_WINDOW}',f'h_split_hard_rate_{SPLIT_WINDOW}']+metric_names('p',PITCHER_WINDOWS)+[f'p_split_pa_{SPLIT_WINDOW}',f'p_split_hr_rate_{SPLIT_WINDOW}',f'p_split_barrel_rate_{SPLIT_WINDOW}',f'p_split_hard_rate_{SPLIT_WINDOW}']

def safe_div(a,b): return a/b.replace(0,np.nan)

def read_sources(srcdir,start,end):
    frames=[]
    for p in sorted(Path(srcdir).glob('*')):
        if p.suffix.lower() not in {'.parquet','.pq'}: continue
        print('reading',p,flush=True)
        df=pd.read_parquet(p,columns=NEEDED)
        df['game_date']=pd.to_datetime(df['game_date'],errors='coerce').dt.normalize()
        df=df[df['game_date'].between(pd.Timestamp(start),pd.Timestamp(end))]
        if 'game_type' in df: df=df[df['game_type'].astype(str).eq('R')]
        frames.append(df.drop(columns=['game_type'],errors='ignore'))
    if not frames: raise RuntimeError('no source parquet files found')
    out=pd.concat(frames,ignore_index=True,sort=False)
    out=out.drop_duplicates(subset=['game_pk','at_bat_number','pitch_number','batter','pitcher'],keep='last')
    print('pitch rows',len(out),'dates',out.game_date.min(),out.game_date.max(),flush=True)
    return out

def build_pa(p):
    p=p.sort_values(['game_date','game_pk','at_bat_number','pitch_number'])
    pa=p.groupby(['game_pk','at_bat_number'],sort=False,as_index=False).tail(1).copy()
    pa=pa[pa.events.notna()].copy()
    pa['is_hr']=pa.events.eq('home_run').astype('int8')
    ls=pd.to_numeric(pa.launch_speed,errors='coerce'); la=pd.to_numeric(pa.launch_angle,errors='coerce'); lsa=pd.to_numeric(pa.launch_speed_angle,errors='coerce')
    pa['is_bbe']=ls.notna().astype('int8'); pa['barrel']=lsa.eq(6).astype('int8'); pa['hard_hit']=(ls>=95).fillna(False).astype('int8'); pa['sweet_spot']=la.between(8,32,inclusive='both').fillna(False).astype('int8'); pa['ev_sum']=ls.fillna(0.).astype('float32')
    top=pa.inning_topbot.astype(str).str.lower().str.startswith('top')
    pa['bat_team']=np.where(top,pa.away_team,pa.home_team); pa['fld_team']=np.where(top,pa.home_team,pa.away_team); pa['is_home']=(~top).astype('int8')
    first=(pa.groupby(['game_pk','bat_team','batter'],as_index=False).at_bat_number.min().sort_values(['game_pk','bat_team','at_bat_number','batter']))
    first['lineup_slot']=first.groupby(['game_pk','bat_team']).cumcount()+1
    pa=pa.merge(first[['game_pk','bat_team','batter','lineup_slot']],on=['game_pk','bat_team','batter'],how='left')
    starter=(pa.groupby(['game_pk','fld_team','pitcher'],as_index=False).at_bat_number.min().sort_values(['game_pk','fld_team','at_bat_number']).groupby(['game_pk','fld_team'],as_index=False).first()[['game_pk','fld_team','pitcher']].rename(columns={'pitcher':'starter_pitcher'}))
    pa=pa.merge(starter,on=['game_pk','fld_team'],how='left'); pa['faced_starter']=pa.pitcher.eq(pa.starter_pitcher).astype('int8')
    return pa.reset_index(drop=True)

def game_agg(pa,entity,extra=None):
    keys=[entity,'game_date','game_pk']+(extra or [])
    g=pa.groupby(keys,as_index=False).agg(pa=('is_hr','size'),hr=('is_hr','sum'),bbe=('is_bbe','sum'),barrel=('barrel','sum'),hard=('hard_hit','sum'),sweet=('sweet_spot','sum'),ev_sum=('ev_sum','sum'))
    return g.sort_values([entity]+(extra or [])+['game_date','game_pk'])

def add_roll(g,groups,wins,prefix):
    out=g.copy()
    for w in wins:
        for c in ['pa','hr','bbe','barrel','hard','sweet','ev_sum']:
            out[f'_{c}_{w}']=out.groupby(groups,sort=False)[c].transform(lambda s:s.shift(1).rolling(w,min_periods=3).sum())
        out[f'{prefix}_pa_{w}']=out[f'_pa_{w}']; out[f'{prefix}_hr_rate_{w}']=safe_div(out[f'_hr_{w}'],out[f'_pa_{w}']); out[f'{prefix}_barrel_rate_{w}']=safe_div(out[f'_barrel_{w}'],out[f'_bbe_{w}']); out[f'{prefix}_hard_rate_{w}']=safe_div(out[f'_hard_{w}'],out[f'_bbe_{w}']); out[f'{prefix}_ev_{w}']=safe_div(out[f'_ev_sum_{w}'],out[f'_bbe_{w}']); out[f'{prefix}_sweet_rate_{w}']=safe_div(out[f'_sweet_{w}'],out[f'_bbe_{w}'])
        out.drop(columns=[f'_{c}_{w}' for c in ['pa','hr','bbe','barrel','hard','sweet','ev_sum']],inplace=True)
    return out

def add_split(g,groups,prefix,w=30):
    out=g.copy()
    for c in ['pa','hr','bbe','barrel','hard']:
        out[f'_{c}']=out.groupby(groups,sort=False)[c].transform(lambda s:s.shift(1).rolling(w,min_periods=3).sum())
    out[f'{prefix}_pa_{w}']=out._pa; out[f'{prefix}_hr_rate_{w}']=safe_div(out._hr,out._pa); out[f'{prefix}_barrel_rate_{w}']=safe_div(out._barrel,out._bbe); out[f'{prefix}_hard_rate_{w}']=safe_div(out._hard,out._bbe)
    return out.drop(columns=['_pa','_hr','_bbe','_barrel','_hard'])

def make_features(pa):
    x=pa.copy()
    hg=add_roll(game_agg(x,'batter'),['batter'],HITTER_WINDOWS,'h'); x=x.merge(hg[['batter','game_pk']+metric_names('h',HITTER_WINDOWS)],on=['batter','game_pk'],how='left')
    hs=add_split(game_agg(x,'batter',['p_throws']),['batter','p_throws'],'h_split'); hc=['batter','game_pk','p_throws']+[f'h_split_{m}_{SPLIT_WINDOW}' for m in ['pa','hr_rate','barrel_rate','hard_rate']]; x=x.merge(hs[hc],on=['batter','game_pk','p_throws'],how='left')
    pg=add_roll(game_agg(x,'pitcher'),['pitcher'],PITCHER_WINDOWS,'p'); x=x.merge(pg[['pitcher','game_pk']+metric_names('p',PITCHER_WINDOWS)],on=['pitcher','game_pk'],how='left')
    ps=add_split(game_agg(x,'pitcher',['stand']),['pitcher','stand'],'p_split'); pc=['pitcher','game_pk','stand']+[f'p_split_{m}_{SPLIT_WINDOW}' for m in ['pa','hr_rate','barrel_rate','hard_rate']]; x=x.merge(ps[pc],on=['pitcher','game_pk','stand'],how='left')
    x['batter_L']=x.stand.eq('L').astype('int8'); x['batter_R']=x.stand.eq('R').astype('int8'); x['batter_S']=x.stand.eq('S').astype('int8'); x['pitcher_R']=x.p_throws.eq('R').astype('int8'); x['platoon_adv']=(x.stand.eq('S')|x.stand.ne(x.p_throws)).astype('int8')
    return x[x.lineup_slot.between(1,9,inclusive='both')].sort_values(['game_date','game_pk','at_bat_number']).reset_index(drop=True)

def new_model():
    return XGBClassifier(objective='binary:logistic',eval_metric='logloss',n_estimators=700,max_depth=4,learning_rate=.035,min_child_weight=10,subsample=.85,colsample_bytree=.80,reg_alpha=.20,reg_lambda=2.0,tree_method='hist',random_state=SEED,n_jobs=max(1,(os.cpu_count() or 2)-1))

def calibrator_fit(raw,y):
    p=np.clip(raw,1e-6,1-1e-6); z=np.log(p/(1-p)).reshape(-1,1); c=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000); c.fit(z,y); return c

def calibrator_apply(c,raw):
    p=np.clip(raw,1e-6,1-1e-6); z=np.log(p/(1-p)).reshape(-1,1); return c.predict_proba(z)[:,1]

def fit_bundle(df):
    d=df.sort_values('game_date').copy(); dates=np.array(sorted(d.game_date.unique())); split=max(1,int(len(dates)*.85)); split_date=dates[min(split,len(dates)-1)]
    tr=d[d.game_date<split_date]; cal=d[d.game_date>=split_date]
    med=tr[FEATURES].median(numeric_only=True).reindex(FEATURES).fillna(0.).to_dict(); X=tr[FEATURES].fillna(med).astype('float32'); y=tr.is_hr.astype(int)
    m=new_model(); m.fit(X,y); raw=m.predict_proba(cal[FEATURES].fillna(med).astype('float32'))[:,1]; c=calibrator_fit(raw,cal.is_hr.astype(int).to_numpy())
    sr=d.loc[d.faced_starter.eq(1),'is_hr'].mean(); rr=d.loc[d.faced_starter.eq(0),'is_hr'].mean(); ratio=float(np.clip(rr/sr if sr and np.isfinite(sr) else 1.,.70,1.35))
    return m,c,med,ratio,str(pd.Timestamp(split_date).date())

def pred(bundle,df):
    m,c,med,_,_=bundle; raw=m.predict_proba(df[FEATURES].fillna(med).astype('float32'))[:,1]; return calibrator_apply(c,raw)

def exposure(pa):
    g=pa[pa.lineup_slot.between(1,9,inclusive='both')].groupby(['game_date','game_pk','batter','lineup_slot'],as_index=False).agg(total_pa=('is_hr','size'),starter_pa=('faced_starter','sum'))
    g['total_pa']=g.total_pa.clip(1,7); g['starter_pa']=np.minimum(g.starter_pa,g.total_pa).clip(0,4); return g[['lineup_slot','total_pa','starter_pa']].astype(int)

def metrics(y,p):
    o={'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),'pr_auc':float(average_precision_score(y,p))}
    if len(np.unique(y))>1:o['roc_auc']=float(roc_auc_score(y,p))
    return o

def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); val=0.
    for i in range(bins):
        mask=(p>=edges[i]) & (p < edges[i+1] if i<bins-1 else p<=edges[i+1])
        if mask.any(): val += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(val)

def game_eval(frame,p_pa,exp):
    t=frame[['game_date','game_pk','at_bat_number','batter','lineup_slot','is_hr']].copy(); t['p_pa']=p_pa
    first=t.sort_values(['game_date','game_pk','at_bat_number']).groupby(['game_date','game_pk','batter'],as_index=False).first(); truth=t.groupby(['game_date','game_pk','batter'],as_index=False).is_hr.max(); g=first.drop(columns='is_hr').merge(truth,on=['game_date','game_pk','batter'])
    em=exp.groupby('lineup_slot').total_pa.mean().to_dict(); g['expected_pa']=g.lineup_slot.round().astype(int).map(em).fillna(4.2); g['p_game']=1-np.power(1-np.clip(g.p_pa,1e-6,.5),g.expected_pa)
    return g

def topk(g,k):
    days=[]
    for _,d in g.groupby('game_date'):
        q=d.nlargest(k,'p_game'); days.append((int(q.is_hr.max()>0),float(q.is_hr.mean())))
    a=np.array(days,float); return {f'top_{k}_days_with_hit':float(a[:,0].mean()),f'top_{k}_pick_precision':float(a[:,1].mean()),'days':int(len(a))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-dir',default='source'); ap.add_argument('--outdir',default='trained/2026-09-17'); ap.add_argument('--start',default='2023-03-30'); ap.add_argument('--end',default='2026-09-17'); a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    pitches=read_sources(a.source_dir,a.start,a.end); pa=build_pa(pitches); del pitches
    print('plate appearances',len(pa),'HR',int(pa.is_hr.sum()),flush=True)
    feat=make_features(pa); print('feature rows',len(feat),flush=True)
    cutoff=pd.Timestamp('2026-01-01'); train=feat[feat.game_date<cutoff]; test=feat[feat.game_date>=cutoff]
    hold_bundle=fit_bundle(train); ppa=pred(hold_bundle,test); exp_train=exposure(pa[pa.game_date<cutoff]); g=game_eval(test,ppa,exp_train)
    report={'data_start':str(pa.game_date.min().date()),'data_end':str(pa.game_date.max().date()),'pitch_rows_after_filter':None,'plate_appearances':int(len(pa)),'feature_rows':int(len(feat)),'holdout_start':'2026-01-01','holdout_pa_rows':int(len(test)),'holdout_hr_rate_pa':float(test.is_hr.mean()),'holdout_pa_metrics':metrics(test.is_hr.to_numpy(),ppa),'holdout_game_metrics':metrics(g.is_hr.to_numpy(),g.p_game.to_numpy()),'holdout_game_ece_10bin':ece(g.is_hr.to_numpy(),g.p_game.to_numpy()),'top5':topk(g,5),'top10':topk(g,10),'holdout_calibration_split_date':hold_bundle[4]}
    base_pa=feat.loc[test.index,'h_hr_rate_40'].fillna(train.is_hr.mean()).clip(1e-6,.5).to_numpy(); gb=game_eval(test,base_pa,exp_train); report['baseline_40game_game_metrics']=metrics(gb.is_hr.to_numpy(),gb.p_game.to_numpy())
    final_bundle=fit_bundle(feat); m,c,med,ratio,split_date=final_bundle; exp_all=exposure(pa)
    joblib.dump(m,out/'xgb_model.joblib'); joblib.dump(c,out/'calibrator.joblib'); (out/'medians.json').write_text(json.dumps(med,indent=2)); exp_all.to_csv(out/'pa_exposure.csv',index=False)
    source_status={}
    sp=Path(a.source_dir)/'source_status.json'
    if sp.exists():
        source_status=json.loads(sp.read_text())
    report['source_status']=source_status
    meta={'train_start':report['data_start'],'train_end':report['data_end'],'rows_pa':int(len(feat)),'hr_rate_pa':float(feat.is_hr.mean()),'features':FEATURES,'relief_vs_starter_hr_ratio':ratio,'model':'XGBClassifier + chronological sigmoid calibration','calibration_split_date':split_date,'source':'SportsDataverse monthly Statcast archive + September patch','source_status':source_status,'regular_season_only':True,'random_seed':SEED}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); (out/'backtest_report.json').write_text(json.dumps(report,indent=2));
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
