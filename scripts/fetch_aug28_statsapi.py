import gzip, json, math
from pathlib import Path
import pandas as pd
import requests

DATE='2026-08-28'
OUT=Path('output'); OUT.mkdir(exist_ok=True)
TEAM_NAME={
'Arizona Diamondbacks':'ARI','Atlanta Braves':'ATL','Baltimore Orioles':'BAL','Boston Red Sox':'BOS','Chicago Cubs':'CHN','Chicago White Sox':'CHA','Cincinnati Reds':'CIN','Cleveland Guardians':'CLE','Colorado Rockies':'COL','Detroit Tigers':'DET','Houston Astros':'HOU','Kansas City Royals':'KCA','Los Angeles Angels':'ANA','Los Angeles Dodgers':'LAN','Miami Marlins':'MIA','Milwaukee Brewers':'MIL','Minnesota Twins':'MIN','New York Mets':'NYN','New York Yankees':'NYA','Athletics':'ATH','Philadelphia Phillies':'PHI','Pittsburgh Pirates':'PIT','San Diego Padres':'SDN','Seattle Mariners':'SEA','San Francisco Giants':'SFN','St. Louis Cardinals':'SLN','Tampa Bay Rays':'TBA','Texas Rangers':'TEX','Toronto Blue Jays':'TOR','Washington Nationals':'WAS'
}

def get(url,params=None):
    r=requests.get(url,params=params,timeout=30); r.raise_for_status(); return r.json()

def ipouts(ip):
    if ip is None: return 0
    s=str(ip)
    if '.' in s:
        a,b=s.split('.',1); return int(a)*3+int(b[:1] or 0)
    return int(float(s))*3

sched=get('https://statsapi.mlb.com/api/v1/schedule',{'sportId':1,'date':DATE})
games=[]
for d in sched.get('dates',[]): games.extend(d.get('games',[]))

bat=[]; pit=[]; gi=[]
for g in games:
    pk=g['gamePk']; box=get(f'https://statsapi.mlb.com/api/v1/game/{pk}/boxscore')
    away=g['teams']['away']['team']; home=g['teams']['home']['team']
    ac=TEAM_NAME.get(away.get('name')); hc=TEAM_NAME.get(home.get('name'))
    if not ac or not hc: raise RuntimeError(f'unknown team mapping {away} {home}')
    gid=f'{hc}{DATE.replace("-","")}0'
    ar=g['teams']['away'].get('score',0) or 0; hr=g['teams']['home'].get('score',0) or 0
    gi.append({'gid':gid,'visteam':ac,'hometeam':hc,'date':int(DATE.replace('-','')),'vruns':ar,'hruns':hr,'wteam':ac if ar>hr else hc,'lteam':hc if ar>hr else ac,'season':2026})
    for side,teamc,opp,vh in [('away',ac,hc,'v'),('home',hc,ac,'h')]:
        t=box['teams'][side]
        players=t.get('players',{})
        batting_order=t.get('battingOrder',[])
        bset=set(int(x) for x in batting_order)
        for key,obj in players.items():
            pid=int(obj['person']['id']); s=obj.get('stats',{}).get('batting',{}) or {}
            pa=s.get('plateAppearances',0) or 0
            if pa<=0: continue
            bat.append({'gid':gid,'id':pid,'team':teamc,'b_pa':pa,'b_ab':s.get('atBats',0) or 0,'b_r':s.get('runs',0) or 0,'b_h':s.get('hits',0) or 0,'b_d':s.get('doubles',0) or 0,'b_t':s.get('triples',0) or 0,'b_hr':s.get('homeRuns',0) or 0,'b_rbi':s.get('rbi',0) or 0,'b_w':s.get('baseOnBalls',0) or 0,'b_k':s.get('strikeOuts',0) or 0,'b_sb':s.get('stolenBases',0) or 0,'b_hbp':s.get('hitByPitch',0) or 0,'b_sf':s.get('sacFlies',0) or 0,'date':int(DATE.replace('-','')),'vishome':vh,'opp':opp,'win':int((teamc==ac and ar>hr) or (teamc==hc and hr>ar)),'loss':int((teamc==ac and ar<hr) or (teamc==hc and hr<ar)),'starter_flag':int(pid in bset)})
        plist=t.get('pitchers',[])
        for j,pid in enumerate(plist):
            pid=int(pid); obj=players.get('ID'+str(pid),{}); s=obj.get('stats',{}).get('pitching',{}) or {}
            if not s: continue
            pit.append({'gid':gid,'id':pid,'team':teamc,'p_ipouts':ipouts(s.get('inningsPitched')),'p_bfp':s.get('battersFaced',0) or 0,'p_h':s.get('hits',0) or 0,'p_hr':s.get('homeRuns',0) or 0,'p_r':s.get('runs',0) or 0,'p_er':s.get('earnedRuns',0) or 0,'p_w':s.get('baseOnBalls',0) or 0,'p_iw':s.get('intentionalWalks',0) or 0,'p_k':s.get('strikeOuts',0) or 0,'p_hbp':s.get('hitBatsmen',0) or 0,'p_wp':s.get('wildPitches',0) or 0,'p_bk':s.get('balks',0) or 0,'p_gs':int(j==0),'p_gf':int(j==len(plist)-1),'p_cg':int(j==0 and len(plist)==1),'wp':0,'lp':0,'save':0,'date':int(DATE.replace('-','')),'vishome':vh,'opp':opp,'win':int((teamc==ac and ar>hr) or (teamc==hc and hr>ar)),'loss':int((teamc==ac and ar<hr) or (teamc==hc and hr<ar))})

pd.DataFrame(bat).to_csv(OUT/'mlb_batting_2026-08-28.csv.gz',index=False,compression='gzip')
pd.DataFrame(pit).to_csv(OUT/'mlb_pitching_2026-08-28.csv.gz',index=False,compression='gzip')
pd.DataFrame(gi).to_csv(OUT/'mlb_gameinfo_2026-08-28.csv.gz',index=False,compression='gzip')
summary={'date':DATE,'games':len(gi),'batting_rows':len(bat),'pitching_rows':len(pit),'starting_batter_rows':sum(r['starter_flag'] for r in bat)}
(OUT/'aug28_statsapi_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
