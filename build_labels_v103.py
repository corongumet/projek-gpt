#!/usr/bin/env python
# AI Swing Structure V1.03 - Label Dataset Validation
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HORIZON=24

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def latest(folder, pattern):
    xs=sorted(folder.glob(pattern), key=lambda p:p.stat().st_mtime, reverse=True)
    if not xs: raise FileNotFoundError(f'Tidak menemukan {pattern} di {folder}')
    return xs[0]

def times(df, cols, name):
    c=next((x for x in cols if x in df.columns),None)
    if c is None: raise ValueError(f'Kolom timestamp tidak ditemukan pada {name}')
    s=df[c]
    if pd.api.types.is_numeric_dtype(s):
        v=pd.to_numeric(s,errors='coerce'); med=float(v.dropna().abs().median()) if v.notna().any() else 0
        t=pd.to_datetime(v,unit='ms' if med>1e11 else 's',utc=True,errors='coerce')
    else: t=pd.to_datetime(s,utc=True,errors='coerce')
    if t.isna().any(): raise ValueError(f'{name}: {int(t.isna().sum())} timestamp gagal diparse')
    return t

def read_parts(path):
    m=json.loads(path.read_text(encoding='utf-8')); frames=[]
    if not m.get('parts'): raise ValueError('parts manifest tidak memiliki daftar parts')
    for x in m['parts']:
        p=path.parent/x['file']
        if not p.exists(): raise FileNotFoundError(p)
        if p.stat().st_size!=int(x['size_bytes']): raise ValueError(f'Ukuran file tidak cocok: {p.name}')
        if sha256(p).lower()!=x['sha256'].lower(): raise ValueError(f'SHA256 tidak cocok: {p.name}')
        d=pd.read_csv(p,low_memory=False)
        if list(d.columns)!=m['header']: raise ValueError(f'Header tidak cocok: {p.name}')
        if len(d)!=int(x['data_rows']): raise ValueError(f'Jumlah baris tidak cocok: {p.name}')
        frames.append(d)
    out=pd.concat(frames,ignore_index=True)
    if len(out)!=int(m['total_data_rows']): raise ValueError('Total rows parts tidak cocok manifest')
    return out,m

def dist(s, labels):
    n=len(s); c=s.value_counts()
    return {k:{'count':int(c.get(k,0)),'percent':round(100*int(c.get(k,0))/n,6)} for k in labels}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root'); ap.add_argument('--parts-manifest'); ap.add_argument('--feature-manifest'); ap.add_argument('--raw-file'); ap.add_argument('--rows-per-part',type=int,default=10000,help='Jumlah baris per CSV part (default: 10000)')
    a=ap.parse_args();
    if a.rows_per_part < 1: raise ValueError('--rows-per-part harus >= 1')
    root=Path(a.root).resolve() if a.root else Path(__file__).resolve().parent.parent
    proc=root/'data'/'processed'; reports=root/'reports'; proc.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    pm=Path(a.parts_manifest).resolve() if a.parts_manifest else latest(proc,'features_EURUSDm_H1_V102_*_parts_manifest.json')
    fm=Path(a.feature_manifest).resolve() if a.feature_manifest else max((p for p in proc.glob('features_EURUSDm_H1_V102_*_manifest.json') if not p.name.endswith('_parts_manifest.json')),key=lambda p:p.stat().st_mtime)
    manifest=json.loads(fm.read_text(encoding='utf-8'))
    if manifest.get('model_feature_count')!=45 or manifest.get('diagnostic_feature_count')!=4: raise ValueError('Schema V1.02 harus berisi 45 model features + 4 diagnostic columns')
    features,_=read_parts(pm)
    if len(features)!=int(manifest['output_rows']): raise ValueError('Jumlah feature rows tidak cocok feature manifest')
    if a.raw_file: rawpath=Path(a.raw_file).expanduser().resolve()
    else:
        rawname=Path(str(manifest.get('source_file','')).replace('\\','/')).name
        cands=[root/'data'/'raw'/rawname]+sorted((root/'data'/'raw').glob('ohlc_EURUSDm_H1_*.csv'),key=lambda p:p.stat().st_mtime,reverse=True)
        rawpath=next((p for p in cands if p.exists()),None)
        if rawpath is None: raise FileNotFoundError('Raw OHLC tidak ditemukan. Gunakan --raw-file "C:\\AI_Swing_Structure\\data\\raw\\ohlc_....csv"')
    if not rawpath.exists(): raise FileNotFoundError(rawpath)
    raw=pd.read_csv(rawpath,low_memory=False)
    ft=times(features,('time_utc','time'),'Feature CSV'); rt=times(raw,('time_utc','time','timestamp','datetime'),'Raw OHLC')
    if 'close' not in raw or 'atr_14' not in features: raise ValueError('Kolom close (raw) atau atr_14 (features) tidak tersedia')
    if ft.duplicated().any() or rt.duplicated().any(): raise ValueError('Timestamp duplikat ditemukan')
    if not ft.is_monotonic_increasing or not rt.is_monotonic_increasing: raise ValueError('Timestamp tidak monoton naik')
    rawclose=pd.to_numeric(raw['close'],errors='coerce').to_numpy(float)
    if not np.isfinite(rawclose).all(): raise ValueError('Raw close mengandung nilai non-finite')
    rm=pd.DataFrame({'ts':rt,'pos':np.arange(len(raw))})
    mapped=pd.DataFrame({'ts':ft,'fpos':np.arange(len(features))}).merge(rm,on='ts',how='left',validate='one_to_one',indicator=True)
    missing=int((mapped['_merge']!='both').sum())
    if missing: raise ValueError(f'{missing} timestamp feature tidak ditemukan pada raw')
    pos=mapped.sort_values('fpos')['pos'].to_numpy(dtype=np.int64)
    jumps=np.diff(pos); bad=int(np.sum(jumps!=1))
    if bad: raise ValueError(f'Ditemukan {bad} lompatan posisi raw di antara feature rows; label horizon tidak aman')
    n=len(features); valid=n-HORIZON
    if valid<=0: raise ValueError('Feature rows tidak cukup untuk horizon 24')
    start=rawclose[pos[:valid]]; future=rawclose[pos[HORIZON:HORIZON+valid]]
    atr=pd.to_numeric(features['atr_14'],errors='coerce').to_numpy(float)[:valid]
    if not np.isfinite(atr).all() or (atr<=0).any(): raise ValueError('atr_14 harus finite dan > 0')
    net=future-start; move=net/atr
    path=np.zeros(valid)
    for k in range(1,HORIZON+1): path+=np.abs(rawclose[pos[:valid]+k]-rawclose[pos[:valid]+k-1])
    eff=np.divide(np.abs(net),path,out=np.zeros_like(path),where=path>0).clip(0,1)
    am=np.abs(move); regime=np.full(valid,'TRANSITION',dtype=object)
    regime[(am>=1.0)&(eff>=.35)]='TRENDING'; regime[(am<=.50)&(eff<=.25)]='RANGING'
    direction=np.full(valid,'NEUTRAL',dtype=object)
    direction[(move>=.50)&(eff>=.25)]='BULLISH'; direction[(move<=-.50)&(eff>=.25)]='BEARISH'
    out=features.iloc[:valid].copy().reset_index(drop=True)
    out['label_forward_close']=future; out['label_net_move']=net; out['label_move_atr']=move; out['label_path_efficiency']=eff
    out['label_regime']=regime; out['label_direction']=direction
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); sym=manifest.get('symbol','EURUSDm'); tf=manifest.get('timeframe','H1')
    stem=f'labels_{sym}_{tf}_V103_{stamp}'
    mpath=proc/f'{stem}_manifest.json'
    parts_path=proc/f'{stem}_parts_manifest.json'
    rpt=reports/f'label_validation_v103_{sym}_{tf}_{stamp}.txt'; jpath=reports/f'label_validation_v103_{sym}_{tf}_{stamp}.json'

    # Write deterministic row chunks so each CSV remains reasonably small for GitHub.
    part_records=[]
    for part_idx, start_row in enumerate(range(0, len(out), a.rows_per_part), start=1):
        end_row=min(start_row+a.rows_per_part, len(out))
        part_file=f'{stem}_part_{part_idx:03d}.csv'
        part_path=proc/part_file
        out.iloc[start_row:end_row].to_csv(part_path,index=False,float_format='%.12g')
        part_records.append({
            'file':part_file,
            'data_rows':int(end_row-start_row),
            'first_data_row':int(start_row+1),
            'last_data_row':int(end_row),
            'size_bytes':int(part_path.stat().st_size),
            'sha256':sha256(part_path)
        })

    d={'feature_rows':n,'label_rows':valid,'tail_rows_removed':HORIZON,'horizon_raw_bars':HORIZON,'feature_timestamps_missing_in_raw':missing,'internal_raw_position_jumps':bad,'feature_timestamp_duplicates':int(ft.duplicated().sum()),'raw_timestamp_duplicates':int(rt.duplicated().sum()),'first_feature_utc':ft.iloc[0].isoformat(),'last_feature_utc':ft.iloc[-1].isoformat(),'first_label_utc':ft.iloc[0].isoformat(),'last_label_utc':ft.iloc[valid-1].isoformat(),'first_future_target_utc':ft.iloc[HORIZON].isoformat(),'last_future_target_utc':ft.iloc[-1].isoformat(),'regime_distribution':dist(pd.Series(regime),['TRENDING','RANGING','TRANSITION']),'direction_distribution':dist(pd.Series(direction),['BULLISH','BEARISH','NEUTRAL']),'move_atr_min':float(move.min()),'move_atr_max':float(move.max()),'efficiency_min':float(eff.min()),'efficiency_max':float(eff.max())}
    result={'project':'AI_Swing_Structure','label_dataset_version':'1.03-label-dataset-validation','created_utc':datetime.now(timezone.utc).isoformat(),'symbol':sym,'timeframe':tf,'feature_parts_manifest':str(pm),'feature_manifest':str(fm),'raw_source_file':str(rawpath),'model_feature_count':45,'diagnostic_feature_count':4,'label_definitions':{'TRENDING':'abs(move_atr) >= 1.0 AND efficiency >= 0.35','RANGING':'abs(move_atr) <= 0.50 AND efficiency <= 0.25','TRANSITION':'otherwise','BULLISH':'move_atr >= 0.50 AND efficiency >= 0.25','BEARISH':'move_atr <= -0.50 AND efficiency >= 0.25','NEUTRAL':'otherwise','move_atr':'(close[t+24]-close[t])/atr_14[t]','efficiency':'abs(net displacement)/sum(abs(close-to-close changes))'},'validation':d,'output_format':'CSV_PARTS','output_parts_manifest':str(parts_path),'output_part_count':len(part_records),'rows_per_part':a.rows_per_part,'models_trained':False}
    parts_manifest={'dataset_file_prefix':stem,'total_data_rows':int(len(out)),'header_columns':int(len(out.columns)),'header':list(out.columns),'rows_per_part':int(a.rows_per_part),'part_count':len(part_records),'parts':part_records,'values_modified':False,'note':'Parts disimpan berurutan; gabungkan sesuai urutan part_001, part_002, dan seterusnya. Gunakan SHA256 untuk verifikasi integritas.'}
    parts_path.write_text(json.dumps(parts_manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    result['parts_manifest_sha256']=sha256(parts_path)
    mpath.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    status='PASSED' if missing==0 and bad==0 and valid==n-HORIZON else 'FAILED'
    lines=['AI SWING STRUCTURE - V1.03 LABEL DATASET VALIDATION','='*70,f'Status                         : {status}',f'Symbol / timeframe             : {sym} / {tf}',f'Raw OHLC                       : {rawpath}',f'Feature rows                   : {n}',f'Label rows                     : {valid}',f'Tail rows removed              : {HORIZON}',f'Forward horizon                : {HORIZON} consecutive raw bars',f'Feature timestamps missing raw : {missing}',f'Internal raw position jumps    : {bad}',f'Output CSV parts                : {len(part_records)}',f'Max rows per CSV part           : {a.rows_per_part}',f'Parts manifest                 : {parts_path}',f'First label UTC                : {d["first_label_utc"]}',f'Last label UTC                 : {d["last_label_utc"]}','','REGIME DISTRIBUTION']
    for k,v in d['regime_distribution'].items(): lines.append(f'  {k:12s}: {v["count"]:>7,d} ({v["percent"]:>8.4f}%)')
    lines+=['','DIRECTION DISTRIBUTION']
    for k,v in d['direction_distribution'].items(): lines.append(f'  {k:12s}: {v["count"]:>7,d} ({v["percent"]:>8.4f}%)')
    lines+=['','LABEL DEFINITIONS (UNCHANGED FROM V1.01)','  TRENDING: abs(move_atr) >= 1.00 and efficiency >= 0.35','  RANGING: abs(move_atr) <= 0.50 and efficiency <= 0.25','  TRANSITION: otherwise','  BULLISH: move_atr >= 0.50 and efficiency >= 0.25','  BEARISH: move_atr <= -0.50 and efficiency >= 0.25','  NEUTRAL: otherwise','','Horizon counts 24 consecutive rows in raw OHLC, not 24 clock hours.','Weekend/holiday gaps are acceptable; missing raw rows within feature sequence are not.','This stage validates labels only; no model training is performed.','= '*70]
    rpt.write_text('\n'.join(lines),encoding='utf-8'); jpath.write_text(json.dumps({'status':status,**d},indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n'.join(lines)); print(f'\nCSV parts: {len(part_records)} files in {proc}\nParts manifest: {parts_path}\nDataset manifest: {mpath}\nReport: {rpt}\nJSON: {jpath}')
    if status!='PASSED': sys.exit(2)

if __name__=='__main__': main()
