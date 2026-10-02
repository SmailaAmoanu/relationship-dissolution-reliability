import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from common import load_data, outcomes, FEATURES, DISPLAY, preprocessor, RESULTS

OUT=RESULTS/'06_reliability_analysis'; OUT.mkdir(exist_ok=True)
raw=load_data(); _,y20,y22=outcomes(raw)

def rf(seed=2027):
    return Pipeline([('pre',preprocessor()),('rf',RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=4,max_features='sqrt',class_weight='balanced_subsample',random_state=seed,n_jobs=-1))])

def cq(scores,alpha=.10):
    n=len(scores); level=min(1,math.ceil((n+1)*(1-alpha))/n)
    return float(np.quantile(scores,level,method='higher'))

# Formal SHAP rank stability from archived repeated full-refit rankings.
ranks=pd.read_csv(RESULTS/'04_robustness'/'full_refit_importance_ranks.csv')
seeds=sorted(set(ranks.seed))
within=[]
for h in ['3-year','5-year']:
    z=ranks[ranks.horizon==h].pivot(index='variable',columns='seed',values='rank')
    vals=[]
    for i,s1 in enumerate(seeds):
        for s2 in seeds[i+1:]: vals.append(spearmanr(z[s1],z[s2]).statistic)
    within.append([h,np.mean(vals),np.std(vals,ddof=1),np.min(vals),np.max(vals)])
pd.DataFrame(within,columns=['horizon','mean_pairwise_spearman','sd','min','max']).to_csv(OUT/'01_within_horizon_shap_rank_stability.csv',index=False)

cross=[]
for s in seeds:
    a=ranks[(ranks.horizon=='3-year')&(ranks.seed==s)].set_index('variable')['rank']
    b=ranks[(ranks.horizon=='5-year')&(ranks.seed==s)].set_index('variable')['rank']
    common=a.index.intersection(b.index)
    rho=spearmanr(a.loc[common],b.loc[common]).statistic
    top5a=set(a.nsmallest(5).index); top5b=set(b.nsmallest(5).index)
    top10a=set(a.nsmallest(10).index); top10b=set(b.nsmallest(10).index)
    cross.append([s,rho,len(top5a&top5b)/len(top5a|top5b),len(top10a&top10b)/len(top10a|top10b)])
pd.DataFrame(cross,columns=['seed','spearman','top5_jaccard','top10_jaccard']).to_csv(OUT/'02_cross_horizon_repeated_shap_stability.csv',index=False)

rho=np.nan

summary=[]; pred_frames=[]
for horizon,y in [('3-year',y20),('5-year',y22)]:
    idx=y.notna(); X=raw.loc[idx,FEATURES]; yy=y.loc[idx].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=.25,random_state=2027,stratify=yy)
    Xfit,Xtmp,yfit,ytmp=train_test_split(Xtr,ytr,test_size=.30,random_state=3031,stratify=ytr)
    Xcal,Xconf,ycal,yconf=train_test_split(Xtmp,ytmp,test_size=.50,random_state=3032,stratify=ytmp)
    base=rf(); base.fit(Xfit,yfit)
    pc0=base.predict_proba(Xcal)[:,1]
    platt=LogisticRegression().fit(pc0.reshape(-1,1),ycal)
    def cp(Z):
        z=base.predict_proba(Z)[:,1]
        return platt.predict_proba(z.reshape(-1,1))[:,1]
    pconf=cp(Xconf); ptest=cp(Xte)
    # calibration intercept/slope on held-out test using logit calibrated probability
    eps=1e-8; lp=np.log(np.clip(ptest,eps,1-eps)/(1-np.clip(ptest,eps,1-eps)))
    calmod=LogisticRegression(C=1e8,solver='lbfgs').fit(lp.reshape(-1,1),yte)
    cal_intercept=float(calmod.intercept_[0]); cal_slope=float(calmod.coef_[0,0])
    # calibration-in-the-large with slope fixed 1, solve intercept by Newton
    a=0.0
    for _ in range(100):
        pr=1/(1+np.exp(-(a+lp))); score=np.sum(yte.values-pr); info=np.sum(pr*(1-pr)); step=score/info; a+=step
        if abs(step)<1e-10: break
    # pooled conformal
    scores=1-np.where(yconf.values==1,pconf,1-pconf); q=cq(scores); cut=1-q
    inc0=(1-ptest)>=cut; inc1=ptest>=cut
    cov=np.where(yte.values==1,inc1,inc0); size=inc0.astype(int)+inc1.astype(int)
    # class conditional/Mondrian
    s0=1-(1-pconf[yconf.values==0]); s1=1-pconf[yconf.values==1]
    q0=cq(s0); q1=cq(s1); cut0=1-q0; cut1=1-q1
    m0=(1-ptest)>=cut0; m1=ptest>=cut1
    mcov=np.where(yte.values==1,m1,m0); msize=m0.astype(int)+m1.astype(int)
    for method,cc,ss in [('Pooled',cov,size),('Class-conditional',mcov,msize)]:
        summary.append([horizon,method,len(yte),int(yte.sum()),cc.mean(),cc[yte.values==0].mean(),cc[yte.values==1].mean(),(ss==1).mean(),(ss==2).mean(),int((yconf==0).sum()),int((yconf==1).sum()),cal_intercept,cal_slope,a,brier_score_loss(yte,ptest)])
    pred_frames.append(pd.DataFrame({'horizon':horizon,'y':yte.values,'calibrated_probability':ptest,'pooled_covered':cov,'pooled_set_size':size,'class_conditional_covered':mcov,'class_conditional_set_size':msize},index=yte.index))

cols=['horizon','method','test_n','test_events','overall_coverage','continuation_coverage','dissolution_coverage','singleton_rate','ambiguous_rate','conformal_cal_n_continue','conformal_cal_n_dissolve','calibration_intercept','calibration_slope','calibration_intercept_slope1','calibrated_brier']
pd.DataFrame(summary,columns=cols).to_csv(OUT/'03_calibration_conformal_diagnostics.csv',index=False)
pd.concat(pred_frames).to_csv(OUT/'04_test_predictions_extended.csv')

pd.DataFrame([{'original_cross_horizon_spearman':rho,'mean_repeated_cross_horizon_spearman':np.mean([x[1] for x in cross]),'mean_repeated_top5_jaccard':np.mean([x[2] for x in cross]),'mean_repeated_top10_jaccard':np.mean([x[3] for x in cross])}]).to_csv(OUT/'05_shap_stability_summary.csv',index=False)
print(pd.read_csv(OUT/'01_within_horizon_shap_rank_stability.csv').to_string(index=False))
print(pd.read_csv(OUT/'05_shap_stability_summary.csv').to_string(index=False))
print(pd.read_csv(OUT/'03_calibration_conformal_diagnostics.csv').to_string(index=False))
