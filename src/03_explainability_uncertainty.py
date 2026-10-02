import math, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import shap
from common import load_data, outcomes, FEATURES, DISPLAY, preprocessor, RESULTS

raw=load_data(); _,y20,y22=outcomes(raw)
out=RESULTS/"03_explainability_uncertainty"; out.mkdir(exist_ok=True)

def conformal_q(scores,alpha=.10):
    n=len(scores); level=min(1,math.ceil((n+1)*(1-alpha))/n)
    return float(np.quantile(scores,level,method="higher"))

rows=[]
for horizon,y in [("3-year",y20),("5-year",y22)]:
    idx=y.notna(); X=raw.loc[idx,FEATURES]; yy=y.loc[idx].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=.25,random_state=2027,stratify=yy)

    model=Pipeline([("pre",preprocessor()),("rf",RandomForestClassifier(
        n_estimators=300,max_depth=8,min_samples_leaf=4,max_features="sqrt",
        class_weight="balanced_subsample",random_state=2027,n_jobs=-1))])
    model.fit(Xtr,ytr); rawp=model.predict_proba(Xte)[:,1]

    Xt=model.named_steps["pre"].transform(Xte)
    if hasattr(Xt,"toarray"): Xt=Xt.toarray()
    names=list(model.named_steps["pre"].get_feature_names_out())
    n=min(150,len(Xte)); ids=np.random.default_rng(2027).choice(len(Xte),n,replace=False)
    sv=shap.TreeExplainer(model.named_steps["rf"]).shap_values(Xt[ids])
    if isinstance(sv,list): sv=sv[1]
    elif getattr(sv,"ndim",0)==3: sv=sv[:,:,1]
    meanabs=np.abs(sv).mean(axis=0)
    agg={v:0.0 for v in FEATURES}
    for nm,val in zip(names,meanabs):
        bare=nm.split("__",1)[-1]
        for v in FEATURES:
            if bare==v or bare.startswith(v+"_"):
                agg[v]+=float(val); break
    pd.DataFrame({"variable":agg.keys(),"feature":[DISPLAY[v] for v in agg],
                  "mean_abs_shap":agg.values()}).sort_values("mean_abs_shap",ascending=False).to_csv(
        out/f"shap_{horizon}.csv",index=False)

    # independent fit / calibration / conformal subsets from training
    Xfit,Xtmp,yfit,ytmp=train_test_split(Xtr,ytr,test_size=.30,random_state=3031,stratify=ytr)
    Xcal,Xconf,ycal,yconf=train_test_split(Xtmp,ytmp,test_size=.50,random_state=3032,stratify=ytmp)
    base=Pipeline([("pre",preprocessor()),("rf",RandomForestClassifier(
        n_estimators=300,max_depth=8,min_samples_leaf=4,max_features="sqrt",
        class_weight="balanced_subsample",random_state=2027,n_jobs=-1))])
    base.fit(Xfit,yfit)
    pc0=base.predict_proba(Xcal)[:,1]
    platt=LogisticRegression().fit(pc0.reshape(-1,1),ycal)
    def cp(Z):
        z=base.predict_proba(Z)[:,1]
        return platt.predict_proba(z.reshape(-1,1))[:,1]
    pconf=cp(Xconf); ptest=cp(Xte)
    q=conformal_q(1-np.where(yconf.values==1,pconf,1-pconf))
    cutoff=1-q
    inc0=(1-ptest)>=cutoff; inc1=ptest>=cutoff
    covered=np.where(yte.values==1,inc1,inc0)
    size=inc0.astype(int)+inc1.astype(int)

    rows.append([horizon,roc_auc_score(yte,rawp),average_precision_score(yte,rawp),
                 brier_score_loss(yte,rawp),brier_score_loss(yte,ptest),
                 covered.mean(),(size==1).mean(),(size==2).mean(),cutoff])

pd.DataFrame(rows,columns=["horizon","roc_auc","pr_auc","brier_raw","brier_calibrated",
                           "conformal_coverage","singleton_rate","ambiguous_rate","probability_cutoff"]).to_csv(
    out/"uncertainty_summary.csv",index=False)
