import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss
from common import load_data, outcomes, FEATURES, DISPLAY, preprocessor, best_balanced_threshold, RESULTS

raw=load_data(); _,y20,y22=outcomes(raw)
out=RESULTS/"04_robustness"; out.mkdir(exist_ok=True)
rows=[]; imps=[]

for horizon,y in [("3-year",y20),("5-year",y22)]:
    idx=y.notna(); X=raw.loc[idx,FEATURES]; yy=y.loc[idx].astype(int)
    for seed in range(101,111):
        Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=.25,random_state=seed,stratify=yy)
        m=Pipeline([("pre",preprocessor()),("rf",RandomForestClassifier(
            n_estimators=180,max_depth=8,min_samples_leaf=4,max_features="sqrt",
            class_weight="balanced_subsample",random_state=seed,n_jobs=-1))])
        m.fit(Xtr,ytr)
        ptr=m.predict_proba(Xtr)[:,1]; p=m.predict_proba(Xte)[:,1]
        t=best_balanced_threshold(ytr.values,ptr)
        rows.append([horizon,seed,roc_auc_score(yte,p),average_precision_score(yte,p),
                     balanced_accuracy_score(yte,(p>=t).astype(int)),brier_score_loss(yte,p)])

        names=list(m.named_steps["pre"].get_feature_names_out())
        vals=m.named_steps["rf"].feature_importances_
        agg={v:0.0 for v in FEATURES}
        for nm,val in zip(names,vals):
            bare=nm.split("__",1)[-1]
            for v in FEATURES:
                if bare==v or bare.startswith(v+"_"):
                    agg[v]+=float(val); break
        order=sorted(agg,key=agg.get,reverse=True)
        for rank,v in enumerate(order,1):
            imps.append([horizon,seed,v,DISPLAY[v],rank,agg[v]])

pd.DataFrame(rows,columns=["horizon","seed","roc_auc","pr_auc","balanced_accuracy","brier"]).to_csv(
    out/"repeated_split_performance.csv",index=False)
pd.DataFrame(imps,columns=["horizon","seed","variable","feature","rank","importance"]).to_csv(
    out/"full_refit_importance_ranks.csv",index=False)
