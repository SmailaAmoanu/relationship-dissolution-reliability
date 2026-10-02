import json, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from common import load_data, outcomes, FEATURES, preprocessor, RESULTS

raw=load_data(); _,y20,y22=outcomes(raw)
out=RESULTS/"05_nested_tuning"; out.mkdir(exist_ok=True)

grid={
"rf__n_estimators":[200,400],
"rf__max_depth":[4,8,None],
"rf__min_samples_leaf":[2,5,10],
"rf__max_features":["sqrt",.5]
}
rows=[]; params=[]
for horizon,y in [("3-year",y20),("5-year",y22)]:
    idx=y.notna(); X=raw.loc[idx,FEATURES]; yy=y.loc[idx].astype(int)
    for seed in [101,103,105,107,109]:
        Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=.25,random_state=seed,stratify=yy)
        cv=StratifiedKFold(3,shuffle=True,random_state=seed+1000)
        model=Pipeline([("pre",preprocessor()),("rf",RandomForestClassifier(
            class_weight="balanced_subsample",random_state=seed,n_jobs=-1))])
        gs=GridSearchCV(model,grid,scoring="average_precision",cv=cv,n_jobs=-1,refit=True)
        gs.fit(Xtr,ytr)
        p=gs.predict_proba(Xte)[:,1]
        rows.append([horizon,seed,roc_auc_score(yte,p),average_precision_score(yte,p),
                     brier_score_loss(yte,p),gs.best_score_,json.dumps(gs.best_params_)])
        print(horizon,seed,gs.best_params_)

pd.DataFrame(rows,columns=["horizon","outer_seed","roc_auc","pr_auc","brier",
                           "inner_cv_pr_auc","best_params"]).to_csv(
    out/"nested_tuning_outer_performance.csv",index=False)
