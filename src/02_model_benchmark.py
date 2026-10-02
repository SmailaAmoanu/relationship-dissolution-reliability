import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from common import load_data, outcomes, FEATURES, preprocessor, RESULTS

raw=load_data(); _,y20,y22=outcomes(raw)
out=RESULTS/"02_model_benchmark"; out.mkdir(exist_ok=True)

models={
"Logistic":(LogisticRegression(max_iter=4000,class_weight="balanced",solver="liblinear"),
            {"model__C":[.1,1,10]}),
"Elastic Net":(LogisticRegression(max_iter=5000,class_weight="balanced",solver="saga",penalty="elasticnet"),
               {"model__C":[.1,.5,1,3],"model__l1_ratio":[.1,.5,.9]}),
"Random Forest":(RandomForestClassifier(n_estimators=300,class_weight="balanced_subsample",random_state=2027,n_jobs=-1),
                 {"model__max_depth":[4,8,None],"model__min_samples_leaf":[2,5,10],"model__max_features":["sqrt",.5]}),
"XGBoost":(XGBClassifier(n_estimators=300,objective="binary:logistic",eval_metric="logloss",random_state=2027,n_jobs=-1),
           {"model__max_depth":[2,3,4],"model__learning_rate":[.03,.07],"model__min_child_weight":[1,5]})
}

rows=[]
for horizon,y in [("3-year",y20),("5-year",y22)]:
    idx=y.notna(); X=raw.loc[idx,FEATURES]; yy=y.loc[idx].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=.25,random_state=2027,stratify=yy)
    cv=StratifiedKFold(5,shuffle=True,random_state=2027)
    for name,(est,grid) in models.items():
        if name=="XGBoost":
            est.set_params(scale_pos_weight=(ytr==0).sum()/(ytr==1).sum())
        gs=GridSearchCV(Pipeline([("pre",preprocessor()),("model",est)]),
                        grid,scoring="average_precision",cv=cv,n_jobs=-1,refit=True)
        gs.fit(Xtr,ytr)
        p=gs.predict_proba(Xte)[:,1]
        rows.append([horizon,name,roc_auc_score(yte,p),average_precision_score(yte,p),
                     brier_score_loss(yte,p),json.dumps(gs.best_params_)])
        print(horizon,name,rows[-1][2:5])
pd.DataFrame(rows,columns=["horizon","model","roc_auc","pr_auc","brier","best_params"]).to_csv(
    out/"model_benchmark.csv",index=False)
