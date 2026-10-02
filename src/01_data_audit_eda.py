import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import load_data, outcomes, FEATURES, DISPLAY, RESULTS

raw=load_data()
baseline,y20,y22=outcomes(raw)
out=RESULTS/"01_data_audit_eda"; out.mkdir(exist_ok=True)

cohort=pd.DataFrame([
["All 2017 respondents",len(raw)],
["Partnered in 2017",int(baseline.sum())],
["Surveyed in 2020",int((baseline&(raw.w2_surveyed==1)).sum())],
["Usable 3-year outcomes",int(y20.notna().sum())],
["3-year dissolutions",int((y20==1).sum())],
["Surveyed in 2022",int((baseline&(raw.w3_surveyed==1)).sum())],
["Usable cumulative 5-year outcomes",int(y22.notna().sum())],
["Cumulative 5-year dissolutions",int((y22==1).sum())]
],columns=["stage","N"])
cohort.to_csv(out/"cohort_flow.csv",index=False)

audit=[]
for v in FEATURES:
    s=raw.loc[baseline,v]
    audit.append([v,DISPLAY[v],int(s.isna().sum()),100*s.isna().mean(),int(s.nunique(dropna=True))])
pd.DataFrame(audit,columns=["variable","feature","missing_n","missing_pct","unique_nonmissing"]).to_csv(
    out/"predictor_audit.csv",index=False)

plt.figure(figsize=(9,5))
plt.barh(cohort.stage,cohort.N)
plt.gca().invert_yaxis()
plt.xlabel("Respondents")
plt.title("HCMST forecasting cohort flow")
plt.tight_layout()
plt.savefig(out/"cohort_flow.png",dpi=180)
plt.close()

print(cohort.to_string(index=False))
