from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "raw" / "HCMST 2017 to 2022 small public version 2.2.dta"
RESULTS = PROJECT / "reproduced_results"
RESULTS.mkdir(exist_ok=True)

EXPECTED_SHA256 = "a76c1ef8e14a69951db3d71142dfb3866063c7af8d0da67942b744e2727aeb56"

SPECS = [
("w1_ppage","Age","num"),("w1_ppgender","Gender","cat"),("w1_ppethm","Race/ethnicity","cat"),
("w1_ppeducat","Education","cat"),("w1_ppincimp","Household income","cat"),("w1_ppwork","Employment","cat"),
("w1_pphhsize","Household size","num"),("w1_PPREG4","Region","cat"),("w1_ppp20072","Religious-service attendance","cat"),
("w1_q9","Partner age","num"),("w1_partner_yrsed","Partner education","num"),
("w1_partnership_status_cohab","Relationship arrangement","cat"),
("w1_relate_duration_in2017_years","Relationship duration","num"),("w1_q34_reduced","Relationship quality","cat"),
("w1_sex_frequency","Sex frequency","cat"),("w1_q23","Relative earnings","cat"),("w1_q19","Lives with partner","cat"),
("w1_q17","Prior marriages","cat"),("w1_either_internet_adjusted","Met online","cat"),
("w1_q24_met_through_friend","Met through friend","cat"),("w1_q24_met_through_family","Met through family","cat"),
("w1_q24_met_as_through_cowork","Met through coworker","cat"),("w1_q24_church","Met through church","cat"),
("w1_interracial_5cat","Interracial couple","cat"),("w1_same_sex_couple","Same-sex couple","cat")
]
FEATURES=[x[0] for x in SPECS]
DISPLAY={x[0]:x[1] for x in SPECS}
NUMS=[x[0] for x in SPECS if x[2]=="num"]
CATS=[x[0] for x in SPECS if x[2]=="cat"]

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing data file: {DATA}")
    return pd.read_stata(DATA, convert_categoricals=False)

def outcomes(raw):
    baseline=raw["w1_partnership_status"].isin([1,2])

    y20=pd.Series(np.nan,index=raw.index,name="dissolved_by_2020")
    y20[baseline & (raw.w2_surveyed==1) & (raw.w2_relationship_end==0)] = 0
    y20[baseline & (raw.w2_surveyed==1) & raw.w2_relationship_end.isin([1,2])] = 1

    y22=pd.Series(np.nan,index=raw.index,name="dissolved_by_2022")
    y22[baseline & raw.w2_relationship_end.isin([1,2])] = 1
    risk=baseline & ~raw.w2_relationship_end.isin([1,2,3])
    y22[risk & (raw.w3_surveyed==1) & (raw.w3_partner_source==1)] = 0
    y22[risk & (raw.w3_surveyed==1) & (raw.w3_breakup_source==1)
        & raw.w3_relationship_end_combo.isin([1,2])] = 1
    return baseline,y20,y22

def preprocessor(features=None):
    features = FEATURES if features is None else features
    nums=[v for v in NUMS if v in features]
    cats=[v for v in CATS if v in features]
    return ColumnTransformer([
        ("num",Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler())
        ]),nums),
        ("cat",Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ]),cats)
    ])

def best_balanced_threshold(y,p):
    from sklearn.metrics import balanced_accuracy_score
    ts=np.linspace(.05,.95,91)
    vals=[balanced_accuracy_score(y,(p>=t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(vals))])
