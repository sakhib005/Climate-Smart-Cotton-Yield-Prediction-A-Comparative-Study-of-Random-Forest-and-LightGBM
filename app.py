import os, warnings, joblib, numpy as np, pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR,"templates"), static_folder=os.path.join(BASE_DIR,"static"))

DATA_PATH = os.path.join(BASE_DIR,"data","dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR,"models","model.pkl")
META_PATH = os.path.join(BASE_DIR,"models","meta.json")

DEFAULT_TARGET = "Yield"
EXCLUDE_FEATURES = {"Harvest_Date","Planting_Date","Harvest_Year"}
MAX_SELECT_OPTIONS = 200

def nse(y_true,y_pred):
    num=np.sum((y_true-y_pred)**2); den=np.sum((y_true-np.mean(y_true))**2); return 1-num/den if den!=0 else np.nan
def detect_target(df):
    if DEFAULT_TARGET in df.columns: return DEFAULT_TARGET
    nums=df.select_dtypes(include=[np.number]).columns.tolist(); return nums[-1] if nums else df.columns[-1]

def build_schema_with_validation(df,target_col):
    feats=[c for c in df.columns if c!=target_col and c not in EXCLUDE_FEATURES]
    nums=df[feats].select_dtypes(include=[np.number]).columns.tolist(); cats=[c for c in feats if c not in nums]
    schema=[]
    for c in nums:
        series=pd.to_numeric(df[c],errors="coerce")
        if series.notna().any():
            p1=float(np.nanpercentile(series,1)); p99=float(np.nanpercentile(series,99)); median=float(np.nanmedian(series))
        else:
            p1,p99,median=0.0,0.0,0.0
        min_allowed=0.0 if p1>=0 else p1; buffer=max(abs(p99-p1)*0.1,1e-6); max_allowed=p99+buffer
        schema.append({"name":c,"type":"number","default":median,"min":min_allowed,"max":max_allowed,"step":"any"})
    for c in cats:
        vals=df[c].astype(str).fillna("").replace("nan","").tolist()
        opts=sorted([v for v in pd.Series(vals).unique().tolist() if v!=""]); default=opts[0] if opts else ""
        if 0<len(opts)<=MAX_SELECT_OPTIONS: schema.append({"name":c,"type":"select","options":opts,"default":default})
        else: schema.append({"name":c,"type":"text","default":default})
    return schema,feats,nums,cats

def train_or_load():
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        pipe=joblib.load(MODEL_PATH); meta=pd.read_json(META_PATH,typ="series").to_dict(); return pipe,meta
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df=pd.read_csv(DATA_PATH)
    if df.shape[1]<2: raise ValueError("Dataset must have at least 2 columns (features + target).")
    tgt=detect_target(df); X=df.drop(columns=[tgt]); X=X[[c for c in X.columns if c not in EXCLUDE_FEATURES]]; y=df[tgt]
    num=X.select_dtypes(include=[np.number]).columns.tolist(); cat=[c for c in X.columns if c not in num]
    pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),
                           ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    model=LGBMRegressor(n_estimators=1500,num_leaves=64,subsample=0.9,colsample_bytree=0.9,learning_rate=0.05,random_state=42)
    pipe=Pipeline([("preprocess",pre),("model",model)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42); pipe.fit(Xtr,ytr); yhat=pipe.predict(Xte)
    meta={"target":tgt,"features":X.columns.tolist(),"r2":float(r2_score(yte,yhat)),"rmse":float(np.sqrt(mean_squared_error(yte,yhat))),"nse":float(nse(yte,yhat)),"excluded":list(EXCLUDE_FEATURES)}
    os.makedirs(os.path.dirname(MODEL_PATH),exist_ok=True); joblib.dump(pipe,MODEL_PATH); pd.Series(meta).to_json(META_PATH)
    return pipe,meta

PIPELINE,META=train_or_load()

def get_schema(): df=pd.read_csv(DATA_PATH); schema,feats,_,_=build_schema_with_validation(df,META["target"]); return schema,META["target"],feats

@app.route("/")
def home():
    metrics=type("M",(),dict(r2=META.get("r2"),rmse=META.get("rmse"))) if META else None
    return render_template("home.html",metrics=metrics)

@app.route("/predictor",methods=["GET","POST"])
def predictor():
    prediction=None; error=None; schema,target,features=get_schema()
    if request.method=="POST":
        try:
            row={}
            for field in schema:
                name=field["name"]; raw=request.form.get(name)
                if raw is None or raw=="": raise ValueError(f"Missing value for '{name}'")
                if field["type"]=="number":
                    val=float(raw); minv=field["min"]; maxv=field["max"]
                    if val<minv or val>maxv: raise ValueError(f"Value for '{name}' must be between {minv} and {maxv}")
                    row[name]=val
                elif field["type"]=="select":
                    row[name]=raw
                else:
                    row[name]=raw
            X_infer=pd.DataFrame([row],columns=features); prediction=float(PIPELINE.predict(X_infer)[0])
        except Exception as e:
            error=str(e)
    metrics=type("M",(),dict(r2=META.get("r2"),rmse=META.get("rmse"))) if META else None
    return render_template("predictor.html",schema=schema,target=target,features=features,prediction=prediction,error=error,metrics=metrics)

@app.route("/about")
def about():
    return render_template("about.html",meta=META)

@app.route("/contact")
def contact():
    return render_template("contact.html",phone="+91-90000-00000",email="support@yourdomain.com")

if __name__=="__main__":
    app.run(debug=True)
