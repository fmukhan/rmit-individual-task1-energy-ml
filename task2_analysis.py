"""Individual Task 2, Part 2: re-evaluation of the Task 1 energy models.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from fairlearn.metrics import MetricFrame

def add_time_features(df, col):
    t = df[col]
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*t.dt.hour/24), np.cos(2*np.pi*t.dt.hour/24)
    df["dow_sin"], df["dow_cos"] = np.sin(2*np.pi*t.dt.dayofweek/7), np.cos(2*np.pi*t.dt.dayofweek/7)

def load_appliances():
    df = pd.read_csv("energydata_complete.csv"); df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True); add_time_features(df, "date")
    df["month"] = df["date"].dt.month
    return df["date"], df.drop(columns=["date", "Appliances", "rv1", "rv2"]), df["Appliances"].astype(float)

def load_tetouan(dayfirst_bug=False):
    df = pd.read_csv("Tetuan City power consumption.csv")
    if dayfirst_bug:   
        df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["DateTime"])
    else:              
        df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("DateTime").reset_index(drop=True); add_time_features(df, "DateTime")
    df["month_sin"] = np.sin(2*np.pi*df["DateTime"].dt.month/12)
    df["month_cos"] = np.cos(2*np.pi*df["DateTime"].dt.month/12)
    target = "Zone 1 Power Consumption"
    drop = ["DateTime", target, "Zone 2  Power Consumption", "Zone 3  Power Consumption"]
    return df["DateTime"], df.drop(columns=drop), df[target].astype(float)

def make_model(name, seed=42):
    if name == "Linear SVR":
        est = LinearSVR(C=1.0, epsilon=0.1, loss="squared_epsilon_insensitive", random_state=seed, max_iter=20000)
    else:
        est = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.0005, max_iter=250, early_stopping=True,
                           validation_fraction=0.10, n_iter_no_change=20, random_state=seed)
    return TransformedTargetRegressor(Pipeline([("scale", StandardScaler()), ("m", est)]), transformer=StandardScaler())

def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5, R2=r2_score(y, p))
show = lambda d: {k: round(v, 3) for k, v in d.items()}

for name, loader in [("Appliances", load_appliances), ("Tetouan Z1", load_tetouan)]:
    t, X, y = loader(); n = len(X); sp = int(0.8 * n)              # chronological 80/20 split
    Xtr, Xte, ytr, yte = X.iloc[:sp], X.iloc[sp:], y.iloc[:sp], y.iloc[sp:]
    print(f"\n===== {name}: {n} rows, test window {t.iloc[sp]} to {t.iloc[-1]}")

    # baselines: training mean, and 'same time yesterday' (144 ten-minute steps)
    print("baseline training mean     ", show(metrics(yte, np.full(len(yte), ytr.mean()))))
    ok = y.shift(144).iloc[sp:].notna()
    print("baseline same time yesterday", show(metrics(yte[ok], y.shift(144).iloc[sp:][ok])))

    for model in ["Linear SVR", "MLP"]:
        print(f"--- {model}")
        print("hold-out (seed 42)   ", show(metrics(yte, make_model(model).fit(Xtr, ytr).predict(Xte))))
        if model == "MLP":                                             # seed sensitivity
            r = pd.DataFrame([metrics(yte, make_model(model, s).fit(Xtr, ytr).predict(Xte)) for s in range(5)])
            print("hold-out, 5 seeds     mean", show(r.mean().to_dict()), "sd", show(r.std().to_dict()))
        for cv_name, cv in [("time-series CV(5)", TimeSeriesSplit(5)), ("shuffled CV(5)   ", KFold(5, shuffle=True, random_state=42))]:
            r = pd.DataFrame([metrics(y.iloc[b], make_model(model).fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b]))
                              for a, b in cv.split(X)])
            print(cv_name, "mean", show(r.mean().to_dict()), "sd", show(r.std().to_dict()))
        print("learning curve (earliest n rows -> fixed test set):  n, train RMSE, test RMSE, test R2")
        for f in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]:
            k = int(sp * f); m = make_model(model).fit(Xtr.iloc[:k], ytr.iloc[:k])
            print(f"  {f:4.2f} {k:6d} {metrics(ytr.iloc[:k], m.predict(Xtr.iloc[:k]))['RMSE']:9.1f}"
                  f" {metrics(yte, m.predict(Xte))['RMSE']:9.1f} {metrics(yte, m.predict(Xte))['R2']:8.3f}")

    # Fairlearn: disaggregated error by operating context (no protected attributes in these data)
    best = "Linear SVR" if name == "Appliances" else "MLP"
    pred = make_model(best).fit(Xtr, ytr).predict(Xte); tt = t.iloc[sp:]
    groups = {"time of day": pd.cut(tt.dt.hour, [-1, 5, 11, 17, 23], labels=["night", "morning", "afternoon", "evening"]),
              "weekend": np.where(tt.dt.dayofweek >= 5, "weekend", "weekday"),
              "month": tt.dt.month.astype(str), "demand quartile": pd.qcut(yte, 4, labels=["Q1", "Q2", "Q3", "Q4"])}
    for g, sf in groups.items():
        mf = MetricFrame(metrics={"MAE": mean_absolute_error, "RMSE": lambda a, b: mean_squared_error(a, b) ** 0.5,
                                  "mean_error": lambda a, b: float(np.mean(b - a)), "n": lambda a, b: len(a)},
                         y_true=yte.values, y_pred=pred, sensitive_features=np.asarray(sf))
        print(f"\n[{name} / {best}] Fairlearn by {g}\n", mf.by_group.round(1).to_string())

# Replica of the Task 1 notebook's faulty date parsing (reproduces the submitted Tetouan numbers)
t, X, y = load_tetouan(dayfirst_bug=True); sp = int(0.8 * len(X))
print(f"\n===== Task 1 replica (faulty parsing): {len(X)} of 52416 rows kept")
for model in ["Linear SVR", "MLP"]:
    print(model, show(metrics(y.iloc[sp:], make_model(model).fit(X.iloc[:sp], y.iloc[:sp]).predict(X.iloc[sp:]))))
