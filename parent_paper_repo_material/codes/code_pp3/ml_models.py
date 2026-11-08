"""Machine learning part of parent paper 3 (Munir et al., 2026), Tables 16-20 and Figs. 2-3.

    dml_effects           Table 16  partially linear DML for each driver in turn: Y = ln CO2 per capita,
                                    D = that driver standardized (so theta is per 1 SD), X = the other six drivers.
                                    Nuisance: gradient boosting (learning rate 0.1, max depth 3, 100 trees), 5 folds
                                    grouped by country ("stratified by country to preserve temporal structure").
    fold stability        Table 17  theta re-estimated inside each fold from the cross-fitted residuals; mean and SD
    counterfactuals       Table 18  Delta ln CO2 for +1 SD: the DML theta (what the paper's Table 18 reports; its
                                    numbers equal Table 16) and the model-based f(X + 1 sd_j) - f(X) from section 4.4.3;
                                    percentage change exp(Delta) - 1 for +1 SD and for -1 SD
    predictive_models     Tables 19-20  random forest and gradient boosting on all seven drivers, 80/20 random split,
                                    test R2 and RMSE, mean |SHAP| (test set), permutation importance (test set, R2 drop)
"""
import doubleml as dml
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

SEED = 42


def gb(seed=SEED):
    return GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=seed)


def dml_effects(df, y, drivers, n_folds=5):
    groups = df.country.to_numpy()
    folds = list(GroupKFold(n_splits=n_folds).split(df, groups=groups))
    effects, stability = [], []
    for d in drivers:
        data = df[[y] + drivers].copy()
        data[d] = (data[d] - data[d].mean()) / data[d].std()
        X = [c for c in drivers if c != d]
        obj = dml.DoubleMLData(data.reset_index(drop=True), y_col=y, d_cols=d, x_cols=X)
        m = dml.DoubleMLPLR(obj, ml_l=gb(), ml_m=gb(), n_folds=n_folds, draw_sample_splitting=False)
        m.set_sample_splitting([[(tr, te) for tr, te in folds]])
        m.fit(store_predictions=True)
        ry = data[y].to_numpy() - m.predictions["ml_l"][:, 0, 0]
        rd = data[d].to_numpy() - m.predictions["ml_m"][:, 0, 0]
        per_fold = [float(rd[te] @ ry[te] / (rd[te] @ rd[te])) for _, te in folds]
        theta, se = float(m.coef[0]), float(m.se[0])
        effects.append(dict(variable=d, estimate=theta, se=se, t=theta / se, p=2 * stats.norm.sf(abs(theta / se)),
                            ci_low=theta - 1.96 * se, ci_high=theta + 1.96 * se,
                            sd_of_driver=df[d].std()))
        stability.append(dict(variable=d, mean=np.mean(per_fold), std=np.std(per_fold, ddof=1),
                              **{f"fold_{k + 1}": v for k, v in enumerate(per_fold)}))
    return pd.DataFrame(effects), pd.DataFrame(stability)


def counterfactuals(df, y, drivers, effects):
    model = gb().fit(df[drivers], df[y])
    base = model.predict(df[drivers]).mean()
    rows = []
    for _, e in effects.iterrows():
        d = e.variable
        bumped = df[drivers].copy()
        bumped[d] = bumped[d] + df[d].std()
        ml_delta = model.predict(bumped).mean() - base
        rows.append(dict(variable=d, delta_ln_co2e_dml=e.estimate,
                         pct_change_plus_1sd=100 * (np.exp(e.estimate) - 1),
                         pct_change_minus_1sd=100 * (np.exp(-e.estimate) - 1),
                         delta_ln_co2e_model=ml_delta))
    return pd.DataFrame(rows)


def predictive_models(df, y, drivers, test_size=0.2):
    X, Y = df[drivers], df[y]
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=test_size, random_state=SEED)
    out = {}
    for name, model in [("random_forest", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),
                        ("gradient_boosting", gb())]:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        sv = shap.TreeExplainer(model).shap_values(Xte)
        perm = permutation_importance(model, Xte, yte, n_repeats=10, random_state=SEED, n_jobs=-1)
        imp = pd.DataFrame(dict(mean_abs_shap=np.abs(sv).mean(0), perm_mean=perm.importances_mean,
                                perm_std=perm.importances_std), index=drivers)
        out[name] = dict(model=model, r2_test=r2_score(yte, pred), rmse_test=np.sqrt(mean_squared_error(yte, pred)),
                         r2_train=r2_score(ytr, model.predict(Xtr)), importances=imp, shap_values=sv, X_test=Xte)
    return out
