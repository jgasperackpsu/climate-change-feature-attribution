"""Econometric tests and models for parent paper 3 (Munir et al., 2026), Tables 2-15.

The paper ran these in EViews 13. EViews' exact settings are not reported, so each function says which
standard estimator it implements. All functions take a balanced long panel `df` with columns
country, year and the variables, sorted by country then year.

    descriptives          Table 2   mean, median, max, min, SD, skewness, kurtosis (EViews: biased, non-excess)
    correlations          Table 3   Pearson r with p-values
    panel_unit_root       Table 4   Fisher-ADF (Maddala-Wu chi2) and Choi inverse-normal Z; per-country lag by AIC
    var_lag_selection     Table 5   pooled VAR on stacked panel (lags built within country), LR/FPE/AIC/SC/HQ
    kao_test              Table 6   Kao (1999) residual ADF test, residual variance and HAC (long-run) variance
    pmg_ardl              Table 7   pooled mean group ARDL(p, q) (Pesaran, Shin & Smith, 1999), back-substitution ML
    bounds_wald           Table 8   Wald test that all long-run coefficients are zero
    dumitrescu_hurlin     Table 9   W-bar, Z-bar tilde (small-T), two-sided p
    panel_var_irf_fevd    Tables 10-11  pooled VAR(p), Cholesky IRF of y and FEVD of y
    hausman               Table 12  FE vs RE
    fixed_effects         Table 13  within estimator with intercept, LSDV R2, Durbin-Watson
    vif                   Table 14  centered VIF
    wooldridge_serial     Table 15  Wooldridge (2002) test, clustered by country
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller


def _by_country(df):
    return [g for _, g in df.groupby("country", sort=False)]


def _lag(g, col, k):
    return g[col].shift(k)


# ---------------------------------------------------------------- Tables 2-4
def descriptives(df, cols):
    rows = {}
    for c in cols:
        x = df[c].dropna()
        rows[c] = dict(Mean=x.mean(), Median=x.median(), Maximum=x.max(), Minimum=x.min(), **{"Std. Dev.": x.std()},
                       Skewness=stats.skew(x, bias=True), Kurtosis=stats.kurtosis(x, fisher=False, bias=True))
    return pd.DataFrame(rows)


def correlations(df, cols):
    r = df[cols].corr()
    p = pd.DataFrame(np.ones_like(r), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[:i]:
            p.loc[a, b] = p.loc[b, a] = stats.pearsonr(df[a], df[b]).pvalue
    return r, p


def panel_unit_root(df, col, trend=False, diff=False, maxlag=2):
    """Fisher-type panel ADF: ADF per country (lag by AIC up to maxlag), combined p-values."""
    pv = []
    for g in _by_country(df):
        x = g[col].diff().dropna() if diff else g[col].dropna()
        if x.std() == 0 or len(x) < 8:
            continue
        pv.append(adfuller(x, maxlag=maxlag, autolag="AIC", regression="ct" if trend else "c")[1])
    pv = np.clip(np.array(pv), 1e-12, 1 - 1e-12)
    n = len(pv)
    fisher = -2 * np.log(pv).sum()
    choi = stats.norm.ppf(pv).sum() / np.sqrt(n)
    return dict(fisher_chi2=fisher, fisher_df=2 * n, fisher_p=stats.chi2.sf(fisher, 2 * n),
                choi_z=choi, choi_p=stats.norm.cdf(choi), n_countries=n)


# ---------------------------------------------------------------- Table 5
def _stack_lags(df, cols, p, maxlag):
    """Stacked Y (t) and [1, lags 1..p] using the common sample t > maxlag within each country."""
    Ys, Xs = [], []
    for g in _by_country(df):
        g = g.reset_index(drop=True)
        Y = g[cols].iloc[maxlag:].to_numpy()
        parts = [np.ones((len(Y), 1))]
        for k in range(1, p + 1):
            parts.append(g[cols].shift(k).iloc[maxlag:].to_numpy())
        Ys.append(Y)
        Xs.append(np.hstack(parts))
    return np.vstack(Ys), np.vstack(Xs)


def var_lag_selection(df, cols, maxlag=8):
    k = len(cols)
    rows, prev_logdet = [], None
    for p in range(maxlag + 1):
        Y, X = _stack_lags(df, cols, p, maxlag)
        T = len(Y)
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        E = Y - X @ B
        S = E.T @ E / T
        logdet = np.linalg.slogdet(S)[1]
        n = k * X.shape[1]
        logl = -T * k / 2 * (1 + np.log(2 * np.pi)) - T / 2 * logdet
        m = X.shape[1]
        lr = np.nan if prev_logdet is None else (T - m) * (prev_logdet - logdet)
        rows.append(dict(Lag=p, LogL=logl, LR=lr, LR_p=np.nan if p == 0 else stats.chi2.sf(lr, k * k),
                         FPE=np.exp(logdet) * ((T + m) / (T - m)) ** k,
                         AIC=-2 * logl / T + 2 * n / T, SC=-2 * logl / T + n * np.log(T) / T,
                         HQ=-2 * logl / T + 2 * n * np.log(np.log(T)) / T))
        prev_logdet = logdet
    out = pd.DataFrame(rows)
    # EViews' LR is sequential: the selected lag is the last one before the first insignificant test at 5%
    sig = out.LR_p < 0.05
    lr_pick = 0
    for p in range(1, maxlag + 1):
        if not sig[p]:
            break
        lr_pick = p
    out["selected_by"] = ""
    for crit, pick in [("LR", lr_pick), ("FPE", out.FPE.idxmin()), ("AIC", out.AIC.idxmin()),
                       ("SC", out.SC.idxmin()), ("HQ", out.HQ.idxmin())]:
        out.loc[pick, "selected_by"] = (out.loc[pick, "selected_by"] + " " + crit).strip()
    return out


# ---------------------------------------------------------------- Table 6
def _within(df, cols):
    return df[cols] - df.groupby("country")[cols].transform("mean")


def _bartlett_lrv(blocks, L):
    """Long-run covariance averaged over countries; blocks: list of (T_i x k) arrays."""
    k = blocks[0].shape[1]
    S, O, n = np.zeros((k, k)), np.zeros((k, k)), 0
    for w in blocks:
        T = len(w)
        G0 = w.T @ w
        Om = G0.copy()
        for l in range(1, L + 1):
            Gl = w[l:].T @ w[:-l]
            Om += (1 - l / (L + 1)) * (Gl + Gl.T)
        S += G0
        O += Om
        n += T
    return S / n, O / n


def kao_test(df, y, xs, max_adf_lag=2):
    """Kao (1999) ADF residual cointegration test. Returns ADF*, p (left tail), residual and HAC variance."""
    W = _within(df, [y] + xs)
    beta = np.linalg.lstsq(W[xs].to_numpy(), W[y].to_numpy(), rcond=None)[0]
    d = df[["country"]].copy()
    d["u"] = W[y].to_numpy() - W[xs].to_numpy() @ beta
    # ADF on pooled residuals, lag chosen by SIC
    best = None
    for p in range(max_adf_lag + 1):
        rows = []
        for _, g in d.groupby("country", sort=False):
            u = g.u.to_numpy()
            du = np.diff(u)
            for t in range(p + 1, len(u)):
                rows.append([u[t], u[t - 1]] + [du[t - 1 - j] for j in range(1, p + 1)])
        A = np.array(rows)
        Yv, Xv = A[:, 0], A[:, 1:]
        b = np.linalg.lstsq(Xv, Yv, rcond=None)[0]
        e = Yv - Xv @ b
        s2 = e @ e / (len(e) - Xv.shape[1])
        sic = np.log(e @ e / len(e)) + Xv.shape[1] * np.log(len(e)) / len(e)
        se = np.sqrt(s2 * np.linalg.inv(Xv.T @ Xv)[0, 0])
        if best is None or sic < best[0]:
            best = (sic, p, (b[0] - 1) / se)
    _, p_adf, t_adf = best
    # long-run variances of (u, dx)
    T = df.groupby("country").size().iloc[0]
    L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    blocks = []
    for (_, g), (_, gu) in zip(df.groupby("country", sort=False), d.groupby("country", sort=False)):
        dx = g[xs].diff().to_numpy()[1:]
        blocks.append(np.column_stack([gu.u.to_numpy()[1:], dx]))
    S, O = _bartlett_lrv(blocks, L)
    sv2 = S[0, 0] - S[0, 1:] @ np.linalg.solve(S[1:, 1:], S[1:, 0])
    s0v2 = O[0, 0] - O[0, 1:] @ np.linalg.solve(O[1:, 1:], O[1:, 0])
    N = df.country.nunique()
    sv, s0v = np.sqrt(sv2), np.sqrt(s0v2)
    adf_star = (t_adf + np.sqrt(6 * N) * sv / (2 * s0v)) / np.sqrt(s0v2 / (2 * sv2) + 3 * sv2 / (10 * s0v2))
    return dict(adf_t_stat=adf_star, p=stats.norm.cdf(adf_star), residual_variance=sv2, hac_variance=s0v2,
                adf_lag=p_adf, bandwidth=L, raw_t=t_adf)


# ---------------------------------------------------------------- Tables 7-8
def pmg_ardl(df, y, xs, p=2, max_iter=500, tol=1e-9):
    """Pooled mean group ARDL(p, 1): dy_t = phi_i (y_{t-1} - theta'x_t) + sum_{j<p} delta_ij dy_{t-j}
    + gamma_i' dx_t + mu_i + e. Long-run theta common (ML by back-substitution); short-run = mean group."""
    blocks = []
    for g in _by_country(df):
        g = g.reset_index(drop=True)
        dy = g[y].diff()
        parts = {"dy": dy, "ylag": g[y].shift(1)}
        for j in range(1, p):
            parts[f"D({y.upper()}(-{j}))"] = dy.shift(j)
        for x in xs:
            parts[f"x_{x}"] = g[x]
            parts[f"D({x.upper()})"] = g[x].diff()
        b = pd.DataFrame(parts).dropna()
        Wc = [c for c in b if c.startswith("D(")]
        blocks.append(dict(dy=b.dy.to_numpy(), ylag=b.ylag.to_numpy(), X=b[[f"x_{x}" for x in xs]].to_numpy(),
                           W=np.column_stack([b[Wc].to_numpy(), np.ones(len(b))]), wnames=Wc + ["C"]))
    # start: pooled fixed-effects ECM
    Z = np.vstack([np.column_stack([bl["ylag"], bl["X"], bl["W"][:, :-1]]) for bl in blocks])
    Yv = np.concatenate([bl["dy"] for bl in blocks])
    ids = np.concatenate([[i] * len(bl["dy"]) for i, bl in enumerate(blocks)])
    Zd = Z - pd.DataFrame(Z).groupby(ids).transform("mean").to_numpy()
    Yd = Yv - pd.Series(Yv).groupby(ids).transform("mean").to_numpy()
    c0 = np.linalg.lstsq(Zd, Yd, rcond=None)[0]
    theta = -c0[1:1 + len(xs)] / c0[0]
    for it in range(max_iter):
        A = np.zeros((len(xs), len(xs)))
        bvec = np.zeros(len(xs))
        for bl in blocks:
            xi = bl["ylag"] - bl["X"] @ theta
            R = np.column_stack([xi, bl["W"]])
            coef = np.linalg.lstsq(R, bl["dy"], rcond=None)[0]
            e = bl["dy"] - R @ coef
            bl["phi"], bl["short"], bl["s2"] = coef[0], coef[1:], e @ e / len(e)
            H = np.eye(len(e)) - bl["W"] @ np.linalg.pinv(bl["W"])
            A += bl["phi"] ** 2 * bl["X"].T @ H @ bl["X"] / bl["s2"]
            bvec += bl["phi"] * bl["X"].T @ H @ (bl["dy"] - bl["phi"] * bl["ylag"]) / bl["s2"]
        new = -np.linalg.solve(A, bvec)
        if np.max(np.abs(new - theta)) < tol:
            theta = new
            break
        theta = new
    V = np.linalg.inv(A)
    N = len(blocks)
    lr = pd.DataFrame(dict(coef=theta, se=np.sqrt(np.diag(V))), index=[x.upper() for x in xs])
    phis = np.array([bl["phi"] for bl in blocks])
    shorts = np.array([bl["short"] for bl in blocks])
    sr_names = ["COINTEQ01"] + blocks[0]["wnames"]
    sr_vals = np.column_stack([phis, shorts])
    sr = pd.DataFrame(dict(coef=sr_vals.mean(0), se=sr_vals.std(0, ddof=1) / np.sqrt(N)), index=sr_names)
    for t_ in (lr, sr):
        t_["t"] = t_.coef / t_.se
        t_["p"] = 2 * stats.norm.sf(np.abs(t_.t))
    n_obs = sum(len(bl["dy"]) for bl in blocks)
    n_par = len(xs) + N * (1 + blocks[0]["W"].shape[1])
    return dict(long_run=lr, short_run=sr, V=V, iterations=it + 1, n_obs=n_obs, n_params=n_par,
                phi_by_country=phis)


def bounds_wald(res):
    th = res["long_run"].coef.to_numpy()
    w = float(th @ np.linalg.solve(res["V"], th))
    k = len(th)
    df2 = res["n_obs"] - res["n_params"]
    return pd.DataFrame([dict(test="F-statistic", value=w / k, df=f"({k}, {df2})", p=stats.f.sf(w / k, k, df2)),
                         dict(test="Chi-square", value=w, df=str(k), p=stats.chi2.sf(w, k))])


# ---------------------------------------------------------------- Table 9
def dumitrescu_hurlin(df, x, y, K=2):
    """H0: x does not homogeneously Granger-cause y."""
    W = []
    T_used = None
    for g in _by_country(df):
        g = g.reset_index(drop=True)
        lags = {f"y{k}": g[y].shift(k) for k in range(1, K + 1)} | {f"x{k}": g[x].shift(k) for k in range(1, K + 1)}
        b = pd.DataFrame({"y": g[y], **lags}).dropna()
        T = len(b)
        T_used = T
        Xu = np.column_stack([np.ones(T), b[[f"y{k}" for k in range(1, K + 1)] + [f"x{k}" for k in range(1, K + 1)]]])
        Xr = Xu[:, :1 + K]
        ssr = lambda X: np.sum((b.y - X @ np.linalg.lstsq(X, b.y, rcond=None)[0]) ** 2)  # noqa: E731
        su, sr = ssr(Xu), ssr(Xr)
        W.append((T - 2 * K - 1) * (sr - su) / su)
    W = np.array(W)
    N, T = len(W), T_used
    wbar = W.mean()
    zbar_tilde = (np.sqrt(N / (2 * K) * (T - 2 * K - 5) / (T - K - 3))
                  * ((T - 2 * K - 3) / (T - 2 * K - 1) * wbar - K))
    return dict(w_stat=wbar, zbar_stat=zbar_tilde, p=2 * stats.norm.sf(abs(zbar_tilde)))


# ---------------------------------------------------------------- Tables 10-11
def panel_var_irf_fevd(df, cols, p=2, horizon=10):
    """Pooled VAR(p) with intercept; Cholesky in the order of `cols`; IRF and FEVD for cols[0]."""
    Y, X = _stack_lags(df, cols, p, p)
    k = len(cols)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    E = Y - X @ B
    S = E.T @ E / (len(Y) - X.shape[1])
    A = [B[1 + (j - 1) * k:1 + j * k].T for j in range(1, p + 1)]   # A_j (k x k): y_t = c + sum A_j y_{t-j}
    Phi = [np.eye(k)]
    for h in range(1, horizon):
        Phi.append(sum(A[j - 1] @ Phi[h - j] for j in range(1, min(h, p) + 1)))
    P = np.linalg.cholesky(S)
    theta = np.array([Ph @ P for Ph in Phi])                          # horizon x k x k
    irf = pd.DataFrame(theta[:, 0, :], columns=[c.upper() for c in cols])
    irf.index = pd.RangeIndex(1, horizon + 1, name="Period")
    cum = np.cumsum(theta[:, 0, :] ** 2, axis=0)
    tot = cum.sum(1)
    fevd = pd.DataFrame(100 * cum / tot[:, None], columns=[c.upper() for c in cols])
    fevd.insert(0, "S.E.", np.sqrt(tot))
    fevd.index = irf.index
    return irf, fevd


# ---------------------------------------------------------------- Tables 12-15
def _panel(df, cols):
    return df.set_index(["country", "year"])[cols]


def hausman(df, y, xs):
    d = _panel(df, [y] + xs).assign(const=1.0)
    fe = PanelOLS(d[y], d[["const"] + xs], entity_effects=True).fit(cov_type="unadjusted")
    re = RandomEffects(d[y], d[["const"] + xs]).fit(cov_type="unadjusted")
    b = (fe.params - re.params)[xs]
    V = (fe.cov - re.cov).loc[xs, xs]
    H = float(b @ np.linalg.pinv(V) @ b)
    return pd.DataFrame([dict(test="Cross-section random", chi2=H, df=len(xs), p=stats.chi2.sf(H, len(xs)))])


def fixed_effects(df, y, xs, cov="unadjusted"):
    """Within estimator with an intercept (EViews-style 'C'); cov in {'unadjusted','clustered','kernel'}."""
    d = _panel(df, [y] + xs).assign(const=1.0)
    mod = PanelOLS(d[y], d[["const"] + xs], entity_effects=True)
    kw = dict(cov_type=cov)
    if cov == "clustered":
        kw["cluster_entity"] = True
    res = mod.fit(**kw)
    tab = pd.DataFrame(dict(coef=res.params, se=res.std_errors, t=res.tstats, p=res.pvalues))
    tab.index = [("C" if i == "const" else i.upper()) for i in tab.index]
    tab = pd.concat([tab.drop(index="C"), tab.loc[["C"]]])
    n, k, N = res.nobs, len(xs), df.country.nunique()
    r2 = res.rsquared_inclusive
    e = res.resids.groupby(level=0)
    dw = float(sum((np.diff(g.to_numpy()) ** 2).sum() for _, g in e) / (res.resids ** 2).sum())
    fit = dict(r2_lsdv=r2, adj_r2_lsdv=1 - (1 - r2) * (n - 1) / (n - k - N), r2_within=res.rsquared_within,
               durbin_watson=dw, n_obs=n, cov=cov)
    return tab, fit


def vif(df, xs):
    X = np.column_stack([np.ones(len(df)), df[xs].to_numpy()])
    return pd.DataFrame(dict(VIF=[variance_inflation_factor(X, i + 1) for i in range(len(xs))]),
                        index=[x.upper() for x in xs])


def wooldridge_serial(df, y, xs):
    """Regress first differences, then the residual on its own lag; H0: coefficient = -0.5. Clustered by country."""
    rows = []
    for g in _by_country(df):
        dd = g[[y] + xs].diff().dropna()
        rows.append(dd.assign(country=g.country.iloc[0]))
    D = pd.concat(rows)
    b = np.linalg.lstsq(D[xs].to_numpy(), D[y].to_numpy(), rcond=None)[0]
    D["e"] = D[y].to_numpy() - D[xs].to_numpy() @ b
    D["elag"] = D.groupby("country").e.shift(1)
    D = D.dropna(subset=["elag"])
    x, e = D.elag.to_numpy(), D.e.to_numpy()
    rho = (x @ e) / (x @ x)
    u = e - rho * x
    meat = sum((x[m] @ u[m]) ** 2 for m in [D.country.to_numpy() == c for c in D.country.unique()])
    G = D.country.nunique()
    se = np.sqrt(meat / (x @ x) ** 2 * G / (G - 1))
    t = (rho + 0.5) / se
    return pd.DataFrame([dict(test="t-statistic", value=t, df=G - 1, p=2 * stats.t.sf(abs(t), G - 1)),
                         dict(test="F-statistic", value=t * t, df=f"(1, {G - 1})", p=stats.f.sf(t * t, 1, G - 1)),
                         dict(test="Chi-square", value=t * t, df="1", p=stats.chi2.sf(t * t, 1))]), rho
