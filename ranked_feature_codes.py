def regime_shift_detection(log_returns: torch.Tensor, window: int=20) -> torch.Tensor:
    if window <= 1:
        return torch.zeros_like(log_returns[:, 0])
    if window > log_returns.shape[-1]:
        window = log_returns.shape[-1]
    windows = log_returns.unfold(-1, window, 1)
    means = windows.mean(dim=-1)
    current_mean = means[:, -1]
    past_mean = means[:, :-1].mean(dim=-1)
    shift = current_mean - past_mean
    return shift

def volatility_regime_indicator(log_returns: torch.Tensor, window: int=30) -> torch.Tensor:
    n = log_returns.shape[-1]
    if window > n:
        window = n
    if window <= 1:
        return torch.zeros_like(log_returns[:, 0])
    windows = log_returns.unfold(-1, window, 1)
    vol = windows.std(dim=-1)
    current_vol = vol[:, -1]
    past_vol = vol[:, :-1].mean(dim=-1)
    return (current_vol - past_vol) / (past_vol + 1e-08)

def memory_effect_ratio(log_returns: torch.Tensor, short_lag: int=5, long_lag: int=20) -> torch.Tensor:
    if short_lag >= log_returns.shape[-1]:
        short_lag = log_returns.shape[-1] - 1
    if long_lag >= log_returns.shape[-1]:
        long_lag = log_returns.shape[-1] - 1
    lead_short = log_returns[:, :-short_lag]
    lagged_short = log_returns[:, short_lag:]
    if lead_short.shape[-1] == 0 or lagged_short.shape[-1] == 0:
        short_autocor = torch.zeros_like(log_returns[:, 0])
    else:
        short_autocor = (lead_short * lagged_short).mean(dim=-1)
    lead_long = log_returns[:, :-long_lag]
    lagged_long = log_returns[:, long_lag:]
    if lead_long.shape[-1] == 0 or lagged_long.shape[-1] == 0:
        long_autocor = torch.zeros_like(log_returns[:, 0])
    else:
        long_autocor = (lead_long * lagged_long).mean(dim=-1)
    ratio = (short_autocor - long_autocor) / (long_autocor + 1e-08)
    return ratio

def information_decay_rate(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    alpha = 1.0 - 1.0 / n ** 0.5
    ema = log_returns.clone()
    for i in range(1, n):
        ema[:, i] = alpha * ema[:, i - 1] + (1 - alpha) * log_returns[:, i]
    delta_ema = (ema[:, -1] - ema[:, 0]).abs()
    return delta_ema

def momentum_driven_risk(log_returns: torch.Tensor, lag: int=1) -> torch.Tensor:
    lead_returns = log_returns[:, lag:]
    lag_returns = log_returns[:, :-lag]
    autocov = (lead_returns * lag_returns).mean(dim=-1)
    return autocov

def nonlinear_autocorrelation(log_returns: torch.Tensor, power: int=3) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    if n == 1:
        return log_returns[:, 0]
    lead = log_returns[:, :-1].pow(power)
    lagged = log_returns[:, 1:].pow(power)
    autocor = (lead * lagged).mean(dim=-1)
    return autocor

def return_impact(log_returns: torch.Tensor) -> torch.Tensor:
    mean = log_returns.mean(dim=-1, keepdim=True)
    deviation = (log_returns - mean).abs()
    return deviation.mean(dim=-1) / (log_returns.abs().mean(dim=-1) + 1e-08)

def extreme_return_ratio(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    k = int(n * 0.1)
    if k == 0:
        k = 1
    positive = torch.topk(log_returns, k, dim=-1)[0]
    negative = torch.topk(-log_returns, k, dim=-1)[0]
    ratio = positive.sum(dim=-1) / (negative.abs().sum(dim=-1) + 1e-08)
    return ratio

def average_autocorrelation(log_returns: torch.Tensor, lags: int=10) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    if lags >= n:
        lags = n - 1
    means = log_returns.mean(dim=-1, keepdim=True)
    centered = (log_returns - means) / (torch.std(log_returns, dim=-1, keepdim=True) + 1e-08)
    autocors = []
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        lead = centered[:, :-lag]
        lagged = centered[:, lag:]
        if lead.shape[-1] == 0 or lagged.shape[-1] == 0:
            continue
        autocor = (lead * lagged).mean(dim=-1)
        autocors.append(autocor)
    if not autocors:
        return torch.zeros_like(means.squeeze())
    return torch.stack(autocors).mean(dim=0)

def risk_adjusted_momentum(log_returns: torch.Tensor, window: int=20) -> torch.Tensor:
    if window == 0:
        return torch.zeros_like(log_returns[:, 0])
    n = log_returns.shape[-1]
    if window > n:
        window = n
    current_returns = log_returns[:, -1]
    past_returns = log_returns[:, :-window]
    momentum = current_returns - past_returns.mean(dim=-1)
    volatility = past_returns.std(dim=-1)
    return momentum / (volatility + 1e-08)

def asymmetric_volatility_contribution(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    alpha_pos = 0.1
    alpha_neg = 0.2
    vol_pos = log_returns.clone()
    vol_neg = log_returns.clone()
    for i in range(1, n):
        vol_pos[:, i] = alpha_pos * vol_pos[:, i - 1] + (1 - alpha_pos) * (log_returns[:, i] * (log_returns[:, i] > 0).float())
        vol_neg[:, i] = alpha_neg * vol_neg[:, i - 1] + (1 - alpha_neg) * (log_returns[:, i] * (log_returns[:, i] < 0).float()).abs()
    return vol_pos[:, -1] - vol_neg[:, -1]


def sum_(log_returns: torch.Tensor) -> torch.Tensor:
    return log_returns.sum(dim=-1)


def max_drawdown(log_returns: torch.Tensor) -> torch.Tensor:
    if log_returns.shape[-1] == 0:
        return torch.zeros_like(log_returns[:, 0])
    cumulative = torch.cumsum(log_returns, dim=-1)
    max_so_far = torch.cummax(cumulative, dim=-1).values
    drawdown = cumulative - max_so_far
    max_dd = drawdown.min(dim=-1).values
    return max_dd

def tail_spread(log_returns: torch.Tensor, threshold: float=0.05) -> torch.Tensor:
    if threshold <= 0 or threshold >= 1:
        threshold = 0.05
    n = log_returns.shape[-1]
    lower = log_returns.topk(int(threshold * n), dim=-1, largest=False)[0]
    upper = log_returns.topk(int((1 - threshold) * n), dim=-1, largest=True)[0]
    lower_mean = lower.mean(dim=-1)
    upper_mean = upper.mean(dim=-1)
    spread = upper_mean - lower_mean
    vol = log_returns.std(dim=-1)
    scaled_spread = spread / (vol + 1e-08)
    return scaled_spread

def kurtosis(log_returns: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(log_returns, dim=-1, keepdim=True)
    standardized = (log_returns - mean) / torch.std(log_returns, dim=-1, keepdim=True)
    kurtosis = torch.mean(torch.pow(standardized, 4), dim=-1) - 3
    return kurtosis

def skewness(log_returns: torch.Tensor) -> torch.Tensor:
    mean = log_returns.mean(dim=-1, keepdim=True)
    standardized = (log_returns - mean) / (torch.std(log_returns, dim=-1, keepdim=True) + 1e-08)
    skew = torch.mean(standardized ** 3, dim=-1)
    return skew

def upside_potential_ratio(log_returns: torch.Tensor) -> torch.Tensor:
    positive_returns = log_returns * (log_returns > 0).float()
    negative_returns = log_returns * (log_returns < 0).float()
    mean_positive = positive_returns.mean(dim=-1)
    mean_negative = negative_returns.abs().mean(dim=-1)
    ratio = mean_positive / (mean_negative + 1e-08)
    return ratio

def asymmetric_tail_risk(log_returns: torch.Tensor, threshold: float=0.1) -> torch.Tensor:
    lower = log_returns.topk(int(threshold * log_returns.shape[-1]), dim=-1, largest=False)[0]
    upper = log_returns.topk(int((1 - threshold) * log_returns.shape[-1]), dim=-1, largest=True)[0]
    lower_mean = lower.mean(dim=-1)
    upper_mean = upper.mean(dim=-1)
    spread = upper_mean - lower_mean
    vol = log_returns.std(dim=-1)
    scaled_spread = spread / (vol + 1e-08)
    return scaled_spread

def turnover_based_liquidity_risk(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    turnover = log_returns.abs().sum(dim=-1)
    vol = log_returns.std(dim=-1)
    return turnover / (vol + 1e-08)

def zero_return_frequency(log_returns: torch.Tensor, threshold: float=0.0001) -> torch.Tensor:
    near_zero = (log_returns.abs() < threshold).float()
    return near_zero.sum(dim=-1)

def autocovariance_decay(log_returns: torch.Tensor, lags: int=10) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    autocovs = []
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        lead = log_returns[:, :-lag]
        lagged = log_returns[:, lag:]
        if lead.shape[-1] == 0 or lagged.shape[-1] == 0:
            continue
        autocov = (lead * lagged).mean(dim=-1)
        autocovs.append(autocov)
    if not autocovs:
        return torch.zeros_like(log_returns[:, 0])
    return torch.stack(autocovs).mean(dim=0)

def volatility_of_volatility(log_returns: torch.Tensor, window: int=30) -> torch.Tensor:
    n = log_returns.shape[-1]
    if window > n:
        window = n
    if window <= 1:
        return torch.zeros_like(log_returns[:, 0])
    windows = log_returns.unfold(-1, window, 1)
    vol = windows.std(dim=-1)
    vol_vol = vol.std(dim=-1)
    return vol_vol


def minimum(log_returns: torch.Tensor) -> torch.Tensor:
    return log_returns.min(dim=-1).values


def asymmetric_volatility(log_returns: torch.Tensor) -> torch.Tensor:
    mean = log_returns.mean(dim=-1, keepdim=True)
    pos_vol = log_returns.where(log_returns > mean, torch.zeros_like(log_returns)).std(dim=-1)
    neg_vol = log_returns.where(log_returns < mean, torch.zeros_like(log_returns)).std(dim=-1)
    return pos_vol / (neg_vol + 1e-08)

def realized_skewness_kurtosis_ratio(log_returns: torch.Tensor) -> torch.Tensor:
    if log_returns.shape[-1] == 0:
        return torch.zeros_like(log_returns[:, 0])
    mean = log_returns.mean(dim=-1, keepdim=True)
    standardized = (log_returns - mean) / (log_returns.std(dim=-1, keepdim=True) + 1e-08)
    skew = (standardized ** 3).mean(dim=-1)
    kurtosis = (standardized ** 4).mean(dim=-1) - 3
    return skew / (kurtosis + 1e-08)


def maximum(log_returns: torch.Tensor) -> torch.Tensor:
    return log_returns.max(dim=-1).values



def median(log_returns: torch.Tensor) -> torch.Tensor:
    return log_returns.median(dim=-1).values



def range_(log_returns: torch.Tensor) -> torch.Tensor:
    return maximum(log_returns) - minimum(log_returns)


def quantile_range(log_returns: torch.Tensor, window: int=30) -> torch.Tensor:
    if window <= 0:
        window = 1
    n = log_returns.shape[-1]
    if window > n:
        window = n
    windows = log_returns.unfold(-1, window, 1)
    lower = windows.quantile(0.25, dim=-1)
    upper = windows.quantile(0.75, dim=-1)
    q_range = (upper - lower).mean(dim=-1)
    return q_range


def mad(log_returns: torch.Tensor) -> torch.Tensor:
    mean_val = log_returns.mean(dim=-1, keepdim=True)
    return (log_returns - mean_val).abs().mean(dim=-1)


def composite_risk_measure(log_returns: torch.Tensor) -> torch.Tensor:
    mean = log_returns.mean(dim=-1, keepdim=True)
    volatility = log_returns.std(dim=-1, keepdim=True)
    skew = torch.mean((log_returns - mean) ** 3, dim=-1, keepdim=True) / (volatility ** 3 + 1e-08)
    kurtosis = torch.mean((log_returns - mean) ** 4, dim=-1, keepdim=True) / (volatility ** 4 + 1e-08)
    return torch.cat([volatility, skew, kurtosis], dim=-1).mean(dim=-1)


def mean(log_returns: torch.Tensor) -> torch.Tensor:
    return log_returns.mean(dim=-1)


def hill_tail_risk_estimator(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    k = max(int(n * 0.1), 1)
    (sorted_returns, _) = log_returns.sort(dim=-1)
    top_returns = sorted_returns[:, -k:]
    mean_top = top_returns.mean(dim=-1)
    hill_estimator = mean_top * k
    return hill_estimator


def std(log_returns: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    return torch.std(log_returns, dim=-1, unbiased=unbiased)


def autocorrelation_profile(log_returns: torch.Tensor, lags: int=10) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    if lags >= n:
        lags = n - 1
    autocors = []
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        lead = log_returns[:, :-lag]
        lagged = log_returns[:, lag:]
        if lead.shape[-1] == 0 or lagged.shape[-1] == 0:
            continue
        current_autocor = (lead * lagged).mean(dim=-1)
        autocors.append(current_autocor)
    if not autocors:
        return torch.zeros_like(log_returns[:, 0])
    return torch.stack(autocors).mean(dim=0)

def tail_spread_ratio(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    lower = log_returns.topk(int(0.05 * n), dim=-1, largest=False)[0]
    upper = log_returns.topk(int(0.95 * n), dim=-1, largest=True)[0]
    lower_mean = lower.mean(dim=-1)
    upper_mean = upper.mean(dim=-1)
    spread = upper_mean - lower_mean
    return spread

def realized_semivariance(log_returns: torch.Tensor) -> torch.Tensor:
    mean = log_returns.mean(dim=-1, keepdim=True)
    semivariance = (log_returns - mean).where(log_returns < mean, torch.zeros_like(log_returns)).pow(2).mean(dim=-1)
    return semivariance

def high_frequency_impact(log_returns: torch.Tensor, window: int=5) -> torch.Tensor:
    if window <= 1:
        return torch.zeros_like(log_returns[:, 0])
    if window > log_returns.shape[-1]:
        window = log_returns.shape[-1]
    windows = log_returns.unfold(-1, window, 1)
    var = windows.var(dim=-1)
    hfi = var.mean(dim=-1)
    return hfi


def absolute_sum(log_returns: torch.Tensor) -> torch.Tensor:
    return log_returns.abs().sum(dim=-1)



def variance(log_returns: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    return log_returns.var(dim=-1, unbiased=unbiased)


def tail_risk_asymmetry(log_returns: torch.Tensor, threshold: float=0.05) -> torch.Tensor:
    if threshold <= 0 or threshold >= 1:
        threshold = 0.05
    n = log_returns.shape[-1]
    lower = log_returns.topk(int(threshold * n), dim=-1, largest=False)[0]
    upper = log_returns.topk(int((1 - threshold) * n), dim=-1, largest=True)[0]
    lower_mean = lower.mean(dim=-1)
    upper_mean = upper.mean(dim=-1)
    spread = upper_mean - lower_mean
    return spread / (spread.abs() + 1e-08)

def hurst_exponent(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    rs = (log_returns.mean(dim=-1, keepdim=True) - log_returns).sum(dim=-1)
    cum_diffs = torch.cumsum(log_returns, dim=-1)
    cum_diffs_mean = cum_diffs.mean(dim=-1, keepdim=True)
    cum_diffs_dev = cum_diffs - cum_diffs_mean
    abs_cum_diffs = torch.abs(cum_diffs_dev)
    w = torch.mean(abs_cum_diffs, dim=-1)
    h = 0.5 * torch.log(torch.mean(torch.pow(w, 2.0) / torch.pow(w.mean(dim=-1, keepdim=True), 2.0), dim=-1))
    return h

def conditional_value_at_risk(log_returns: torch.Tensor, alpha: float=0.95) -> torch.Tensor:
    (sorted_returns, _) = torch.sort(log_returns, dim=-1)
    var = torch.index_select(sorted_returns, dim=-1, index=torchLongTensor([int(alpha * len(sorted_returns))]))
    cvar = torch.mean(var)
    return cvar

def tail_risk(log_returns: torch.Tensor, alpha: float=0.95) -> torch.Tensor:
    (sorted_returns, _) = torch.sort(log_returns, dim=-1)
    tail_length = int((1 - alpha) * len(sorted_returns))
    if tail_length == 0:
        return torch.mean(sorted_returns)
    tail = sorted_returns[:, :tail_length]
    return torch.mean(tail)

def garch_volatility(log_returns: torch.Tensor, alpha: float=0.94, beta: float=0.05, burn_in: int=50) -> torch.Tensor:
    n = log_returns.shape[-1]
    var = torch.zeros_like(log_returns)
    var[:, burn_in] = torch.var(log_returns[:, :burn_in], dim=-1)
    for t in range(burn_in, n):
        var[:, t] = alpha * log_returns[:, t - 1] ** 2 + beta * var[:, t - 1]
    return torch.sqrt(var)

def rolling_sharpe_ratio(log_returns: torch.Tensor, window: int=30) -> torch.Tensor:
    n = log_returns.shape[-1]
    means = log_returns.rolling(window).mean()
    stds = log_returns.rolling(window).std()
    sharpes = (means / (stds + 1e-08))[:, -window:]
    return sharpes.mean(dim=-1)

def volatility_clustering(log_returns: torch.Tensor, window: int=30) -> torch.Tensor:
    n = log_returns.shape[-1]
    squared_returns = (log_returns - log_returns.rolling(window).mean()) ** 2
    clustering = squared_returns.rolling(window).mean()[:, -window:]
    return clustering.mean(dim=-1)

def return_reversal(log_returns: torch.Tensor) -> torch.Tensor:
    shifted = torch.cat([log_returns, log_returns[:, :-1]], dim=-1)
    product = log_returns * shifted[:, :-log_returns.shape[-1]]
    return product.mean(dim=-1) / (log_returns.std(dim=-1, unbiased=False) + 1e-08)

def windowed_correlation(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n <= 1:
        return torch.zeros_like(log_returns[:, 0])
    window = n // 2
    windows_past = log_returns.unfold(-1, window, 1)
    windows_future = log_returns[:, :, None, :].unfold(-1, window, 1)
    covariance = torch.mean(windows_past * windows_future, dim=(-2, -1))
    past_var = torch.var(windows_past, dim=-1, unbiased=False)
    future_var = torch.var(windows_future, dim=-1, unbiased=False)
    corr = covariance / (past_var * future_var + 1e-08).sqrt()
    return corr.mean(dim=-1)

def timing_of_reversals(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n <= 1:
        return torch.zeros_like(log_returns[:, 0])
    diffs = torch.diff(torch.arange(n), device=log_returns.device).float()
    high_to_low = (log_returns[:, :-1] > log_returns[:, 1:]).float() * diffs[:, None]
    low_to_high = (log_returns[:, :-1] < log_returns[:, 1:]).float() * diffs[:, None]
    total_time = high_to_low.sum(dim=-1) + low_to_high.sum(dim=-1)
    return total_time / (n - 1)

def variance_of_variances(log_returns: torch.Tensor) -> torch.Tensor:
    window_size = log_returns.shape[-1] // 3
    if window_size <= 1:
        return torch.zeros_like(log_returns[:, 0])
    windows = log_returns.unfold(-1, window_size, 1)
    variances = torch.var(windows, dim=-1, unbiased=False)
    return torch.var(variances, unbiased=False)

def temporal_correlation(log_returns: torch.Tensor, window: int=20, lag: int=5) -> torch.Tensor:
    if window <= 0:
        window = 1
    if lag >= window:
        lag = window - 1
    n = log_returns.shape[-1]
    if window > n:
        window = n
    if window <= 1:
        return torch.zeros_like(log_returns[:, 0])
    means = log_returns.mean(dim=-1, keepdim=True)
    centered = log_returns - means
    windows = centered.unfold(-1, window, 1)
    corr_coeff = []
    for i in range(lag):
        if i + window > n:
            break
        lead = windows[:, i, :]
        lagged = windows[:, i + lag, :] if i + lag < windows.size(1) else windows[:, -1, :]
        if lead.shape[-1] != lagged.shape[-1]:
            continue
        cov = (lead * lagged).mean(dim=-1)
        var_lead = lead.var(dim=-1)
        var_lagged = lagged.var(dim=-1)
        with torch.no_grad():
            if var_lead.mean() == 0 or var_lagged.mean() == 0:
                corr = torch.zeros_like(cov)
            else:
                corr = cov / (torch.sqrt(var_lead * var_lagged) + 1e-08)
        corr_coeff.append(corr.mean(dim=-1))
    if not corr_coeff:
        return torch.zeros_like(means.squeeze())
    return torch.stack(corr_coeff).mean(dim=0)

def momentum_reversal_ratio(log_returns: torch.Tensor, short_window: int=10, long_window: int=30) -> torch.Tensor:
    if short_window <= 0:
        short_window = 1
    if long_window <= 0:
        long_window = 1
    n = log_returns.shape[-1]
    if short_window > n:
        short_window = n
    if long_window > n:
        long_window = n
    short_returns = log_returns.unfold(-1, short_window, 1).mean(dim=-1)
    long_returns = log_returns.unfold(-1, long_window, 1).mean(dim=-1)
    ratio = (short_returns - long_returns) / (long_returns + 1e-08)
    return ratio.mean(dim=-1)

def wavelet_variance(log_returns: torch.Tensor, level: int=2) -> torch.Tensor:
    if level <= 0:
        level = 1
    with torch.no_grad():
        n = log_returns.shape[-1]
        if n <= 1:
            return torch.zeros_like(log_returns[:, 0])
        wavelet = torch.tensor([1, -1], dtype=torch.float) / torch.sqrt(torch.tensor(2, dtype=torch.float))
        w = log_returns.clone()
        for _ in range(level):
            w = torch.cat([w[:, :-1], w[:, 1:]], dim=-1)
            w = torch.mv(orthonormal_matrix(w))
        var = w.var(dim=-1)
    return var

def liquidity_adjusted_volatility(log_returns: torch.Tensor, volumes: torch.Tensor) -> torch.Tensor:
    if 'volumes' not in locals():
        volumes = torch.ones_like(log_returns)
    vol = log_returns.std(dim=-1, keepdim=True)
    liquidity = volumes.mean(dim=-1, keepdim=True)
    return vol / (liquidity + 1e-08)

def weighted_realized_volatility(log_returns: torch.Tensor, window: int=30) -> torch.Tensor:
    n = log_returns.shape[-1]
    if window > n:
        window = n
    weights = torch.arange(1, window + 1, dtype=torch.float32, device=log_returns.device)
    weights = weights / weights.sum()
    windows = log_returns.unfold(-1, window, 1)
    vol = windows.std(dim=-1)
    wvol = (vol * weights).sum(dim=-1)
    return wvol

def conditional_autocorrelation_dynamics(log_returns: torch.Tensor, lags: int=5, window: int=20) -> torch.Tensor:
    n = log_returns.shape[-1]
    if window > n:
        window = n
    if window <= 1:
        return torch.zeros_like(log_returns[:, 0])
    if lags >= window:
        lags = window - 1
    autocors = []
    for lag in range(1, lags + 1):
        lead = log_returns[:, :-lag]
        lagged = log_returns[:, lag:]
        if lead.shape[-1] == 0 or lagged.shape[-1] == 0:
            continue
        autocor = (lead * lagged).mean(dim=-1)
        autocors.append(autocor)
    if not autocors:
        return torch.zeros_like(log_returns[:, 0])
    autocor_series = torch.stack(autocors, dim=-1)
    moving_autocor = autocor_series.unfold(-1, window, 1).mean(dim=-1)
    cad = moving_autocor[:, -1] - moving_autocor[:, :-1].mean(dim=-1)
    return cad

def momentum_neutral_risk(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    (u, s, v) = torch.linalg.svd(log_returns)
    pc = u[:, :, None] * s[:, None, None] * v[:, None, :]
    pc = pc.squeeze(-1)
    X = pc
    XTX_inv = torch.inverse(torch.mm(X.T, X)) if X.shape[1] <= X.shape[0] else torch.pinverse(X)
    beta = torch.mm(torch.mm(XTX_inv, X.T), log_returns.T).T
    residuals = log_returns - torch.mm(X, beta)
    return residuals.std(dim=-1)

def time_scale_stretching(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    scales = [2, 4, 8, 16]
    correlations = []
    for scale in scales:
        if scale >= n:
            break
        resampled = log_returns[:, ::scale]
        correlations.append(torch.corr(resampled, log_returns, dim=-1).mean(dim=-1))
    if not correlations:
        return torch.zeros_like(log_returns[:, 0])
    return torch.stack(correlations).mean(dim=0)

def power_law_autocorrelation_decay(log_returns: torch.Tensor) -> torch.Tensor:
    n = log_returns.shape[-1]
    if n == 0:
        return torch.zeros_like(log_returns[:, 0])
    max_lag = min(100, n // 2)
    lags = torch.arange(1, max_lag + 1, dtype=torch.float32)
    means = log_returns.mean(dim=-1, keepdim=True)
    centered = (log_returns - means) / (log_returns.std(dim=-1, keepdim=True) + 1e-08)
    autocors = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            continue
        lead = centered[:, :-lag]
        lagged = centered[:, lag:]
        if lead.shape[-1] == 0 or lagged.shape[-1] == 0:
            continue
        autocor = (lead * lagged).mean(dim=-1)
        autocors.append(autocor)
    if not autocors:
        return torch.zeros_like(means.squeeze())
    autocor_matrix = torch.stack(autocors)
    decay_rate = (autocor_matrix.log().mean(dim=0) / lags[None, :].to(log_returns.device)).mean()
    return decay_rate

def leverage_effect_risk_indicator(log_returns: torch.Tensor, window: int=20) -> torch.Tensor:
    if window <= 0:
        return torch.zeros_like(log_returns[:, 0])
    n = log_returns.shape[-1]
    if window > n:
        window = n
    if window <= 1:
        return torch.zeros_like(log_returns[:, 0])
    windows = log_returns.unfold(-1, window, 1)
    negative_returns = (windows[:, :, -1] < 0).float()
    current_returns = windows[:, :, 0]
    lead_vols = current_returns.std(dim=-1)
    neg_corr = (negative_returns.mean(dim=-1) * lead_vols).mean(dim=-1)
    return neg_corr

def positive_return_ratio(log_returns: torch.Tensor, window: int=20) -> torch.Tensor:
    if window <= 0:
        window = 1
    n = log_returns.shape[-1]
    if window > n:
        window = n
    windows = log_returns.unfold(-1, window, 1)
    pos_count = (windows > 0).sum(dim=-1)
    return pos_count / window

def tail_recovery_indicator(log_returns: torch.Tensor, window: int=20) -> torch.Tensor:
    if window <= 0:
        window = 1
    n = log_returns.shape[-1]
    if window > n:
        window = n
    windows = log_returns.unfold(-1, window, 1)
    tail_events = (windows < windows.mean(dim=-1, keepdim=True) - windows.std(dim=-1, keepdim=True) * 2).float()
    next_returns = log_returns[:, window:]
    if next_returns.shape[-1] == 0:
        return torch.zeros_like(log_returns[:, 0])
    recovery = (next_returns * tail_events.mean(dim=-1)).mean(dim=-1)
    return recovery

def volatility_change_indicator(log_returns: torch.Tensor, short_window: int=10, long_window: int=30) -> torch.Tensor:
    if short_window <= 0:
        short_window = 1
    if long_window <= 0:
        long_window = 1
    n = log_returns.shape[-1]
    if short_window > n:
        short_window = n
    if long_window > n:
        long_window = n
    short_vol = log_returns.unfold(-1, short_window, 1).std(dim=-1)
    long_vol = log_returns.unfold(-1, long_window, 1).std(dim=-1)
    if short_vol.shape[-1] != long_vol.shape[-1]:
        long_vol = long_vol[:, :short_vol.shape[-1]]
    return (short_vol - long_vol).mean(dim=-1)

def volatility_persistence_ratio(log_returns: torch.Tensor, window: int=20) -> torch.Tensor:
    if window <= 0:
        window = 1
    n = log_returns.shape[-1]
    if window > n:
        window = n
    windows = log_returns.unfold(-1, window, 1)
    current_vol = windows.std(dim=-1)
    past_vol = windows[:, :-1].mean(dim=-1)
    persistence = (current_vol - past_vol) / (past_vol + 1e-08)
    return persistence.mean(dim=-1)

