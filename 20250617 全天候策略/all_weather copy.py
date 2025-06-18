"""
All-Weather Strategy (中国版全天候策略)
========================================
This script constructs and back-tests a Chinese All-Weather portfolio inspired by Bridgewater's All-Weather
strategy and the factor-risk-parity framework.

Key Characteristics
-------------------
1. Macro Quadrant framework (Growth ↑ / ↓, Inflation ↑ / ↓)
2. Quadrant portfolios are constructed based on weighted allocations to asset classes.
3. Inside each asset class, assets are equal-weighted.
4. Risk for each quadrant portfolio is measured via the exponentially weighted
   moving average (EWMA) semi-covariance matrix (downside risk only).
5. Monthly re-balancing to achieve risk parity **across quadrants** (not across
   single assets).
6. No ex-ante leverage (notional weights sum to 1). Transaction fee is 1 bps
   (单边万分之一) per trade.
7. Back-test window: 2013-12-31 – 2025-04-30 (inclusive, monthly).

Dependencies
------------
- akshare >= 1.11.0
- pandas, numpy, matplotlib, scipy (optional but recommended for optimisation)

Usage
-----
$ python all_weather.py

The script will fetch historical data using **akshare**. First calls might be
slow due to network latency; subsequent runs read from on-disk CSV cache in the
`data/` folder.
"""

from __future__ import annotations

import math
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# try/except so the file is importable even if akshare is not installed yet.
try:
    import akshare as ak  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "akshare is required: pip install akshare --upgrade --quiet") from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# NOTE: To include futures contracts, please upgrade your akshare library
# pip install akshare --upgrade
# ---------------------------------------------------------------------------

START_DATE = "2010-01-01"
END_DATE = "2025-06-15"
FEE_RATE = 0.0001  # 单边万分之一
EWMA_LAMBDA = 0.80  # decay parameter for EWMA
WINDOW_MONTHS = 20  # rolling window for risk estimation
REBALANCE_FREQ = "W"  # 调仓频率: "W"=周末, "ME"=月末, "QE"=季末, "YE"=年末

# Asset Universe -----------------------------------------------------------------------
ASSET_NAMES: Dict[str, str] = {
    # 股票
    "510300.SH": "沪深300ETF",
    "510500.SH": "中证500ETF",
    "512100.SH": "中证1000ETF",
    "159920.SZ": "恒生ETF",
    "159740.SZ": "恒生科技ETF",
    # 债券
    "511090.SH": "30年国债ETF",
    "511010.SH": "5年国债ETF",
    "511260.SH": "10年国债ETF",
    # 商品
    "159980.SZ": "有色ETF",
    "159981.SZ": "能源化工ETF",
    "M.DCE": "豆粕期货",
    "160723.SZ": "嘉实原油LOF",
    "RB.SHF": "螺纹钢期货",
    # 红利
    "510880.SH": "红利ETF",
    "513630.SH": "港股红利指数ETF",
    # 黄金
    "518880.SH": "黄金ETF",
}

# Asset Classes ------------------------------------------------------------------------
ASSET_CLASSES: Dict[str, List[str]] = {
    "股票": [
        "510300.SH", "510500.SH", "512100.SH", "159920.SZ", "159740.SZ"
    ],
    "债券": ["511090.SH", "511010.SH", "511260.SH"],
    "商品": ["159980.SZ", "159981.SZ", "M.DCE", "160723.SZ", "RB.SHF"],
    "红利": ["510880.SH", "513630.SH"],
    "黄金": ["518880.SH"],
}


# Quadrant asset allocation (Asset Class Weights) ------------------------------------
QUADRANTS: Dict[str, Dict[str, float]] = {
    "通胀超预期_增长超预期": {"股票": 2, "商品": 1},
    "通胀超预期_增长不及预期": {"商品": 3, "黄金": 2, "红利": 1},
    "通胀不及预期_增长超预期": {"股票": 3, "债券": 1},
    "通胀不及预期_增长不及预期": {"债券": 3, "红利": 1},
}

# Normalize quadrant weights to sum to 1
for quad, weights in QUADRANTS.items():
    total_weight = sum(weights.values())
    QUADRANTS[quad] = {cls: w / total_weight for cls, w in weights.items()}


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _cache_path(symbol: str) -> Path:
    # Sanitize symbol for filename
    safe_symbol = symbol.replace('.', '_')
    return DATA_DIR / f"{safe_symbol}.pkl"


def fetch_price_series(ticker: str) -> pd.Series:
    """Fetch daily *adjusted* close price for a single asset.

    Handles ETFs, LOFs, and Futures continuous contracts.
    Data are cached locally (pickle)."""
    cache_file = _cache_path(ticker)
    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception:  # Handle corrupted cache
            pass

    print(f"Fetching data for {ticker} ({ASSET_NAMES.get(ticker, 'Unknown')})...")
    price = None
    df = pd.DataFrame()

    try:
        if ticker.endswith((".SH", ".SZ")):
            symbol_code = ticker.split('.')[0]
            # For LOF, use fund_lof_hist_em; for ETFs, use fund_etf_hist_em
            if ticker == "160723.SZ":
                df = ak.fund_lof_hist_em(symbol=symbol_code, start_date="20100101", end_date="20251231")
            else:
                df = ak.fund_etf_hist_em(symbol=symbol_code)

            if not df.empty:
                df['date'] = pd.to_datetime(df['日期'])
                df = df.set_index('date').sort_index()
                if ticker == "160723.SZ" and '单位净值' in df.columns:
                    price = df['单位净值'].astype(float)
                elif '收盘' in df.columns:
                    price = df['收盘'].astype(float)
                elif 'close' in df.columns:
                    price = df['close'].astype(float)

        elif ticker == "M.DCE":  # 豆粕期货
            start_dt_str = START_DATE.replace('-', '')
            end_dt_str = END_DATE.replace('-', '')
            df = ak.futures_main_sina(symbol="M0", start_date=start_dt_str, end_date=end_dt_str)
            if not df.empty:
                df['date'] = pd.to_datetime(df['日期'])
                df = df.set_index('date').sort_index()
                price = df['收盘价'].astype(float)

        elif ticker == "RB.SHF":  # 螺纹钢期货
            start_dt_str = START_DATE.replace('-', '')
            end_dt_str = END_DATE.replace('-', '')
            df = ak.futures_main_sina(symbol="RB0", start_date=start_dt_str, end_date=end_dt_str)
            if not df.empty:
                df['date'] = pd.to_datetime(df['日期'])
                df = df.set_index('date').sort_index()
                price = df['收盘价'].astype(float)

    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)

    if price is None or df.empty:
        print(f"Failed to fetch or process data for {ticker}. DataFrame was empty or price could not be determined.")
        return pd.Series(dtype=float, name=ticker)

    price = price.rename(ticker)
    # Persist cache
    price.to_pickle(cache_file)
    return price


def get_all_prices(tickers: List[str]) -> pd.DataFrame:
    series = [fetch_price_series(t) for t in tickers]
    prices = pd.concat(series, axis=1).sort_index()
    return prices


# ---------------------------------------------------------------------------
# Risk calculations
# ---------------------------------------------------------------------------

def ewma_downside_cov(ret_window: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """Compute EWMA semi-covariance matrix (downside only).

    Parameters
    ----------
    ret_window : 2-D ndarray (T, N)
        Historical returns (row: time, col: assets) for the rolling window.
    lam : float
        Decay factor (0 < lam < 1). Higher = more weight on recent obs.
    """
    # Keep only negative returns; positives -> 0
    R = np.minimum(ret_window, 0.0)

    T = R.shape[0]
    weights = lam ** np.arange(T - 1, -1, -1)
    weights = weights / weights.sum()

    # Weighted demeaned returns (mean is not zero b/c we truncated)
    R_w = R * np.sqrt(weights[:, None])
    cov = R_w.T @ R_w  # (N, N)
    return cov


def risk_parity_weights(sigma: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Simple iterative risk-parity (equal risk budget) solver.

    sigma : (N,N) positive-definite covariance matrix"""
    n = sigma.shape[0]
    w = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        marginal = sigma @ w  # (N,)
        rc = w * marginal     # risk contributions
        avg_rc = rc.mean()
        diff = rc - avg_rc
        if np.linalg.norm(diff) < tol:
            break
        # Update rule (multiplicative)
        w *= avg_rc / rc
        # Sanitise & renormalise
        w = np.clip(w, 1e-8, 1.0)
        w /= w.sum()
    return w

# ---------------------------------------------------------------------------
# Back-test engine
# ---------------------------------------------------------------------------

def backtest() -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        portfolio_returns: 月度收益序列
        asset_weights: 资产权重历史
        quadrant_weights: 象限权重历史
        quadrant_returns: 象限收益历史
    """
    all_tickers = list(ASSET_NAMES.keys())
    prices_daily = get_all_prices(all_tickers)

    # Restrict sample window and forward-fill (ETFs may start trading later).
    prices_daily = prices_daily.loc[START_DATE:END_DATE].ffill()

    # Resample to specified frequency last price & compute simple returns.
    prices_m = prices_daily.resample(REBALANCE_FREQ).last()
    rets = prices_m.pct_change().dropna(how="all")

    # Build quadrant returns
    quad_ret = pd.DataFrame(index=rets.index)
    for q_name, class_weights in QUADRANTS.items():
        q_return = pd.Series(0.0, index=rets.index)
        for class_name, weight in class_weights.items():
            class_assets = ASSET_CLASSES[class_name]
            
            # Check for available return data for the assets in the class
            available_assets = [a for a in class_assets if a in rets.columns and not rets[a].isnull().all()]
            if not available_assets:
                continue

            # Equal-weight assets within the class
            class_return = rets[available_assets].mean(axis=1)
            q_return += class_return.fillna(0) * weight
        quad_ret[q_name] = q_return

    # Ensure consistent numeric np.ndarray views
    quad_ret.dropna(how='all', inplace=True)
    # Align returns dataframe to the new quadrant return index
    rets = rets.loc[quad_ret.index]


    # Prepare result containers
    port_ret = pd.Series(dtype=float, index=quad_ret.index)
    asset_weights_hist = pd.DataFrame(index=quad_ret.index, columns=rets.columns)
    quadrant_weights_hist = pd.DataFrame(index=quad_ret.index, columns=quad_ret.columns)

    prev_asset_w = pd.Series(0.0, index=rets.columns)

    start_idx = WINDOW_MONTHS
    if start_idx >= len(quad_ret) -1:
        raise ValueError(
            "Not enough data for backtest window. "
            f"Need at least {WINDOW_MONTHS + 2} months of data, but got {len(quad_ret)}."
        )


    print(f"Starting backtest from {quad_ret.index[WINDOW_MONTHS]} to {quad_ret.index[-2]}...")

    # Iterate until the *second-to-last* observation so that t_idx+1 exists.
    for t_idx in range(start_idx, len(quad_ret) - 1):
        date = quad_ret.index[t_idx]
        window = quad_ret.iloc[t_idx - WINDOW_MONTHS: t_idx]
        
        # Skip if window has NaNs, which can happen with new assets
        if window.isnull().values.any():
            continue

        sigma = ewma_downside_cov(window.to_numpy())
        q_weights = risk_parity_weights(sigma)  # ndarray
        q_weights = pd.Series(q_weights, index=quad_ret.columns, name=date)

        # Map quadrant weights back to individual assets
        asset_w = pd.Series(0.0, index=rets.columns)
        for q_name, q_weight in q_weights.items():
            class_weights = QUADRANTS[q_name]
            for class_name, c_weight in class_weights.items():
                class_assets = ASSET_CLASSES[class_name]
                # Distribute quadrant weight to assets
                num_assets_in_class = len(class_assets)
                if num_assets_in_class > 0:
                    per_asset_weight = q_weight * c_weight / num_assets_in_class
                    asset_w[class_assets] += per_asset_weight
        
        asset_w /= asset_w.sum()  # renormalise to 1.0

        # Transaction cost
        turnover = (asset_w - prev_asset_w).abs().sum()
        cost = turnover * FEE_RATE

        # Realised portfolio return is *next* month's asset returns (t_idx + 1)
        ret_vec = rets.iloc[t_idx + 1].fillna(0) # Fill NaN returns with 0
        gross = (asset_w * ret_vec).sum()
        net = gross - cost
        port_ret.iloc[t_idx + 1] = net  # align with realised period

        # Store weights
        asset_weights_hist.iloc[t_idx] = asset_w
        quadrant_weights_hist.iloc[t_idx] = q_weights

        prev_asset_w = asset_w

    port_ret.dropna(inplace=True)
    asset_weights_hist.dropna(how='all', inplace=True)
    quadrant_weights_hist.dropna(how='all', inplace=True)

    return port_ret, asset_weights_hist, quadrant_weights_hist, quad_ret


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def performance_summary(ret: pd.Series) -> pd.Series:
    cum = (1 + ret).cumprod()
    # 根据调仓频率确定年化因子
    freq_to_factor = {"W": 52, "ME": 12, "QE": 4, "YE": 1}
    ann_factor = freq_to_factor.get(REBALANCE_FREQ, 12)
    cagr = cum.iloc[-1] ** (ann_factor / len(ret)) - 1
    vol = ret.std() * math.sqrt(ann_factor)
    sharpe = cagr / vol if vol else np.nan
    # Max drawdown
    roll_max = cum.cummax()
    mdd = (cum / roll_max - 1).min()
    
    # 计算更多指标
    positive_months = (ret > 0).sum()
    negative_months = (ret < 0).sum()
    win_rate = positive_months / len(ret)
    
    # 计算月度最大收益和最大亏损
    max_monthly_gain = ret.max()
    max_monthly_loss = ret.min()
    
    stats = pd.Series({
        "年化收益率": f"{cagr:.2%}",
        "年化波动率": f"{vol:.2%}",
        "夏普比率": f"{sharpe:.3f}",
        "最大回撤": f"{mdd:.2%}",
        "胜率": f"{win_rate:.2%}",
        "正收益月数": positive_months,
        "负收益月数": negative_months,
        "月度最大收益": f"{max_monthly_gain:.2%}",
        "月度最大亏损": f"{max_monthly_loss:.2%}",
        "总月数": len(ret),
    })
    return stats


def calculate_annual_performance(returns: pd.Series) -> pd.DataFrame:
    """计算年度绩效表现"""
    annual_data = []
    freq_to_factor = {"W": 52, "ME": 12, "QE": 4, "YE": 1}
    ann_factor = freq_to_factor.get(REBALANCE_FREQ, 12)
    
    for year in range(returns.index[0].year, returns.index[-1].year + 1):
        year_returns = returns[returns.index.year == year]
        if len(year_returns) > 0:
            year_cum = (1 + year_returns).cumprod().iloc[-1] - 1
            year_vol = year_returns.std() * math.sqrt(ann_factor)
            year_sharpe = year_cum / year_vol if year_vol > 0 else np.nan
            year_mdd = calculate_max_drawdown(year_returns)
            
            annual_data.append({
                "年份": year,
                "年度收益率": f"{year_cum:.2%}",
                "年度波动率": f"{year_vol:.2%}",
                "夏普比率": f"{year_sharpe:.3f}",
                "最大回撤": f"{year_mdd:.2%}",
            })
    
    return pd.DataFrame(annual_data)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum / roll_max - 1).min()
    return drawdown


def save_results(port_ret: pd.Series, asset_weights: pd.DataFrame,
                quadrant_weights: pd.DataFrame, quadrant_returns: pd.DataFrame):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存投资组合收益
    port_nav = (1 + port_ret).cumprod()
    results_df = pd.DataFrame({
        "日期": port_ret.index,
        "月度收益率": port_ret.values,
        "净值": port_nav.values,
    })
    results_df.to_csv(OUTPUT_DIR / f"portfolio_performance_{timestamp}.csv",
                     index=False, encoding='utf-8-sig')

    # 保存资产权重历史（添加中文名称）
    asset_weights_cn = asset_weights.copy()
    asset_weights_cn.columns = [f"{col}_{ASSET_NAMES.get(col, 'Unknown')}" for col in asset_weights_cn.columns]
    asset_weights_cn.to_csv(OUTPUT_DIR / f"asset_weights_{timestamp}.csv",
                           encoding='utf-8-sig')

    # 保存象限权重历史
    quadrant_weights.to_csv(OUTPUT_DIR / f"quadrant_weights_{timestamp}.csv",
                              encoding='utf-8-sig')

    # 保存象限收益历史
    quadrant_returns.to_csv(OUTPUT_DIR / f"quadrant_returns_{timestamp}.csv",
                              encoding='utf-8-sig')

    print(f"\n结果已保存到 output/ 目录，时间戳: {timestamp}")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("中国版全天候策略回测系统")
    print("=" * 60)
    print("Fetching data & running back-test … this may take a while on first run.")
    
    # 运行回测
    port_ret, asset_weights, quadrant_weights, quadrant_returns = backtest()
    
    # 计算绩效统计
    stats = performance_summary(port_ret)
    annual_perf = calculate_annual_performance(port_ret)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("投资组合整体表现")
    print("=" * 60)
    print(stats.to_string())
    
    print("\n" + "=" * 60)
    print("年度表现明细")
    print("=" * 60)
    print(annual_perf.to_string(index=False))
    
    # 显示最新权重分配
    print("\n" + "=" * 60)
    print("最新资产权重分配")
    print("=" * 60)
    latest_weights = asset_weights.iloc[-1]
    for ticker, weight in latest_weights.items():
        if weight > 1e-6: # Only show assets with meaningful weight
            print(f"{ASSET_NAMES.get(ticker, ticker):12} ({ticker}): {weight:.2%}")

    print("\n" + "=" * 60)
    print("最新象限权重分配")
    print("=" * 60)
    latest_quad_weights = quadrant_weights.iloc[-1]
    for quad, weight in latest_quad_weights.items():
        print(f"{quad:20}: {weight:.2%}")
    
    # 保存结果
    save_results(port_ret, asset_weights, quadrant_weights, quadrant_returns)
    
    # 绘制图表
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib import rcParams
        
        # 设置中文字体
        rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 净值曲线
        cum_curve = (1 + port_ret).cumprod()
        axes[0, 0].plot(cum_curve.index, cum_curve.values, linewidth=2, label='全天候策略')
        axes[0, 0].set_title('策略净值曲线', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('净值')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 回撤曲线
        roll_max = cum_curve.cummax()
        drawdown = (cum_curve / roll_max - 1) * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title('回撤曲线', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('回撤 (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 象限权重变化
        for quad in quadrant_weights.columns:
            axes[1, 0].plot(quadrant_weights.index, quadrant_weights[quad],
                           label=quad, linewidth=1.5)
        axes[1, 0].set_title('象限权重变化', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('权重')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 月度收益分布
        axes[1, 1].hist(port_ret * 100, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(port_ret.mean() * 100, color='red', linestyle='--', 
                          label=f'平均收益: {port_ret.mean():.2%}')
        axes[1, 1].set_title('月度收益分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('月度收益率 (%)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("matplotlib not installed; skipping plot. Install via 'pip install matplotlib'.")
    
    print(f"\n回测完成！数据范围: {port_ret.index[0].strftime('%Y-%m')} 至 {port_ret.index[-1].strftime('%Y-%m')}")


if __name__ == "__main__":
    main() 