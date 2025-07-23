"""
All-Weather Strategy (中国版全天候策略)
========================================
This script constructs and back-tests a Chinese All-Weather portfolio inspired by Bridgewater's All-Weather
strategy and the factor-risk-parity framework.

Key Characteristics
-------------------
1. Macro Quadrant framework (Growth ↑ / ↓, Inflation ↑ / ↓)
2. Quadrant portfolios are equal-weighted baskets of domestic ETFs.
3. Risk for each quadrant portfolio is measured via the exponentially weighted
   moving average (EWMA) semi-covariance matrix (downside risk only).
4. Monthly re-balancing to achieve risk parity **across quadrants** (not across
   single assets). Inside each quadrant, assets remain equal-weighted.
5. No ex-ante leverage (notional weights sum to 1). Transaction fee is 5 bps
   (单边万分之五) per trade.
6. Back-test window: 2013-12-31 – 2025-04-30 (inclusive, monthly).

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
START_DATE = "2011-01-01"
END_DATE = "2025-07-16"
FEE_RATE = 0.0001  # 单边万分之五
EWMA_LAMBDA = 0.96  # decay parameter for EWMA
WINDOW_MONTHS = 36  # rolling window for risk estimation
REBALANCE_FREQ = "ME"  # 调仓频率: "W"=周末, "ME"=月末, "QE"=季末, "YE"=年末

# Mapping from Bloomberg-like ticker to akshare symbol ---------------------------------
TICKER_MAP: Dict[str, str] = {
    "510300.SH": "sh510300",  # 沪深 300 ETF 股票
    "512100.SH": "sh512100",  # 中证 1000 ETF 股票
    "512890.SH": "sh512890",  # 红利低波 ETF 红利
    "511260.SH": "sh511260",  # 10Y 国债 ETF 债券
    "511090.SH": "sh511090",  # 30Y 国债 ETF 债券
    "159980.SZ": "sz159980",  # 有色 ETF 商品
    "159981.SZ": "sz159981",  # 能化 ETF 商品
    "159985.SZ": "sz159985",  # 豆粕 ETF 商品
    "518880.SH": "sh518880",  # 黄金 ETF 商品
}

# ETF名称映射
ETF_NAMES: Dict[str, str] = {
    "510300.SH": "沪深300ETF",
    "512100.SH": "中证1000ETF", 
    "512890.SH": "红利低波ETF",
    "511260.SH": "10年国债ETF",
    "511090.SH": "30年国债ETF",
    "159980.SZ": "有色ETF",
    "159981.SZ": "能化ETF", 
    "159985.SZ": "豆粕ETF",
    "518880.SH": "黄金ETF",
}

# 资产类别定义
ASSET_CLASSES: Dict[str, List[str]] = {
    "股票": ["510300.SH", "512100.SH"],           # 沪深300ETF, 中证1000ETF
    "债券": ["511260.SH", "511090.SH"],           # 10年国债ETF, 30年国债ETF
    "商品": ["159980.SZ", "159981.SZ", "159985.SZ"], # 有色ETF, 能化ETF, 豆粕ETF
    "黄金": ["518880.SH"],                       # 黄金ETF
    "红利": ["512890.SH"],                       # 红利低波ETF
}

# 象限内大类资产配比定义
QUADRANT_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    "Growth_Up": {      # 股票和商品1:1
        "股票": 0.5,
        "商品": 0.5,
    },
    "Inflation_Up": {   # 商品和黄金1:1
        "商品": 0.5,
        "黄金": 0.5,
    },
    "Growth_Down": {    # 债券、黄金、红利1:1:1
        "债券": 1/3,
        "黄金": 1/3,
        "红利": 1/3,
    },
    "Inflation_Down": { # 债券和黄金1:1
        "债券": 0.5,
        "黄金": 0.5,
    },
}

# 从配比生成每个象限的ETF列表（保持兼容性）
QUADRANTS: Dict[str, List[str]] = {}
for quadrant, allocations in QUADRANT_ALLOCATIONS.items():
    etf_list = []
    for asset_class in allocations.keys():
        etf_list.extend(ASSET_CLASSES[asset_class])
    QUADRANTS[quadrant] = etf_list

# 象限中文名称
QUADRANT_NAMES: Dict[str, str] = {
    "Growth_Up": "经济增长超预期",
    "Inflation_Up": "通胀超预期", 
    "Growth_Down": "经济增长不及预期",
    "Inflation_Down": "通胀不及预期",
}

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _cache_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}.pkl"


def fetch_price_series(ticker: str) -> pd.Series:
    """Fetch daily *adjusted* close price for a single ETF.

    Data are cached locally (pickle)."""
    ak_symbol = TICKER_MAP[ticker]
    cache_file = _cache_path(ak_symbol)
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    print(f"Fetching data for {ticker} ({ETF_NAMES[ticker]})...")
    df = ak.fund_etf_hist_sina(symbol=ak_symbol)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    # Use *close*; adjust if necessary. Sina data are already adjusted.
    price = df["close"].astype(float).rename(ticker)
    price.sort_index(inplace=True)

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
    tic_list = list(TICKER_MAP)
    prices_daily = get_all_prices(tic_list)

    # Restrict sample window and forward-fill (ETFs may start trading later).
    prices_daily = prices_daily.loc[START_DATE:END_DATE].ffill()

    # Resample to specified frequency last price & compute simple returns.
    prices_m = prices_daily.resample(REBALANCE_FREQ).last()
    rets = prices_m.pct_change().dropna(how="all")

    # 分析数据可用性（动态逻辑，支持部分ETF缺失）
    print("启用动态资产类别权重调整逻辑...")

    # Build quadrant returns (按大类资产配比计算，允许跳过缺失的资产类别)
    quad_ret = pd.DataFrame(index=rets.index)
    
    for q, allocations in QUADRANT_ALLOCATIONS.items():
        quadrant_return = pd.Series(index=rets.index, dtype=float)
        
        print(f"\n计算 {q} ({QUADRANT_NAMES[q]}) 象限收益:")
        
        for date_idx in rets.index:
            available_classes = {}
            total_available_weight = 0.0
            
            # 检查每个资产类别在当前时间点是否有数据
            for asset_class, weight in allocations.items():
                asset_etfs = ASSET_CLASSES[asset_class]
                # 检查该类别在当前时间点是否有至少一个ETF可用
                available_etfs = [etf for etf in asset_etfs if etf in rets.columns and not pd.isna(rets.loc[date_idx, etf])]
                
                if available_etfs:  # 只要有至少一个ETF可用就认为该类别可用
                    available_classes[asset_class] = weight
                    total_available_weight += weight
            
            # 重新归一化权重并计算当前时间点的象限收益
            if total_available_weight > 0:
                date_return = 0.0
                for asset_class, original_weight in available_classes.items():
                    # 归一化权重
                    normalized_weight = original_weight / total_available_weight
                    # 计算该资产类别的收益（使用该类别内可用ETF的平均值）
                    asset_etfs = ASSET_CLASSES[asset_class]
                    available_etfs = [etf for etf in asset_etfs if etf in rets.columns and not pd.isna(rets.loc[date_idx, etf])]
                    asset_return = rets.loc[date_idx, available_etfs].mean()
                    date_return += normalized_weight * asset_return
                
                quadrant_return.loc[date_idx] = date_return
            else:
                quadrant_return.loc[date_idx] = np.nan
        
        # 显示该象限的配比情况
        print(f"  配比: {' + '.join([f'{cls} {weight:.1%}' for cls, weight in allocations.items()])}")
        
        quad_ret[q] = quadrant_return

    # Ensure consistent numeric np.ndarray views
    quad_ret.dropna(inplace=True)

    # Prepare result containers
    port_ret = pd.Series(dtype=float, index=quad_ret.index)
    asset_weights_hist = pd.DataFrame(index=quad_ret.index, columns=rets.columns)
    quadrant_weights_hist = pd.DataFrame(index=quad_ret.index, columns=quad_ret.columns)

    prev_asset_w = pd.Series(0.0, index=rets.columns)

    print(f"Starting backtest from {quad_ret.index[WINDOW_MONTHS]} to {quad_ret.index[-2]}...")
    
    # Iterate until the *second-to-last* observation so that t_idx+1 exists.
    for t_idx in range(WINDOW_MONTHS, len(quad_ret) - 1):
        date = quad_ret.index[t_idx]
        window = quad_ret.iloc[t_idx - WINDOW_MONTHS: t_idx]

        sigma = ewma_downside_cov(window.to_numpy())
        q_weights = risk_parity_weights(sigma)  # ndarray
        q_weights = pd.Series(q_weights, index=quad_ret.columns, name=date)

        # Map quadrant weights back to individual assets (按大类资产配比，跳过缺失数据)
        asset_w = pd.Series(0.0, index=rets.columns)
        next_date = rets.index[t_idx + 1]  # 下个月的数据
        
        for q, allocations in QUADRANT_ALLOCATIONS.items():
            quadrant_weight = q_weights[q]
            
            # 检查下个时间点该象限有哪些资产类别可用
            available_classes = {}
            total_available_weight = 0.0
            
            for asset_class, class_weight in allocations.items():
                asset_etfs = ASSET_CLASSES[asset_class]
                # 检查该类别在下个时间点是否有至少一个ETF可用
                available_etfs = [etf for etf in asset_etfs if etf in rets.columns and not pd.isna(rets.loc[next_date, etf])]
                
                if available_etfs:  # 只要有至少一个ETF可用就认为该类别可用
                    available_classes[asset_class] = class_weight
                    total_available_weight += class_weight
            
            # 重新归一化权重并分配到具体ETF
            if total_available_weight > 0:
                for asset_class, original_weight in available_classes.items():
                    # 归一化权重
                    normalized_weight = original_weight / total_available_weight
                    # 该大类资产在组合中的权重
                    class_total_weight = quadrant_weight * normalized_weight
                    # 该大类资产内的ETF等权重分配（只分配给可用的ETF）
                    asset_etfs = ASSET_CLASSES[asset_class]
                    available_etfs = [etf for etf in asset_etfs if etf in rets.columns and not pd.isna(rets.loc[next_date, etf])]
                    for etf in available_etfs:
                        asset_w[etf] += class_total_weight / len(available_etfs)
                        
        asset_w /= asset_w.sum()  # renormalise to 1.0

        # Transaction cost
        turnover = (asset_w - prev_asset_w).abs().sum()
        cost = turnover * FEE_RATE

        # Realised portfolio return is *next* month's asset returns (t_idx + 1)
        ret_vec = rets.iloc[t_idx + 1]
        gross = (asset_w * ret_vec).sum()
        net = gross - cost
        port_ret.iloc[t_idx + 1] = net  # align with realised period

        # Store weights
        asset_weights_hist.iloc[t_idx] = asset_w
        quadrant_weights_hist.iloc[t_idx] = q_weights
        
        prev_asset_w = asset_w

    port_ret.dropna(inplace=True)
    asset_weights_hist.dropna(inplace=True)
    quadrant_weights_hist.dropna(inplace=True)
    
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

    # 获取日度数据
    tic_list = list(TICKER_MAP)
    prices_daily = get_all_prices(tic_list)
    prices_daily = prices_daily.loc[pd.to_datetime(START_DATE):pd.to_datetime(END_DATE)].ffill()
    
    # 计算日度资产权重（通过月度权重forward fill得到）
    asset_weights_daily = asset_weights.reindex(prices_daily.index, method='ffill')
    
    # 计算日度收益和净值
    daily_returns = (prices_daily.pct_change() * asset_weights_daily).sum(axis=1)
    daily_nav = (1 + daily_returns).cumprod()

    # 保存日度数据
    daily_results_df = pd.DataFrame({
        "日期": daily_returns.index,
        "日度收益率": daily_returns.values,
        "净值": daily_nav.values,
    })
    daily_results_df.to_csv(OUTPUT_DIR / f"portfolio_daily_performance_{timestamp}.csv",
                     index=False, encoding='utf-8-sig')

    # 保存月度数据
    monthly_results_df = pd.DataFrame({
        "日期": port_ret.index,
        "月度收益率": port_ret.values,
        "月度净值": (1 + port_ret).cumprod().values,
    })
    monthly_results_df.to_csv(OUTPUT_DIR / f"portfolio_monthly_performance_{timestamp}.csv",
                     index=False, encoding='utf-8-sig')
    
    # 保存资产权重历史（添加中文名称）
    asset_weights_cn = asset_weights.copy()
    asset_weights_cn.columns = [f"{col}_{ETF_NAMES[col]}" for col in asset_weights_cn.columns]
    asset_weights_cn.to_csv(OUTPUT_DIR / f"asset_weights_{timestamp}.csv", 
                           encoding='utf-8-sig')
    
    # 保存象限权重历史（添加中文名称）
    quadrant_weights_cn = quadrant_weights.copy()
    quadrant_weights_cn.columns = [f"{col}_{QUADRANT_NAMES[col]}" for col in quadrant_weights_cn.columns]
    quadrant_weights_cn.to_csv(OUTPUT_DIR / f"quadrant_weights_{timestamp}.csv", 
                              encoding='utf-8-sig')
    
    # 保存象限收益历史
    quadrant_returns_cn = quadrant_returns.copy()
    quadrant_returns_cn.columns = [f"{col}_{QUADRANT_NAMES[col]}" for col in quadrant_returns_cn.columns]
    quadrant_returns_cn.to_csv(OUTPUT_DIR / f"quadrant_returns_{timestamp}.csv", 
                              encoding='utf-8-sig')
    
    print(f"\n结果已保存到 output/ 目录，时间戳: {timestamp}")
    print(f"- 日度数据：portfolio_daily_performance_{timestamp}.csv")
    print(f"- 月度数据：portfolio_monthly_performance_{timestamp}.csv")
    print(f"- 资产权重：asset_weights_{timestamp}.csv")
    print(f"- 象限权重：quadrant_weights_{timestamp}.csv")
    print(f"- 象限收益：quadrant_returns_{timestamp}.csv")


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
        print(f"{ETF_NAMES[ticker]:12} ({ticker}): {weight:.2%}")
    
    print("\n" + "=" * 60)
    print("最新象限权重分配")
    print("=" * 60)
    latest_quad_weights = quadrant_weights.iloc[-1]
    for quad, weight in latest_quad_weights.items():
        print(f"{QUADRANT_NAMES[quad]:12}: {weight:.2%}")
    
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
                           label=QUADRANT_NAMES[quad], linewidth=1.5)
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