"""
全天候策略参数优化器
===================
This script optimizes the parameters of the All-Weather strategy using grid search.

Parameters to optimize:
- EWMA_LAMBDA: 0.88, 0.90, 0.92, 0.94, 0.96
- WINDOW_MONTHS: 24, 30, 36, 42, 48
- FEE_RATE: 0.0001, 0.0003, 0.0005, 0.0010
- REBALANCE_FREQ: "W", "ME", "QE"

Optimization objectives:
- Sharpe Ratio (primary)
- CAGR / Max Drawdown ratio (secondary)
- Total Return (tertiary)

Usage:
------
python parameter_optimization.py
"""

import itertools
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

# Import the strategy components
import sys
sys.path.append(str(Path(__file__).parent))

# We'll import the functions we need from all_weather.py
from all_weather import (
    get_all_prices, TICKER_MAP, QUADRANTS, ewma_downside_cov, 
    risk_parity_weights, calculate_max_drawdown
)

# ---------------------------------------------------------------------------
# Parameter Optimization Configuration
# ---------------------------------------------------------------------------

# Parameters to optimize
PARAM_GRID = {
    "EWMA_LAMBDA": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96],
    "WINDOW_MONTHS": [6, 12, 24, 30, 36, 42, 48],
    "FEE_RATE": [0.0001,],
    "REBALANCE_FREQ": ["W", "ME",],
}

# Fixed parameters
START_DATE = "2015-12-31"
END_DATE = "2025-06-15"

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent / "optimization_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Strategy Backtesting for Optimization
# ---------------------------------------------------------------------------

def backtest_with_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run backtest with specific parameter set
    
    Returns:
        Dictionary with performance metrics and parameters
    """
    try:
        # Extract parameters
        ewma_lambda = params["EWMA_LAMBDA"]
        window_months = params["WINDOW_MONTHS"]
        fee_rate = params["FEE_RATE"]
        rebalance_freq = params["REBALANCE_FREQ"]
        
        # Get price data (cached from previous runs)
        tic_list = list(TICKER_MAP.keys())
        prices_daily = get_all_prices(tic_list)
        prices_daily = prices_daily.loc[START_DATE:END_DATE].ffill()
        
        # Resample to specified frequency
        prices_m = prices_daily.resample(rebalance_freq).last()
        rets = prices_m.pct_change().dropna(how="all")
        
        # Build quadrant returns
        quad_ret = pd.DataFrame(index=rets.index)
        for q, members in QUADRANTS.items():
            quad_ret[q] = rets[members].mean(axis=1)
        quad_ret.dropna(inplace=True)
        
        # Check if we have enough data
        if len(quad_ret) <= window_months + 2:
            return {**params, "error": "insufficient_data"}
        
        # Prepare result containers
        port_ret = pd.Series(dtype=float, index=quad_ret.index)
        prev_asset_w = pd.Series(0.0, index=rets.columns)
        
        # Backtest loop
        for t_idx in range(window_months, len(quad_ret) - 1):
            window = quad_ret.iloc[t_idx - window_months: t_idx]
            
            # Risk calculation with current parameters
            sigma = ewma_downside_cov(window.to_numpy(), lam=ewma_lambda)
            q_weights = risk_parity_weights(sigma)
            q_weights = pd.Series(q_weights, index=quad_ret.columns)
            
            # Map to asset weights
            asset_w = pd.Series(0.0, index=rets.columns)
            for q, members in QUADRANTS.items():
                asset_w[members] += q_weights[q] / len(members)
            asset_w /= asset_w.sum()
            
            # Transaction cost
            turnover = (asset_w - prev_asset_w).abs().sum()
            cost = turnover * fee_rate
            
            # Portfolio return
            ret_vec = rets.iloc[t_idx + 1]
            gross = (asset_w * ret_vec).sum()
            net = gross - cost
            port_ret.iloc[t_idx + 1] = net
            
            prev_asset_w = asset_w
        
        port_ret.dropna(inplace=True)
        
        # Calculate performance metrics
        if len(port_ret) == 0:
            return {**params, "error": "no_returns"}
            
        cum_ret = (1 + port_ret).cumprod()
        
        # Annualization factor
        freq_to_factor = {"W": 52, "ME": 12, "QE": 4, "YE": 1}
        ann_factor = freq_to_factor.get(rebalance_freq, 12)
        
        total_return = cum_ret.iloc[-1] - 1
        cagr = cum_ret.iloc[-1] ** (ann_factor / len(port_ret)) - 1
        vol = port_ret.std() * np.sqrt(ann_factor)
        sharpe = cagr / vol if vol > 0 else 0
        max_dd = calculate_max_drawdown(port_ret)
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0
        
        # Win rate
        win_rate = (port_ret > 0).mean()
        
        return {
            **params,
            "total_return": total_return,
            "cagr": cagr,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "num_periods": len(port_ret),
            "error": None
        }
        
    except Exception as e:
        return {**params, "error": str(e)}


def optimize_parameters(max_workers: int = None) -> pd.DataFrame:
    """
    Run parameter optimization using multiprocessing
    """
    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    print(f"Parameters: {param_names}")
    
    # Convert to list of dictionaries
    param_dicts = []
    for combo in param_combinations:
        param_dict = dict(zip(param_names, combo))
        param_dicts.append(param_dict)
    
    # Run optimization
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(param_dicts))
    
    print(f"Using {max_workers} workers for parallel processing...")
    
    start_time = time.time()
    
    with mp.Pool(processes=max_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap(backtest_with_params, param_dicts)):
            results.append(result)
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{len(param_dicts)} combinations...")
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.1f} seconds")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out errors
    valid_results = results_df[results_df['error'].isna()].copy()
    error_results = results_df[results_df['error'].notna()].copy()
    
    print(f"Valid results: {len(valid_results)}")
    if len(error_results) > 0:
        print(f"Failed combinations: {len(error_results)}")
        print("Error summary:")
        print(error_results['error'].value_counts())
    
    return valid_results


def analyze_results(results_df: pd.DataFrame) -> None:
    """
    Analyze and display optimization results
    """
    if len(results_df) == 0:
        print("No valid results to analyze!")
        return
    
    print("\n" + "="*80)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("="*80)
    
    # Top 10 by Sharpe Ratio
    print("\n📈 Top 10 by Sharpe Ratio:")
    print("-" * 50)
    top_sharpe = results_df.nlargest(10, 'sharpe_ratio')[
        ['EWMA_LAMBDA', 'WINDOW_MONTHS', 'FEE_RATE', 'REBALANCE_FREQ', 
         'sharpe_ratio', 'cagr', 'max_drawdown', 'calmar_ratio']
    ]
    for i, (_, row) in enumerate(top_sharpe.iterrows(), 1):
        print(f"{i:2d}. λ={row['EWMA_LAMBDA']:.2f}, W={row['WINDOW_MONTHS']:2d}, "
              f"Fee={row['FEE_RATE']:.4f}, Freq={row['REBALANCE_FREQ']}, "
              f"Sharpe={row['sharpe_ratio']:.3f}, CAGR={row['cagr']:.2%}, "
              f"MDD={row['max_drawdown']:.2%}")
    
    # Top 10 by Calmar Ratio
    print("\n🛡️  Top 10 by Calmar Ratio (CAGR/MaxDD):")
    print("-" * 50)
    top_calmar = results_df.nlargest(10, 'calmar_ratio')[
        ['EWMA_LAMBDA', 'WINDOW_MONTHS', 'FEE_RATE', 'REBALANCE_FREQ', 
         'calmar_ratio', 'cagr', 'max_drawdown', 'sharpe_ratio']
    ]
    for i, (_, row) in enumerate(top_calmar.iterrows(), 1):
        print(f"{i:2d}. λ={row['EWMA_LAMBDA']:.2f}, W={row['WINDOW_MONTHS']:2d}, "
              f"Fee={row['FEE_RATE']:.4f}, Freq={row['REBALANCE_FREQ']}, "
              f"Calmar={row['calmar_ratio']:.3f}, CAGR={row['cagr']:.2%}, "
              f"MDD={row['max_drawdown']:.2%}")
    
    # Parameter sensitivity analysis
    print("\n🔍 Parameter Sensitivity Analysis:")
    print("-" * 50)
    
    for param in ['EWMA_LAMBDA', 'WINDOW_MONTHS', 'FEE_RATE', 'REBALANCE_FREQ']:
        print(f"\n{param}:")
        param_analysis = results_df.groupby(param).agg({
            'sharpe_ratio': ['mean', 'std', 'max'],
            'cagr': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        print(param_analysis.to_string())
    
    # Best overall parameters
    print("\n🏆 RECOMMENDED PARAMETERS:")
    print("-" * 50)
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    print("Based on highest Sharpe Ratio:")
    print(f"  EWMA_LAMBDA: {best_sharpe['EWMA_LAMBDA']}")
    print(f"  WINDOW_MONTHS: {best_sharpe['WINDOW_MONTHS']}")
    print(f"  FEE_RATE: {best_sharpe['FEE_RATE']}")
    print(f"  REBALANCE_FREQ: {best_sharpe['REBALANCE_FREQ']}")
    print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
    print(f"  CAGR: {best_sharpe['cagr']:.2%}")
    print(f"  Max Drawdown: {best_sharpe['max_drawdown']:.2%}")


def save_results(results_df: pd.DataFrame) -> None:
    """
    Save optimization results to files
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    results_file = OUTPUT_DIR / f"optimization_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Full results saved to: {results_file}")
    
    # Save top performers
    top_results = results_df.nlargest(20, 'sharpe_ratio')
    top_file = OUTPUT_DIR / f"top_20_results_{timestamp}.csv"
    top_results.to_csv(top_file, index=False, encoding='utf-8-sig')
    print(f"📊 Top 20 results saved to: {top_file}")
    
    # Save parameter summary
    summary_data = []
    for param in ['EWMA_LAMBDA', 'WINDOW_MONTHS', 'FEE_RATE', 'REBALANCE_FREQ']:
        param_stats = results_df.groupby(param)['sharpe_ratio'].agg(['mean', 'std', 'max'])
        for value, stats in param_stats.iterrows():
            summary_data.append({
                'Parameter': param,
                'Value': value,
                'Mean_Sharpe': stats['mean'],
                'Std_Sharpe': stats['std'],
                'Max_Sharpe': stats['max']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = OUTPUT_DIR / f"parameter_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"📈 Parameter summary saved to: {summary_file}")


def main():
    """
    Main optimization routine
    """
    print("🚀 Starting All-Weather Strategy Parameter Optimization")
    print("=" * 60)
    
    try:
        # Run optimization
        results_df = optimize_parameters()
        
        if len(results_df) > 0:
            # Analyze results
            analyze_results(results_df)
            
            # Save results
            save_results(results_df)
            
            # Optional: Create visualization
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                from matplotlib import rcParams
                
                # Set Chinese font
                rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
                rcParams['axes.unicode_minus'] = False
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Sharpe ratio distribution
                axes[0, 0].hist(results_df['sharpe_ratio'], bins=30, alpha=0.7)
                axes[0, 0].set_title('夏普比率分布')
                axes[0, 0].set_xlabel('夏普比率')
                axes[0, 0].set_ylabel('频次')
                
                # CAGR vs Max Drawdown
                scatter = axes[0, 1].scatter(results_df['max_drawdown'] * 100, 
                                           results_df['cagr'] * 100, 
                                           c=results_df['sharpe_ratio'], 
                                           cmap='viridis', alpha=0.6)
                axes[0, 1].set_xlabel('最大回撤 (%)')
                axes[0, 1].set_ylabel('年化收益率 (%)')
                axes[0, 1].set_title('收益 vs 风险')
                plt.colorbar(scatter, ax=axes[0, 1], label='夏普比率')
                
                # Parameter impact on Sharpe
                param_sharpe = results_df.groupby('EWMA_LAMBDA')['sharpe_ratio'].mean()
                axes[1, 0].plot(param_sharpe.index, param_sharpe.values, 'o-')
                axes[1, 0].set_title('EWMA Lambda 对夏普比率的影响')
                axes[1, 0].set_xlabel('EWMA Lambda')
                axes[1, 0].set_ylabel('平均夏普比率')
                
                # Window impact
                window_sharpe = results_df.groupby('WINDOW_MONTHS')['sharpe_ratio'].mean()
                axes[1, 1].plot(window_sharpe.index, window_sharpe.values, 'o-')
                axes[1, 1].set_title('窗口长度对夏普比率的影响')
                axes[1, 1].set_xlabel('窗口长度 (月)')
                axes[1, 1].set_ylabel('平均夏普比率')
                
                plt.tight_layout()
                
                # Save plot
                plot_file = OUTPUT_DIR / 'optimization_analysis.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"📊 Analysis plots saved to: {plot_file}")
                
            except ImportError:
                print("📊 matplotlib/seaborn not available for plotting")
        
        else:
            print("❌ No valid results obtained!")
            
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 