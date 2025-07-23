# https://mp.weixin.qq.com/s/f0YqHAT9Wn1qpKb8-4oj_w
# 修改版本：支持每月月初和月中调仓（每月两次调仓）
# 主要修改：
# 1. monthly_rebalancing_backtest函数：支持每月多个调仓日期
# 2. run_monthly_analysis函数：新增rebalance_days参数
# 3. 默认调仓日期设置为[1, 15]，即每月1日和15日
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class EfficientFrontierETF:
    def __init__(self):
        # ETF代码和名称
        # self.etf_codes = {
        #     '159934': '黄金ETF',
        #     '515450': '红利低波50ETF', 
        #     '513100': '纳指ETF',
        #     '161716': '招商双债LOF',
        #     '159985': '豆粕ETF',
        #     '513880': '日经225ETF',
        #     '510300': '沪深300ETF',
        #     '159920': '恒生ETF',
        #     '159740': '恒生科技ETF',
        #     '511090': '30年国债ETF',
        #     '159980': '有色ETF',
        #     '160723': '嘉实原油LOF'
        # }
        self.etf_codes = {
            '159934': '黄金ETF',
            '510880': '红利ETF', 
            '513630': '港股红利指数ETF',
            '513100': '纳指ETF',
            '588000': '科创50ETF',
            '510500': '中证500ETF',
            '513880': '日经225ETF',
            '510300': '沪深300ETF',
            '159920': '恒生ETF',
            '159740': '恒生科技ETF',
            '511090': '30年国债ETF',
            '159980': '有色ETF',
            '160723': '嘉实原油LOF'
        }

        
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def get_etf_data(self, start_date='20200101', end_date=None):
        """
        获取ETF历史数据
        
        注意：
        - 使用后复权价格(hfq)计算收益率，这样能更准确反映真实投资回报
        - 后复权价格已考虑分红、拆分等因素对历史价格的影响
        - 对于投资组合优化和风险评估，使用复权价格是更合理的选择
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        print("正在获取ETF数据...")
        all_data = {}
        
        for code, name in self.etf_codes.items():
            try:
                print(f"获取 {code} {name} 数据...")
                # 获取ETF历史数据（使用后复权价格）
                df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
                if df is not None and len(df) > 0:
                    # 只保留后复权收盘价
                    df = df[['日期', '收盘']].copy()
                    df['日期'] = pd.to_datetime(df['日期'])
                    df.set_index('日期', inplace=True)
                    df.columns = [name]
                    all_data[name] = df
                    print(f"成功获取 {name} 数据，共 {len(df)} 条记录")
                else:
                    print(f"获取 {name} 数据失败")
            except Exception as e:
                print(f"获取 {code} {name} 数据时出错: {e}")
                
        if len(all_data) == 0:
            raise ValueError("没有成功获取任何ETF数据")
            
        # 合并所有数据
        self.data = pd.concat(all_data.values(), axis=1, join='inner')
        print(f"数据合并完成，共 {len(self.data)} 个交易日")
        
        return self.data
    
    def calculate_returns(self):
        """
        计算收益率
        """
        if self.data is None:
            raise ValueError("请先获取数据")
            
        # 计算日收益率
        self.returns = self.data.pct_change().dropna()
        
        # 计算年化均值收益率和协方差矩阵
        self.mean_returns = self.returns.mean() * 252  # 年化
        self.cov_matrix = self.returns.cov() * 252     # 年化
        
        print("收益率计算完成")
        print(f"年化收益率:\n{self.mean_returns}")
        
        return self.returns
    
    def portfolio_performance(self, weights):
        """
        计算投资组合表现
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def calculate_portfolio_returns(self, weights):
        """
        计算投资组合的历史收益率序列
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        return portfolio_returns
    
    def calculate_max_drawdown(self, weights):
        """
        计算最大回撤
        """
        portfolio_returns = self.calculate_portfolio_returns(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # 计算最大回撤
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)
    
    def calculate_calmar_ratio(self, weights):
        """
        计算卡玛比率 (年化收益率 / 最大回撤)
        """
        portfolio_return, _, _ = self.portfolio_performance(weights)
        max_drawdown = self.calculate_max_drawdown(weights)
        
        calmar_ratio = portfolio_return / max_drawdown if max_drawdown > 0 else 0
        return calmar_ratio
    
    def negative_sharpe_ratio(self, weights):
        """
        负夏普比率 (用于最小化)
        """
        _, _, sharpe = self.portfolio_performance(weights)
        return -sharpe
    
    def negative_calmar_ratio(self, weights):
        """
        负卡玛比率 (用于最小化)
        """
        calmar = self.calculate_calmar_ratio(weights)
        return -calmar
    
    def portfolio_volatility(self, weights):
        """
        投资组合波动率
        """
        _, volatility, _ = self.portfolio_performance(weights)
        return volatility
    
    def generate_efficient_frontier(self, num_portfolios=10000):
        """
        生成有效前沿
        """
        if self.returns is None:
            raise ValueError("请先计算收益率")
            
        num_assets = len(self.etf_codes)
        
        # 约束条件：权重和为1，每个权重在0-1之间
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # 存储结果
        results = np.zeros((4, num_portfolios))  # returns, volatility, sharpe, weights
        
        print("正在生成有效前沿...")
        
        # 随机生成投资组合
        for i in range(num_portfolios):
            # 生成随机权重并标准化
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # 计算投资组合表现
            portfolio_return, portfolio_vol, sharpe = self.portfolio_performance(weights)
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_vol
            results[2, i] = sharpe
            
        self.efficient_frontier_data = results
        print("有效前沿生成完成")
        
        return results
    
    def find_optimal_portfolios(self):
        """
        找到三个最优投资组合
        """
        num_assets = len(self.etf_codes)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # 初始猜测 - 等权重
        initial_guess = np.array([1/num_assets] * num_assets)
        
        # 1. 最大夏普比率组合
        print("寻找最大夏普比率组合...")
        max_sharpe_result = minimize(
            self.negative_sharpe_ratio, 
            initial_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        max_sharpe_weights = max_sharpe_result.x
        
        # 2. 最大卡玛比率组合
        print("寻找最大卡玛比率组合...")
        max_calmar_result = minimize(
            self.negative_calmar_ratio, 
            initial_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        max_calmar_weights = max_calmar_result.x
        
        # 3. 控制回撤的高收益组合
        print("寻找控制回撤的高收益组合...")
        # 添加最大回撤约束
        def max_drawdown_constraint(weights):
            return 0.10 - self.calculate_max_drawdown(weights)  # 最大回撤不超过10%
        
        drawdown_constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': max_drawdown_constraint}
        ]
        
        # 最大化收益率（最小化负收益率）
        def negative_return(weights):
            ret, _, _ = self.portfolio_performance(weights)
            return -ret
        
        high_return_result = minimize(
            negative_return, 
            initial_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=drawdown_constraints
        )
        high_return_weights = high_return_result.x
        
        # 如果优化失败，使用备选方案
        if not high_return_result.success:
            print("高收益组合优化失败，使用备选方案...")
            # 尝试不同的初始值
            for _ in range(10):
                random_initial = np.random.random(num_assets)
                random_initial /= np.sum(random_initial)
                
                result = minimize(
                    negative_return, 
                    random_initial, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=drawdown_constraints
                )
                
                if result.success:
                    high_return_weights = result.x
                    break
            else:
                # 如果还是失败，使用风险平价权重
                high_return_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        self.optimal_portfolios = {
            'max_sharpe': max_sharpe_weights,
            'max_calmar': max_calmar_weights,
            'high_return': high_return_weights
        }
        
        return self.optimal_portfolios
    
    def analyze_portfolios(self):
        """
        分析三个最优投资组合
        """
        if not hasattr(self, 'optimal_portfolios'):
            raise ValueError("请先找到最优投资组合")
            
        portfolio_names = ['最大夏普比率组合', '最大卡玛比率组合', '控制回撤高收益组合']
        portfolio_keys = ['max_sharpe', 'max_calmar', 'high_return']
        
        analysis_results = {}
        
        print("\n" + "="*80)
        print("投资组合分析结果")
        print("="*80)
        
        for i, (name, key) in enumerate(zip(portfolio_names, portfolio_keys)):
            weights = self.optimal_portfolios[key]
            
            # 计算各项指标
            portfolio_return, portfolio_vol, sharpe = self.portfolio_performance(weights)
            max_drawdown = self.calculate_max_drawdown(weights)
            calmar_ratio = self.calculate_calmar_ratio(weights)
            
            analysis_results[key] = {
                'name': name,
                'weights': weights,
                'annual_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio
            }
            
            print(f"\n📌 {name}")
            print("-" * 50)
            print(f"年化收益率: {portfolio_return:.2%}")
            print(f"年化波动率: {portfolio_vol:.2%}")
            print(f"夏普比率: {sharpe:.3f}")
            print(f"最大回撤: {max_drawdown:.2%}")
            print(f"卡玛比率: {calmar_ratio:.3f}")
            print("\n资产配置:")
            
            for j, (code, etf_name) in enumerate(self.etf_codes.items()):
                print(f"  {etf_name}: {weights[j]:.1%}")
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def plot_efficient_frontier(self, save_path=None):
        """
        绘制有效前沿图
        """
        if not hasattr(self, 'efficient_frontier_data'):
            raise ValueError("请先生成有效前沿")
            
        plt.figure(figsize=(14, 10))
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 有效前沿散点图
        returns = self.efficient_frontier_data[0]
        volatilities = self.efficient_frontier_data[1]
        sharpe_ratios = self.efficient_frontier_data[2]
        
        scatter = ax1.scatter(volatilities, returns, c=sharpe_ratios, 
                            cmap='viridis', alpha=0.6, s=10)
        ax1.set_xlabel('年化波动率')
        ax1.set_ylabel('年化收益率')
        ax1.set_title('有效前沿')
        plt.colorbar(scatter, ax=ax1, label='夏普比率')
        
        # 标记三个最优组合
        if hasattr(self, 'optimal_portfolios'):
            colors = ['red', 'blue', 'green']
            labels = ['最大夏普比率', '最大卡玛比率', '控制回撤高收益']
            
            for i, (key, color, label) in enumerate(zip(['max_sharpe', 'max_calmar', 'high_return'], 
                                                       colors, labels)):
                weights = self.optimal_portfolios[key]
                ret, vol, _ = self.portfolio_performance(weights)
                ax1.scatter(vol, ret, color=color, s=100, marker='*', 
                           label=label, edgecolors='white', linewidth=2)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 权重分布对比
        if hasattr(self, 'analysis_results'):
            portfolio_names = ['最大夏普比率', '最大卡玛比率', '控制回撤高收益']
            portfolio_keys = ['max_sharpe', 'max_calmar', 'high_return']
            
            # 准备数据
            weights_data = []
            for key in portfolio_keys:
                weights_data.append(self.analysis_results[key]['weights'])
            
            weights_df = pd.DataFrame(weights_data, 
                                    columns=list(self.etf_codes.values()),
                                    index=portfolio_names)
            
            # 绘制堆叠柱状图
            weights_df.plot(kind='bar', stacked=True, ax=ax2, 
                           colormap='Set3', width=0.8)
            ax2.set_title('投资组合权重分布')
            ax2.set_ylabel('权重')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_xticklabels(portfolio_names, rotation=45)
            
        # 子图3: 风险收益散点图
        if hasattr(self, 'analysis_results'):
            returns_list = []
            volatilities_list = []
            names_list = []
            
            for key in portfolio_keys:
                result = self.analysis_results[key]
                returns_list.append(result['annual_return'])
                volatilities_list.append(result['volatility'])
                names_list.append(result['name'])
            
            ax3.scatter(volatilities_list, returns_list, 
                       c=['red', 'blue', 'green'], s=200, alpha=0.7)
            
            for i, name in enumerate(names_list):
                ax3.annotate(name, (volatilities_list[i], returns_list[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax3.set_xlabel('年化波动率')
            ax3.set_ylabel('年化收益率')
            ax3.set_title('最优投资组合对比')
            ax3.grid(True, alpha=0.3)
        
        # 子图4: 关键指标对比
        if hasattr(self, 'analysis_results'):
            metrics = ['夏普比率', '卡玛比率', '最大回撤']
            portfolio_metrics = {name: [] for name in portfolio_names}
            
            for key, name in zip(portfolio_keys, portfolio_names):
                result = self.analysis_results[key]
                portfolio_metrics[name].append(result['sharpe_ratio'])
                portfolio_metrics[name].append(result['calmar_ratio'])
                portfolio_metrics[name].append(result['max_drawdown'])
            
            x = np.arange(len(metrics))
            width = 0.25
            
            for i, (name, values) in enumerate(portfolio_metrics.items()):
                ax4.bar(x + i*width, values, width, label=name, alpha=0.8)
            
            ax4.set_xlabel('指标')
            ax4.set_ylabel('数值')
            ax4.set_title('关键指标对比')
            ax4.set_xticks(x + width)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def run_complete_analysis(self, start_date='20200101', save_results=True):
        """
        运行完整分析流程
        """
        print("开始有效前沿ETF策略分析...")
        print("="*80)
        
        # 1. 获取数据
        self.get_etf_data(start_date=start_date)
        
        # 2. 计算收益率
        self.calculate_returns()
        
        # 3. 生成有效前沿
        self.generate_efficient_frontier()
        
        # 4. 找到最优投资组合
        self.find_optimal_portfolios()
        
        # 5. 分析结果
        self.analyze_portfolios()
        
        # 6. 绘制图表
        if save_results:
            save_path = 'efficient_frontier_analysis.png'
            self.plot_efficient_frontier(save_path=save_path)
        else:
            self.plot_efficient_frontier()
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        
        return self.analysis_results
    
    def split_sample_data(self, in_sample_start='20200101', in_sample_end='20231231', 
                         out_sample_start='20240101', out_sample_end=None):
        """
        分割样本内外数据
        
        Parameters:
        - in_sample_start: 样本内开始日期
        - in_sample_end: 样本内结束日期
        - out_sample_start: 样本外开始日期
        - out_sample_end: 样本外结束日期
        """
        if out_sample_end is None:
            out_sample_end = datetime.now().strftime('%Y%m%d')
        
        print("分割样本内外数据...")
        
        # 获取完整数据
        full_start = min(in_sample_start, out_sample_start)
        self.get_etf_data(start_date=full_start, end_date=out_sample_end)
        
        # 分割数据
        in_sample_start_dt = pd.to_datetime(in_sample_start)
        in_sample_end_dt = pd.to_datetime(in_sample_end)
        out_sample_start_dt = pd.to_datetime(out_sample_start)
        out_sample_end_dt = pd.to_datetime(out_sample_end)
        
        # 样本内数据
        self.in_sample_data = self.data[
            (self.data.index >= in_sample_start_dt) & 
            (self.data.index <= in_sample_end_dt)
        ].copy()
        
        # 样本外数据
        self.out_sample_data = self.data[
            (self.data.index >= out_sample_start_dt) & 
            (self.data.index <= out_sample_end_dt)
        ].copy()
        
        print(f"样本内数据: {len(self.in_sample_data)} 个交易日 ({in_sample_start} - {in_sample_end})")
        print(f"样本外数据: {len(self.out_sample_data)} 个交易日 ({out_sample_start} - {out_sample_end})")
        
        return self.in_sample_data, self.out_sample_data
    
    def train_on_in_sample(self):
        """
        在样本内数据上训练模型，选择最优投资组合
        """
        print("\n" + "="*60)
        print("样本内分析 (2020-2023)")
        print("="*60)
        
        # 使用样本内数据计算收益率
        original_data = self.data
        self.data = self.in_sample_data
        
        # 计算收益率
        self.calculate_returns()
        
        # 生成有效前沿
        self.generate_efficient_frontier()
        
        # 找到最优投资组合
        self.find_optimal_portfolios()
        
        # 分析结果
        in_sample_results = self.analyze_portfolios()
        
        # 保存样本内最优权重
        self.in_sample_optimal_portfolios = self.optimal_portfolios.copy()
        
        # 恢复原始数据
        self.data = original_data
        
        return in_sample_results
    
    def backtest_portfolio(self, weights, data, portfolio_name="投资组合"):
        """
        回测单个投资组合
        """
        # 计算收益率
        returns = data.pct_change().dropna()
        
        # 计算投资组合收益率序列
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # 计算累计收益率
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # 计算各项指标
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 计算最大回撤
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # 计算卡玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # 计算其他指标
        total_return = cumulative_returns.iloc[-1] - 1
        win_rate = (portfolio_returns > 0).mean()
        
        backtest_result = {
            'portfolio_name': portfolio_name,
            'weights': weights,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate
        }
        
        return backtest_result
    
    def out_sample_backtest(self):
        """
        样本外回测
        """
        print("\n" + "="*60)
        print("样本外回测 (2024至今)")
        print("="*60)
        
        if not hasattr(self, 'in_sample_optimal_portfolios'):
            raise ValueError("请先进行样本内分析")
        
        portfolio_names = ['最大夏普比率组合', '最大卡玛比率组合', '控制回撤高收益组合']
        portfolio_keys = ['max_sharpe', 'max_calmar', 'high_return']
        
        self.out_sample_results = {}
        
        for name, key in zip(portfolio_names, portfolio_keys):
            weights = self.in_sample_optimal_portfolios[key]
            
            # 样本外回测
            backtest_result = self.backtest_portfolio(
                weights, self.out_sample_data, name
            )
            
            self.out_sample_results[key] = backtest_result
            
            print(f"\n📌 {name} - 样本外表现")
            print("-" * 45)
            print(f"总收益率: {backtest_result['total_return']:.2%}")
            print(f"年化收益率: {backtest_result['annual_return']:.2%}")
            print(f"年化波动率: {backtest_result['annual_volatility']:.2%}")
            print(f"夏普比率: {backtest_result['sharpe_ratio']:.3f}")
            print(f"最大回撤: {backtest_result['max_drawdown']:.2%}")
            print(f"卡玛比率: {backtest_result['calmar_ratio']:.3f}")
            print(f"胜率: {backtest_result['win_rate']:.1%}")
        
        return self.out_sample_results
    
    def compare_in_out_sample(self):
        """
        对比样本内外表现
        """
        print("\n" + "="*80)
        print("样本内外表现对比")
        print("="*80)
        
        portfolio_names = ['最大夏普比率组合', '最大卡玛比率组合', '控制回撤高收益组合']
        portfolio_keys = ['max_sharpe', 'max_calmar', 'high_return']
        
        comparison_data = []
        
        for name, key in zip(portfolio_names, portfolio_keys):
            # 样本内表现
            in_sample = self.analysis_results[key]
            
            # 样本外表现
            out_sample = self.out_sample_results[key]
            
            comparison_data.append({
                '投资组合': name,
                '样本内年化收益率': f"{in_sample['annual_return']:.2%}",
                '样本外年化收益率': f"{out_sample['annual_return']:.2%}",
                '样本内夏普比率': f"{in_sample['sharpe_ratio']:.3f}",
                '样本外夏普比率': f"{out_sample['sharpe_ratio']:.3f}",
                '样本内最大回撤': f"{in_sample['max_drawdown']:.2%}",
                '样本外最大回撤': f"{out_sample['max_drawdown']:.2%}",
                '样本内卡玛比率': f"{in_sample['calmar_ratio']:.3f}",
                '样本外卡玛比率': f"{out_sample['calmar_ratio']:.3f}"
            })
        
        # 创建对比表格
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_backtest_results(self, save_path=None):
        """
        绘制回测结果图表
        """
        if not hasattr(self, 'out_sample_results'):
            raise ValueError("请先进行样本外回测")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        portfolio_names = ['最大夏普比率组合', '最大卡玛比率组合', '控制回撤高收益组合']
        portfolio_keys = ['max_sharpe', 'max_calmar', 'high_return']
        colors = ['red', 'blue', 'green']
        
        # 子图1: 样本外累计收益率曲线
        ax1 = axes[0, 0]
        for key, name, color in zip(portfolio_keys, portfolio_names, colors):
            cum_returns = self.out_sample_results[key]['cumulative_returns']
            ax1.plot(cum_returns.index, cum_returns, label=name, color=color, linewidth=2)
        
        ax1.set_title('样本外累计收益率曲线')
        ax1.set_ylabel('累计收益率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 样本内外年化收益率对比
        ax2 = axes[0, 1]
        in_sample_returns = [self.analysis_results[key]['annual_return'] for key in portfolio_keys]
        out_sample_returns = [self.out_sample_results[key]['annual_return'] for key in portfolio_keys]
        
        x = np.arange(len(portfolio_names))
        width = 0.35
        
        ax2.bar(x - width/2, in_sample_returns, width, label='样本内', alpha=0.8)
        ax2.bar(x + width/2, out_sample_returns, width, label='样本外', alpha=0.8)
        
        ax2.set_ylabel('年化收益率')
        ax2.set_title('样本内外年化收益率对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name.replace('组合', '') for name in portfolio_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 样本内外夏普比率对比
        ax3 = axes[1, 0]
        in_sample_sharpe = [self.analysis_results[key]['sharpe_ratio'] for key in portfolio_keys]
        out_sample_sharpe = [self.out_sample_results[key]['sharpe_ratio'] for key in portfolio_keys]
        
        ax3.bar(x - width/2, in_sample_sharpe, width, label='样本内', alpha=0.8)
        ax3.bar(x + width/2, out_sample_sharpe, width, label='样本外', alpha=0.8)
        
        ax3.set_ylabel('夏普比率')
        ax3.set_title('样本内外夏普比率对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels([name.replace('组合', '') for name in portfolio_names], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 样本内外最大回撤对比
        ax4 = axes[1, 1]
        in_sample_drawdown = [self.analysis_results[key]['max_drawdown'] for key in portfolio_keys]
        out_sample_drawdown = [self.out_sample_results[key]['max_drawdown'] for key in portfolio_keys]
        
        ax4.bar(x - width/2, in_sample_drawdown, width, label='样本内', alpha=0.8)
        ax4.bar(x + width/2, out_sample_drawdown, width, label='样本外', alpha=0.8)
        
        ax4.set_ylabel('最大回撤')
        ax4.set_title('样本内外最大回撤对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels([name.replace('组合', '') for name in portfolio_names], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"回测图表已保存到: {save_path}")
        
        plt.show()
    
    def run_in_out_sample_analysis(self, in_sample_start='20210101', in_sample_end='20230630',
                                  out_sample_start='20240101', out_sample_end=None, save_results=True):
        """
        运行完整的样本内外分析
        """
        print("开始样本内外回测分析...")
        print("="*80)
        
        # 1. 分割数据
        self.split_sample_data(in_sample_start, in_sample_end, out_sample_start, out_sample_end)
        
        # 2. 样本内分析
        in_sample_results = self.train_on_in_sample()
        
        # 3. 样本外回测
        out_sample_results = self.out_sample_backtest()
        
        # 4. 对比分析
        comparison_df = self.compare_in_out_sample()
        
        # 5. 绘制图表
        if save_results:
            save_path = 'in_out_sample_backtest.png'
            self.plot_backtest_results(save_path=save_path)
        else:
            self.plot_backtest_results()
        
        print("\n" + "="*80)
        print("样本内外分析完成！")
        print("="*80)
        
        return {
            'in_sample_results': in_sample_results,
            'out_sample_results': out_sample_results,
            'comparison': comparison_df
        }
    
    def monthly_rebalancing_backtest(self, start_date='20210101', end_date=None, 
                                   lookback_months=12, rebalance_days=[1, 15]):
        """
        月度调仓回测 - 支持每月多次调仓
        
        Parameters:
        - start_date: 回测开始日期
        - end_date: 回测结束日期
        - lookback_months: 优化时使用的历史数据长度(月)
        - rebalance_days: 每月调仓日期列表，默认[1, 15]表示每月1日和15日调仓
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        print("\n" + "="*80)
        print("月度动态调仓回测分析")
        print("="*80)
        print(f"回测期间: {start_date} - {end_date}")
        print(f"历史数据窗口: {lookback_months} 个月")
        print(f"调仓频率: 每月 {rebalance_days} 日")
        
        # 获取完整数据
        full_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=lookback_months + 2)
        self.get_etf_data(start_date=full_start_date.strftime('%Y%m%d'), end_date=end_date)
        
        # 生成调仓日期
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        rebalance_dates = []
        
        # 确定开始年月
        current_year = start_dt.year
        current_month = start_dt.month
        
        while True:
            # 为当前月份生成所有调仓日期
            for rebalance_day in rebalance_days:
                try:
                    current_date = pd.Timestamp(year=current_year, month=current_month, day=rebalance_day)
                except ValueError:  # 处理某些月份没有对应日期的情况（如2月30日）
                    # 如果指定日期不存在，使用当月最后一天
                    import calendar
                    last_day = calendar.monthrange(current_year, current_month)[1]
                    current_date = pd.Timestamp(year=current_year, month=current_month, 
                                              day=min(rebalance_day, last_day))
                
                # 只考虑在回测期间内的日期
                if current_date >= start_dt and current_date <= end_dt:
                    # 找到最近的交易日
                    search_date = current_date
                    found_trading_day = False
                    
                    # 向前向后各搜索5个工作日
                    for offset in range(-5, 6):
                        candidate_date = search_date + pd.Timedelta(days=offset)
                        if candidate_date in self.data.index and candidate_date <= end_dt:
                            rebalance_dates.append(candidate_date)
                            found_trading_day = True
                            break
                    
                    if not found_trading_day:
                        print(f"警告: 未找到 {current_date.strftime('%Y-%m-%d')} 附近的交易日")
            
            # 移动到下个月
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1
            
            # 检查是否超出结束日期
            if pd.Timestamp(year=current_year, month=current_month, day=1) > end_dt:
                break
        
        # 移除重复日期并排序
        rebalance_dates = sorted(list(set(rebalance_dates)))
        
        print(f"调仓次数: {len(rebalance_dates)} 次 (每月{len(rebalance_days)}次)")
        
        # 存储每次调仓的权重和表现
        self.monthly_weights_history = {
            'dates': [],
            'max_sharpe_weights': [],
            'max_calmar_weights': [], 
            'high_return_weights': []
        }
        
        self.monthly_portfolio_returns = {
            'max_sharpe': [],
            'max_calmar': [],
            'high_return': []
        }
        
        # 逐月进行优化和回测
        for i, rebalance_date in enumerate(rebalance_dates):
            print(f"\n第 {i+1} 次调仓: {rebalance_date.strftime('%Y-%m-%d')}")
            
            # 确定训练数据窗口
            train_start = rebalance_date - pd.DateOffset(months=lookback_months)
            train_end = rebalance_date - pd.Timedelta(days=1)
            
            # 获取训练数据
            train_data = self.data[
                (self.data.index >= train_start) & 
                (self.data.index <= train_end)
            ].copy()
            
            if len(train_data) < 60:  # 至少需要60个交易日
                print(f"训练数据不足，跳过本次调仓")
                continue
            
            # 在训练数据上进行优化
            original_data = self.data
            self.data = train_data
            
            try:
                # 计算收益率和优化
                self.calculate_returns()
                self.find_optimal_portfolios()
                
                # 只有成功优化时才保存权重和日期
                self.monthly_weights_history['dates'].append(rebalance_date)
                self.monthly_weights_history['max_sharpe_weights'].append(
                    self.optimal_portfolios['max_sharpe'].copy()
                )
                self.monthly_weights_history['max_calmar_weights'].append(
                    self.optimal_portfolios['max_calmar'].copy()
                )
                self.monthly_weights_history['high_return_weights'].append(
                    self.optimal_portfolios['high_return'].copy()
                )
                
                print(f"优化完成，训练数据: {len(train_data)} 个交易日")
                
                # 输出各策略的权重分配
                print("\n🔍 本次调仓权重分配：")
                strategies = [
                    ('最大夏普比率组合', self.optimal_portfolios['max_sharpe']),
                    ('最大卡玛比率组合', self.optimal_portfolios['max_calmar']),
                    ('控制回撤高收益组合', self.optimal_portfolios['high_return'])
                ]
                
                for strategy_name, weights in strategies:
                    print(f"\n📊 {strategy_name}:")
                    for j, (etf_code, etf_name) in enumerate(self.etf_codes.items()):
                        if j < len(weights) and weights[j] > 0.001:  # 只显示权重大于0.1%的资产
                            print(f"   {etf_name:15} ({etf_code}): {weights[j]:7.1%}")
                
                print("-" * 60)
                
            except Exception as e:
                print(f"优化失败: {e}")
                # 使用等权重作为备选
                num_assets = len(self.etf_codes)
                equal_weights = np.array([1/num_assets] * num_assets)
                
                self.monthly_weights_history['dates'].append(rebalance_date)
                self.monthly_weights_history['max_sharpe_weights'].append(equal_weights)
                self.monthly_weights_history['max_calmar_weights'].append(equal_weights)
                self.monthly_weights_history['high_return_weights'].append(equal_weights)
                print(f"使用等权重作为备选方案")
                
                # 输出等权重分配
                print("\n🔍 本次调仓权重分配（等权重备选）：")
                print(f"📊 所有策略均采用等权重:")
                for etf_code, etf_name in self.etf_codes.items():
                    print(f"   {etf_name:15} ({etf_code}): {1/num_assets:7.1%}")
                print("-" * 60)
            
            finally:
                self.data = original_data
        
        print(f"\n最终调仓统计:")
        print(f"总调仓尝试次数: {len(rebalance_dates)}")
        print(f"成功调仓次数: {len(self.monthly_weights_history['dates'])}")
        print(f"跳过次数: {len(rebalance_dates) - len(self.monthly_weights_history['dates'])}")
        
        # 计算月度调仓策略的收益率
        self._calculate_monthly_returns(start_dt, end_dt)
        
        # 分析月度调仓结果
        self._analyze_monthly_results()
        
        return self.monthly_backtest_results
    
    def _calculate_monthly_returns(self, start_date, end_date):
        """
        计算月度调仓策略的收益率
        """
        # 获取回测期间的数据
        backtest_data = self.data[
            (self.data.index >= start_date) & 
            (self.data.index <= end_date)
        ].copy()
        
        returns = backtest_data.pct_change().dropna()
        
        # 初始化日度数据存储
        portfolio_daily_returns = {
            'max_sharpe': pd.Series(index=returns.index, dtype=float),
            'max_calmar': pd.Series(index=returns.index, dtype=float),
            'high_return': pd.Series(index=returns.index, dtype=float)
        }
        
        portfolio_cumulative_returns = {
            'max_sharpe': pd.Series(index=returns.index, dtype=float),
            'max_calmar': pd.Series(index=returns.index, dtype=float),
            'high_return': pd.Series(index=returns.index, dtype=float)
        }
        
        # 初始化组合价值
        portfolio_values = {
            'max_sharpe': 1.0,
            'max_calmar': 1.0,
            'high_return': 1.0
        }
        
        current_weights = {
            'max_sharpe': None,
            'max_calmar': None,
            'high_return': None
        }
        
        rebalance_dates = self.monthly_weights_history['dates']
        weight_index = 0
        
        # 确保权重列表长度一致
        min_weights_length = min(
            len(self.monthly_weights_history['max_sharpe_weights']),
            len(self.monthly_weights_history['max_calmar_weights']),
            len(self.monthly_weights_history['high_return_weights']),
            len(rebalance_dates)
        )
        
        for date in returns.index:
            # 检查是否需要调仓
            if (weight_index < min_weights_length and 
                weight_index < len(rebalance_dates) and
                date >= rebalance_dates[weight_index]):
                
                current_weights['max_sharpe'] = self.monthly_weights_history['max_sharpe_weights'][weight_index]
                current_weights['max_calmar'] = self.monthly_weights_history['max_calmar_weights'][weight_index]
                current_weights['high_return'] = self.monthly_weights_history['high_return_weights'][weight_index]
                weight_index += 1
            
            # 计算当日收益率
            if current_weights['max_sharpe'] is not None:
                daily_returns = returns.loc[date]
                
                for strategy in ['max_sharpe', 'max_calmar', 'high_return']:
                    weights = current_weights[strategy]
                    portfolio_return = np.sum(daily_returns * weights)
                    portfolio_values[strategy] *= (1 + portfolio_return)
                    portfolio_daily_returns[strategy].loc[date] = portfolio_return
                    portfolio_cumulative_returns[strategy].loc[date] = portfolio_values[strategy]
        
        # 保存日度和累计收益数据
        self.monthly_portfolio_returns = portfolio_cumulative_returns
        self.daily_portfolio_returns = portfolio_daily_returns
        
        # 计算各项指标
        self.monthly_backtest_results = {}
        
        for strategy in ['max_sharpe', 'max_calmar', 'high_return']:
            cum_returns = portfolio_cumulative_returns[strategy].dropna()
            daily_rets = portfolio_daily_returns[strategy].dropna()
            
            if len(cum_returns) > 0:
                total_return = cum_returns.iloc[-1] - 1
                annual_return = daily_rets.mean() * 252
                annual_volatility = daily_rets.std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                
                # 计算最大回撤
                peak = cum_returns.expanding().max()
                drawdown = (cum_returns - peak) / peak
                max_drawdown = abs(drawdown.min())
                
                calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
                win_rate = (daily_rets > 0).mean()
                
                self.monthly_backtest_results[strategy] = {
                    'cumulative_returns': cum_returns,
                    'daily_returns': daily_rets,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'annual_volatility': annual_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': calmar_ratio,
                    'win_rate': win_rate
                }
    
    def _analyze_monthly_results(self):
        """
        分析月度调仓结果
        """
        print("\n" + "="*80)
        print("月度调仓策略回测结果")
        print("="*80)
        
        strategy_names = {
            'max_sharpe': '最大夏普比率组合',
            'max_calmar': '最大卡玛比率组合', 
            'high_return': '控制回撤高收益组合'
        }
        
        for strategy, name in strategy_names.items():
            if strategy in self.monthly_backtest_results:
                result = self.monthly_backtest_results[strategy]
                
                print(f"\n📌 {name} - 月度调仓表现")
                print("-" * 50)
                print(f"总收益率: {result['total_return']:.2%}")
                print(f"年化收益率: {result['annual_return']:.2%}")
                print(f"年化波动率: {result['annual_volatility']:.2%}")
                print(f"夏普比率: {result['sharpe_ratio']:.3f}")
                print(f"最大回撤: {result['max_drawdown']:.2%}")
                print(f"卡玛比率: {result['calmar_ratio']:.3f}")
                print(f"胜率: {result['win_rate']:.1%}")
                
        # 输出最新调仓信息
        self._print_latest_weights()
        
        # 添加调试信息
        print(f"\n调试信息:")
        print(f"调仓日期数量: {len(self.monthly_weights_history['dates'])}")
        print(f"最大夏普权重数量: {len(self.monthly_weights_history['max_sharpe_weights'])}")
        print(f"最大卡玛权重数量: {len(self.monthly_weights_history['max_calmar_weights'])}")
        print(f"高收益权重数量: {len(self.monthly_weights_history['high_return_weights'])}")
    
    def _print_latest_weights(self):
        """
        输出最新一期的权重分配
        """
        if (hasattr(self, 'monthly_weights_history') and 
            len(self.monthly_weights_history['dates']) > 0):
            
            print(f"\n" + "="*80)
            print("📋 最新一期权重分配详情")
            print("="*80)
            
            latest_date = self.monthly_weights_history['dates'][-1]
            print(f"调仓日期: {latest_date.strftime('%Y-%m-%d')}")
            
            # 策略名称映射
            strategies = {
                'max_sharpe': ('最大夏普比率组合', self.monthly_weights_history['max_sharpe_weights'][-1]),
                'max_calmar': ('最大卡玛比率组合', self.monthly_weights_history['max_calmar_weights'][-1]),
                'high_return': ('控制回撤高收益组合', self.monthly_weights_history['high_return_weights'][-1])
            }
            
            for strategy_key, (strategy_name, weights) in strategies.items():
                print(f"\n📊 {strategy_name}:")
                print("-" * 50)
                
                # 按权重大小排序输出
                weight_pairs = []
                for j, (etf_code, etf_name) in enumerate(self.etf_codes.items()):
                    if j < len(weights):
                        weight_pairs.append((weights[j], etf_name, etf_code))
                
                # 按权重从大到小排序
                weight_pairs.sort(reverse=True)
                
                for weight, etf_name, etf_code in weight_pairs:
                    if weight > 0.001:  # 只显示权重大于0.1%的资产
                        print(f"   {etf_name:15} ({etf_code}): {weight:7.1%}")
            
            print("="*80)
    
    def plot_monthly_backtest_results(self, save_path=None):
        """
        绘制月度调仓回测结果图表
        """
        if not hasattr(self, 'monthly_backtest_results'):
            raise ValueError("请先进行月度调仓回测")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        strategy_names = {
            'max_sharpe': '最大夏普比率组合',
            'max_calmar': '最大卡玛比率组合', 
            'high_return': '控制回撤高收益组合'
        }
        colors = ['red', 'blue', 'green']
        
        # 1. 累计收益率曲线对比
        ax1 = fig.add_subplot(gs[0, :])
        for i, (strategy, name) in enumerate(strategy_names.items()):
            if strategy in self.monthly_backtest_results:
                cum_returns = self.monthly_backtest_results[strategy]['cumulative_returns']
                ax1.plot(cum_returns.index, cum_returns, label=name, 
                        color=colors[i], linewidth=2)
        
        ax1.set_title('月度调仓策略累计收益率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('累计收益率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 权重变化热力图（以最大夏普比率组合为例）
        if len(self.monthly_weights_history['max_sharpe_weights']) > 0:
            ax2 = fig.add_subplot(gs[1, 0])
            weights_df = pd.DataFrame(
                self.monthly_weights_history['max_sharpe_weights'],
                index=[d.strftime('%Y-%m') for d in self.monthly_weights_history['dates']],
                columns=list(self.etf_codes.values())
            )
            
            im = ax2.imshow(weights_df.T, cmap='RdYlBu', aspect='auto')
            ax2.set_title('最大夏普比率组合权重变化')
            ax2.set_ylabel('资产')
            ax2.set_xlabel('调仓时间')
            ax2.set_yticks(range(len(self.etf_codes)))
            ax2.set_yticklabels(list(self.etf_codes.values()), fontsize=8)
            ax2.set_xticks(range(0, len(weights_df), max(1, len(weights_df)//6)))
            ax2.set_xticklabels([weights_df.index[i] for i in range(0, len(weights_df), max(1, len(weights_df)//6))], 
                              rotation=45, fontsize=8)
            plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 3. 年化收益率对比
        ax3 = fig.add_subplot(gs[1, 1])
        annual_returns = [self.monthly_backtest_results[s]['annual_return'] 
                         for s in strategy_names.keys() if s in self.monthly_backtest_results]
        strategy_labels = [strategy_names[s] for s in strategy_names.keys() 
                          if s in self.monthly_backtest_results]
        
        bars = ax3.bar(range(len(annual_returns)), annual_returns, 
                      color=colors[:len(annual_returns)], alpha=0.7)
        ax3.set_title('年化收益率对比')
        ax3.set_ylabel('年化收益率')
        ax3.set_xticks(range(len(strategy_labels)))
        ax3.set_xticklabels([s.replace('组合', '') for s in strategy_labels], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, annual_returns):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1%}', ha='center', va='bottom')
        
        # 4. 夏普比率对比
        ax4 = fig.add_subplot(gs[1, 2])
        sharpe_ratios = [self.monthly_backtest_results[s]['sharpe_ratio'] 
                        for s in strategy_names.keys() if s in self.monthly_backtest_results]
        
        bars = ax4.bar(range(len(sharpe_ratios)), sharpe_ratios, 
                      color=colors[:len(sharpe_ratios)], alpha=0.7)
        ax4.set_title('夏普比率对比')
        ax4.set_ylabel('夏普比率')
        ax4.set_xticks(range(len(strategy_labels)))
        ax4.set_xticklabels([s.replace('组合', '') for s in strategy_labels], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, sharpe_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 5. 最大回撤对比
        ax5 = fig.add_subplot(gs[2, 0])
        max_drawdowns = [self.monthly_backtest_results[s]['max_drawdown'] 
                        for s in strategy_names.keys() if s in self.monthly_backtest_results]
        
        bars = ax5.bar(range(len(max_drawdowns)), max_drawdowns, 
                      color=colors[:len(max_drawdowns)], alpha=0.7)
        ax5.set_title('最大回撤对比')
        ax5.set_ylabel('最大回撤')
        ax5.set_xticks(range(len(strategy_labels)))
        ax5.set_xticklabels([s.replace('组合', '') for s in strategy_labels], rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, max_drawdowns):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{value:.1%}', ha='center', va='bottom')
        
        # 6. 滚动回撤曲线
        ax6 = fig.add_subplot(gs[2, 1:])
        for i, (strategy, name) in enumerate(strategy_names.items()):
            if strategy in self.monthly_backtest_results:
                cum_returns = self.monthly_backtest_results[strategy]['cumulative_returns']
                peak = cum_returns.expanding().max()
                drawdown = (cum_returns - peak) / peak
                ax6.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color=colors[i])
                ax6.plot(drawdown.index, drawdown, label=name, color=colors[i], linewidth=1)
        
        ax6.set_title('滚动回撤曲线')
        ax6.set_ylabel('回撤')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('月度动态调仓策略分析报告', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"月度调仓图表已保存到: {save_path}")
        
        plt.show()
    
    def run_monthly_analysis(self, start_date='20210101', end_date=None, 
                           lookback_months=12, rebalance_days=[1, 15], save_results=True):
        """
        运行完整的月度调仓分析
        
        Parameters:
        - start_date: 回测开始日期
        - end_date: 回测结束日期
        - lookback_months: 优化时使用的历史数据长度(月)
        - rebalance_days: 每月调仓日期列表，默认[1, 15]表示每月1日和15日调仓
        - save_results: 是否保存结果
        """
        # 执行月度调仓回测
        monthly_results = self.monthly_rebalancing_backtest(
            start_date=start_date, 
            end_date=end_date,
            lookback_months=lookback_months,
            rebalance_days=rebalance_days
        )
        
        # 绘制分析图表
        if save_results:
            save_path = 'monthly_rebalancing_analysis.png'
            self.plot_monthly_backtest_results(save_path=save_path)
        else:
            self.plot_monthly_backtest_results()
        
        # 保存详细数据
        if save_results:
            self.save_monthly_results(start_date, end_date, lookback_months, rebalance_days)
        
        return monthly_results
    
    def save_monthly_results(self, start_date, end_date, lookback_months, rebalance_days=[1, 15]):
        """
        保存月度调仓分析的详细结果
        """
        print("\n正在保存分析结果...")
        
        # 1. 保存策略表现汇总
        strategy_names = {
            'max_sharpe': '最大夏普比率组合',
            'max_calmar': '最大卡玛比率组合', 
            'high_return': '控制回撤高收益组合'
        }
        
        summary_data = []
        for strategy, name in strategy_names.items():
            if strategy in self.monthly_backtest_results:
                result = self.monthly_backtest_results[strategy]
                summary_data.append({
                    '策略名称': name,
                    '策略代码': strategy,
                    '总收益率': f"{result['total_return']:.4f}",
                    '年化收益率': f"{result['annual_return']:.4f}",
                    '年化波动率': f"{result['annual_volatility']:.4f}",
                    '夏普比率': f"{result['sharpe_ratio']:.4f}",
                    '最大回撤': f"{result['max_drawdown']:.4f}",
                    '卡玛比率': f"{result['calmar_ratio']:.4f}",
                    '胜率': f"{result['win_rate']:.4f}",
                    '回测开始日期': start_date,
                    '回测结束日期': end_date or datetime.now().strftime('%Y%m%d'),
                    '历史数据窗口_月': lookback_months,
                    '调仓日期': str(rebalance_days)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('月度调仓策略表现汇总.csv', index=False, encoding='utf-8-sig')
        print("✅ 策略表现汇总已保存: 月度调仓策略表现汇总.csv")
        
        # 2. 保存历史持仓明细
        if hasattr(self, 'monthly_weights_history') and len(self.monthly_weights_history['dates']) > 0:
            holdings_data = []
            
            # 确保数据长度一致
            dates = self.monthly_weights_history['dates']
            min_length = min(
                len(dates),
                len(self.monthly_weights_history['max_sharpe_weights']),
                len(self.monthly_weights_history['max_calmar_weights']),
                len(self.monthly_weights_history['high_return_weights'])
            )
            
            for i in range(min_length):
                date = dates[i]
                # 为每个策略和每个资产创建记录
                for strategy_key, strategy_name in strategy_names.items():
                    weights = None
                    if strategy_key == 'max_sharpe':
                        weights = self.monthly_weights_history['max_sharpe_weights'][i]
                    elif strategy_key == 'max_calmar':
                        weights = self.monthly_weights_history['max_calmar_weights'][i]
                    elif strategy_key == 'high_return':
                        weights = self.monthly_weights_history['high_return_weights'][i]
                    
                    if weights is not None:
                        for j, (etf_code, etf_name) in enumerate(self.etf_codes.items()):
                            if j < len(weights):
                                holdings_data.append({
                                    '调仓日期': date.strftime('%Y-%m-%d'),
                                    '策略名称': strategy_name,
                                    '策略代码': strategy_key,
                                    'ETF代码': etf_code,
                                    'ETF名称': etf_name,
                                    '权重': f"{weights[j]:.6f}",
                                    '权重百分比': f"{weights[j]*100:.2f}%"
                                })
            
            if holdings_data:
                holdings_df = pd.DataFrame(holdings_data)
                holdings_df.to_csv('月度调仓历史持仓明细.csv', index=False, encoding='utf-8-sig')
                print("✅ 历史持仓明细已保存: 月度调仓历史持仓明细.csv")
            else:
                print("⚠️ 没有持仓数据可保存")
        
        # 3. 保存每日收益明细
        if hasattr(self, 'monthly_backtest_results') and self.monthly_backtest_results:
            daily_returns_data = []
            
            for strategy, name in strategy_names.items():
                if strategy in self.monthly_backtest_results:
                    cum_returns = self.monthly_backtest_results[strategy]['cumulative_returns']
                    daily_returns = self.monthly_backtest_results[strategy]['daily_returns']
                    
                    for date in cum_returns.index:
                        daily_returns_data.append({
                            '日期': date.strftime('%Y-%m-%d'),
                            '策略名称': name,
                            '策略代码': strategy,
                            '日度收益率': f"{daily_returns.loc[date]:.6f}" if date in daily_returns.index else "0.000000",
                            '日度收益率百分比': f"{daily_returns.loc[date]*100:.4f}%" if date in daily_returns.index else "0.0000%",
                            '累计净值': f"{cum_returns.loc[date]:.6f}",
                            '累计收益率': f"{(cum_returns.loc[date]-1)*100:.2f}%"
                        })
            
            if daily_returns_data:
                daily_returns_df = pd.DataFrame(daily_returns_data)
                daily_returns_df.to_csv('月度调仓每日收益明细.csv', index=False, encoding='utf-8-sig')
                print("✅ 每日收益明细已保存: 月度调仓每日收益明细.csv")
            else:
                print("⚠️ 没有收益数据可保存")
        
        # 4. 保存权重变化透视表（便于分析）
        if hasattr(self, 'monthly_weights_history') and len(self.monthly_weights_history['dates']) > 0:
            dates = self.monthly_weights_history['dates']
            min_length = min(
                len(dates),
                len(self.monthly_weights_history['max_sharpe_weights']),
                len(self.monthly_weights_history['max_calmar_weights']),
                len(self.monthly_weights_history['high_return_weights'])
            )
            
            # 为每个策略创建权重变化表
            for strategy_key, strategy_name in strategy_names.items():
                weights_list = None
                if strategy_key == 'max_sharpe':
                    weights_list = self.monthly_weights_history['max_sharpe_weights']
                elif strategy_key == 'max_calmar':
                    weights_list = self.monthly_weights_history['max_calmar_weights']
                elif strategy_key == 'high_return':
                    weights_list = self.monthly_weights_history['high_return_weights']
                
                if weights_list and len(weights_list) > 0:
                    weights_pivot_data = []
                    for i in range(min(min_length, len(weights_list))):
                        date = dates[i]
                        row = {'调仓日期': date.strftime('%Y-%m-%d')}
                        weights = weights_list[i]
                        for j, (etf_code, etf_name) in enumerate(self.etf_codes.items()):
                            if j < len(weights):
                                row[f"{etf_name}({etf_code})"] = f"{weights[j]:.4f}"
                        weights_pivot_data.append(row)
                    
                    if weights_pivot_data:
                        weights_pivot_df = pd.DataFrame(weights_pivot_data)
                        filename = f'{strategy_name}_权重变化表.csv'
                        weights_pivot_df.to_csv(filename, index=False, encoding='utf-8-sig')
                        print(f"✅ {strategy_name}权重变化表已保存: {filename}")
                    else:
                        print(f"⚠️ {strategy_name}没有权重数据可保存")
        
        # 5. 保存分析参数和元数据
        metadata = {
            '分析参数': [
                f"回测开始日期: {start_date}",
                f"回测结束日期: {end_date or datetime.now().strftime('%Y%m%d')}",
                f"历史数据窗口: {lookback_months} 个月",
                f"调仓频率: 每月{rebalance_days}日",
                f"调仓次数: 每月{len(rebalance_days)}次",
                f"ETF数量: {len(self.etf_codes)}",
                f"策略数量: {len(strategy_names)}",
                f"分析生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
        }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv('分析参数和元数据.csv', index=False, encoding='utf-8-sig', header=False)
        print("✅ 分析参数和元数据已保存: 分析参数和元数据.csv")
        
        print(f"\n🎉 所有分析结果已保存完成！")
        print("📁 生成的文件列表:")
        print("   1. 月度调仓策略表现汇总.csv - 策略整体表现")
        print("   2. 月度调仓历史持仓明细.csv - 每次调仓的详细持仓")
        print("   3. 月度调仓每日收益明细.csv - 每日收益率和净值变化")
        print("   4. [策略名称]_权重变化表.csv - 各策略权重变化透视表")
        print("   5. 分析参数和元数据.csv - 分析配置信息")
        print("   6. monthly_rebalancing_analysis.png - 可视化分析图表")

def main():
    """
    主函数
    """
    # 创建有效前沿分析器
    ef_analyzer = EfficientFrontierETF()
    
    # 运行样本内外分析
    try:
        # 方法1: 运行样本内外分析 (推荐)
        # results = ef_analyzer.run_in_out_sample_analysis(
        #     in_sample_start='20210101', 
        #     in_sample_end='20230630',
        #     out_sample_start='20240101'
        # )
        
        # print("\n" + "="*80)
        # print("开始月度动态调仓分析...")
        # print("="*80)
        
        # 方法3: 运行月度动态调仓回测 (新增) - 每月月初和月中调仓
        monthly_results = ef_analyzer.run_monthly_analysis(
            start_date='20200101',      # 调仓回测开始日期
            lookback_months=12,         # 使用12个月历史数据进行优化
            rebalance_days=[1, 15],     # 每月1日和15日调仓
            save_results=True
        )
        
        print("\n投资建议总结:")
        print("="*80)
        print("📌 组合一：稳中求胜（最大夏普比率组合）")
        print("   特点：风险调整后收益最优，适合稳健型投资者")
        print("   适用场景：追求长期稳定收益，不希望承受过大波动")
        
        print("\n📌 组合二：控制回撤（最大卡玛比率组合）") 
        print("   特点：在控制回撤的前提下获得较好收益")
        print("   适用场景：注重资产保值，能接受适中收益")
        
        print("\n📌 组合三：小心激进（控制回撤的高收益组合）")
        print("   特点：在10%回撤限制下追求更高收益")
        print("   适用场景：风险承受能力较强，追求较高收益")
        
        print("\n注意：")
        print("- 样本内外表现存在差异是正常现象")
        print("- 建议关注样本外的夏普比率和最大回撤控制")
        print("- 可根据实际风险偏好选择合适的投资组合")
        print("- 每月月初和月中调仓策略可以更好地适应市场变化")
        print("- 增加调仓频率有助于及时调整投资组合，但可能增加交易成本")
        
        # 方法2: 也可以运行传统的全样本分析
        # results = ef_analyzer.run_complete_analysis(start_date='20200101')
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查网络连接和数据获取情况")

if __name__ == "__main__":
    main()

