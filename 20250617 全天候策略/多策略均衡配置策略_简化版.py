import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
import matplotlib.font_manager as fm

# 获取系统可用的中文字体
def get_chinese_font():
    """获取系统中可用的中文字体"""
    font_list = [font.name for font in fm.fontManager.ttflist]
    
    # 优先选择的中文字体列表
    preferred_fonts = [
        'Microsoft YaHei', 'Microsoft YaHei UI',  # 微软雅黑
        'SimHei',  # 黑体
        'SimSun',  # 宋体
        'KaiTi',   # 楷体
        'FangSong',  # 仿宋
        'STSong',    # 华文宋体
        'STKaiti',   # 华文楷体
        'STHeiti',   # 华文黑体
        'WenQuanYi Micro Hei',  # 文泉驿微米黑(Linux)
        'Droid Sans Fallback',   # Android中文字体
        'Noto Sans CJK SC',      # Google Noto字体
        'Source Han Sans SC'     # 思源黑体
    ]
    
    # 查找第一个可用的中文字体
    for font in preferred_fonts:
        if font in font_list:
            print(f"使用中文字体: {font}")
            return font
    
    print("警告: 未找到合适的中文字体，可能无法正确显示中文")
    return 'DejaVu Sans'

# 设置中文字体
chinese_font = get_chinese_font()
plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置matplotlib的默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# 设置seaborn样式
plt.style.use('default')  # 使用默认样式而不是seaborn
sns.set_palette("husl")
sns.set_style("whitegrid")

# 确保中文显示正常
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.sans-serif': [chinese_font, 'DejaVu Sans'],
    'axes.unicode_minus': False
})

def load_strategy_data():
    """加载三个策略的数据"""
    print("正在加载策略数据...")
    
    # 加载全天候策略1
    strategy1 = pd.read_csv('20250617 全天候策略/output/portfolio_daily_performance_20250715_172155.csv')
    strategy1['日期'] = pd.to_datetime(strategy1['日期'])
    strategy1 = strategy1.rename(columns={'日度收益率': '策略1收益率', '净值': '策略1净值'})
    
    # 加载全天候策略2  
    strategy2 = pd.read_csv('20250617 全天候策略/output/portfolio_daily_performance_20250716_133751.csv')
    strategy2['日期'] = pd.to_datetime(strategy2['日期'])
    strategy2 = strategy2.rename(columns={'日度收益率': '策略2收益率', '净值': '策略2净值'})
    
    # 加载有效前沿策略
    strategy3 = pd.read_csv('20250617 全天候策略/月度调仓每日收益明细.csv')
    strategy3['日期'] = pd.to_datetime(strategy3['日期'])
    # 提取日度收益率（使用当日收益率列）
    strategy3 = strategy3[['日期', '当日收益率', '净值']].copy()
    strategy3 = strategy3.rename(columns={'当日收益率': '策略3收益率', '净值': '策略3净值'})
    
    print(f"策略1数据范围: {strategy1['日期'].min()} 到 {strategy1['日期'].max()}")
    print(f"策略2数据范围: {strategy2['日期'].min()} 到 {strategy2['日期'].max()}")  
    print(f"策略3数据范围: {strategy3['日期'].min()} 到 {strategy3['日期'].max()}")
    
    return strategy1, strategy2, strategy3

def find_effective_start_date(strategy1, strategy2, strategy3):
    """找到所有策略都开始有有效数据的日期"""
    print("\n正在寻找有效数据开始日期...")
    
    # 找到策略1开始有非零收益的日期（排除极小的精度误差）
    strategy1_start = strategy1[abs(strategy1['策略1收益率']) > 1e-6]['日期'].min()
    
    # 找到策略2开始有非零收益的日期
    strategy2_start = strategy2[abs(strategy2['策略2收益率']) > 1e-6]['日期'].min()
    
    # 策略3从第一天就有数据
    strategy3_start = strategy3['日期'].min()
    
    # 取最晚的开始日期作为共同开始日期
    effective_start = max(strategy1_start, strategy2_start, strategy3_start)
    
    print(f"策略1有效数据开始日期: {strategy1_start}")
    print(f"策略2有效数据开始日期: {strategy2_start}")
    print(f"策略3有效数据开始日期: {strategy3_start}")
    print(f"共同有效数据开始日期: {effective_start}")
    
    return effective_start

def align_and_merge_data(strategy1, strategy2, strategy3, start_date):
    """对齐并合并三个策略的数据"""
    print(f"\n正在从 {start_date} 开始对齐数据...")
    
    # 筛选数据到有效开始日期
    s1 = strategy1[strategy1['日期'] >= start_date].copy()
    s2 = strategy2[strategy2['日期'] >= start_date].copy()
    s3 = strategy3[strategy3['日期'] >= start_date].copy()
    
    # 合并数据
    merged = pd.merge(s1[['日期', '策略1收益率']], s2[['日期', '策略2收益率']], on='日期', how='outer')
    merged = pd.merge(merged, s3[['日期', '策略3收益率']], on='日期', how='outer')
    
    # 排序并填充缺失值
    merged = merged.sort_values('日期').reset_index(drop=True)
    merged = merged.fillna(0)
    
    print(f"合并后数据量: {len(merged)} 天")
    print(f"数据范围: {merged['日期'].min()} 到 {merged['日期'].max()}")
    
    return merged

def calculate_equal_weight_portfolio(merged_data):
    """计算等权重组合策略"""
    print("\n正在计算等权重组合策略...")
    
    # 等权重配置（1/3 + 1/3 + 1/3）
    merged_data['组合收益率'] = (
                              merged_data['策略2收益率'] + 
                              merged_data['策略3收益率']) / 2
    
    # 计算累计净值
    merged_data['组合净值'] = (1 + merged_data['组合收益率']).cumprod()
    
    # 计算各策略净值（重新计算确保一致性）
    merged_data['策略1净值'] = (1 + merged_data['策略1收益率']).cumprod()
    merged_data['策略2净值'] = (1 + merged_data['策略2收益率']).cumprod()
    merged_data['策略3净值'] = (1 + merged_data['策略3收益率']).cumprod()
    
    return merged_data

def calculate_performance_metrics(returns_series, strategy_name):
    """计算策略绩效指标"""
    returns = returns_series.dropna()
    
    if len(returns) == 0:
        return {}
    
    # 基本统计
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 夏普比率（假设无风险利率为3%）
    risk_free_rate = 0.03
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # 卡玛比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 胜率
    win_rate = (returns > 0).mean()
    
    return {
        '策略名称': strategy_name,
        '总收益率': f"{total_return:.4f}",
        '年化收益率': f"{annual_return:.4f}",
        '年化波动率': f"{annual_volatility:.4f}",
        '夏普比率': f"{sharpe_ratio:.4f}",
        '最大回撤': f"{max_drawdown:.4f}",
        '卡玛比率': f"{calmar_ratio:.4f}",
        '胜率': f"{win_rate:.4f}",
        '交易天数': len(returns)
    }

def generate_performance_report(data):
    """生成绩效报告"""
    print("\n正在生成绩效报告...")
    
    # 计算各策略绩效
    metrics_list = []
    
    # 策略1
    metrics_list.append(calculate_performance_metrics(data['策略1收益率'], '全天候策略1'))
    
    # 策略2  
    metrics_list.append(calculate_performance_metrics(data['策略2收益率'], '全天候策略2'))
    
    # 策略3
    metrics_list.append(calculate_performance_metrics(data['策略3收益率'], '有效前沿策略'))
    
    # 组合策略
    metrics_list.append(calculate_performance_metrics(data['组合收益率'], '等权重组合策略'))
    
    # 创建DataFrame
    performance_df = pd.DataFrame(metrics_list)
    
    return performance_df

def calculate_drawdowns(returns_series):
    """计算回撤序列"""
    cumulative_returns = (1 + returns_series).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return drawdowns

def create_comprehensive_analysis(data):
    """创建全面的图形分析"""
    print("\n正在生成综合图形分析...")
    
    # 获取当前的中文字体
    chinese_font = plt.rcParams['font.sans-serif'][0]
    
    # 确保中文显示
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建一个大的图形画布
    fig = plt.figure(figsize=(20, 24))
    
    # 1. 净值走势对比 (第1行，占两列)
    ax1 = plt.subplot(4, 3, (1, 2))
    strategies = ['策略1净值', '策略2净值', '策略3净值', '组合净值']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['全天候策略1', '全天候策略2', '有效前沿策略', '等权重组合策略']
    
    for i, (strategy, color, label) in enumerate(zip(strategies, colors, labels)):
        linewidth = 3 if '组合' in label else 1.5
        alpha = 1.0 if '组合' in label else 0.8
        ax1.plot(data['日期'], data[strategy], label=label, color=color, 
                linewidth=linewidth, alpha=alpha)
    
    ax1.set_title('策略净值走势对比', fontsize=16, fontweight='bold')
    ax1.set_ylabel('净值', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 收益率分布 (第1行第3列)
    ax2 = plt.subplot(4, 3, 3)
    return_cols = ['策略1收益率', '策略2收益率', '策略3收益率', '组合收益率']
    return_labels = ['全天候策略1', '全天候策略2', '有效前沿策略', '等权重组合策略']
    
    for i, (col, label, color) in enumerate(zip(return_cols, return_labels, colors)):
        ax2.hist(data[col] * 100, bins=50, alpha=0.6, label=label, 
                color=color, density=True)
    
    ax2.set_title('日收益率分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日收益率 (%)', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 滚动收益率 (第2行第1列)
    ax3 = plt.subplot(4, 3, 4)
    window = 60
    for col, label, color in zip(return_cols, return_labels, colors):
        rolling_return = data[col].rolling(window).mean() * 252 * 100
        linewidth = 2.5 if '组合' in label else 1.5
        ax3.plot(data['日期'], rolling_return, label=label, color=color, linewidth=linewidth)
    
    ax3.set_title(f'{window}日滚动年化收益率', fontsize=14, fontweight='bold')
    ax3.set_ylabel('年化收益率 (%)', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 滚动波动率 (第2行第2列)
    ax4 = plt.subplot(4, 3, 5)
    for col, label, color in zip(return_cols, return_labels, colors):
        rolling_vol = data[col].rolling(window).std() * np.sqrt(252) * 100
        linewidth = 2.5 if '组合' in label else 1.5
        ax4.plot(data['日期'], rolling_vol, label=label, color=color, linewidth=linewidth)
    
    ax4.set_title(f'{window}日滚动年化波动率', fontsize=14, fontweight='bold')
    ax4.set_ylabel('年化波动率 (%)', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. 最大回撤 (第2行第3列)
    ax5 = plt.subplot(4, 3, 6)
    for col, label, color in zip(return_cols, return_labels, colors):
        drawdowns = calculate_drawdowns(data[col]) * 100
        linewidth = 2.5 if '组合' in label else 1.5
        ax5.plot(data['日期'], drawdowns, label=label, color=color, linewidth=linewidth)
    
    ax5.set_title('策略回撤走势', fontsize=14, fontweight='bold')
    ax5.set_ylabel('回撤 (%)', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 6. 收益率相关性热力图 (第3行第1列)
    ax6 = plt.subplot(4, 3, 7)
    returns_df = data[return_cols].rename(columns=dict(zip(return_cols, return_labels)))
    correlation_matrix = returns_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, ax=ax6, cbar_kws={'label': '相关系数'})
    ax6.set_title('策略收益率相关性', fontsize=14, fontweight='bold')
    
    # 7. 风险收益散点图 (第3行第2列)
    ax7 = plt.subplot(4, 3, 8)
    annual_returns = []
    annual_vols = []
    
    for col in return_cols:
        returns = data[col].dropna()
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_vol = returns.std() * np.sqrt(252)
        annual_returns.append(annual_return * 100)
        annual_vols.append(annual_vol * 100)
    
    scatter = ax7.scatter(annual_vols, annual_returns, c=colors, s=200, alpha=0.7)
    
    for i, label in enumerate(return_labels):
        ax7.annotate(label, (annual_vols[i], annual_returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax7.set_title('风险收益散点图', fontsize=14, fontweight='bold')
    ax7.set_xlabel('年化波动率 (%)', fontsize=12)
    ax7.set_ylabel('年化收益率 (%)', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # 8. 月度收益率热力图 (第3行第3列)
    ax8 = plt.subplot(4, 3, 9)
    
    # 计算组合策略的月度收益率
    data_monthly = data.copy()
    data_monthly['年月'] = data_monthly['日期'].dt.to_period('M')
    monthly_returns = data_monthly.groupby('年月')['组合收益率'].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    monthly_returns['年'] = monthly_returns['年月'].dt.year
    monthly_returns['月'] = monthly_returns['年月'].dt.month
    
    # 创建透视表
    pivot_monthly = monthly_returns.pivot(index='年', columns='月', values='组合收益率')
    pivot_monthly = pivot_monthly * 100  # 转换为百分比
    
    sns.heatmap(pivot_monthly, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax8, cbar_kws={'label': '月收益率 (%)'})
    ax8.set_title('等权重组合策略月度收益率热力图', fontsize=14, fontweight='bold')
    ax8.set_xlabel('月份', fontsize=12)
    ax8.set_ylabel('年份', fontsize=12)
    
    # 9. 年度收益率对比 (第4行第1列)
    ax9 = plt.subplot(4, 3, 10)
    
    # 计算年度收益率
    data_yearly = data.copy()
    data_yearly['年份'] = data_yearly['日期'].dt.year
    
    yearly_returns = {}
    for col, label in zip(return_cols, return_labels):
        yearly_ret = data_yearly.groupby('年份')[col].apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        yearly_returns[label] = yearly_ret
    
    yearly_df = pd.DataFrame(yearly_returns)
    yearly_df.plot(kind='bar', ax=ax9, width=0.8)
    
    ax9.set_title('年度收益率对比', fontsize=14, fontweight='bold')
    ax9.set_xlabel('年份', fontsize=12)
    ax9.set_ylabel('年收益率 (%)', fontsize=12)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(axis='x', rotation=45)
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 10. 策略权重饼图 (第4行第2列)
    ax10 = plt.subplot(4, 3, 11)
    weights = [1/3, 1/3, 1/3]
    strategy_names = ['全天候策略1\n(33.33%)', '全天候策略2\n(33.33%)', '有效前沿策略\n(33.33%)']
    
    wedges, texts, autotexts = ax10.pie(weights, labels=strategy_names, autopct='',
                                       colors=colors[:3], startangle=90,
                                       explode=(0.05, 0.05, 0.05))
    
    ax10.set_title('等权重组合策略权重分配', fontsize=14, fontweight='bold')
    
    # 11. 累计收益率对比 (第4行第3列)
    ax11 = plt.subplot(4, 3, 12)
    
    for strategy, color, label in zip(strategies, colors, labels):
        cumulative_return = (data[strategy] - 1) * 100
        linewidth = 3 if '组合' in label else 1.5
        alpha = 1.0 if '组合' in label else 0.8
        ax11.plot(data['日期'], cumulative_return, label=label, color=color,
                 linewidth=linewidth, alpha=alpha)
    
    ax11.set_title('累计收益率对比', fontsize=14, fontweight='bold')
    ax11.set_ylabel('累计收益率 (%)', fontsize=12)
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)
    ax11.tick_params(axis='x', rotation=45)
    ax11.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout(pad=3.0)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'多策略均衡配置综合分析_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"综合分析图表已保存到: {filename}")
    
    plt.close()

def set_chinese_labels(ax, title=None, xlabel=None, ylabel=None):
    """设置图表的中文标签，确保字体正确显示"""
    chinese_font = plt.rcParams['font.sans-serif'][0]
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', 
                    fontproperties=fm.FontProperties(family=chinese_font))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, 
                     fontproperties=fm.FontProperties(family=chinese_font))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, 
                     fontproperties=fm.FontProperties(family=chinese_font))

def create_detailed_performance_charts(data, performance_df):
    """创建详细的绩效分析图表"""
    print("\n正在生成详细绩效分析图表...")
    
    # 确保中文显示
    chinese_font = plt.rcParams['font.sans-serif'][0]
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 绩效指标雷达图
    categories = ['年化收益率', '夏普比率', '卡玛比率', '胜率']
    
    # 提取数值数据
    strategies_data = []
    strategy_names = []
    
    for _, row in performance_df.iterrows():
        if row['策略名称'] != '等权重组合策略':  # 先处理三个基础策略
            strategy_names.append(row['策略名称'])
            values = [
                float(row['年化收益率']) * 5,  # 放大收益率便于显示
                float(row['夏普比率']),
                float(row['卡玛比率']),
                float(row['胜率']) * 100  # 转换为百分比
            ]
            strategies_data.append(values)
    
    # 添加组合策略
    combo_row = performance_df[performance_df['策略名称'] == '等权重组合策略'].iloc[0]
    strategy_names.append(combo_row['策略名称'])
    combo_values = [
        float(combo_row['年化收益率']) * 5,
        float(combo_row['夏普比率']),
        float(combo_row['卡玛比率']),
        float(combo_row['胜率']) * 100
    ]
    strategies_data.append(combo_values)
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (strategy_data, name, color) in enumerate(zip(strategies_data, strategy_names, colors)):
        values = strategy_data + [strategy_data[0]]  # 闭合数据
        linewidth = 3 if '组合' in name else 2
        alpha = 0.25 if '组合' not in name else 0.4
        ax1.plot(angles, values, 'o-', linewidth=linewidth, label=name, color=color)
        ax1.fill(angles, values, alpha=alpha, color=color)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_title('策略绩效雷达图', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax1.grid(True)
    
    # 2. 月度收益率箱线图
    data_copy = data.copy()
    data_copy['月份'] = data_copy['日期'].dt.month
    
    monthly_data = []
    months = sorted(data_copy['月份'].unique())
    
    for month in months:
        month_data = data_copy[data_copy['月份'] == month]
        monthly_data.append(month_data['组合收益率'] * 100)
    
    bp = ax2.boxplot(monthly_data, labels=[f'{m}月' for m in months], patch_artist=True)
    
    # 设置箱线图颜色
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(monthly_data)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('等权重组合策略月度收益率分布', fontsize=14, fontweight='bold')
    ax2.set_ylabel('日收益率 (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 3. 收益率分布对比（小提琴图）
    return_data = []
    return_labels = []
    
    for col in ['策略1收益率', '策略2收益率', '策略3收益率', '组合收益率']:
        return_data.append(data[col] * 100)
        if col == '策略1收益率':
            return_labels.append('全天候策略1')
        elif col == '策略2收益率':
            return_labels.append('全天候策略2')
        elif col == '策略3收益率':
            return_labels.append('有效前沿策略')
        else:
            return_labels.append('等权重组合策略')
    
    parts = ax3.violinplot(return_data, positions=range(1, len(return_data) + 1), 
                          showmeans=True, showmedians=True)
    
    # 设置小提琴图颜色
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax3.set_xticks(range(1, len(return_labels) + 1))
    ax3.set_xticklabels(return_labels, rotation=45)
    ax3.set_title('策略收益率分布对比（小提琴图）', fontsize=14, fontweight='bold')
    ax3.set_ylabel('日收益率 (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 4. 策略绩效指标条形图
    metrics = ['总收益率', '年化收益率', '夏普比率', '卡玛比率']
    x = np.arange(len(performance_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [float(val) for val in performance_df[metric]]
        if metric in ['总收益率', '年化收益率']:
            values = [v * 100 for v in values]  # 转换为百分比
        
        bars = ax4.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        # 在条形图上添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_xlabel('策略', fontsize=12)
    ax4.set_ylabel('指标值', fontsize=12)
    ax4.set_title('策略绩效指标对比', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(performance_df['策略名称'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'多策略详细绩效分析_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"详细绩效分析图表已保存到: {filename}")
    
    plt.close()

def save_results(data, performance_df):
    """保存结果"""
    print("\n正在保存结果...")
    
    # 保存日度数据
    output_data = data[['日期', '策略1收益率', '策略2收益率', '策略3收益率', '组合收益率',
                       '策略1净值', '策略2净值', '策略3净值', '组合净值']].copy()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'多策略均衡配置日度数据_{timestamp}.csv'
    output_data.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    # 保存绩效汇总
    performance_filename = f'多策略均衡配置绩效汇总_{timestamp}.csv'
    performance_df.to_csv(performance_filename, index=False, encoding='utf-8-sig')
    
    print(f"日度数据已保存到: {output_filename}")
    print(f"绩效汇总已保存到: {performance_filename}")

def main():
    """主函数"""
    print("=" * 60)
    print("多策略均衡配置分析（增强图形版）")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        strategy1, strategy2, strategy3 = load_strategy_data()
        
        # 2. 找到有效开始日期
        start_date = find_effective_start_date(strategy1, strategy2, strategy3)
        
        # 3. 对齐并合并数据
        merged_data = align_and_merge_data(strategy1, strategy2, strategy3, start_date)
        
        # 4. 计算等权重组合
        portfolio_data = calculate_equal_weight_portfolio(merged_data)
        
        # 5. 生成绩效报告
        performance_df = generate_performance_report(portfolio_data)
        
        # 6. 显示绩效报告
        print("\n" + "=" * 80)
        print("绩效指标汇总")
        print("=" * 80)
        print(performance_df.to_string(index=False))
        
        # 7. 生成综合图形分析
        create_comprehensive_analysis(portfolio_data)
        
        # 8. 生成详细绩效分析图表
        create_detailed_performance_charts(portfolio_data, performance_df)
        
        # 9. 保存结果
        save_results(portfolio_data, performance_df)
        
        # 10. 显示总结信息
        print("\n" + "=" * 80)
        print("分析总结")
        print("=" * 80)
        
        final_portfolio_value = portfolio_data['组合净值'].iloc[-1]
        final_strategy1_value = portfolio_data['策略1净值'].iloc[-1]
        final_strategy2_value = portfolio_data['策略2净值'].iloc[-1] 
        final_strategy3_value = portfolio_data['策略3净值'].iloc[-1]
        
        print(f"分析期间: {start_date.strftime('%Y-%m-%d')} 到 {portfolio_data['日期'].max().strftime('%Y-%m-%d')}")
        print(f"交易天数: {len(portfolio_data)} 天")
        print(f"\n最终净值:")
        print(f"  全天候策略1: {final_strategy1_value:.4f}")
        print(f"  全天候策略2: {final_strategy2_value:.4f}")
        print(f"  有效前沿策略: {final_strategy3_value:.4f}")
        print(f"  等权重组合策略: {final_portfolio_value:.4f}")
        
        print("\n📊 已生成的图表文件:")
        print("  - 多策略均衡配置综合分析_[时间戳].png")
        print("  - 多策略详细绩效分析_[时间戳].png")
        
        print("\n" + "=" * 80)
        print("分析完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 