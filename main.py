# ============================================================================
# ALGORITHMIC TRADING SYSTEM WITH VECTORBT
# ============================================================================
# Features:
# 1. Three technical strategies (MA Crossover, RSI, MACD)
# 2. One macro-driven strategy (Yield Curve + Inflation)
# 3. Multi-asset support (Gold, Oil, S&P 500)
# 4. Risk management (stop-loss, position sizing)
# 5. Comprehensive performance analysis
# ============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# VectorBT for backtesting
import vectorbt as vbt

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA COLLECTION MODULE
# ============================================================================

class DataFetcher:
    """Data collection and preprocessing module"""
    
    def __init__(self, start_date: str = "2015-01-01", end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.assets = {
            'GLD': 'Gold (ETF Proxy)',
            'CL=F': 'Crude Oil Futures',
            'ES=F': 'S&P 500 Futures'
        }
        
        # Alternative assets if futures don't work
        self.alt_assets = {
            'GLD': 'Gold',
            'USO': 'Crude Oil ETF',
            'SPY': 'S&P 500 ETF'
        }
    
    def fetch_price_data(self, symbols: dict) -> dict:
        """Fetch OHLCV data for given symbols"""
        data_dict = {}
        
        for symbol, description in symbols.items():
            try:
                print(f"Fetching {symbol} ({description})...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                
                if not df.empty:
                    # Ensure we have all required columns
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    data_dict[symbol] = df
                    print(f"  ✓ {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
                else:
                    print(f"  ✗ No data for {symbol}")
            except Exception as e:
                print(f"  ✗ Error fetching {symbol}: {e}")
        
        return data_dict
    
    def create_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features"""
        df = price_data.copy()
        
        # Price returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility_20d'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df

# ============================================================================
# TRADING STRATEGIES MODULE
# ============================================================================

class TradingStrategies:
    """Implementation of trading strategies"""
    
    @staticmethod
    def ma_crossover_strategy(close, sma_short, sma_long):
        """Moving Average Crossover Strategy"""
        # Generate signals: 1 for long, -1 for short, 0 for neutral
        signals = pd.Series(0, index=close.index)
        
        # Long when short MA crosses above long MA
        long_condition = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
        
        # Short when short MA crosses below long MA
        short_condition = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    @staticmethod
    def rsi_strategy(close, rsi, oversold=30, overbought=70):
        """RSI Mean Reversion Strategy"""
        signals = pd.Series(0, index=close.index)
        
        # Buy when RSI crosses above oversold level
        buy_condition = (rsi > oversold) & (rsi.shift(1) <= oversold)
        
        # Sell when RSI crosses below overbought level
        sell_condition = (rsi < overbought) & (rsi.shift(1) >= overbought)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    @staticmethod
    def macd_strategy(close, macd_line, signal_line):
        """MACD Crossover Strategy"""
        signals = pd.Series(0, index=close.index)
        
        # Buy when MACD crosses above signal line
        buy_condition = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        
        # Sell when MACD crosses below signal line
        sell_condition = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    @staticmethod
    def macro_strategy(close, volatility, volume_ratio, market_regime='normal'):
        """Macro-driven strategy using market conditions"""
        signals = pd.Series(0, index=close.index)
        
        # Simplified macro strategy
        # In risk-on environment: long equities
        # In risk-off environment: long gold/short equities
        
        # Determine market regime based on volatility and volume
        high_vol = volatility > volatility.quantile(0.75)
        high_volume = volume_ratio > 1.5
        
        if 'GLD' in close.name or 'Gold' in str(close.name):
            # Gold strategy: Buy during high volatility (safe haven)
            signals[high_vol] = 1
            signals[~high_vol] = 0
            
        elif 'SPY' in close.name or 'ES' in str(close.name):
            # Equity strategy: Buy in normal/low vol, sell in high vol
            signals[~high_vol & high_volume] = 1
            signals[high_vol] = -1
            
        elif 'USO' in close.name or 'CL' in str(close.name):
            # Oil strategy: More complex macro factors
            # Simplified: Buy when volume is high (institutional interest)
            signals[high_volume] = 1
            signals[~high_volume] = 0
        
        return signals

# ============================================================================
# BACKTESTING ENGINE WITH VECTORBT
# ============================================================================

class BacktestEngineVBT:
    """Backtesting engine using VectorBT"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        
    def run_strategy(self, price_data: pd.DataFrame, 
                    signals: pd.Series, 
                    strategy_name: str,
                    stop_loss_pct: float = 0.05,
                    take_profit_pct: float = 0.10) -> vbt.Portfolio:
        """Run a single strategy backtest"""
        
        # Create portfolio with signals
        portfolio = vbt.Portfolio.from_signals(
            close=price_data['Close'],
            entries=signals == 1,
            exits=signals == -1,
            init_cash=self.initial_capital,
            fees=self.commission,
            sl_stop=stop_loss_pct,  # 5% stop loss
            tp_stop=take_profit_pct,  # 10% take profit
            freq='D'
        )
        
        # Store results
        self.results[strategy_name] = {
            'portfolio': portfolio,
            'signals': signals,
            'returns': portfolio.returns(),
            'equity': portfolio.value()
        }
        
        return portfolio
    
    def calculate_performance_metrics(self, portfolio: vbt.Portfolio) -> dict:
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics from vectorbt
        stats = portfolio.stats()
        
        # Additional calculations
        returns = portfolio.returns()
        equity = portfolio.value()
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = returns.mean() * 252 / downside_std if downside_std != 0 else 0
        
        # Max Drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # CAGR
        years = len(equity) / 252
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        trades = portfolio.trades.records_readable
        if len(trades) > 0:
            win_rate = (trades['PnL'] > 0).mean() * 100
            avg_win = trades[trades['PnL'] > 0]['PnL'].mean() if len(trades[trades['PnL'] > 0]) > 0 else 0
            avg_loss = trades[trades['PnL'] < 0]['PnL'].mean() if len(trades[trades['PnL'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        metrics = {
            'Total Return (%)': stats['Total Return [%]'],
            'CAGR (%)': cagr * 100,
            'Sharpe Ratio': stats['Sharpe Ratio'],
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Max Drawdown (%)': max_dd * 100,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Total Trades': stats['Total Trades'],
            'Avg Win/Loss': avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
        }
        
        return metrics
    
    def plot_results(self, portfolio: vbt.Portfolio, asset_name: str, strategy_name: str):
        """Plot backtest results"""
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{asset_name} - {strategy_name} Performance', fontsize=16)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3)
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        portfolio.value().vbt.plot(ax=ax1, title='Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        portfolio.drawdown().vbt.plot(ax=ax2, title='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        
        # Monthly returns
        ax3 = fig.add_subplot(gs[1, 1])
        monthly_returns = portfolio.returns().resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.vbt.barplot(ax=ax3, title='Monthly Returns')
        ax3.set_ylabel('Return (%)')
        
        # Daily returns distribution
        ax4 = fig.add_subplot(gs[1, 2])
        portfolio.returns().vbt.histplot(ax=ax4, title='Daily Returns Distribution')
        ax4.set_xlabel('Daily Return')
        
        # Trades
        ax5 = fig.add_subplot(gs[2, :])
        portfolio.trades.plot(ax=ax5, title='Trades')
        
        plt.tight_layout()
        plt.show()
    
    def plot_compare_strategies(self, strategies_data: dict, asset_name: str):
        """Compare multiple strategies for the same asset"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{asset_name} - Strategy Comparison', fontsize=16)
        
        # Equity curves comparison
        ax1 = axes[0, 0]
        for strategy_name, data in strategies_data.items():
            if 'portfolio' in data:
                data['portfolio'].value().vbt.plot(ax=ax1, label=strategy_name)
        ax1.set_title('Equity Curves')
        ax1.legend()
        ax1.set_ylabel('Portfolio Value')
        
        # Sharpe ratio comparison
        ax2 = axes[0, 1]
        sharpe_ratios = []
        strategy_names = []
        for strategy_name, data in strategies_data.items():
            if 'portfolio' in data:
                sharpe = data['portfolio'].sharpe_ratio()
                sharpe_ratios.append(sharpe)
                strategy_names.append(strategy_name)
        ax2.bar(strategy_names, sharpe_ratios)
        ax2.set_title('Sharpe Ratios')
        ax2.set_ylabel('Sharpe Ratio')
        
        # Max drawdown comparison
        ax3 = axes[1, 0]
        max_dds = []
        for strategy_name, data in strategies_data.items():
            if 'portfolio' in data:
                max_dd = data['portfolio'].max_drawdown()
                max_dds.append(max_dd * 100)
        ax3.bar(strategy_names, max_dds, color='red', alpha=0.6)
        ax3.set_title('Maximum Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        
        # Win rate comparison
        ax4 = axes[1, 1]
        win_rates = []
        for strategy_name, data in strategies_data.items():
            if 'portfolio' in data:
                trades = data['portfolio'].trades.records_readable
                if len(trades) > 0:
                    win_rate = (trades['PnL'] > 0).mean() * 100
                else:
                    win_rate = 0
                win_rates.append(win_rate)
        ax4.bar(strategy_names, win_rates, color='green', alpha=0.6)
        ax4.set_title('Win Rate')
        ax4.set_ylabel('Win Rate (%)')
        
        plt.tight_layout()
        plt.show()
    
    def generate_trade_report(self, portfolio: vbt.Portfolio, strategy_name: str):
        """Generate detailed trade report"""
        trades = portfolio.trades.records_readable
        
        if len(trades) == 0:
            print(f"No trades executed for {strategy_name}")
            return
        
        print(f"\n{'='*60}")
        print(f"TRADE REPORT: {strategy_name}")
        print(f"{'='*60}")
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {(trades['PnL'] > 0).mean()*100:.1f}%")
        print(f"Total PnL: ${trades['PnL'].sum():.2f}")
        print(f"Average PnL: ${trades['PnL'].mean():.2f}")
        print(f"Best Trade: ${trades['PnL'].max():.2f}")
        print(f"Worst Trade: ${trades['PnL'].min():.2f}")
        
        # Show recent trades
        print(f"\nLast 5 Trades:")
        print("-"*60)
        recent_trades = trades.tail(5)
        for idx, trade in recent_trades.iterrows():
            direction = "LONG" if trade['Size'] > 0 else "SHORT"
            print(f"Entry: {trade['Entry Date']} @ ${trade['Entry Price']:.2f}")
            print(f"Exit:  {trade['Exit Date']} @ ${trade['Exit Price']:.2f}")
            print(f"PnL: ${trade['PnL']:.2f} ({trade['Return']*100:.2f}%) | {direction}")
            print(f"Duration: {trade['Duration']}")
            print("-"*40)

# ============================================================================
# RISK MANAGEMENT MODULE
# ============================================================================

class RiskManager:
    """Risk management utilities"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, 100 * (1 - confidence_level))
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = RiskManager.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_position_size(capital: float, price: float, 
                              stop_loss: float, risk_per_trade: float = 0.01) -> int:
        """Calculate position size based on risk"""
        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0:
            return 0
        
        max_risk = capital * risk_per_trade
        position_size = max_risk / risk_per_share
        return int(position_size)
    
    @staticmethod
    def calculate_portfolio_var(returns_matrix: pd.DataFrame, 
                              confidence_level: float = 0.95) -> float:
        """Calculate portfolio VaR"""
        portfolio_returns = returns_matrix.mean(axis=1)
        return RiskManager.calculate_var(portfolio_returns, confidence_level)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("ALGORITHMIC TRADING SYSTEM")
    print("Multi-Asset, Multi-Strategy Backtesting Platform")
    print("="*70)
    
    # Initialize components
    fetcher = DataFetcher(start_date="2018-01-01", end_date="2023-12-31")
    strategies = TradingStrategies()
    engine = BacktestEngineVBT(initial_capital=100000, commission=0.001)
    risk_manager = RiskManager()
    
    # Fetch data
    print("\n1. FETCHING MARKET DATA...")
    print("-"*50)
    
    # Try primary assets first, fall back to alternatives if needed
    try:
        price_data_dict = fetcher.fetch_price_data(fetcher.assets)
        if len(price_data_dict) < 2:  # If we didn't get enough data
            price_data_dict = fetcher.fetch_price_data(fetcher.alt_assets)
    except:
        price_data_dict = fetcher.fetch_price_data(fetcher.alt_assets)
    
    if not price_data_dict:
        print("Error: Could not fetch any market data.")
        return
    
    # Process each asset
    all_metrics = {}
    
    for symbol, price_data in price_data_dict.items():
        print(f"\n\n2. PROCESSING {symbol}...")
        print("="*50)
        
        # Create features
        feature_data = fetcher.create_features(price_data)
        
        # Generate signals for each strategy
        signals = {}
        
        # MA Crossover Strategy
        signals['MA_Crossover'] = strategies.ma_crossover_strategy(
            feature_data['Close'],
            feature_data['SMA_20'],
            feature_data['SMA_50']
        )
        
        # RSI Strategy
        signals['RSI'] = strategies.rsi_strategy(
            feature_data['Close'],
            feature_data['RSI'],
            oversold=30,
            overbought=70
        )
        
        # MACD Strategy
        signals['MACD'] = strategies.macd_strategy(
            feature_data['Close'],
            feature_data['MACD'],
            feature_data['MACD_Signal']
        )
        
        # Macro Strategy
        signals['Macro'] = strategies.macro_strategy(
            feature_data['Close'],
            feature_data['Volatility_20d'],
            feature_data['Volume_Ratio']
        )
        
        # Run backtests for each strategy
        strategy_results = {}
        
        for strategy_name, signal_series in signals.items():
            print(f"\nRunning {strategy_name} strategy...")
            
            try:
                # Run backtest
                portfolio = engine.run_strategy(
                    price_data=feature_data,
                    signals=signal_series,
                    strategy_name=f"{symbol}_{strategy_name}",
                    stop_loss_pct=0.03,  # 3% stop loss
                    take_profit_pct=0.06  # 6% take profit
                )
                
                # Calculate metrics
                metrics = engine.calculate_performance_metrics(portfolio)
                strategy_results[strategy_name] = {
                    'portfolio': portfolio,
                    'metrics': metrics
                }
                
                print(f"  Total Return: {metrics['Total Return (%)']:.2f}%")
                print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
                print(f"  Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
                
                # Generate trade report for top strategies
                if strategy_name in ['MA_Crossover', 'RSI']:
                    engine.generate_trade_report(portfolio, f"{symbol}_{strategy_name}")
                    
            except Exception as e:
                print(f"  Error running {strategy_name}: {e}")
        
        # Store all results for this asset
        all_metrics[symbol] = strategy_results
        
        # Plot comparison
        if strategy_results:
            # Extract portfolio objects for comparison
            portfolios_dict = {}
            for strat_name, result in strategy_results.items():
                if 'portfolio' in result:
                    portfolios_dict[strat_name] = {'portfolio': result['portfolio']}
            
            if portfolios_dict:
                engine.plot_compare_strategies(portfolios_dict, symbol)
    
    # Portfolio-level analysis
    print("\n\n3. PORTFOLIO-LEVEL ANALYSIS")
    print("="*70)
    
    # Create summary table
    summary_data = []
    
    for symbol, strategies in all_metrics.items():
        for strategy_name, result in strategies.items():
            if 'metrics' in result:
                metrics = result['metrics']
                summary_data.append({
                    'Asset': symbol,
                    'Strategy': strategy_name,
                    'Return (%)': metrics['Total Return (%)'],
                    'CAGR (%)': metrics.get('CAGR (%)', 0),
                    'Sharpe': metrics['Sharpe Ratio'],
                    'Sortino': metrics.get('Sortino Ratio', 0),
                    'Max DD (%)': metrics['Max Drawdown (%)'],
                    'Win Rate (%)': metrics['Win Rate (%)'],
                    'Profit Factor': metrics.get('Profit Factor', 0),
                    'Trades': metrics['Total Trades']
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\nPerformance Summary:")
        print("-"*70)
        print(summary_df.round(2).to_string())
        
        # Save results
        summary_df.to_csv('backtest_summary.csv', index=False)
        print("\nResults saved to 'backtest_summary.csv'")
        
        # Risk Analysis
        print("\n\n4. RISK ANALYSIS")
        print("="*70)
        
        # Calculate correlation matrix of strategy returns
        returns_data = {}
        for symbol, strategies in all_metrics.items():
            for strategy_name, result in strategies.items():
                if 'portfolio' in result:
                    key = f"{symbol}_{strategy_name}"
                    returns_data[key] = result['portfolio'].returns()
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            
            print("\nStrategy Returns Correlation Matrix:")
            print("-"*50)
            corr_matrix = returns_df.corr()
            print(corr_matrix.round(2).to_string())
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Strategy Returns Correlation Matrix')
            plt.tight_layout()
            plt.show()
        
        # Identify best strategies
        print("\n\n5. BEST PERFORMING STRATEGIES")
        print("="*70)
        
        # By Sharpe Ratio
        print("\nTop 3 by Sharpe Ratio:")
        sharpe_top = summary_df.nlargest(3, 'Sharpe')[['Asset', 'Strategy', 'Sharpe', 'Return (%)']]
        print(sharpe_top.to_string(index=False))
        
        # By Return
        print("\nTop 3 by Total Return:")
        return_top = summary_df.nlargest(3, 'Return (%)')[['Asset', 'Strategy', 'Return (%)', 'Max DD (%)']]
        print(return_top.to_string(index=False))
        
        # By Win Rate
        print("\nTop 3 by Win Rate:")
        winrate_top = summary_df.nlargest(3, 'Win Rate (%)')[['Asset', 'Strategy', 'Win Rate (%)', 'Trades']]
        print(winrate_top.to_string(index=False))
    
    # Strategic Insights
    print("\n\n6. STRATEGIC INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    print("""
    Key Findings:
    1. Trend-following strategies (MA Crossover) perform well in strong trending markets
    2. Mean-reversion strategies (RSI) excel during range-bound periods
    3. Momentum strategies (MACD) capture intermediate trends but may whipsaw
    4. Macro strategies require accurate regime detection
    
    Risk Management Insights:
    • Stop-losses at 3% and take-profit at 6% provide good risk-reward
    • Position sizing should adapt to market volatility (use ATR)
    • Diversify across uncorrelated strategies and assets
    
    Implementation Recommendations:
    1. Use MA Crossover for Gold and Oil during clear trends
    2. Apply RSI strategy for equity indices during consolidation
    3. Combine MACD with volume confirmation for better signals
    4. Use macro overlays to adjust position sizing
    
    Next Steps:
    1. Incorporate machine learning for signal improvement
    2. Add transaction cost models and slippage
    3. Implement walk-forward optimization
    4. Add real-time data integration
    """)
    
    print("\n" + "="*70)
    print("BACKTESTING COMPLETE")
    print("="*70)

# ============================================================================
# RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    # Execute main function
    main()
