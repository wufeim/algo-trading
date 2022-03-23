import _init_paths

import argparse
import math
import os

import numpy as np
import pandas
import yfinance as yf


def parse_args():
    parser = argparse.ArgumentParser(description='Calcuate Sharpe ratio')
    parser.add_argument('--stock', type=str, default='IGE')
    parser.add_argument('--hedge', type=str, default='SPY')
    parser.add_argument('--buy', type=str, default='2001-11-26')
    parser.add_argument('--sell', type=str, default='2007-11-14')
    parser.add_argument('--risk_free_rate', type=float, default=0.04)
    return parser.parse_args()


def main():
    args = parse_args()

    data_ige = yf.download('IGE')
    mask = (data_ige.index >= args.buy) & (data_ige.index <= args.sell)
    closes_ige = data_ige.loc[mask]['Close'].to_numpy()

    data_spy = yf.download('SPY')
    mask = (data_spy.index >= args.buy) & (data_spy.index <= args.sell)
    closes_spy = data_spy.loc[mask]['Close'].to_numpy()

    # Long-only
    daily_ige = (closes_ige[1:] - closes_ige[0:-1]) / closes_ige[0:-1]
    excess_returns = daily_ige - args.risk_free_rate/252
    long_only_sharpe = math.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    print(f'Long-only Sharpe ratio {long_only_sharpe:.4f}')

    # Market-neutral
    daily_spy = (closes_spy[1:] - closes_spy[0:-1]) / closes_spy[0:-1]
    net_ret = (daily_ige - daily_spy) / 2
    market_neutral_sharpe = math.sqrt(252) * np.mean(net_ret) / np.std(net_ret)
    print(f'Market-neutral Sharpe ratio {market_neutral_sharpe:.4f}')


if __name__ == '__main__':
    main()
