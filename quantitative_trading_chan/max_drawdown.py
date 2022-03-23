import _init_paths

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import yfinance as yf


def parse_args():
    parser = argparse.ArgumentParser(description='Calcuate Sharpe ratio')
    parser.add_argument('--stock', type=str, default='IGE')
    parser.add_argument('--hedge', type=str, default='SPY')
    parser.add_argument('--buy', type=str, default='2001-11-26')
    parser.add_argument('--sell', type=str, default='2007-11-14')
    return parser.parse_args()


def calculate_max_drawdown(cum_ret):
    high_watermark = np.zeros(len(cum_ret), dtype=np.float32)
    drawdown = np.zeros(len(cum_ret), dtype=np.float32)
    drawdown_duration = np.zeros(len(cum_ret), dtype=np.int32)
    for t in range(1, len(cum_ret)):
        high_watermark[t] = max(high_watermark[t-1], cum_ret[t])
        drawdown[t] = (1+high_watermark[t]) / (1+cum_ret[t]) - 1
        if drawdown[t] <= 0:
            drawdown_duration[t] = 0
        else:
            drawdown_duration[t] = drawdown_duration[t-1] + 1
    return np.max(drawdown), np.max(drawdown_duration)


def main():
    args = parse_args()

    data_ige = yf.download('IGE')
    mask = (data_ige.index >= args.buy) & (data_ige.index <= args.sell)
    closes_ige = data_ige.loc[mask]['Close'].to_numpy()

    data_spy = yf.download('SPY')
    mask = (data_spy.index >= args.buy) & (data_spy.index <= args.sell)
    closes_spy = data_spy.loc[mask]['Close'].to_numpy()

    daily_ige = (closes_ige[1:] - closes_ige[0:-1]) / closes_ige[0:-1]
    daily_spy = (closes_spy[1:] - closes_spy[0:-1]) / closes_spy[0:-1]
    net_ret = (daily_ige - daily_spy) / 2

    cum_ret = np.cumprod(1+net_ret) - 1
    max_drawdown, max_drawdown_duration = calculate_max_drawdown(cum_ret)
    print(f'max_drawdown {max_drawdown:.4f}')
    print(f'max_drawdown_duration {max_drawdown_duration:d}')

    plt.plot(np.arange(len(cum_ret)), cum_ret)
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns')
    plt.tight_layout()
    plt.savefig('drawdown.png', dpi=240)


if __name__ == '__main__':
    main()
