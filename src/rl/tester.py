from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.rl.algs.mvo import modern_portfolio_theory, top_stocks
from src.rl.env import PortfolioOptimizationEnv
from src.rl.loaders import get_start_end_dates
from src.rl.models import PolicyGradient
from src.rl.nets_zoo import EI3
from src.rl.trainer import load_dataset, build_env, time_window


def validation(predict_fn: Callable[[pd.DataFrame], np.ndarray], env: PortfolioOptimizationEnv):
    test_obs, _ = env.reset()
    while True:
        action = predict_fn(test_obs)
        test_obs, _, dones, _, _ = env.step(action)
        if dones:
            break
    return env.get_portfolio_size_history()


def validation_hold(predict_fn: Callable[[pd.DataFrame], np.ndarray], env: PortfolioOptimizationEnv):
    test_obs, _ = env.reset()
    action = predict_fn(test_obs)
    while True:
        test_obs, _, dones, _, info = env.step(action)
        action = info['real_weights']
        if dones:
            break
    return env.get_portfolio_size_history()


def backtest():
    train, trade = load_dataset()
    env_trade = build_env(trade, verbose=2)
    print(get_start_end_dates(trade))

    # trained_model = PPO.load('trained_models/agent_ppo')
    # model_predict_fun = lambda obs: trained_model.predict(obs, deterministic=True)[0]

    policy = EI3(time_window=time_window, initial_features=len(env_trade._window_features))
    policy.load_state_dict(torch.load('trained_models/policy_EI3.pt'))
    PolicyGradient(env_trade).test(env_trade, online_training_period=999990, policy=policy)

    mvo_result = modern_portfolio_theory(train)  # Обучаемся на train
    mvo_fun = lambda obs: mvo_result

    portfolio_size = trade['tic'].nunique()
    ubah_fun = lambda obs: [0] + [1 / portfolio_size] * portfolio_size

    # Тут именно берём топ акций, которые вырастут на trade,
    # отсекаем первое окно, поскольку оно не участвует в сравнении
    top_5_stocks = top_stocks(trade, 5, skip_first=time_window)
    top_5_fun = lambda obs: top_5_stocks

    top_1_stocks = top_stocks(trade, 1, skip_first=time_window)
    top_1_fun = lambda obs: top_1_stocks

    # profiler = LineProfiler()
    # profiler.add_function(validation)
    # profiler.add_function(env_trade.step)
    # profiler.add_function(env_trade._rebalance_portfolio)
    # profiler.add_function(env_trade._allocate_with_fee)
    # profiler.add_function(env_trade._get_state_and_info_from_time_index)
    # profiler.add_function(discrete_allocation_custom)
    # profiler.add_function(RNNPolicyNetwork.forward)
    #
    # profiler_wrapper = profiler(validation)
    # account_value_model = profiler_wrapper(model_predict_fun, env_trade)
    account_value_model = env_trade.get_portfolio_size_history()
    account_value_mvo = validation_hold(mvo_fun, env_trade)
    account_value_ubah = validation_hold(ubah_fun, env_trade)
    account_value_top_5 = validation_hold(top_5_fun, env_trade)
    account_value_top_1 = validation_hold(top_1_fun, env_trade)

    plt.plot(account_value_model, label="PPO model")
    plt.plot(account_value_mvo, label="MVO")
    plt.plot(account_value_ubah, label="UBAH")
    plt.plot(account_value_top_5, label="TOP_5")
    plt.plot(account_value_top_1, label="TOP_1")
    plt.xlabel("Times")
    plt.ylabel("Portfolio Value")
    plt.title("Performance")
    plt.legend()
    plt.show()

    # profiler.print_stats()


if __name__ == '__main__':
    backtest()
