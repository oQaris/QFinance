import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from src.rl.algs.mvo import top_stocks, modern_portfolio_theory
from src.rl.envs.base_env import BaseEnv
from src.rl.loaders import get_start_end_dates
from src.rl.traint_test.env_builder import load_datasets, time_window
from src.rl.traint_test.trainer import agent_class, env_build, exp_name


def validation(model: BaseAlgorithm,
               lstm_states_start: Optional[tuple[np.ndarray, ...]],
               env: BaseEnv) -> tuple[list[float], Optional[tuple[np.ndarray, ...]]]:
    obs, _ = env.reset()
    lstm_states = lstm_states_start
    episode_starts = np.ones((1,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs,
                                            state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=True)
        obs, _, done, _, _ = env.step(action)
        episode_starts[0] = done
        if done:
            break
        env.render()
    return env.get_portfolio_size_history(), lstm_states


def validation_hold(first_action: np.ndarray, env: BaseEnv):
    old_ttr = env.transaction_threshold_ratio
    env.transaction_threshold_ratio = 0.0
    env.reset()
    action = first_action
    while True:
        obs, _, dones, _, info = env.step(action)
        # action = np.zeros_like(first_action)
        action = obs['portfolio_dist']
        if dones:
            break
    env.transaction_threshold_ratio = old_ttr
    return env.get_portfolio_size_history()


def backtest():
    train, trade = load_datasets()
    env_trade = env_build(trade, env_check=False, verbose=2)
    print(get_start_end_dates(trade))

    trained_model = agent_class.load(f'trained_models/{exp_name}/best_model.zip')

    # todo универсализировать
    # env_train = env_build(train, env_check=False, verbose=0)
    # _, lstm_states = validation(trained_model, None, env_train)
    with open(f'trained_models/{exp_name}/best_lstm_states.pkl', 'rb') as input_file:
        lstm_states = pickle.load(input_file).pi

    # Обучаемся на train
    mvo_result = modern_portfolio_theory(train)

    portfolio_size = trade['tic'].nunique()
    ubah_result = np.array([0] + [1 / portfolio_size] * portfolio_size)

    # Тут именно берём топ акций, которые вырастут на trade,
    # отсекаем первое окно, поскольку оно не участвует в сравнении
    top_10_stocks = top_stocks(trade, 10, skip_first=time_window)

    print('------------------------------------ model ------------------------------------')
    account_value_model, _ = validation(trained_model, lstm_states, env_trade)
    plt.show()  # Визуализируем результат из render()
    print('------------------------------------- mvo -------------------------------------')
    account_value_mvo = validation_hold(mvo_result, env_trade)
    print('------------------------------------- ubah -------------------------------------')
    account_value_ubah = validation_hold(ubah_result, env_trade)
    print('------------------------------------ top_10 ------------------------------------')
    account_value_top_5 = validation_hold(top_10_stocks, env_trade)

    plt.figure(figsize=(12, 6))
    plt.plot(account_value_model, label='Model', color='green')
    plt.plot(account_value_mvo, label='MVO', color='blue')
    plt.plot(account_value_ubah, label='UBAH', color='orange')
    plt.plot(account_value_top_5, label='TOP_10', color='red')

    plt.title('Performance')
    plt.xlabel('Times')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == '__main__':
    backtest()
