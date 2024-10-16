import matplotlib.pyplot as plt
from line_profiler import LineProfiler
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.rl.env import PortfolioOptimizationEnv, distribute_optimally
from src.rl.trainer import load_dataset, build_env


def prediction(model: BaseAlgorithm, env: PortfolioOptimizationEnv, deterministic=True):
    test_obs, _ = env.reset()
    while True:
        action, _states = model.predict(test_obs, deterministic=deterministic)
        test_obs, _, dones, _, _ = env.step(action)
        if dones:
            break
    return env.get_portfolio_size_history()


def backtest():
    _, trade = load_dataset()
    env_trade = build_env(trade)
    trained_model = PPO.load('trained_models/agent_ppo')

    profiler = LineProfiler()
    profiler.add_function(prediction)
    profiler.add_function(env_trade.step)
    profiler.add_function(env_trade._rebalance_portfolio)
    profiler.add_function(env_trade.calc_fee_portfolio)
    profiler.add_function(env_trade._get_state_and_info_from_time_index)
    profiler.add_function(distribute_optimally)

    profiler_wrapper = profiler(prediction)
    account_value = profiler_wrapper(trained_model, env_trade, deterministic=False)

    plt.plot(account_value, label="PPO model")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.title("Performance")
    plt.legend()
    plt.show()

    profiler.print_stats()


if __name__ == '__main__':
    backtest()
