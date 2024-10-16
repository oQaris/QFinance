import pandas as pd
from stable_baselines3 import PPO

from src.rl.env import PortfolioOptimizationEnv
from src.rl.loaders import split

time_window = 100


def load_dataset():
    dataset = pd.read_csv('C:/Users/oQaris/Desktop/Git/QFinance/data/pre/2023-10-01_2024-10-04_DAY_final.csv', sep=',')
    dataset['close_orig'] = dataset['close'].copy()
    return split(dataset, train_ratio=0.8, stratification=time_window)


def build_env(dataset):
    features = ['open',
                'close',
                'high',
                'low',
                'volume']
    env_kwargs = {
        'initial_amount': 1000000,
        'comission_fee_pct': 0.003,
        # 'return_last_action':True,
        'time_window': time_window,
        'features': features,
        # 'normalize_df': 'by_fist_time_window_value',
        'normalize_df': None,
    }
    return PortfolioOptimizationEnv(
        dataset,
        **env_kwargs
    )


if __name__ == '__main__':
    total_timesteps = 500000
    train, _ = load_dataset()
    env_train = build_env(train)
    agent = PPO(policy='MlpPolicy', env=env_train)
    trained_model = agent.learn(
        total_timesteps=total_timesteps,
        # callback=TensorboardCallback(),
        progress_bar=True
    )
    trained_model.save('trained_models/agent_ppo')
