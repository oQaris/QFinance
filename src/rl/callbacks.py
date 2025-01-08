import os
import pickle
import warnings
from typing import List, Dict, Any, Optional, Callable

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


def get_terminal_stats(dones: list[bool], infos: dict) -> dict[Any, int] | None:
    if not all(dones):
        if any(dones):
            raise ValueError('Episode lengths should be the same in all training environments')
        return None

    aggregated_stats = {}

    for info in infos:
        terminal_stats = info.get('terminal_stats', None)
        if terminal_stats is None:
            raise ValueError('"terminal_stats" does not exist in environment information')

        # Суммируем значения для каждого ключа
        for key, value in terminal_stats.items():
            if key not in aggregated_stats:
                aggregated_stats[key] = 0
            aggregated_stats[key] += value

    # Вычисляем средние значения
    num_envs = len(infos)
    for key, _ in aggregated_stats.items():
        aggregated_stats[key] /= num_envs
    return aggregated_stats

class EnvTerminalStatsLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EnvTerminalStatsLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', {})

        aggregated_stats = get_terminal_stats(dones, infos)
        if aggregated_stats is not None:
            for key, mean_value in aggregated_stats.items():
                self.logger.record(f'env_train/{key}', mean_value)
        return True


class CustomEvalCallback(EventCallback):
    BEST_MODEL_PREFIX = 'best_model'
    LSTM_STATE_SUFFIX = '_lstm_states.pkl'
    DELIMITER = '-'
    ZIP = '.zip'

    def __init__(
            self,
            eval_env: gym.Env | VecEnv,
            by_stats: tuple[str] = ('sharpe_ratio', 'sortino_ratio', 'profit', 'mean_reward'),
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,  # todo рендерить каждые n эпизодов
            verbose: int = 1
    ):
        super().__init__(verbose=verbose)

        if any(self.DELIMITER in stat for stat in by_stats):
            raise ValueError(
                f"Statistics in by_stats cannot contain the '{self.DELIMITER}' character, as it is used as a delimiter in file names.")

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        if len(os.listdir(best_model_save_path)) > 0:
            warnings.warn(f'The directory for saving models is not empty, the results will be overwritten.')

        self.eval_env = eval_env
        self.by_stats = by_stats
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.render = render

        self.best_reward_by_stat = [-np.inf for _ in range(len(by_stats))]
        self.n_eval_episodes = 1
        self.is_recurrent = False

        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []

    def _init_callback(self) -> None:
        self.is_recurrent = isinstance(self.model, RecurrentPPO)
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _create_name_by_stats(self, stats):
        """Create new model name based on updated stats."""
        return f'{self.BEST_MODEL_PREFIX}{self.DELIMITER}{self.DELIMITER.join(stats)}'

    def _get_lstm_file_by_model(self, model_file):
        return os.path.join(self.best_model_save_path, f'{model_file.removesuffix(self.ZIP)}{self.LSTM_STATE_SUFFIX}')

    def _save_model(self, model_name):
        """Save the model and its LSTM states (if recurrent)."""
        if self.best_model_save_path is not None:
            self.model.save(os.path.join(self.best_model_save_path, model_name))
            if self.is_recurrent:
                with open(self._get_lstm_file_by_model(model_name), "wb") as fp:
                    pickle.dump(self.model._last_lstm_states, fp)  # type: ignore

    def _update_model_files(self, updated_stats):
        """Update model files based on updated statistics."""
        if not self.best_model_save_path:
            return

        existing_files = os.listdir(self.best_model_save_path)
        model_files = [f for f in existing_files if f.startswith(self.BEST_MODEL_PREFIX) and f.endswith(self.ZIP)]

        for file in model_files:
            stats_in_file = (file
                             .removeprefix(self.BEST_MODEL_PREFIX + self.DELIMITER)
                             .removesuffix(self.ZIP)
                             .split(self.DELIMITER))
            remaining_stats = [stat for stat in stats_in_file if stat not in updated_stats]
            old_path = os.path.join(self.best_model_save_path, file)

            if remaining_stats:
                new_name = self._create_name_by_stats(remaining_stats)

                if new_name + self.ZIP != file:  # Avoid overwriting the same file
                    new_path = os.path.join(self.best_model_save_path, new_name + self.ZIP)
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(
                        old_path,
                        new_path
                    )
                    if self.is_recurrent:
                        new_lstm_path = self._get_lstm_file_by_model(new_name)
                        if os.path.exists(new_lstm_path):
                            os.remove(new_lstm_path)
                        os.rename(
                            self._get_lstm_file_by_model(file),
                            new_lstm_path
                        )
            else:
                os.remove(old_path)
                if self.is_recurrent:
                    os.remove(self._get_lstm_file_by_model(file))

    def _evaluate(self, aggregated_stats: dict) -> None:
        updated_stats = []
        for i, stat in enumerate(self.by_stats):
            mean_reward = aggregated_stats[stat]
            if mean_reward > self.best_reward_by_stat[i]:
                self.best_reward_by_stat[i] = mean_reward
                updated_stats.append(stat)

        if self.verbose >= 1:
            print(f'Eval num_timesteps={self.num_timesteps}, ' + ", ".join(
                [f'{stat}={aggregated_stats[stat]:.5f}' for stat in self.by_stats]
            ))

        if updated_stats:
            if self.verbose >= 1:
                print(f'New best mean reward! Updated stats: {updated_stats}')
            # Update existing model files first
            self._update_model_files(updated_stats)
            # Save the updated model
            self._save_model(self._create_name_by_stats(updated_stats))
        else:
            plt.close(plt.gcf())  # Закрываем фигуру, отрисованную в render()

    def _success_callback(self, locals_: Dict[str, Any]) -> None:
        """
        Callback passed to the ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        """
        if self.render:
            self.eval_env.render()
        dones = locals_.get('dones', [])
        infos = locals_.get('infos', {})

        aggregated_stats = get_terminal_stats(dones, infos)
        if aggregated_stats is None:
            return

        for key, mean_value in aggregated_stats.items():
            self.logger.record(f'env_test/{key}', mean_value)

        self._evaluate(aggregated_stats)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        if not all(dones) and self.n_calls != 1:
            if any(dones):
                raise ValueError('Episode lengths should be the same in all training environments')
            return True

        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    'Training and eval env are not wrapped the same way, '
                    'see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback '
                    'and warning above.'
                ) from e

        # Не ориентируемся на среднюю награду, считаем терминальные статистики, обновление и рендеринг происходят в callback
        _, _ = custom_evaluate_policy(
            self.model,  # type: ignore
            self.eval_env,
            start_states=self.model._last_lstm_states.pi if self.is_recurrent else None,  # type:  ignore
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            callback=self._success_callback,
        )
        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def custom_evaluate_policy(
        model: BaseAlgorithm,
        env: gym.Env | VecEnv,
        start_states: RNNStates = None,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[list[str], list[int | Any]]:
    """
    See `stable_baselines3.common.evaluation.evaluate_policy` for more details.
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    n_envs = env.num_envs
    if n_envs != 1:
        raise ValueError('Working with sub environments is not yet supported')
    episode_rewards = []
    episode_lengths = []
    episode_count = 0

    current_length = 0
    observations = env.reset()
    states = start_states
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while episode_count < n_eval_episodes:
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_length += 1

        # unpack values so that the callback can access the local variables
        reward = rewards.item()
        done = dones.item()
        info = infos[0]
        episode_starts[0] = done

        if callback is not None:
            callback(locals())

        episode_rewards.append(reward)
        if done:
            episode_lengths.append(current_length)
            current_length = 0
            episode_count += 1

        observations = new_observations

    return episode_rewards, episode_lengths
