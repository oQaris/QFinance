import os
import pickle
from typing import List, Dict, Any, Union, Optional, Tuple, Callable

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


def log_terminal_stats(locals_dict, logger, marker='env'):
    dones = locals_dict.get('dones', [])
    infos = locals_dict.get('infos', [])

    if not all(dones):
        if any(dones):
            raise ValueError('Episode lengths should be the same in all training environments')
        return True

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

    # Вычисляем средние значения и записываем в логгер
    num_envs = len(infos)
    for key, total_value in aggregated_stats.items():
        mean_value = total_value / num_envs
        logger.record(f'{marker}/{key}', mean_value)


class EnvTerminalStatsLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EnvTerminalStatsLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        log_terminal_stats(self.locals, self.logger, marker='env_train')
        return True


class CustomEvalCallback(EventCallback):
    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            n_eval_episodes: int = 1,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,  # todo рендерить каждые n эпизодов
            verbose: int = 1,
            warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []
        self.is_recurrent: bool = False

    def _init_callback(self) -> None:
        self.is_recurrent = isinstance(self.model, RecurrentPPO)

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        """
        log_terminal_stats(locals_, self.logger, marker='env_test')

        if locals_['done']:
            maybe_is_success = locals_['info'].get('is_success')
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

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

        # Reset success rate buffer
        self._is_success_buffer = []

        if self.is_recurrent:
            episode_rewards, episode_lengths = recurrent_evaluate_policy(
                self.model,  # type: ignore
                self.eval_env,
                self.model._last_lstm_states.pi,  # type: ignore
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
        else:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,  # type: ignore
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=lambda l, g: self._log_success_callback(l),
            )

        if self.log_path is not None:
            assert isinstance(episode_rewards, list)
            assert isinstance(episode_lengths, list)
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,  # type: ignore
                ep_lengths=self.evaluations_length,  # type: ignore
                **kwargs,
            )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = float(mean_reward)

        if self.verbose >= 1:
            print(
                f'Eval num_timesteps={self.num_timesteps}, ' f'episode_reward={mean_reward:.2f} +/- {std_reward:.2f}')
            print(f'Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}')
        # Add to current Logger
        self.logger.record('eval/mean_reward', float(mean_reward))
        self.logger.record('eval/mean_ep_length', mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f'Success rate: {100 * success_rate:.2f}%')
            self.logger.record('eval/success_rate', success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record('time/total_timesteps', self.num_timesteps, exclude='tensorboard')
        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print(f'New best mean reward!')
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                if self.is_recurrent:
                    with open(os.path.join(self.best_model_save_path, 'best_lstm_states'), "wb") as fp:
                        pickle.dump(self.model._last_lstm_states, fp)
            self.best_mean_reward = float(mean_reward)

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def recurrent_evaluate_policy(
        model: RecurrentPPO,
        env: Union[gym.Env, VecEnv],
        start_states: RNNStates,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    See `stable_baselines3.common.evaluation.evaluate_policy` for more details.
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype='int')
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype='int')

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype='int')
    observations = env.reset()
    states = start_states
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals())

                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations
        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: ' f'{mean_reward:.2f} < {reward_threshold:.2f}'
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
