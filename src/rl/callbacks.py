from stable_baselines3.common.callbacks import BaseCallback


class EnvTerminalStatsLoggingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(EnvTerminalStatsLoggingCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self):
        # Проверяем, перейдёт ли среда в терминальное состояние
        # Сделано так, поскольку _on_step() вызывается после reset() при достижении конца эпизода
        is_next_terminal = self.env._time_index >= len(self.env._sorted_times) - 2
        if is_next_terminal:
            profit, max_draw_down, sharpe_ratio, fee_ratio, num_of_transactions = self.env.get_terminal_stats()
            self.logger.record("environment/profit", profit)
            self.logger.record("environment/max_draw_down", max_draw_down)
            self.logger.record("environment/sharpe_ratio", sharpe_ratio)
            self.logger.record("environment/fee_ratio", fee_ratio)
            self.logger.record("environment/num_of_transactions", num_of_transactions)
        return True
