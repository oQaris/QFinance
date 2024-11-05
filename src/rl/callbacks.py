from stable_baselines3.common.callbacks import BaseCallback


class EnvTerminalStatsLoggingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(EnvTerminalStatsLoggingCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self):
        # Проверяем, перейдёт ли среда в терминальное состояние
        # Сделано так, поскольку _on_step() вызывается после reset() при достижении конца эпизода
        if self.env.is_pred_terminal_state():
            profit, max_draw_down, sharpe_ratio, fee_ratio, mean_position_tic, mean_transactions = self.env.get_terminal_stats()
            self.logger.record("environment/profit", profit)
            self.logger.record("environment/max_draw_down", max_draw_down)
            self.logger.record("environment/sharpe_ratio", sharpe_ratio)
            self.logger.record("environment/fee_ratio", fee_ratio)
            self.logger.record("environment/mean_position_tic", mean_position_tic)
            self.logger.record("environment/mean_transactions", mean_transactions)
        return True
