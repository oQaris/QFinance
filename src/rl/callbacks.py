from stable_baselines3.common.callbacks import BaseCallback

class EnvTerminalStatsLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EnvTerminalStatsLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for idx, done in enumerate(dones):
            if done:
                terminal_stats = infos[idx].get("terminal_stats", {})
                for key, value in terminal_stats.items():
                    self.logger.record(f"env/{key}", value)
        return True
