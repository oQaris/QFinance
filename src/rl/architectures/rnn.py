import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.architectures.base import BaseNetwork


class RNNPolicyNetwork(nn.Module, BaseNetwork):
    def __init__(self, tic_count, window_size, window_features_num, indicators_num, common_num):
        super().__init__()
        out_common_dim = 256
        BaseNetwork.__init__(self, out_common_dim, out_common_dim)

        self.tic_count = tic_count
        self.window_size = window_size
        self.window_features_num = window_features_num
        self.indicators_num = indicators_num
        self.common_num = common_num

        # Обработка исторических данных по акциям с помощью GRU
        self.rnn_hist = nn.GRU(input_size=self.window_features_num, hidden_size=128, num_layers=1, batch_first=True)

        # Обработка технических и фундаментальных индикаторов по акциям
        self.fc_indicators = nn.Linear(self.indicators_num, 128)

        # Объединение признаков по акциям
        self.fc_stock = nn.Linear(128 + 128, 128)

        # Обработка общих данных рынка и текущего времени
        self.fc_common = nn.Linear(self.common_num, 128)

        # Обработка текущего распределения портфеля
        self.fc_portfolio = nn.Linear(self.tic_count + 1, 128)

        # Финальные слои для вывода действий
        self.fc_final = nn.Linear(128 * 3, out_common_dim)

    def forward(self, x):
        # Извлечение данных из словаря
        price_data, indicators, common_data, portfolio_dist = self.extract_features_from_dict(x)
        batch_size = price_data.size(0)

        # Перестановка и изменение размерности для обработки GRU
        # [batch_size, tic_count, window_size, window_features_num]
        hist_data = price_data.permute(0, 2, 3, 1)
        # [batch_size * tic_count, window_size, window_features_num]
        hist_data = hist_data.reshape(-1, self.window_size, self.window_features_num)

        # Пропуск через GRU
        rnn_out, _ = self.rnn_hist(hist_data)
        # Используем последний выход GRU
        hist_features = rnn_out[:, -1, :]

        # Обработка индикаторов
        # [batch_size * tic_count, indicators_num]
        indicators = indicators.reshape(-1, self.indicators_num)
        indicators_features = F.relu(self.fc_indicators(indicators))

        # Объединение признаков по акциям
        stock_features = torch.cat([hist_features, indicators_features], dim=1)
        stock_features = F.relu(self.fc_stock(stock_features))

        # Восстановление размерности и агрегирование по акциям
        stock_features = stock_features.view(batch_size, self.tic_count, -1)
        aggregated_stock_features = torch.mean(stock_features, dim=1)

        # Обработка общих данных рынка
        common_features = F.relu(self.fc_common(common_data))

        # Обработка текущего распределения портфеля
        portfolio_features = F.relu(self.fc_portfolio(portfolio_dist))

        # Объединение всех признаков
        combined_features = torch.cat([aggregated_stock_features, common_features, portfolio_features], dim=1)

        # Финальные слои для получения вероятностей действий
        x = F.relu(self.fc_final(combined_features))
        # action_logits = self.fc_action(x)
        # action_probs = F.softmax(action_logits, dim=1)

        return x, x

    def forward_actor(self, input_dict):
        return self.forward(input_dict)[0]

    def forward_critic(self, input_dict):
        return self.forward(input_dict)[1]
