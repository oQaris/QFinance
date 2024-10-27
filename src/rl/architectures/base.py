class BaseNetwork:
    def __init__(self, latent_dim_pi, latent_dim_vf):
        # Выходные размерности. Необходимы для работы ActorCriticPolicy
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

    def extract_features_from_dict(self, x):
        price_data = x['price_data']  # Размерность: [batch_size, window_features_num, tic_count, window_size]
        indicators = x['indicators']  # Размерность: [batch_size, tic_count, indicators_num]
        common_data = x['common_data']  # Размерность: [batch_size, common_num]
        portfolio_dist = x['portfolio_dist']  # Размерность: [batch_size, tic_count + 1]
        return price_data, indicators, common_data, portfolio_dist
