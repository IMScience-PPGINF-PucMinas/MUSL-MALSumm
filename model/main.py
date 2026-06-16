from configs.configs import get_config
from data.data_loader import get_loader
from .solver import Solver


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model."""
    config      = get_config(mode='train')
    test_config = get_config(mode='test')

    print(config)
    print(test_config)
    print('Currently selected split_index:', config.split_index)

    train_loader = get_loader(config.mode, config.video_type, config.split_index, seed=config.seed)
    test_loader  = get_loader(test_config.mode, test_config.video_type, test_config.split_index)

    solver = Solver(config, train_loader, test_loader)

    solver.build()
    solver.evaluate(-1)   # avalia com pesos aleatórios iniciais
    solver.train()