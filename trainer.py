import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Union
from pathlib import Path

class Trainer:
    def __init__(self, dataset, model, cfg="./default.yaml") -> None:
        print("Trainer constructor")
        self.args, self.hyps = self._get_config(cfg)

    def _get_config(self, config: Union[str, Path, DictConfig] = None):
        """
        Accepts yaml file name or DictConfig containing experiment configuration.
        Returns train and hyps namespace
        :param conf: Optional file name or DictConfig object
        """
        try:
            if isinstance(config, str) or isinstance(config, Path):
                config = OmegaConf.load(config)
            print(f"args: \n {config.args} \n hyps: \n {config.hyps}")
            return config.args, config.hyps
        except KeyError:
            raise Exception("Missing key(s) in config")

    def fit(self):
        model = self.get_model(self.args.model)
        train_loader = self.get_dataloader(self.args.trainset)
        optimizer = self.build_optimizer(self.args.optimizer, self.hyps)
        
        for epoch in self.args.epochs:
            for data, label in train_loader:
                loss = self.criterion(model(data), label) # forward pass
                loss.backward() # backward pass
                optimizer.step()
        
    def get_dataloader(self, trainset):
        pass

    def get_model(self, model):
        pass

    def build_optimizer(self, optimizer):
        pass


@hydra.main(version_base=None, config_path=".", config_name="default")
def train(cfg):
    trainer = Trainer("", "", cfg)



if __name__ == "__main__":
    train()