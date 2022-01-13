from get_config import get_config
from get_dataloader import get_dataloader
from get_model import get_model
import wandb

from network import Network

config = get_config()

if config['use_wandb']:
	run = wandb.init(project="floodNet-baseline", 
					 entity="guangnan", 
					 config=config)

model = get_model(config)
net = Network(model, config)
dataloader_tr, dataloader_va = get_dataloader(config, mode="train")

if config['mode']=="train":
	net.train(dataloader_tr, dataloader_va)
	net.test(dataloader_va, mode="test", vis_all = True)
if config['mode']=="test":
	net.test(dataloader_va, mode="test", vis_all = True)
if config['use_wandb']:
	wandb.finish()