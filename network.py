import os
import torch
from losses import dice_loss_logits, iou_logits
from utils import label_to_one_hot, vis_pred
import wandb

SHOW_IMAGES = True

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = model
        self.num_classes = config['num_classes']
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config['exponential_gamma'])
        self.loss_type = config['loss_type']
        if self.config['use_cuda']: 
            self.model.cuda()
        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        if self.config['use_wandb']:
            if self.config['mode'] == "train":
                self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
            elif self.config['mode'] == "test":
                self.log_dir = os.path.join(self.config['log_dir'], self.config['test_dir'])
            else:
                raise NotImplementedError
        else:
            self.log_dir = self.config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stopping strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")
 
    
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        # Loading optimizer.
        self.optimizer.load_state_dict(load_res["optimizer"])

    
    def _get_loss(self, pred, data):
        """ Compute loss function based on configs """
        if self.loss_type == "Dice":
            loss = dice_loss_logits(pred, data)
        else:
            raise NotImplementedError
        return loss
            
    def train(self, loader_tr, loader_va):
        """Training pipeline."""
        # Switch model into train mode.
        self.model.train()
        best_va_acc = 0.0 # Record the best validation metrics.

        for epoch in range(self.config['num_epochs']):
            losses = []
            for data in loader_tr:
                # Transfer data from CPU to GPU.
                if self.config['use_cuda']:
                    for key in data.keys():
                        data[key] = data[key].cuda()
                
                # get the logit output
                pred = self.model(data["image"]) #BxCxHxW
                if self.config['model'] == "deeplab_resnet50":
                    pred = pred['out']
                target = label_to_one_hot(data["mask"], self.num_classes).contiguous()
                loss = self._get_loss(pred, target)
                losses += [loss]

                # Calculate the gradient.
                loss.backward()
                # Update the parameters according to the gradient.
                self.optimizer.step()
                # Zero the parameter gradients in the optimizer
                self.optimizer.zero_grad()

            loss_avg = torch.mean(torch.stack(losses)).item()
            acc, val_loss = self.test(loader_va, mode="valid")
            self.scheduler.step()
            
            # Save model every epoch.
            self._save(self.checkpts_file)
            if self.config['use_wandb']:
                wandb.log({"tr_loss":loss_avg, "val_loss":val_loss, "val_iou":acc})

            # Early stopping strategy.
            if acc > best_va_acc:
                # Save model with the best accuracy on validation set.
                best_va_acc = acc
                self._save(self.bestmodel_file)
            print(
                "Epoch: %3d, loss_avg: %.5f, val loss_avg: %.5f, val IoU: %.5f, lowest val IoU: %.5f"
                % (epoch, loss_avg, val_loss, acc, best_va_acc)
            )

            # Recording training losses and validation performance.
            self.train_losses += [loss_avg]
            self.valid_oas += [acc]
            self.idx_steps += [epoch]

    @torch.no_grad()
    def test(self, loader_te, mode="test", vis_all=False):
        """Estimating the performance of model on the given dataset."""
        # Choose which model to evaluate.
        if mode == "test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()

        accs = []
        losses = []
        num_samples = 0
        for data in loader_te:
            if self.config['use_cuda']:
                for key in data.keys():
                    data[key] = data[key].cuda()
            batch_size = len(data["image"])
            pred = self.model(data["image"])
            if self.config['model'] == "deeplab_resnet50":
                pred = pred['out']
            target = label_to_one_hot(data["mask"], self.num_classes).contiguous()
            
            acc = iou_logits(pred, target)
            accs += [acc * batch_size]
            
            loss = self._get_loss(pred, target)
            losses += [loss]
            
            num_samples += batch_size
            
            if SHOW_IMAGES:
                if vis_all:
                    out_segmap = torch.argmax(pred, dim=1, keepdim=True).detach()
                    for i in range(batch_size):
                        vis_pred(data["image"][i], out_segmap[i], data["mask"][i])

        avg_acc = torch.stack(accs).sum() / num_samples
        avg_loss = torch.mean(torch.stack(losses)).item()
        
        # print one instance of the results
        if SHOW_IMAGES:
            if not vis_all:
                out_segmap = torch.argmax(pred, dim=1, keepdim=True).detach()
                vis_pred(data["image"][0], out_segmap[0], data["mask"][0])

        # Switch the model into training mode
        self.model.train()
        return avg_acc, avg_loss
        


# if __name__ == "__main__":
    # """Main for mock testing."""
    # from get_config import get_config
    # from get_dataloader import get_dataloader
    # from get_model import get_model

    # config = get_config()
    
    # if config['use_wandb']:
        # run = wandb.init(project="floodNet-baseline", 
                         # entity="guangnan", 
                         # config=config)
    
    # model = get_model(config)
    # net = Network(model, config)
    # dataloader_tr, dataloader_va = get_dataloader(config, mode="train")
    
    # data = next(iter(dataloader_tr)) 
    # image, mask = data["image"].cuda(), data["mask"].cuda()
    # target = label_to_one_hot(mask, config['num_classes).contiguous()
    # pred = model(image)["out"] #BxCxHxW
    # loss = dice_loss_logits(pred, target)
    # iou = iou_logits(pred, target)
    # out_segmap = torch.argmax(pred, dim=1, keepdim=True).detach()
    # vis_pred(image[0], out_segmap[0], mask[0])
    
    # if config['mode']=="train":
        # net.train(dataloader_tr, dataloader_va)
        # net.test(dataloader_va, mode="test", vis_all = True)
    # if config['mode']=="test":
        # net.test(dataloader_va, mode="test", vis_all = True)
    # if config['use_wandb']:
        # wandb.finish()