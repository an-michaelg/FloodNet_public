import ml_collections

def get_config():
    """Get the hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    config.use_wandb = True

    # Hyperparameters for dataset. 
    config.ratio_tr_data = 0.8 
    config.num_workers = 0 
    config.num_classes = 10  
    config.image_dim = (200, 300)

    # Hyperparameters for models.
    config.model = "deeplab_resnet50"
    config.pretrained = True
    config.loss_type = "Dice"

    # Hyperparameters for training.
    config.log_dir = "logs"
    config.use_cuda = True
    config.batch_size = 8
    config.num_epochs = 50
    config.lr = 1e-3 
    config.exponential_gamma = 0.9
 
    return config