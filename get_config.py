def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    config['mode'] = "train"
    config['use_wandb'] = True

    # Hyperparameters for dataset. 
    config['ratio_tr_data'] = 0.8 
    config['num_workers'] = 0 
    config['num_classes'] = 10  
    config['image_dim'] = (256, 256)

    # Hyperparameters for models.
    config['model'] = "UNet"
    config['pretrained'] = False
    config['loss_type'] = "Dice"
    config['num_filters'] = 16

    # Hyperparameters for training.
    config['log_dir'] = "logs"
    config['use_cuda'] = True
    config['batch_size'] = 8
    config['lr'] = 1e-3 
    config['exponential_gamma'] = 0.9
    config['num_epochs'] = 50
 
    return config