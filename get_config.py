def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    
    # Check for either "train" mode or "test" mode
    config['mode'] = "train"
    
    # True if you want to save training results to wandb (default is False)
    config['use_wandb'] = False
    config['log_dir'] = "logs"
    # test directory must be specified for testing, leave as None for training
    config['test_dir'] = "unet-trained"

    # Hyperparameters for dataset. 
    config['ratio_tr_data'] = 0.8 
    config['num_classes'] = 10

    # Hyperparameters for models.
    config['model'] = "UNet" #deeplab_resnet50/UNet
    config['pretrained'] = False

    # Hyperparameters for training.
    config['use_cuda'] = True # put "False" if you have the GPU out of memory error
    config['batch_size'] = 8 # reduce the batch size if you have the GPU out of memory error
    config['lr'] = 1e-3 # Learning rate
    config['exponential_gamma'] = 0.95 # Learning rate exponential decay
    config['num_epochs'] = 50 # Training epochs
    
    # config['loss_type'] = "Dice" # Option is locked for now
    # config['num_filters'] = 16 # Option is locked for now
    # config['image_dim'] = (256, 256) # Option is locked for now
 
    return config