srun --mem 40G --partition amd-longq --gres=gpu python discriminator.py --gpu --test --checkpoint "models/disc_loss_0.681065380573_iter_100_lr_0.0001.bin"
