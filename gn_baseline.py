from task_vector.TaskVectorModel import TaskVectorModel
from task_vector.utils import get_data,get_cifar10_26_gn,set_seed
import logging
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    seeds = [2022,2023,2024]
    corruptions = [
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'glass_blur',
        'motion_blur',
        'zoom_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'elastic_transform',
        'pixelate',
        'jpeg_compression',
    ]
    corruption_df = pd.DataFrame(
        columns=corruptions,
        index=seeds
    )
    writer = SummaryWriter()
    for corruption in corruptions:
        for seed in seeds:
            set_seed(seed)
            model = get_cifar10_26_gn()
            data_loader = get_data(True,corruption,16,seed=seed)
            correct = []
            for step,epoch,batch in data_loader:
                y_hat = model(batch._x).max(dim=1)[1]
                correct.append((y_hat==batch._y).sum().item()/len(y_hat))
                writer.add_scalar('accuracy',correct[-1],step)
                writer.flush()
            corruption_df.loc[seed,corruption] = sum(correct)/len(correct)
    corruption_df.to_csv(os.path.join(os.curdir,'result','baseline.csv'))
