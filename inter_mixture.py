from task_vector.TaskVectorModel import TaskVectorModel
from task_vector.utils import get_data, get_cifar10_26_gn, set_seed, get_inter_mixture_data
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
    seeds = [2022, 2023, 2024]
    corruption_df = pd.DataFrame(
        columns=['baseline', 'task_vector'],
        index=seeds
    )
    for seed in seeds:
        writer = SummaryWriter()
        set_seed(seed)
        model = get_cifar10_26_gn()
        data_loader = get_inter_mixture_data(True, 16, seed)
        correct = []
        for step, epoch, batch in data_loader:
            y_hat = model(batch._x).max(dim=1)[1]
            correct.append((y_hat == batch._y).sum().item() / len(y_hat))
            writer.add_scalar('accuracy', correct[-1], step)
            writer.flush()
        corruption_df.loc[seed, 'baseline'] = sum(correct) / len(correct)


    for seed in seeds:
        writer = SummaryWriter()
        set_seed(seed)
        model = TaskVectorModel(model=get_cifar10_26_gn(), pool_size=8, num_classes=10, img_size=(3, 32, 32),batch_size=16, writer=writer)
        data_loader = get_inter_mixture_data(True, 16, seed)
        correct = []
        for step, epoch, batch in data_loader:
            y_hat = model(batch._x).max(dim=1)[1]
            correct.append((y_hat == batch._y).sum().item() / len(y_hat))
            writer.add_scalar('accuracy', correct[-1], step)
            writer.flush()
        corruption_df.loc[seed, 'task_vector'] = sum(correct) / len(correct)
    corruption_df.to_csv(os.path.join(os.curdir, 'result', 'inter_mixture.csv'))
