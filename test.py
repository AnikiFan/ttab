from task_vector.TaskVectorModel import TaskVectorModel
from task_vector.utils import get_data,get_cifar10_26
import logging
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    writer = SummaryWriter()
    ttm = TaskVectorModel(model=get_cifar10_26(),pool_size=8,num_classes=10,img_size=(3,32,32),batch_size=16,writer=writer)
    data_loader = get_data(True,'gaussian_noise',16)
    correct = []
    for step,epoch,batch in data_loader:
        y_hat = ttm.forward(batch._x).max(dim=1)[1]
        correct.append((y_hat==batch._y).sum().item()/len(y_hat))
        writer.add_scalar('accuracy',correct[-1],step)
        writer.flush()
    print(sum(correct)/len(correct))
