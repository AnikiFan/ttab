from task_vector.TaskVectorModel import TaskVectorModel
from task_vector.utils import get_data,get_cifar10_26

if __name__ == '__main__':
    ttm = TaskVectorModel(model=get_cifar10_26(),pool_size=10,num_classes=10,img_size=(3,32,32),max_sample=64)
    data_loader = get_data(True,'gaussian_noise',1)
    correct = 0
    for step,epoch,batch in data_loader:
        y_hat = ttm(batch._x).max(dim=1)[1]
        correct += (y_hat==batch._y).item()
    print(correct)
