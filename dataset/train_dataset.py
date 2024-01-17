from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse

# 以下是设置的一些参数，这些参数主要是文件的路径
ir = "D:\\use\\ir\\"
vis = "D:\\use\\vis\\"

# dataset类，用来从根目录中加载数据集
def default_loader(path):
    # 格式设置需要注意，如果是彩色图像就用RGB，灰度图像就用L
    return Image.open(path).convert('L')

class MyTrainDataset(Dataset):
    # 初始化参数说明
    # root: 数据保存的根目录
    # number：数量参数，指明数据中有多少张图片，因为我的图片命名方式都是1.png，2.png...，所以根据number就知道根目录所有文件
    # transform: 一组compose的数据转换方法，主要是将数据转换成pytorch能处理的tensor格式
    # loader： 加载数据的方法，其实就是从文件路径中，将数据加载进入
    def __init__(self, irroot='', visroot='', number=10000, transform=None, loader=default_loader):
        super(MyTrainDataset, self).__init__()
        imgs = []
        for i in range(number):
            path1 = irroot + '/' + str(i+1) + '.png'
            path2 = visroot + '/' + str(i+1) + '.png'
            path = [path1, path2]
            imgs.append(path)
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    # index参数好像实在DataLoader的时候会根据batch自动触发，神奇的事情
    def __getitem__(self, index):
        path = self.imgs[index]
        irimg = self.loader(path[0])
        visimg = self.loader(path[1])
        if self.transform is not None:
            irimg = self.transform(irimg)
            visimg = self.transform(visimg)
        image = [irimg, visimg]
        return image

    def __len__(self):
        return len(self.imgs)


tfs = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#
TrainDataset = MyTrainDataset(irroot=ir, visroot=vis, number=16317, transform=tfs)
# train_set = TrainDatasetFromFolder("./data/", crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
ImageLoader = DataLoader(dataset=TrainDataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
