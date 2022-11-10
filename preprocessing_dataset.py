from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as scio

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'



def default_loader(path):
    mat = scio.loadmat(path)
    datamat = mat['new']
    return datamat


class MyDataset(Dataset):
    def __init__(self, txt, transform = None, target_transform = None, loader = default_loader):
        super(MyDataset,self).__init__()#对继承自父类的属性进行初始化
        imgs = []
        with open(txt, 'r') as f:
            rows = f.readlines()
            for line in rows: #迭代该列表#按行循环txt文本中的内
                try:
                    words = line.split(' ')#用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
                    imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
                                                          #很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
                except:
                    continue
                self.imgs = imgs
                self.transform = transform
                self.target_transform = target_transform
                self.loader = loader
    #*************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************
    def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index] # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn) # 按照路径读取图片
        img = img.astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)   #数据标签转换为Tensor
            return img, label #return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    #**********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************
    def __len__(self):
        #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)