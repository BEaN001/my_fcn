from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class FCNDataSet(Dataset):
    def __init__(self, data_label_dir):
        super(FCNDataSet, self).__init__()

        self.data_label_dir = data_label_dir
        self.images_dir = []
        self.labels_dir = []

        self.gen_imgs_labels_dir()

    def gen_imgs_labels_dir(self):
        """
        get absolute directory of all input images and all label images
        :return:
        """
        imgs = os.listdir(self.data_label_dir)
        for im in imgs:
            if im == '.DS_Store' or im.split('.')[-1] == 'json':
                continue
            im_input = im + '/img.png'
            im_label = im + '/label.png'
            self.images_dir.append(os.path.join(self.data_label_dir, im_input))
            self.labels_dir.append(os.path.join(self.data_label_dir, im_label))
        print('total number of image or label is {}'.format(len(self.images_dir)))

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        image_input_dir = self.images_dir[idx]
        image_label_dir = self.labels_dir[idx]
        im_input = Image.open(image_input_dir)
        im_label = Image.open(image_label_dir)
        im_input = transforms.ToTensor()(im_input)
        im_label = transforms.ToTensor()(im_label)

        return im_input.float(), im_label.float(), image_input_dir, image_label_dir


# test
if __name__ == '__main__':
    data_label_dir = '../dataset/label_resize/'

    data = FCNDataSet(data_label_dir)

    print(len(data))

    im_input, im_label, input_dir, label_dir = data[0]
    print(im_input.shape, im_label.shape)
    print(input_dir, label_dir)
    im_input = transforms.ToPILImage()(im_input)
    im_label = transforms.ToPILImage()(im_label)
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(im_input)
    # plt.pause(5)
    plt.subplot(1, 2, 2)
    plt.imshow(im_label)
    plt.show()
    print('finish!')
