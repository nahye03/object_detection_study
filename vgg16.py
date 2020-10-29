import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()  # 네트워크를 초기화해서 실행하는 과정

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, image):
        image = F.relu(self.conv1(image))  # inplace 하면 input 으로 들어온 것 자체를 수정하겠다는 뜻.
        image = F.relu(self.conv2(image))
        image = F.max_pool2d(image, 3, 2, 1)
        print("1-2 : ", image.size())

        image = F.relu(self.conv3(image))
        image = F.relu(self.conv4(image))
        image = F.max_pool2d(image, 3, 2, 1)
        print("3-4 : ", image.size())

        image = F.relu(self.conv5(image))
        image = F.relu(self.conv6(image))
        image = F.max_pool2d(image, 3, 2, 1)
        print("5-6 : ", image.size())

        image = F.relu(self.conv7(image))
        image = F.relu(self.conv8(image))
        image = F.relu(self.conv9(image))
        image = F.max_pool2d(image, 3, 2, 1)
        print("7-9 : ", image.size())

        image = F.relu(self.conv10(image))
        image = F.relu(self.conv11(image))
        image = F.relu(self.conv12(image))
        image = F.max_pool2d(image, 3, 2, 1)
        print("10-12 : ", image.size())

        return image


image = torch.rand([1, 3, 300, 300])
model = VGG16()
model(image)
