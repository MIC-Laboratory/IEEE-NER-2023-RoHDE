import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d,image_size1,image_size2,num_classes):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img+1, features_d , (3,4),(1,2),1,1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, (3,4),(1,2),(1,1)),
            self._block(features_d * 2, features_d * 4, (3,4),(1,2),(1,1)),
            self._block(features_d * 4, features_d * 8, (4,3),(2,1),(1,1)),

            nn.Conv2d(features_d * 8, 1, kernel_size=(4,3), stride=(2,2), padding=0),
        )
        self.image_size1 = image_size1
        self.image_size2 = image_size2
        self.num_classes = num_classes
        self.embbed = nn.Embedding(self.num_classes,image_size1*image_size2)
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x,labels):
        emb = self.embbed(labels).view(labels.size(0),1,self.image_size1,self.image_size2)
        x = torch.cat([x,emb],dim=1)
        
        return self.disc(x)