import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g,image_size1,image_size2,emb_size,num_classes):
        super(Generator, self).__init__()
        # input shape 1x1
        self.net = nn.Sequential(
            self._block(channels_noise+emb_size, features_g * 16, (3,4),(1,1),(0,1)), 
            self._block(features_g * 16, features_g * 8, (3,3),(1,2),(1,1)), 
            self._block(features_g * 8, features_g * 4, (3,4),(1,2),(1,1)), 
            self._block(features_g * 4, features_g * 2, (4,4),(1,2),(1,1)), 
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=(4,4), stride=(2,2), padding=(1,1)
            ),
            nn.Tanh(),
        )
        self.image_size1 = image_size1
        self.image_size2 = image_size2
        self.embbed = nn.Embedding(num_classes,emb_size)
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x,labels):

        embbeding = self.embbed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x,embbeding],dim=1)
        return self.net(x)