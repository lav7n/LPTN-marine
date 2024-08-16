import torch.nn as nn
import torch.nn.functional as F
import torch

# import segmentation_models_pytorch as smp
# from torchinfo import summary

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=2, device=torch.device('cuda')):
        super(Lap_Pyramid_Conv, self).__init__()
        
        self.interpolate_mode = 'bicubic'
        self.num_high = num_high
        self.device = device
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel.to(self.device)

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel.to(self.device), groups=img.shape[1])
        return out

    def pyramid_decom(self, HR):
        """
        Args:
            HR: High Resolution RGB image
        Returns:
            Laplacian pyramid
        """
        current = HR
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered).to(self.device)
            up = self.upsample(down).to(self.device)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]), mode='bicubic').to(self.device)
            diff = current - up
            # diff =  torch.cat([current, diff], 1).to(self.device)
            pyr.append(diff)
            current = down.to(self.device)
        pyr.append(current)
        
        return pyr

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks, num_classes):
        super(Trans_low, self).__init__()

        model = [nn.Conv2d(3, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, num_classes, 3, padding=1),
            nn.LeakyReLU()]

        self.model = nn.Sequential(*model)   
    
    def forward(self, x):
        out = self.model(x)
        return out
    
class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks, num_classes):
        super(Trans_high, self).__init__()
        
        model = [nn.Conv2d(3+2*num_classes, 64, 3, padding=1),
                nn.LeakyReLU()]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]
            
        model += [nn.Conv2d(64, num_classes, 3, padding=1),
            nn.LeakyReLU()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        model_output = self.model(x)
        return model_output

class Trans_Highest(nn.Module):
    def __init__(self, num_residual_blocks, num_classes):
        super(Trans_Highest, self).__init__()

        model = []
        
        model = [nn.Conv2d(3+num_classes, 16, 3, padding=1),
            nn.LeakyReLU()]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(16)]
        
        model += [
            nn.Conv2d(16, num_classes, 3, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

class highBranch(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, num_classes):
        super(highBranch, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels=num_classes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU())

    def forward(self, x):
        x = self.model(x)
        return x

class maskBranch(nn.Module):
    def __init__(self, kernel_size, padding, num_classes):
        super(maskBranch, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU())

    def forward(self, x):
        x = self.model(x)
        return x   

class upperBranch(nn.Module):
    def __init__(self, kernel_size, padding, num_classes):
        super(upperBranch, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU())

    def forward(self, x):
        output = self.model(x)
        return output

    
class LPTNPaper(nn.Module):
    def __init__(self, nrb_low=5, nrb_high=3, nrb_highest=2, in_channels=3, num_high=2, kernel_size=3, padding=1, num_classes=5, device='cpu'):
        super(LPTNPaper, self).__init__()

        self.interpolate_mode = 'bicubic'
        self.device = device
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, self.device)
        trans_low = Trans_low(nrb_low, num_classes)
        trans_high = Trans_high(nrb_high, num_classes)
        upper_branch = upperBranch(kernel_size, padding, num_classes)
        mask_branch = maskBranch(kernel_size, padding, num_classes)
        trans_highest = Trans_Highest(nrb_highest, num_classes)
        high_branch = highBranch(in_channels, kernel_size, padding, num_classes)
        self.upper_branch = upper_branch
        self.mask_branch = mask_branch
        self.trans_low = trans_low
        self.trans_high = trans_high
        self.trans_highest = trans_highest
        self.high_branch = high_branch
        
    def forward(self, input_img):
        """
        Args:
            HR: High Resolution image
        Returns:
        """
        # create Laplacian Pyramid
        pyr = self.lap_pyramid.pyramid_decom(HR=input_img)
        # print("Pyramid length: ", len(pyr))
        # print("pyr[-1] size: ", pyr[-1].shape)
        # print("pyr[-2] size: ", pyr[-2].shape)
        # print("pyr[-3] size: ", pyr[-3].shape)
        
        # manually instantiating residual for readability
        residual = pyr[-1]
        # print("residual shape: ", residual.shape)
        
        # creating mask
        mask = self.trans_low(pyr[-1])
        # print("mask shape: ", mask.shape)
        
        # upsampling
        up_mask = self.mask_branch(mask)
        # print("up_mask shape: ", up_mask.shape)

        # upsampling
        up_residual = self.high_branch(residual)
        # print("up_residual shape - ", up_residual.shape)
        
        # concatenating
        higher = torch.cat([up_mask, pyr[-2], up_residual], 1)
        # print("higher shape: ", higher.shape)
        
        # creating upper image
        trans_img = self.trans_high(higher)
        # print("trans img shape: ", trans_img.shape)
        
        # concatenating upper image with up
        upper = torch.cat([trans_img, up_mask], 1)
        # print("upper shape: ", upper.shape)
        
        # applying transpose convolution
        up_upper = self.upper_branch(upper)
        # print("up_upper shape: ", up_upper.shape)
        
        # concatenating
        highest = torch.cat([up_upper, pyr[-3]], 1)
        # print("highest shape: ", highest.shape)
        
        # final layers
        output = self.trans_highest(highest)
        
        return output