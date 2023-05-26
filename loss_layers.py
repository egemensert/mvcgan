import torch
import torch.nn as nn

class SSIM(nn.Module):
    def __init__(self) -> None:
        super(SSIM, self).__init__()

        self.mean_x_pool = nn.AvgPool2d(3, 1)
        self.mean_y_pool = nn.AvgPool2d(3, 1)
        self.var_x_pool = nn.AvgPool2d(3, 1)
        self.var_y_pool = nn.AvgPool2d(3, 1)
        self.cov_xy_pool = nn.AvgPool2d(3, 1)

        self.reflection = nn.ReflectionPad2d(1)

    def forward(self, x, y, c1=0.01**2, c2=0.03**2):
        x = self.reflection(x)
        y = self.reflection(y)

        mean_x = self.mean_x_pool(x)
        mean_y = self.mean_y_pool(y)

        x2 = x ** 2
        y2 = y ** 2

        mx2 = mean_x ** 2
        my2 = mean_y ** 2
        cov = x * y


        var_x = self.var_x_pool(x2) - mx2
        var_y = self.var_y_pool(y2) - my2
        cov_xy = self.cov_xy_pool(cov) - mean_x * mean_y

        num = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
        den = (mx2 + my2 + c1) * (var_x + var_y + c2)

        out = (1 - num / den) / 2
        return torch.clamp(out, 0, 1)