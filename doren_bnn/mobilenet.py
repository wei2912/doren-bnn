import torch.nn as nn
from torchinfo import summary


class MobileNetV1(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(MobileNetV1, self).__init__()

        def conv(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def conv_ds(in_channels, out_channels, stride):
            def conv_dw(in_channels, stride):
                return nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        3,
                        stride=stride,
                        padding=1,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                )

            def conv_pw(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, 1, stride=1, padding=0, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            return nn.Sequential(
                conv_dw(in_channels, stride), conv_pw(in_channels, out_channels)
            )

        self.model = nn.Sequential(
            conv(in_channels, 32, 2),
            conv_ds(32, 64, 1),
            conv_ds(64, 128, 2),
            conv_ds(128, 128, 1),
            conv_ds(128, 256, 2),
            conv_ds(256, 256, 1),
            conv_ds(256, 512, 2),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 1024, 2),
            conv_ds(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # model check
    model = MobileNetV1(3, 1000)
    summary(model, input_size=(1, 3, 224, 224))
