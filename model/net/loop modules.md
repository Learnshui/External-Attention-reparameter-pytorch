## 目录
  
- [1. Resnet](#1-resnet)
- [2. Shufflenetv2](#1-shufflenetv2)

## 1. Resnet
<img width="900" alt="image" src="https://user-images.githubusercontent.com/63939745/184646366-a3000d5f-d91b-43d6-b9fe-dc11e73caa97.png">

      def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)
## 2. Shufflenetv2
<img width="500" alt="image" src="https://user-images.githubusercontent.com/63939745/184789667-0aca772f-f138-451a-bb14-e8d688697d05.png"><img width="500" alt="image" src="https://user-images.githubusercontent.com/63939745/184789704-ce742989-50b0-4b2a-848f-85b43af9b22f.png">

      # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        #这一块写得好 当stride为1时候，先将输入通过Channel Split在channel维度均分为两份，接着将第一个分支branch1直接与branch2cat，当stride为2时候，由于输入直接送到两个分支中，因此最终输出的channel是翻倍的
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

