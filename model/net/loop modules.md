## 目录
  
- [1. Resnet](#1-resnet)
- [2. Shufflenetv2](#2-shufflenetv2)
- [3. Densenet](#3-densenet)
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
## 3. Densenet
<img width="826" alt="image" src="https://user-images.githubusercontent.com/63939745/184814674-1bb354bf-f46d-4989-92d7-5b7e4f783dc5.png">
Transition层可以产生θm个特征（通过卷积层）,θ∈(0,1] 是压缩系数;由于密集连接方式，DenseNet提升了梯度的反向传播，使得网络更容易训练。

    class DenseNet(nn.Module):
      def __init__(self,
                   growth_rate: int = 32,
                   block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                   num_init_features: int = 64,
                   bn_size: int = 4,
                   drop_rate: float = 0,
                   num_classes: int = 2,
                   memory_efficient: bool = False):
          super(DenseNet, self).__init__()
          # first conv+bn+relu+pool
          self.features = nn.Sequential(OrderedDict([
              ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
              ("norm0", nn.BatchNorm2d(num_init_features)),
              ("relu0", nn.ReLU(inplace=True)),
              ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
          ]))
          # each dense block
          num_features = num_init_features
          for i, num_layers in enumerate(block_config):
              block = _DenseBlock(num_layers=num_layers,
                                  input_c=num_features,
                                  bn_size=bn_size,
                                  growth_rate=growth_rate,
                                  drop_rate=drop_rate,
                                  memory_efficient=memory_efficient)
              self.features.add_module("denseblock%d" % (i + 1), block)
              num_features = num_features + num_layers * growth_rate
              if i != len(block_config) - 1:
                  trans = _Transition(input_c=num_features,
                                      output_c=num_features // 2)
                  self.features.add_module("transition%d" % (i + 1), trans)
                  num_features = num_features // 2

