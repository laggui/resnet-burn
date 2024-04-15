use core::marker::PhantomData;

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    tensor::{backend::Backend, Device, Tensor},
};

/// ResNet [basic residual block](https://paperswithcode.com/method/residual-block) implementation.
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    downsample: Option<Downsample<B>>,
}

/// ResNet [bottleneck residual block](https://paperswithcode.com/method/bottleneck-residual-block)
/// implementation.
#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    downsample: Option<Downsample<B>>,
}

/// Downsample layer applies a 1x1 conv to reduce the resolution (H, W) and adjust the number of channels.
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> Downsample<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self {
        // conv1x1
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init(device);
        let bn = BatchNormConfig::new(out_channels).init(device);

        Self { conv, bn }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        self.bn.forward(out)
    }
}

/// Residual blocks are the building blocks of residual networks (ResNets).
/// The blocks use skip connections to learn residual functions with reference to the layer inputs.
pub trait ResidualBlock<B: Backend> {
    fn init(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self;
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4>;
}

impl<B: Backend> ResidualBlock<B> for BasicBlock<B> {
    fn init(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self {
        // conv3x3
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        let relu = Relu::new();
        // conv3x3
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        let downsample = {
            if in_channels != out_channels {
                Some(Downsample::new(in_channels, out_channels, stride, device))
            } else {
                None
            }
        };

        Self {
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
            downsample,
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Skip connection
        let out = {
            match &self.downsample {
                Some(downsample) => out + downsample.forward(identity),
                None => out + identity,
            }
        };

        // Activation
        self.relu.forward(out)
    }
}

impl<B: Backend> ResidualBlock<B> for Bottleneck<B> {
    fn init(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self {
        // Intermediate output channels w/ expansion = 4
        let int_out_channels = out_channels / 4;
        // conv1x1
        let conv1 = Conv2dConfig::new([in_channels, int_out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(int_out_channels).init(device);
        let relu = Relu::new();
        // conv3x3
        let conv2 = Conv2dConfig::new([int_out_channels, int_out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .init(device);
        let bn2 = BatchNormConfig::new(int_out_channels).init(device);
        // conv1x1
        let conv3 = Conv2dConfig::new([int_out_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init(device);
        let bn3 = BatchNormConfig::new(out_channels).init(device);

        let downsample = {
            if in_channels != out_channels {
                Some(Downsample::new(in_channels, out_channels, stride, device))
            } else {
                None
            }
        };

        Self {
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv3.forward(out);
        let out = self.bn3.forward(out);

        // Skip connection
        let out = {
            match &self.downsample {
                Some(downsample) => out + downsample.forward(identity),
                None => out + identity,
            }
        };

        // Activation
        self.relu.forward(out)
    }
}

/// Collection of sequential residual blocks.
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend, M> {
    blocks: Vec<M>,
    _backend: PhantomData<B>,
}

impl<B: Backend, M: ResidualBlock<B>> LayerBlock<B, M> {
    pub fn new(
        num_blocks: usize,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        device: &Device<B>,
    ) -> Self {
        let blocks = (0..num_blocks)
            .map(|b| {
                if b == 0 {
                    // First block uses the specified stride
                    M::init(in_channels, out_channels, stride, device)
                } else {
                    // Other blocks use a stride of 1
                    M::init(out_channels, out_channels, 1, device)
                }
            })
            .collect();

        Self {
            blocks,
            _backend: PhantomData,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = input;
        for block in &self.blocks {
            out = block.forward(out);
        }
        out
    }
}
