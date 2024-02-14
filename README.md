# **👉 [Check out the full tutorial](https://dev.to/laggui/transitioning-from-pytorch-to-burn-45m) 👈**

## ResNet + Burn ✍️

To use ResNet in your application, take a look at the official Burn implementation
[available on GitHub](https://github.com/tracel-ai/models/tree/main/resnet-burn)! It closely follows
this tutorial's implementation but further extends it to provide an easy interface to load the
pre-trained weights for the whole ResNet family of models.

## Example Usage

1. Download the ResNet-18 pre-trained weights from `torchvision`

```sh
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

2. Download a sample image for inference

```sh
wget https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg
```

3. Run the example

```sh
cargo run --release YellowLabradorLooking_new.jpg
```
