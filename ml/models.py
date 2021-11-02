import torch
import torchvision


class LinearModel(torch.nn.Module):
    def __init__(self, hyperparameters: dict):
        super().__init__()

        # Get model config
        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]
        self.hidden_dims = hyperparameters["hidden_dims"]
        self.negative_slope = hyperparameters.get("negative_slope", 0.2)

        # Create layer list
        self.layers = torch.nn.ModuleList([])
        all_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))

        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.layers[-1](x)
        return torch.nn.functional.softmax(x, dim=-1)


class CNN(torch.nn.Module):
    def __init__(self, hyperparameters: dict):
        super().__init__()

        # parameters
        self.pooling = hyperparameters["pooling"]
        self.channels = hyperparameters["channels"]
        self.kernels = hyperparameters["kernels"]
        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]

        # layers
        self.convs = torch.nn.ModuleList([])
        last_ch = 1
        width = self.input_dim[-1]
        for ch, k in zip(self.channels, self.kernels):
            self.convs.append(torch.nn.Conv2d(last_ch, ch, kernel_size=k, stride=1))
            width = (width - k + 1) // 2
            last_ch = ch
        self.out = torch.nn.Linear(ch * width * width, self.output_dim)

    def forward(self, x):
        x = x.view(-1, *self.input_dim)
        for conv in self.convs:
            x = conv(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return torch.nn.functional.softmax(x, dim=-1)
