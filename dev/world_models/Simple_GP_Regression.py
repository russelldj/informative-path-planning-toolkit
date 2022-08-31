import math
import torch
import gpytorch
from matplotlib import pyplot as plt

print(torch.cuda.is_available())

axis_samples_train = torch.linspace(-10, 10, 100).cuda()
train_x = torch.meshgrid(axis_samples_train, axis_samples_train)
initial_shape = train_x[0].shape
train_x = [x.flatten() for x in train_x]
train_x = torch.vstack(train_x).T.cuda()

train_y = torch.norm(train_x, dim=1).cuda() + torch.randn(
    train_x.shape[0]
).cuda() * math.sqrt(0.04)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()


import os

smoke_test = "CI" in os.environ
training_iter = 2 if smoke_test else 50
training_iter = 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print(
        "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
        % (
            i + 1,
            training_iter,
            loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item(),
        )
    )
    optimizer.step()


model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    axis_samples_test = torch.linspace(-10, 10, 199).cuda()
    test_x = torch.meshgrid(axis_samples_test, axis_samples_test)

    initial_shape = test_x[0].shape
    test_x = [x.flatten() for x in test_x]
    test_x = torch.vstack(test_x).T
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, axs = plt.subplots(1, 3, figsize=(4, 3))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    lower, upper = [torch.reshape(x, initial_shape) for x in (lower, upper)]
    lower, upper = (x.detach().cpu().numpy() for x in (lower, upper))
    cb0 = axs[0].imshow(lower, vmin=-2, vmax=16)
    cb1 = axs[1].imshow(upper, vmin=-2, vmax=16)
    plt.colorbar(cb0, ax=axs[0], orientation="vertical")
    plt.colorbar(cb1, ax=axs[1], orientation="vertical")
    plt.show()
plt.show()
