import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from ipp_toolkit.world_models.gaussian_process_regression import (
    GaussianProcessRegressionWorldModel,
)


print(torch.cuda.is_available())

axis_samples_train = torch.linspace(-10, 10, 100).cuda()
train_x = torch.meshgrid(axis_samples_train, axis_samples_train)
initial_shape = train_x[0].shape
train_x = [x.flatten() for x in train_x]
train_x = torch.vstack(train_x).T.cuda()

train_y = torch.norm(train_x, dim=1).cuda() + torch.randn(
    train_x.shape[0]
).cuda() * math.sqrt(0.04)

gp = GaussianProcessRegressionWorldModel()
for x, y in zip(train_x, train_y):
    gp.add_observation(x, y)
gp.train_model()
gp.test_model(world_size=(20, 20))
