import torch


# define the dataset

x = torch.normal(2.0, 2.0, (1000, 1))
y = (x - 4.0) * (x - 3.0) * x ** 2 + torch.normal(0.0, 1.0, (1000, 1))

dataset = torch.utils.data.TensorDataset(x, y)

# define the model

model = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# define the loss function
criteria = torch.nn.MSELoss()

momentums = [0.0, 0.5, 0.9]
# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=m)

# define the data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

losses = []
# train the model
for epoch in range(100):
    for x_batch, y_batch in data_loader:
        y_pred = model(x_batch)
        loss = criteria(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

# plot the loss
import matplotlib.pyplot as plt

plt.plot(losses)
plt.savefig('loss.png')