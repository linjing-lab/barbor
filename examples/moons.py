import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from barbor import barbor

X, y = make_moons(n_samples=500, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

class NonConvexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

model = NonConvexNet()
criterion = nn.CrossEntropyLoss()

optimizer = barbor(
        model.parameters(),
        lr=1.0,
        method='alternating',
        gamma=1e-8,
        min_step=1e-8,
        max_step=1e3,
        adaptive_restart=True,
        restart_condition='both',
        restart_tol=0.9,
        momentum=0.0,
        nesterov=False
    )

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            test_pred = model(X_test).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean()
            test_acc = (test_pred == y_test).float().mean()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")
        print(optimizer.get_gradient_history_info())

model.eval()
with torch.no_grad():
    test_pred = model(X_test).argmax(dim=1)
    test_acc = (test_pred == y_test).float().mean()
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")