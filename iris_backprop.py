from timeit import default_timer

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Net(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
    ) -> None:
        super().__init__()
        self._linear_1 = nn.Linear(in_feats, hidden_feats)
        self._linear_2 = nn.Linear(hidden_feats, hidden_feats)
        self._linear_3 = nn.Linear(hidden_feats, out_feats)
        self._activation_1 = nn.ReLU()
        self._activation_2 = nn.ReLU()
        self._classification = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._linear_1(inputs)
        x = self._activation_1(x)
        x = self._linear_2(x)
        x = self._activation_2(x)
        x = self._linear_3(x)
        x = self._classification(x)

        return x


if __name__ == '__main__':
    torch.manual_seed(1337)

    dataset = load_iris()
    scaler = StandardScaler()

    data = scaler.fit_transform(dataset['data'])
    labels = dataset['target']

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=13)

    device = torch.device('cpu')

    train_data = torch.from_numpy(train_data).float().to(device)
    test_data = torch.from_numpy(test_data).float().to(device)

    train_labels = torch.from_numpy(train_labels).to(device)
    test_labels = torch.from_numpy(test_labels).to(device)

    NUM_EPOCHS = 14

    model = Net(4, 50, 3).to(device)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, 1 + NUM_EPOCHS):
        start = default_timer()

        model.train()

        outputs = model(train_data)

        train_loss = loss_function(outputs, train_labels)

        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            outputs = model(test_data)

        test_loss = loss_function(outputs, test_labels)
        test_accuracy = (torch.argmax(outputs, dim=1) ==
                         test_labels).type(torch.FloatTensor).mean()

        stop = default_timer()

        print(
            f'Epoch: {epoch:3} Train Loss: {train_loss:.2f} '
            f'Test Loss: {test_loss:.2f} '
            f'Test Accuracy: {test_accuracy * 100:.2f} % '
            f'Epoch Time: {stop - start:.2f} s.'
        )
