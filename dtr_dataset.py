import torch as flow


class ImagenetStyleDataset(flow.utils.data.Dataset):
    def __init__(self, batch_size: int, length: int = 224, device=flow.device("cpu")) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.length = length
        self.device = device

    def __len__(self):
        return 99999999

    def __getitem__(self, idx):
        return flow.ones(self.batch_size, 3, self.length, self.length, dtype=flow.float32, device=self.device), flow.ones(self.batch_size, dtype=flow.int64, device=self.device)