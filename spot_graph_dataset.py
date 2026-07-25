import os
import logging
from typing import List, Optional, Callable

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SpotGraphDataset(Dataset):
    def __init__(
        self,
        graph_dir: str,
        file_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        gene_list: Optional[List[str]] = None,
    ):
        self.graph_dir = graph_dir
        self.transform = transform
        self.gene_list = gene_list

        if file_list is not None:
            self.file_list = sorted(file_list)
        else:
            self.file_list = sorted([
                f for f in os.listdir(graph_dir)
                if f.endswith(".pt")
            ])

        if len(self.file_list) == 0:
            raise FileNotFoundError(f"在 {graph_dir} 中未找到 .pt 文件")

        first = self._load(0)
        self._num_features = int(first.x.size(1))
        self._num_genes = int(first.y.size(1))

        if self.gene_list is not None:
            if len(self.gene_list) != self._num_genes:
                raise ValueError(
                    f"Gene list length ({len(self.gene_list)}) does not match "
                    f"data y dimension ({self._num_genes})."
                )

        for i in range(1, len(self.file_list)):
            data = self._load(i)
            if data.y.size(1) != self._num_genes:
                raise ValueError(
                    f"基因数不一致: 文件 0 有 {self._num_genes} 个基因, "
                    f"文件 {i} ({self.file_list[i]}) 有 {data.y.size(1)} 个基因"
                )
            if data.x.size(1) != self._num_features:
                raise ValueError(
                    f"特征维度不一致: 文件 0 有 {self._num_features} 维, "
                    f"文件 {i} ({self.file_list[i]}) 有 {data.x.size(1)} 维"
                )

        logging.info(
            f"SpotGraphDataset: {len(self.file_list)} graphs, "
            f"num_features={self._num_features}, num_genes={self._num_genes}"
        )

    def _load(self, idx: int):
        path = os.path.join(self.graph_dir, self.file_list[idx])
        data = torch.load(path, map_location="cpu", weights_only=False)

        required_attrs = [
            "x", "y", "pos",
            "edge_index_low", "edge_weight_low",
            "edge_index_high", "edge_weight_high",
        ]
        for field in required_attrs:
            if not hasattr(data, field):
                raise ValueError(
                    "Graph file does not contain MCFG-style low/high graph fields. "
                    "Please rebuild graphs using the updated graph builder."
                )

        assert data.edge_index_low.size(0) == 2
        assert data.edge_weight_low.numel() == data.edge_index_low.size(1)
        assert data.edge_index_high.size(0) == 2
        assert data.edge_weight_high.numel() == data.edge_index_high.size(1)

        if data.edge_index_low.numel() > 0:
            assert data.edge_index_low.max() < data.x.size(0)
        if data.edge_index_high.numel() > 0:
            assert data.edge_index_high.max() < data.x.size(0)

        return data

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = self._load(idx)

        if self.transform is not None:
            data = self.transform(data)

        return data

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def num_genes(self) -> int:
        return self._num_genes

    @property
    def sample_names(self) -> List[str]:
        names = []
        for i in range(len(self.file_list)):
            data = self._load(i)
            names.append(getattr(data, "sample_name", self.file_list[i]))
        return names
