from torch.utils.data import Sampler, RandomSampler, TensorDataset, DataLoader, BatchSampler
import torch
import numpy as np

"""
        BatchSamper  定义batch的indice   Samper定义如何返回batch，之后iter调用batch_samper
"""


class MultitastdataBatchsampler(BatchSampler):
    def __init__(self, sampler, batch_size=1, data_type_idx=-1, drop_last=False, data_type=None):
        self.sampler = sampler
        self.batch_size = batch_size
        self.data_type_idx = data_type_idx
        self.drop_last = drop_last
        self.data_type = data_type

    # def __iter__(self):
    #     """按照data_type的顺序返回batch"""
    #     batch = []
    #     yield_batch = 0
    #     for j in self.data_type:
    #         for idx in self.sampler:
    #             if int(self.sampler.data_source[idx][self.data_type_idx]) == j:
    #                 batch.append(idx)
    #             if len(batch) == self.batch_size:
    #                 yield batch
    #                 yield_batch += 1
    #                 batch = []
    #         if len(batch) > 0 and not self.drop_last:
    #             yield batch
    #             yield_batch += 1
    #             batch = []
        # print(len(self), yield_batch)

    def __iter__(self):
        """随机返回batch"""
        yield_batch = 0
        batch_type_1 = []
        batch_type_2 = []
        for idx in self.sampler:
            if int(self.sampler.data_source[idx][self.data_type_idx]) == self.data_type[0]:
                batch_type_1.append(idx)
            elif int(self.sampler.data_source[idx][self.data_type_idx]) == self.data_type[1]:
                batch_type_2.append(idx)
            if len(batch_type_1) == self.batch_size:
                yield batch_type_1
                yield_batch += 1
                batch_type_1 = []
            if len(batch_type_2) == self.batch_size:
                yield batch_type_2
                batch_type_2 = []
                yield_batch += 1
        if len(batch_type_1) > 0 and not self.drop_last:
            yield batch_type_1
            yield_batch += 1
        if len(batch_type_2) > 0 and not self.drop_last:
            yield batch_type_1
            yield_batch += 1
        print('==================')
        print(len(self),yield_batch)
        # assert len(self) == yield_batch  # 因为最后有可能有两个不够batch的数据类别，所以yield_batch 有可能要比len() 大 1
 
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([0, 1, 1])
# c = np.array([7, 8, 8])
# a = torch.from_numpy(a)
# b = torch.from_numpy(b)
# c = torch.from_numpy(c)

# d = TensorDataset(a, b, c)
# train_sampler = RandomSampler(d)

# multisamper = MultitastdataBatchsampler(train_sampler, batch_size=2, drop_last=False, data_type=[7, 8])
# train_dataloader = DataLoader(d, batch_sampler=multisamper)
# for j in range(2):
#     for i in train_dataloader:
#         a = [int(k) for k in i[-1]]
#         a = set(a)
#         print(a)
#         assert len(a) == 1  # 确保每个batch只有一个数据类型
