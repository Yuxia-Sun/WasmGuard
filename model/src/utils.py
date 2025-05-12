import os
import yaml
import torch
import numpy as np
import random
import argparse
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class logger:
    def __init__(self, path):
        self.path = path

    def print_info_to_log(self, message, end):
        print(message, end=end)
        with open(os.path.join(self.path), 'a') as f:
            f.write(message + end)


# 读取配置文件,所有的参数都在args里
def load_config(config_path):
    parser = argparse.ArgumentParser(description='simMalConv upstream task')

    config = load_yaml_config(config_path)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args([])

    args.use_gpu = True if torch.cuda.is_available() else False

    # args.first_n_byte：2**20 是表达式,需要转成数字
    # args.first_n_byte = args.first_n_byte
    return args


def load_yaml_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_model(model, args):
    out = os.path.join(args.simMal_path + "_checkpoint{}.tar".format(args.current_epoch))
    torch.save(model.state_dict(), out)


def save_to_csv(data_dict: dict, base_dir: str):
    """
    将列表中的nparray按照列表名字保存为csv文件
    :param data: [np_array1, np_array_2, ... ]
    :return:
    """
    for k, v in data_dict.items():
        np.savetxt("/".join([base_dir, k + '.csv']), v, delimiter=',')


def load_np_from_csv(path: str):
    data = np.loadtxt(open(path, 'rb'), delimiter=',', skiprows=0)
    return data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def load_dataset(args):
    # 训练集，不需要添加头部，指定0列(md5)为索引
    tr_label_table = pd.read_csv(args.train_label_path, header=None, index_col=0)
    tr_label_table.index = tr_label_table.index.str.lower()
    tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})

    # 测试集，不需要添加头部，指定0列(md5)为索引
    val_label_table = pd.read_csv(args.valid_label_path, header=None, index_col=0)
    val_label_table.index = val_label_table.index.str.lower()
    val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

    print('Train dataset:')
    print('\tTotal', len(tr_label_table), 'files')
    print('\tTraining Count :\n', tr_label_table['ground_truth'].value_counts())
    print()
    print('Test dataset:')
    print('\tTotal', len(val_label_table), 'files')
    print('\tTest Count :\n', val_label_table['ground_truth'].value_counts())

    return tr_label_table, val_label_table


def tensor_basic_info(x):
    return x.dtype, x.shape


# define dataset and dataloader
def set_dataloader(tr_table, val_table, args):
    # tr_table.index: md5, tr_table.ground_truth:label
    train_dataset = ExeDataset(
        list(tr_table.index),
        args.train_data_path,
        list(tr_table.ground_truth),
        args.first_n_byte
    )
    validate_dataset = ExeDataset(
        list(val_table.index),
        args.valid_data_path,
        list(val_table.ground_truth),
        args.first_n_byte
    )
    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.cpu_num,
    )
    validloader = DataLoader(
        dataset=validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.cpu_num,
    )
    return trainloader, validloader


# def exe_to_numpy(file_path, max_length):
#     with open(file_path, "rb") as f:
#         x = np.zeros((max_length,), dtype=np.int_)
#         # 未padding的样本
#         x_trim = np.frombuffer(f.read()[:max_length], dtype=np.uint8)
#         # 样本的真实长度
#         x_length = len(x_trim)
#         # 给样本的每一个二进制都+1
#         x_trim = x_trim.astype(np.int_) + 1
#         # 将x_trim放到padding后的样本中
#         x[: x_length] = x_trim
#     return x


# def exe_to_numpy_real_length(file_path, max_length):
#     with open(file_path, "rb") as f:
#         # 未padding的样本
#         x_trim = np.frombuffer(f.read(), dtype=np.uint8)
#         # 样本的真实长度
#         x_length = len(x_trim)
#         # 给样本的每一个二进制都+1
#         x_trim = x_trim.astype(np.int_) + 1
#         # x = np.zeros((x_length + max_length,), dtype=np.int32)
#         # 将x_trim放到padding后的样本中
#         # x[: x_length] = x_trim
#     return x_trim


# 用于训练malconvGCT的dataset
class ExeDataset(Dataset):
    def __init__(self, file_path_list, data_dir, label_list, sort_by_size=False):
        self.fp_list = file_path_list
        self.data_dir = data_dir
        self.label_list = label_list
        self.sort_by_size = sort_by_size
        # Tuple (file_path, label, file_size)
        self.all_files = []

        file_len = len(self.fp_list)
        for index in range(file_len):
            file_path = os.path.join(self.data_dir, self.fp_list[index])
            self.all_files.append((file_path, self.label_list[index], os.path.getsize(file_path)))

        if self.sort_by_size:
            self.all_files.sort(key=lambda filename: filename[2])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path, label, _ = self.all_files[idx]
        try:
            with open(file_path, 'rb') as f:
                x = [i + 1 for i in f.read()]
        except:
            with open(file_path.lower(), 'rb') as f:
                x = [i + 1 for i in f.read()]
        return torch.tensor(x), torch.tensor([label])


class RandomChunkSampler(torch.utils.data.sampler.Sampler):
    """
    Samples random "chunks" of a dataset, so that items within a chunk are always loaded together. Useful to keep chunks in similar size groups to reduce runtime. 
    """

    def __init__(self, data_source, batch_size):
        """
        data_source: the souce pytorch dataset object
        batch_size: the size of the chunks to keep together. Should generally be set to the desired batch size during training to minimize runtime. 
        """
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_source)

        data = [x for x in range(n)]
        # Create blocks
        blocks = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        # shuffle the blocks，try don‘t shuffle
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]

        return iter(data)

    def __len__(self):
        return len(self.data_source)


class DistributedRandomChunkSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, data_source, num_replicas, rank, batch_size, shuffle=True, seed=0):
        super(DistributedRandomChunkSampler, self).__init__(
            dataset=data_source,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,  # DistributedSampler 默认会处理 shuffle
            seed=seed,
        )
        self.batch_size = batch_size
        self.shuffle_b = shuffle
        # self.seed = seed

    def __iter__(self):
        # global result
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # if self.epoch % 2 == 0:
        #     n = len(indices)
        #     if n % 2:
        #         n = n - 1
        #     for i in range(0, n, 2):
        #         temp = indices[i]
        #         indices[i] = indices[i + 1]
        #         indices[i + 1] = temp

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # return iter(indices)

        # 创建 blocks
        blocks = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle_b:
            random.seed(self.seed + self.epoch)  # 使用 DistributedSampler 的 seed 和 epoch
            random.shuffle(blocks)
            # for item in blocks:
            #     random.shuffle(item)
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            # index = torch.randperm(len(blocks), generator=g).tolist()  # type: ignore
            # result = []
            # for item in index:
            #     result += blocks[item]
        indices[:] = [b for bs in blocks for b in bs]
        return iter(indices)

    def __len__(self):
        return self.num_samples


# We want to handel true variable length
# Data loader needs equal length. So use special function to padd all the data in a single batch to be of equal length
# to the longest item in the batch
def pad_collate_func(batch):
    """
    This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader
    object in order to pad out files in a batch to the length of the longest item in the batch.
    """
    vecs = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=0)
    # stack will give us (B, 1), so index [:,0] to get to just (B)
    y = torch.stack(labels)[:, 0]

    return x, y


# 用于训练RARA的dataset
class RARADataset(Dataset):
    def __init__(self, file_path_list, data_dir, label_list, pert_data_dir, sort_by_size=False):
        self.fp_list = file_path_list
        self.data_dir = data_dir
        self.label_list = label_list
        self.pert_data_dir = pert_data_dir
        self.sort_by_size = sort_by_size
        # Tuple (file_path, label, file_size)
        self.all_files = []

        self.malware_files = []
        self.benign_files = []

        file_len = len(self.fp_list)
        for index in range(file_len):
            file_path = os.path.join(self.data_dir, self.fp_list[index])
            pert_file_path = os.path.join(self.pert_data_dir, self.fp_list[index])
            if self.label_list[index] == 0:
                self.benign_files.append(
                    (file_path, self.label_list[index], os.path.getsize(file_path), pert_file_path))
            else:
                self.malware_files.append(
                    (file_path, self.label_list[index], os.path.getsize(file_path), pert_file_path))
            # self.all_files.append((file_path, self.label_list[index], os.path.getsize(file_path)))

        if self.sort_by_size:
            # self.all_files.sort(key=lambda filename: filename[2])
            self.benign_files.sort(key=lambda filename: filename[2])
            self.malware_files.sort(key=lambda filename: filename[2])
        # 标签0和标签1轮流排列，确保每个batch中都有标签0和标签1
        # self.all_files = [item for pair in zip(self.malware_files, self.benign_files) for item in pair]

        # 按照1 1 0 0 1 1 0 0的顺序
        self.all_files = []
        i = 0
        j = 0
        while j < len(self.benign_files):
            for _ in range(2):
                if i < len(self.malware_files):
                    self.all_files.append(self.malware_files[i])
                    i += 1
            for _ in range(2):
                if j < len(self.benign_files):
                    self.all_files.append(self.benign_files[j])
                    j += 1

    def __len__(self):
        return len(self.all_files)
        # return len(self.benign_files) + len(self.malware_files)

    def __getitem__(self, idx):
        file_path, label, _, pert_file_path = self.all_files[idx]
        try:
            with open(file_path, 'rb') as f:
                x = [i + 1 for i in f.read()]
        except:
            with open(file_path.lower(), 'rb') as f:
                x = [i + 1 for i in f.read()]
        return torch.tensor(x), torch.tensor([label]), file_path, pert_file_path


# We want to handel true variable length
# Data loader needs equal length. So use special function to padd all the data in a single batch to be of equal length
# to the longest item in the batch
def pad_collate_func_rara(batch):
    """
    This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader
    object in order to pad out files in a batch to the length of the longest item in the batch.
    """
    vecs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    file_path = [x[2] for x in batch]
    pert_file_path = [x[3] for x in batch]

    x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=0)
    # stack will give us (B, 1), so index [:,0] to get to just (B)
    y = torch.stack(labels)[:, 0]

    return x, y, file_path, pert_file_path
