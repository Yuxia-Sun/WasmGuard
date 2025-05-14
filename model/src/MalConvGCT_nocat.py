from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_REWA.src.LowMemConv import LowMemConvBase
from model_REWA.src.MalConvML import MalConvML
from model_REWA.src.checkpoint2 import CheckpointFunction  # This checkpoint implementation is faster than PyTorch's when using


# multiple GPUs


def getParams():
    # Format for this is to make it work easily with Optuna in an automated fashion.
    # variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels': ("suggest_int", {'name': 'channels', 'low': 32, 'high': 1024}),
        'log_stride': ("suggest_int", {'name': 'log2_stride', 'low': 2, 'high': 9}),
        'window_size': ("suggest_int", {'name': 'window_size', 'low': 32, 'high': 256}),
        'layers': ("suggest_int", {'name': 'layers', 'low': 1, 'high': 3}),
        'embd_size': ("suggest_int", {'name': 'embd_size', 'low': 4, 'high': 16}),
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))


def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]

    return MalConvGCT(**new_args)


class MalConvGCT(LowMemConvBase):

    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None,
                 low_mem=False):
        super(MalConvGCT, self).__init__()
        self.low_mem = low_mem
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        if not log_stride is None:
            stride = 2 ** log_stride

        self.context_net = MalConvML(out_size=channels, channels=channels, window_size=window_size, stride=stride,
                                     layers=layers, embd_size=embd_size)
        self.convs = nn.ModuleList([nn.Conv1d(embd_size, channels * 2, window_size, stride=stride, bias=True)] + [
            nn.Conv1d(channels, channels * 2, window_size, stride=1, bias=True) for i in range(layers - 1)])

        # These two objs are not used. They were originally present before the F.glu function existed, and then were
        # accidently left in when we switched over. So the state file provided has unusued states in it. They are left
        # in this definition so that there are no issues loading the file that MalConv was trained on.
        # If you are going to train from scratch, you can delete these two lines.
        # self.convs_1 = nn.ModuleList([nn.Conv1d(channels*2, channels, 1, bias=True) for i in range(layers)])
        # self.convs_atn = nn.ModuleList([nn.Conv1d(channels*2, channels, 1, bias=True) for i in range(layers)])

        self.linear_atn = nn.ModuleList([nn.Linear(channels, channels) for i in range(layers)])

        # one-by-one cons to perform information sharing
        self.convs_share = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for i in range(layers)])

        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    # Over-write the determinRF call to use the base context_net to detemrin RF. We should have the same totla RF, and
    # this will simplify logic significantly.
    def determinRF(self, embeded):
        return self.context_net.determinRF(embeded)

    def processRange(self, x, embeded=False, gct=None):
        if gct is None:
            raise Exception("No Global Context Given")

        if not embeded:
            x = x.long()
            x = self.embd(x)
            # print("malconvGCT embeded x: ", x.shape)
        # x = torch.transpose(x,-1,-2)
        # print("malconvGCT x: ", x.shape)
        x = x.permute(0, 2, 1)

        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            x_len = x.shape[2]
            B = x.shape[0]
            C = x.shape[1]

            sqrt_dim = np.sqrt(x.shape[1])
            # we are going to need a version of GCT with a time dimension, which we will adapt as needed to the right
            # length
            ctnx = torch.tanh(linear_cntx(gct))

            # Size is (B, C), but we need (B, C, 1) to use as a 1d conv filter
            ctnx = torch.unsqueeze(ctnx, dim=2)
            # roll the batches into the channels
            x_tmp = x.view(1, B * C, -1)
            # Now we can apply a conv with B groups, so that each batch gets its own context applied only to what was
            # needed
            x_tmp = F.conv1d(x_tmp, ctnx, groups=B)
            # x_tmp will have a shape of (1, B, L), now we just need to re-order the data back to (B, 1, L)
            x_gates = x_tmp.view(B, 1, -1)

            # Now we effectively apply σ(x_t^T tanh(W c))
            gates = torch.sigmoid(x_gates)
            x = x * gates

        return x

    def forward(self, x):

        if self.low_mem:
            # (B, Channel) ... (70, 128)
            global_context = CheckpointFunction.apply(self.context_net.seq2fix, 2, x, False)
        else:
            global_context = self.context_net.seq2fix(x)

        post_conv = x = self.seq2fix(x, False, pr_args={'gct': global_context})

        penult = x = F.leaky_relu(self.fc_1(x))
        x = self.fc_2(x)

        # zero_257 = torch.arange(0, 257, dtype=torch.int, device="cuda")
        # E1 = self.embd(zero_257)
        # E2 = self.context_net.embd(zero_257)
        # print("E1 == E2: ", E1 == E2)

        return x, penult, post_conv

    # 以下函数了为了兼容Evasion类添加的
    def predict(self, x, return_decision_function=False):
        """

        :param x:
        :param return_decision_function:
        :return:
        """

        scores, _, _ = self.forward(x)

        # The classification label is the label of the class
        # associated with the highest score
        label = scores.argmax(dim=1)
        # print(label.shape)

        return (label, scores) if return_decision_function is True else label

    def get_embedding_size(self):
        return self.embd.embedding_dim

    def embed(self, x, transpose=True):
        """
        获取二进制的词嵌入
        :param x: x的原始输入
        :param transpose: 暂时不用
        :return: 经过嵌入之后的x, [feature_size, embedding_size]
        """
        if x.ndim > 1:
            x = x.squeeze(dim=1)

        # lvwenjian
        # x = torch.LongTensor(x)
        x = x.to(torch.int64)
        # x = x.cuda()

        return self.embd(x)

    def embedd_and_forward(self, x):
        """
        对经过词嵌入之后的x进行前向传播

        :param x: 经过词嵌入后的x
        :return x:模型的预测结果置信度
        """
        if self.low_mem:
            # (B, Channel) ... (70, 128)
            global_context = CheckpointFunction.apply(self.context_net.seq2fix, 1, x)
        else:
            global_context = self.context_net.seq2fix(x, embeded=True)

        post_conv = x = self.seq2fix(x, embeded=True, pr_args={'gct': global_context})

        penult = x = F.leaky_relu(self.fc_1(x))
        x = self.fc_2(x)

        return x, penult, post_conv
