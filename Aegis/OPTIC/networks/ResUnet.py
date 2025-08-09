from torch import nn
import torch
from networks.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from networks.unet import UnetBlock
from einops import rearrange, repeat
from torch.nn import Dropout, Softmax, LayerNorm
import numpy as np
import math
class CrossAttnMem_q_ori(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels

        self.q_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax = Softmax(dim=3)

        self.out_l2u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb):
        emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
        _, N, C = emb_u.size()

        q_l2u = self.q_l2u(emb_l)
        k_l2u = self.k_l2u(emb_u)
        v_l2u = self.v_l2u(emb_u)

        batch_size = q_l2u.size(0)

        k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')
        v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')
        k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=batch_size)
        v_l2u = repeat(v_l2u, 'n bc -> r n bc', r=batch_size)

        q_l2u = q_l2u.unsqueeze(1).transpose(-1, -2)
        k_l2u = k_l2u.unsqueeze(1)
        v_l2u = v_l2u.unsqueeze(1).transpose(-1, -2)

        cross_attn_l2u = torch.matmul(q_l2u, k_l2u)
        cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u)))
        cross_attn_l2u = torch.matmul(cross_attn_l2u, v_l2u)

        cross_attn_l2u = cross_attn_l2u.permute(0, 3, 2, 1).contiguous()
        new_shape_l2u = cross_attn_l2u.size()[:-2] + (self.KV_size,)
        cross_attn_l2u = cross_attn_l2u.view(*new_shape_l2u)

        out_l2u = self.out_l2u(cross_attn_l2u)
        out_l2u = self.proj_dropout(out_l2u)

        return out_l2u

class CrossAttnMem_q_aug(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels

        self.q_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax = Softmax(dim=3)

        self.out_l2u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb):
        # Split the embeddings into two halves
        emb_u, emb_l = torch.split(emb, emb.size(0) // 2, dim=0)

        # Process lower to upper embeddings with cross-attention
        q_l2u = self.q_l2u(emb_l)
        k_l2u = self.k_l2u(emb_u)
        v_l2u = self.v_l2u(emb_u)

        batch_size = q_l2u.size(0)

        k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')
        v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')

        k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=batch_size)
        v_l2u = repeat(v_l2u, 'n bc -> r n bc', r=batch_size)

        q_l2u = q_l2u.unsqueeze(1).transpose(-1, -2)
        k_l2u = k_l2u.unsqueeze(1)
        v_l2u = v_l2u.unsqueeze(1).transpose(-1, -2)

        cross_attn_l2u = torch.matmul(q_l2u, k_l2u)
        cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u)))
        cross_attn_l2u = torch.matmul(cross_attn_l2u, v_l2u)

        cross_attn_l2u = cross_attn_l2u.permute(0, 3, 2, 1).contiguous()
        new_shape_l2u = cross_attn_l2u.size()[:-2] + (self.KV_size,)
        cross_attn_l2u = cross_attn_l2u.view(*new_shape_l2u)

        out_l2u = self.out_l2u(cross_attn_l2u)
        out_l2u = self.proj_dropout(out_l2u)

        return out_l2u

class AllSpark(nn.Module):
    def __init__(self, num_heads, embedding_channels, channel_num_in, channel_num_out,
                 attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.map_in = nn.Sequential(nn.Conv2d(channel_num_in, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn_q_aug = CrossAttnMem_q_aug(num_heads, embedding_channels, attention_dropout_rate)
        self.attn_q_ori = CrossAttnMem_q_ori(num_heads, embedding_channels, attention_dropout_rate)
        self.encoder_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv2d(embedding_channels, channel_num_out, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)
        self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, en):


        _, _, h, w = en.shape
        en = self.map_in(en)
        en = en.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)

        emb = self.attn_norm(en)
        emb_ori = self.attn_q_ori(emb)
        emb_aug = self.attn_q_aug(emb)
        emb = torch.cat((emb_ori, emb_aug))

        emb = emb + en

        out = self.encoder_norm(emb)

        B, n_patch, hidden = out.size()
        out = out.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        out = self.map_out(out)

        return out


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False, mixstyle_layers=[],
                 random_type=None, p=0.5, num_heads=2, num_class=1, in_planes=512, image_size=384):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.mixstyle_layers = mixstyle_layers
        self.res = base_model(pretrained=pretrained, mixstyle_layers=mixstyle_layers, random_type=random_type, p=p)

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

        self.allspark_4 = AllSpark(num_heads=num_heads,
                                   embedding_channels=512,
                                   channel_num_in=512,
                                   channel_num_out=512,
                                   attention_dropout_rate=0.1,
                                   patch_num=(image_size // 32) ** 2,
                                   # 最后一层特征的h*w，image_size//32是算出最后一层特征的大小，这里是被缩小了32倍
                                   num_class=num_class)  # patch_num=(image_size // 32 ) ** 2,



    def forward(self, input):
        x, sfs, sfs_ori = self.res(input)

        x = F.relu(x)
        sfs_ori[4] = F.relu(sfs_ori[4])

        x = torch.cat((sfs_ori[4], x), dim=0)
        sfs[3] = torch.cat((sfs_ori[3], sfs[3]), dim=0)
        sfs[2] = torch.cat((sfs_ori[2], sfs[2]), dim=0)
        sfs[1] = torch.cat((sfs_ori[1], sfs[1]), dim=0)
        sfs[0] = torch.cat((sfs_ori[0], sfs[0]), dim=0)

        x = self.allspark_4(x)



        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))
        seg_output = self.seg_head(head_input)



        return seg_output

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=['layer1'], random_type='Random')
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 3, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())

