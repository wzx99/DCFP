import torch
from .channel_pruner import ChannelPruner


class RandomChannelPruner(ChannelPruner):
    def __init__(self, global_percent=0.8, layer_keep=0.01, except_start_keys=['head.fc'], **kwards):
        super(RandomChannelPruner, self).__init__(except_start_keys=except_start_keys)
        self.layer_keep = layer_keep
        self.global_percent = global_percent

    def gen_channel_mask(self):
        pruned = 0
        total = 0
        for bn_layer, conv_layer in self.norm_conv_links.items():
            channels = self.name2module[bn_layer].weight.shape[0]
            if conv_layer not in self.except_layers:
                min_channel_num = int(channels * self.layer_keep) if int(channels * self.layer_keep) > 0 else 1
                mask = (torch.rand(channels)>self.global_percent)*1.0

                if int(torch.sum(mask)) < min_channel_num: 
                    mask[:min_channel_num]=1. 

                self.name2module[conv_layer].out_mask = mask.reshape(self.name2module[conv_layer].out_mask.shape)

                remain = int(mask.sum())
            else:
                remain = channels
            pruned = pruned + channels - remain
            # print('layer {} \t total channel: {} \t remaining channel: {}'.format(conv_layer, channels, remain))
            
            total += channels

        prune_ratio = pruned / total
        # print('Prune channels: {}\t Prune ratio: {}'.format(pruned, prune_ratio))
