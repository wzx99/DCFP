import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from types import MethodType
from ordered_set import OrderedSet

# These grad_fn pattern are flags of specific a nn.Module
CONV = ('ThnnConv2DBackward', 'CudnnConvolutionBackward',
        'MkldnnConvolutionBackward')
FC = ('ThAddmmBackward', 'AddmmBackward', 'MmBackward')
BN = ('ThnnBatchNormBackward', 'CudnnBatchNormBackward',
      'NativeBatchNormBackward')
GN = ('NativeGroupNormBackward', )
CONCAT = ('CatBackward', )
# the modules which contains NON_PASS grad_fn need to change the parameter size
# according to channels after pruning
NON_PASS = CONV + FC
PASS = BN + GN
NORM = BN + GN

BACKWARD_PARSER_DICT = dict()
MAKE_GROUP_PARSER_DICT = dict()

                
def init_pruned_model(supernet, channel_cfg):
    for name, module in supernet.named_modules():
        if name not in channel_cfg:
            continue

        channels_per_layer = channel_cfg[name]
        requires_grad = module.weight.requires_grad
        out_channels = channels_per_layer['out_channels']
        temp_weight = module.weight[:out_channels]

        if hasattr(module, 'out_channels'):
            module.out_channels = out_channels
        if hasattr(module, 'out_features'):
            module.out_features = out_channels
        if hasattr(module, 'num_features'):
            module.num_features = out_channels
#         if hasattr(module, 'out_mask'):
#             module.out_mask = module.out_mask[:, :out_channels]

        if 'in_channels' in channels_per_layer:
            in_channels = channels_per_layer['in_channels']

            # can also handle depthwise conv
            temp_weight = temp_weight[:, :in_channels]
            if hasattr(module, 'in_channels'):
                module.in_channels = in_channels
            if hasattr(module, 'in_features'):
                module.in_features = in_channels
#             if hasattr(module, 'in_mask'):
#                 module.in_mask = module.in_mask[:, :in_channels]
            # TODO Seems not support GroupConv
            if getattr(module, 'groups', in_channels) > 1:
                module.groups = in_channels

        module.weight = nn.Parameter(temp_weight.data.contiguous())
        module.weight.requires_grad = requires_grad

        if hasattr(module, 'bias') and module.bias is not None:
            module.bias = nn.Parameter(module.bias[:out_channels].data.contiguous())
            module.bias.requires_grad = requires_grad

        if hasattr(module, 'running_mean'):
            module.running_mean = module.running_mean[:out_channels].contiguous()

        if hasattr(module, 'running_var'):
            module.running_var = module.running_var[:out_channels].contiguous()


def register_parser(parser_dict, name=None, force=False):
    """Register a parser function.
    A record will be added to ``parser_dict``, whose key is the specified
    ``name``, and value is the function itself.
    It can be used as a decorator or a normal function.
    Example:
        >>> BACKWARD_PARSER_DICT = dict()
        >>> @register_parser(BACKWARD_PARSER_DICT, 'ThnnConv2DBackward')
        >>> def conv_backward_parser():
        >>>     pass
    Args:
        parser_dict (dict): A dict to map strings to parser functions.
        name (str | None): The function name to be registered. If not
            specified, the function name will be used.
        force (bool, optional): Whether to override an existing function with
            the same name. Default: False.
    """

    def _register(parser_func):
        parser_name = parser_func.__name__ if name is None else name
        if (parser_name not in parser_dict) or force:
            parser_dict[parser_name] = parser_func
        else:
            raise KeyError(
                f'{parser_name} is already registered in task_dict, '
                'add "force=True" if you want to override it')
        return parser_func

    return _register


class ChannelPruner():
    """Base class for structure pruning. This class defines the basic functions
    of a structure pruner. Any pruner that inherits this class should at least
    define its own `sample_subnet` and `set_min_channel` functions. This part
    is being continuously optimized, and there may be major changes in the
    future.
    Reference to https://github.com/jshilong/FisherPruning
    Args:
        except_start_keys (List[str]): the module whose name start with a
            string in except_start_keys will not be prune.
    """

    def __init__(self, except_start_keys=None, **kwards):
        if except_start_keys is None:
            self.except_start_keys = list()
        else:
            self.except_start_keys = except_start_keys

    def trace_shared_module_hook(self, module, inputs, outputs):
        """Trace shared modules. Modules such as the detection head in
        RetinaNet which are visited more than once during :func:`forward` are
        shared modules.
        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        module.cnt += 1
        if module.cnt == 2:
            self.shared_module.append(self.module2name[module])

    def prepare_from_supernet(self, supernet):
        """Prepare for pruning."""

        module2name = OrderedDict()
        name2module = OrderedDict()
        var2module = OrderedDict()

        # record the visited module name during trace path
        visited = dict()
        # Record shared modules which will be visited more than once during
        # forward such as shared detection head in RetinaNet.
        # If a module is not a shared module and it has been visited during
        # forward, its parent modules must have been traced already.
        # However, a shared module will be visited more than once during
        # forward, so it is still need to be traced even if it has been
        # visited.
        self.shared_module = []
        tmp_shared_module_hook_handles = list()

        for name, module in supernet.named_modules():
            if isinstance(module, nn.GroupNorm):
                min_required_version = '1.6.0'
                assert digit_version(torch.__version__) >= digit_version(
                    min_required_version
                ), f'Requires pytorch>={min_required_version} to auto-trace' \
                   f'GroupNorm correctly.'
            if hasattr(module, 'weight'):
                # trace shared modules
                module.cnt = 0
                # the handle is only to remove the corresponding hook later
                handle = module.register_forward_hook(
                    self.trace_shared_module_hook)
                tmp_shared_module_hook_handles.append(handle)

                module2name[module] = name
                name2module[name] = module
                var2module[id(module.weight)] = module
                self.add_pruning_attrs(module)
                visited[name] = False
#             if isinstance(module, SwitchableBatchNorm2d):
#                 name2module[name] = module
        self.name2module = name2module
        self.module2name = module2name

        # Set requires_grad to True. If the `requires_grad` of a module's
        # weight is False, we can not trace this module by parsing backward.
        param_require_grad = dict()
        for param in supernet.parameters():
            param_require_grad[id(param)] = param.requires_grad
            param.requires_grad = True
        
        supernet.eval()
        pseudo_img = torch.randn(2, 3, 224, 224)
        pseudo_loss = supernet.forward(pseudo_img, deepsup=True)
        if isinstance(pseudo_loss, list):
            pseudo_loss = sum([i.sum() for i in pseudo_loss])
        elif isinstance(pseudo_loss, dict):
            pseudo_loss = sum([pseudo_loss[i].sum() for i in pseudo_loss])
        else:
            pseudo_loss = pseudo_loss.sum()

        # `trace_shared_module_hook` and `cnt` are only used to trace the
        # shared modules in a model and need to be remove later
        for name, module in supernet.named_modules():
            if hasattr(module, 'weight'):
                del module.cnt

        for handle in tmp_shared_module_hook_handles:
            handle.remove()

        # We set requires_grad to True to trace the whole architecture
        # topology. So it should be reset after that.
        for param in supernet.parameters():
            param.requires_grad = param_require_grad[id(param)]
        del param_require_grad

        non_pass_paths = list()
        cur_non_pass_path = list()
        self.trace_non_pass_path(pseudo_loss.grad_fn, module2name, var2module,
                                 cur_non_pass_path, non_pass_paths, visited)

        norm_conv_links = dict()
        self.trace_norm_conv_links(pseudo_loss.grad_fn, module2name,
                                   var2module, norm_conv_links, visited)
        self.norm_conv_links = norm_conv_links
        
        self.conv_norm_links = dict()
        for norm, conv in self.norm_conv_links.items():
            self.conv_norm_links[conv] = norm

        # a node can be the name of a conv module or a str like 'concat_{id}'
        node2parents = self.find_node_parents(non_pass_paths)
        self.node2parents = node2parents

        self.same_out_channel_groups = self.make_same_out_channel_groups(
            node2parents, name2module)

        self.module2group = dict()
        for group_name, group in self.same_out_channel_groups.items():
            for module_name in group:
                self.module2group[module_name] = group_name

        self.modules_have_ancest = list()
        for node_name, parents_name in node2parents.items():
            if node_name in name2module and len(parents_name) > 0:
                self.modules_have_ancest.append(node_name)

        self.modules_have_child = OrderedSet()
        for parents_name in node2parents.values():
            for name in parents_name:
                # The node is a module in supernet
                if name in name2module:
                    self.modules_have_child.add(name)

        self.channel_spaces = self.build_channel_spaces(name2module)

#         self._reset_norm_running_stats(supernet)

    def get_space_id(self, module_name):
        """Get the corresponding space_id of the module_name.
        The modules who share the same space_id will share the same out_mask.
        If the module is the output module(there is no other ``nn.Module``
        whose input is its output), this function will return None. As the
        output module can not be pruned.
        If the input of this module is the concatenation of the output
        of several ``nn.Module``, this function will return a dict object.
        If this module is in one of the groups, this function will return the
        group name. As the modules in the same group should share the same
        space_id.
        Otherwise, this function will return the module_name as space_id.
        Args:
            module_name (str): the name of a ``nn.Module``.
        Return:
            str or dict or None: the corresponding space_id of the module_name.
        """
        if 'concat' in module_name and module_name not in self.name2module:
            # each module_name in concat_parents should be in name2module
            if 'item' in module_name:
                space_id = self.get_space_id(self.node2parents[module_name][0])
            else:
                concat_parents = [
                    self.get_space_id(parent)
                    for parent in self.node2parents[module_name]
                ]
                space_id = dict(concat=concat_parents)

        elif module_name not in self.modules_have_child:
            return None
        elif module_name in self.module2group:
            space_id = self.module2group[module_name]
        else:
            space_id = module_name
        return space_id

    def find_make_group_parser(self, node_name, name2module):
        """Find the corresponding make_group_parser according to the
        ``node_name``"""
        if 'concat' in node_name and node_name not in name2module:
            return MAKE_GROUP_PARSER_DICT['concat']
        elif 'chunk' in node_name and node_name not in name2module:
            return MAKE_GROUP_PARSER_DICT['chunk']
        elif (node_name in name2module
              and isinstance(name2module[node_name], nn.Conv2d)
              and name2module[node_name].in_channels
              == name2module[node_name].groups):
            return MAKE_GROUP_PARSER_DICT['depthwise']
        else:
            return MAKE_GROUP_PARSER_DICT['common']

    @register_parser(MAKE_GROUP_PARSER_DICT, 'concat')
    def concat_make_group_parser(self, node_name, parents_name, group_idx,
                                 same_in_channel_groups,
                                 same_out_channel_groups):
        if 'item' in node_name:
            return self.make_group_parser(node_name, parents_name, group_idx,
                                          same_in_channel_groups,
                                          same_out_channel_groups)
        else:
            return group_idx, same_in_channel_groups, same_out_channel_groups

    @register_parser(MAKE_GROUP_PARSER_DICT, 'chunk')
    def chunk_make_group_parser(self, group_idx, same_in_channel_groups,
                                same_out_channel_groups, **kwargs):
        return group_idx, same_in_channel_groups, same_out_channel_groups

    @register_parser(MAKE_GROUP_PARSER_DICT, 'depthwise')
    def depthwise_make_group_parser(self, node_name, parents_name, **kwargs):
        # depth wise conv should be in the same group with its parent
        parents_name.add(node_name)
        return self.make_group_parser(node_name, parents_name, **kwargs)

    @register_parser(MAKE_GROUP_PARSER_DICT, 'common')
    def make_group_parser(self, node_name, parents_name, group_idx,
                          same_in_channel_groups, same_out_channel_groups):
        added = False
        for group_name in same_in_channel_groups:
            group_parents = set(same_out_channel_groups[group_name])
            if len(parents_name.intersection(group_parents)) > 0:
                same_in_channel_groups[group_name].append(node_name)
                same_out_channel_groups[group_name] = list(
                    parents_name.union(group_parents))
                added = True
                break
        if not added:
            group_idx += 1
            same_in_channel_groups[group_idx] = [node_name]
            same_out_channel_groups[group_idx] = list(parents_name)

        return group_idx, same_in_channel_groups, same_out_channel_groups

    def make_same_out_channel_groups(self, node2parents, name2module):
        """Modules have the same child should be in the same group."""
        idx = -1
        # the nodes in same_out_channel_groups are parents of
        # nodes in same_in_channel_groups
        same_in_channel_groups, same_out_channel_groups = {}, {}
        for node_name, parents_name in node2parents.items():
            parser = self.find_make_group_parser(node_name, name2module)
            idx, same_in_channel_groups, same_out_channel_groups = \
                parser(self,
                       node_name=node_name,
                       parents_name=parents_name,
                       group_idx=idx,
                       same_in_channel_groups=same_in_channel_groups,
                       same_out_channel_groups=same_out_channel_groups)

        groups = dict()
        idx = 0
        for group in same_out_channel_groups.values():
            if len(group) > 1:
                group_name = f'group_{idx}'
                groups[group_name] = group
                idx += 1
        
        return groups

    @staticmethod
    def modify_conv_forward(module):
        """Modify the forward method of a conv layer."""
        original_forward = module.forward

        def modified_forward(self, feature):
            feature = feature * self.in_mask
            return original_forward(feature)

        return MethodType(modified_forward, module)

    @staticmethod
    def modify_fc_forward(module):
        """Modify the forward method of a linear layer."""
        original_forward = module.forward

        def modified_forward(self, feature):
            if not len(self.in_mask.shape) == len(self.out_mask.shape):
                self.in_mask = self.in_mask.reshape(self.in_mask.shape[:2])

            feature = feature * self.in_mask
            return original_forward(feature)

        return MethodType(modified_forward, module)

    def add_pruning_attrs(self, module):
        """Add masks to a ``nn.Module``."""
        if isinstance(module, nn.Conv2d):
            module.register_buffer(
                'in_mask',
                module.weight.new_ones((1, module.in_channels, 1, 1), ))
            module.register_buffer(
                'out_mask',
                module.weight.new_ones((1, module.out_channels, 1, 1), ))
            module.forward = self.modify_conv_forward(module)
        if isinstance(module, nn.Linear):
            module.register_buffer(
                'in_mask', module.weight.new_ones((1, module.in_features), ))
            module.register_buffer(
                'out_mask', module.weight.new_ones((1, module.out_features), ))
            module.forward = self.modify_fc_forward(module)
        if (isinstance(module, _BatchNorm)
                or isinstance(module, _InstanceNorm)
                or isinstance(module, GroupNorm)):
            module.register_buffer(
                'out_mask',
                module.weight.new_ones((1, len(module.weight), 1, 1), ))
    
    def find_node_parents(self, paths):
        """Find the parent node of a node.
        A node in the ``paths`` can be a module name or a operation name such
        as `concat_140719322997152`. Note that the string of numbers following
        ``concat`` do not have a particular meaning. It just make the operation
        name unique.
        Args:
            paths (list): The traced paths.
        """
        node2parents = dict()
        for path in paths:
            if len(path) == 0:
                continue
            for i, node_name in enumerate(path[:-1]):
                parent_name = path[i + 1]
                if parent_name in self.end_nodes:
                    continue
                if node_name in node2parents.keys():
                    node2parents[node_name].add(parent_name)
                else:
                    node2parents[node_name] = OrderedSet([parent_name])

            leaf_name = path[-1]
            if leaf_name not in node2parents.keys():
                node2parents[leaf_name] = OrderedSet()
        return node2parents
    
    def build_channel_spaces(self, name2module):
        """Build channel search space.
        Args:
            name2module (dict): A mapping between module_name and module.
        Return:
            dict: The channel search space. The key is space_id and the value
                is the corresponding out_mask.
        """
        search_space = dict()

        for module_name in self.modules_have_child:
#             need_prune = True
#             for key in self.except_start_keys:
#                 if module_name.startswith(key):
#                     need_prune = False
#                     break
#             if not need_prune:
#                 continue
            if module_name in self.module2group:
                space_id = self.module2group[module_name]
            else:
                space_id = module_name
            module = name2module[module_name]
            if space_id not in search_space:
                search_space[space_id] = module.out_mask

        return search_space

    def set_channel_bins(self, channel_bins_dict, max_channel_bins):
        """Set subnet according to the number of channel bins in a layer.
        Args:
            channel_bins_dict (dict): The number of bins in each layer. Key is
                the space_id of each layer and value is the corresponding
                mask of channel bin.
            max_channel_bins (int): The max number of bins in each layer.
        """
        subnet_dict = dict()
        for space_id, bin_mask in channel_bins_dict.items():
            mask = self.channel_spaces[space_id]
            shape = mask.shape
            channel_num = shape[1]
            channels_per_bin = channel_num // max_channel_bins
            new_mask = []
            for mask in bin_mask:
                new_mask.extend([1] * channels_per_bin if mask else [0] *
                                channels_per_bin)
            new_mask.extend([0] * (channel_num % max_channel_bins))
            new_mask = torch.tensor(new_mask).reshape(*shape)
            subnet_dict[space_id] = new_mask
        self.set_subnet(subnet_dict)

    def trace_non_pass_path(self, grad_fn, module2name, var2module, cur_path,
                            result_paths, visited):
        """Trace the topology of all the ``NON_PASS_MODULE``."""
        grad_fn = grad_fn[0] if isinstance(grad_fn, (list, tuple)) else grad_fn

        if grad_fn is not None:
            parser = self.find_backward_parser(grad_fn)
            if parser is not None:
                parser(self, grad_fn, module2name, var2module, cur_path,
                       result_paths, visited)
            else:
                # If the op is AccumulateGrad, parents is (),
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        self.trace_non_pass_path(parent, module2name,
                                                 var2module, cur_path,
                                                 result_paths, visited)
        else:
            result_paths.append(copy.deepcopy(cur_path))

    def trace_norm_conv_links(self, grad_fn, module2name, var2module,
                              norm_conv_links, visited):
        """Get the convolutional layer placed before a normalization layer in
        the model.
        Example:
            >>> conv = nn.Conv2d(3, 3, 3)
            >>> norm = nn.BatchNorm2d(3)
            >>> pseudo_img = torch.rand(1, 3, 224, 224)
            >>> out = norm(conv(pseudo_img))
            >>> print(out.grad_fn)
            <NativeBatchNormBackward object at 0x0000022BC709DB08>
            >>> print(out.grad_fn.next_functions)
            ((<ThnnConv2DBackward object at 0x0000020E40639688>, 0),
            (<AccumulateGrad object at 0x0000020E40639208>, 0),
            (<AccumulateGrad object at 0x0000020E406398C8>, 0))
            >>> # op.next_functions[0][0] is ThnnConv2DBackward means
            >>> # the parent of this NativeBatchNormBackward op is
            >>> # ThnnConv2DBackward
            >>> # op.next_functions[1][0].variable is the weight of this
            >>> # normalization module
            >>> # op.next_functions[2][0].variable is the bias of this
            >>> # normalization module
            >>> # Things are different in InstanceNorm
            >>> conv = nn.Conv2d(3, 3, 3)
            >>> norm = nn.InstanceNorm2d(3, affine=True)
            >>> out = norm(conv(pseudo_img))
            >>> print(out.grad_fn)
            <ViewBackward object at 0x0000022BC709DD48>
            >>> print(out.grad_fn.next_functions)
            ((<NativeBatchNormBackward object at 0x0000022BC81E8A08>, 0),)
            >>> print(out.grad_fn.next_functions[0][0].next_functions)
            ((<ViewBackward object at 0x0000022BC81E8DC8>, 0),
            (<RepeatBackward object at 0x0000022BC81E8D08>, 0),
            (<RepeatBackward object at 0x0000022BC81E81C8>, 0))
            >>> # Hence, a dfs is necessary.
        """

        def is_norm_grad_fn(grad_fn):
            for fn_name in NORM:
                if type(grad_fn).__name__.startswith(fn_name):
                    return True
            return False

        def is_conv_grad_fn(grad_fn):
            for fn_name in CONV:
                if type(grad_fn).__name__.startswith(fn_name):
                    return True
            return False

        def is_leaf_grad_fn(grad_fn):
            if type(grad_fn).__name__ == 'AccumulateGrad':
                return True
            return False

        grad_fn = grad_fn[0] if isinstance(grad_fn, (list, tuple)) else grad_fn
        if grad_fn is not None:
            if is_norm_grad_fn(grad_fn):
                conv_grad_fn = grad_fn.next_functions[0][0]
                while not is_conv_grad_fn(conv_grad_fn):
                    conv_grad_fn = conv_grad_fn.next_functions[0][0]

                leaf_grad_fn = conv_grad_fn.next_functions[1][0]
                while not is_leaf_grad_fn(leaf_grad_fn):
                    leaf_grad_fn = leaf_grad_fn.next_functions[0][0]
                conv_var = leaf_grad_fn.variable

                leaf_grad_fn = grad_fn.next_functions[1][0]
                while not is_leaf_grad_fn(leaf_grad_fn):
                    leaf_grad_fn = leaf_grad_fn.next_functions[0][0]
                bn_var = leaf_grad_fn.variable

                conv_module = var2module[id(conv_var)]
                bn_module = var2module[id(bn_var)]
                conv_name = module2name[conv_module]
                bn_name = module2name[bn_module]
                if visited[bn_name]:
                    pass
                else:
                    visited[bn_name] = True
                    norm_conv_links[bn_name] = conv_name

                    self.trace_norm_conv_links(conv_grad_fn, module2name,
                                               var2module, norm_conv_links,
                                               visited)

            else:
                # If the op is AccumulateGrad, parents is (),
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        self.trace_norm_conv_links(parent, module2name,
                                                   var2module, norm_conv_links,
                                                   visited)

    def find_backward_parser(self, grad_fn):
        for name, parser in BACKWARD_PARSER_DICT.items():
            if type(grad_fn).__name__.startswith(name):
                return parser

    @register_parser(BACKWARD_PARSER_DICT, 'ThnnConv2DBackward')
    @register_parser(BACKWARD_PARSER_DICT, 'CudnnConvolutionBackward')
    @register_parser(BACKWARD_PARSER_DICT, 'MkldnnConvolutionBackward')
    @register_parser(BACKWARD_PARSER_DICT, 'SlowConvDilated2DBackward')
    def conv_backward_parser(self, grad_fn, module2name, var2module, cur_path,
                             result_paths, visited):
        """Parse the backward of a conv layer.
        Example:
            >>> conv = nn.Conv2d(3, 3, 3)
            >>> pseudo_img = torch.rand(1, 3, 224, 224)
            >>> out = conv(pseudo_img)
            >>> print(out.grad_fn.next_functions)
            ((None, 0), (<AccumulateGrad object at 0x0000020E405CBD88>, 0),
            (<AccumulateGrad object at 0x0000020E405CB588>, 0))
            >>> # op.next_functions[0][0] is None means this ThnnConv2DBackward
            >>> # op has no parents
            >>> # op.next_functions[1][0].variable is the weight of this Conv2d
            >>> # module
            >>> # op.next_functions[2][0].variable is the bias of this Conv2d
            >>> # module
        """
        variable = grad_fn.next_functions[1][0].variable
        var_id = id(variable)
        module = var2module[var_id]
        name = module2name[module]
        parent = grad_fn.next_functions[0][0]
        cur_path.append(name)
        # If a module is not a shared module and it has been visited during
        # forward, its parent modules must have been traced already.
        # However, a shared module will be visited more than once during
        # forward, so it is still need to be traced even if it has been
        # visited.
        if visited[name] and name not in self.shared_module:
            result_paths.append(copy.deepcopy(cur_path))
        else:
            visited[name] = True
            self.trace_non_pass_path(parent, module2name, var2module, cur_path,
                                     result_paths, visited)
        cur_path.pop(-1)

    @register_parser(BACKWARD_PARSER_DICT, 'ThAddmmBackward')
    @register_parser(BACKWARD_PARSER_DICT, 'AddmmBackward')
    @register_parser(BACKWARD_PARSER_DICT, 'MmBackward')
    def linear_backward_parser(self, grad_fn, module2name, var2module,
                               cur_path, result_paths, visited):
        """Parse the backward of a conv layer.
        Example:
            >>> fc = nn.Linear(3, 3, bias=True)
            >>> input = torch.rand(3, 3)
            >>> out = fc(input)
            >>> print(out.grad_fn.next_functions)
            ((<AccumulateGrad object at 0x0000020E405F75C8>, 0), (None, 0),
            (<TBackward object at 0x0000020E405F7D48>, 0))
            >>> # op.next_functions[0][0].variable is the bias of this Linear
            >>> # module
            >>> # op.next_functions[1][0] is None means this AddmmBackward op
            >>> # has no parents
            >>> # op.next_functions[2][0] is the TBackward op, and
            >>> # op.next_functions[2][0].next_functions[0][0].variable is
            >>> # the transpose of the weight of this Linear module
        """
        variable = grad_fn.next_functions[-1][0].next_functions[0][0].variable
        var_id = id(variable)
        module = var2module[var_id]
        name = module2name[module]
        parent = grad_fn.next_functions[1][0]
        cur_path.append(name)
        # If a module is not a shared module and it has been visited during
        # forward, its parent modules must have been traced already.
        # However, a shared module will be visited more than once during
        # forward, so it is still need to be traced even if it has been
        # visited.
        if visited[name] and name not in self.shared_module:
            result_paths.append(copy.deepcopy(cur_path))
        else:
            visited[name] = True
            self.trace_non_pass_path(parent, module2name, var2module, cur_path,
                                     result_paths, visited)
        cur_path.pop(-1)

    @register_parser(BACKWARD_PARSER_DICT, 'CatBackward')
    def concat_backward_parser(self, grad_fn, module2name, var2module,
                               cur_path, result_paths, visited):
        """Parse the backward of a concat operation.
        Example:
            >>> conv = nn.Conv2d(3, 3, 3)
            >>> pseudo_img = torch.rand(1, 3, 224, 224)
            >>> out1 = conv(pseudo_img)
            >>> out2 = conv(pseudo_img)
            >>> out = torch.cat([out1, out2], dim=1)
            >>> print(out.grad_fn.next_functions)
            ((<ThnnConv2DBackward object at 0x0000020E405F24C8>, 0),
            (<ThnnConv2DBackward object at 0x0000020E405F2648>, 0))
            >>> # the length of ``out.grad_fn.next_functions`` is two means
            >>> # ``out`` is obtained by concatenating two tensors
        """
        parents = grad_fn.next_functions
        concat_id = '_'.join([str(id(p)) for p in parents])
        name = f'concat_{concat_id}'
        cur_path.append(name)
        # If a module is not a shared module and it has been visited during
        # forward, its parent modules must have been traced already.
        # However, a shared module will be visited more than once during
        # forward, so it is still need to be traced even if it has been
        # visited.
        if (name in visited and visited[name]
                and name not in self.shared_module):
            result_paths.append(copy.deepcopy(cur_path))
        else:
            visited[name] = True
            for i, parent in enumerate(parents):
                cur_path.append(f'{name}_item_{i}')
                self.trace_non_pass_path(parent, module2name, var2module,
                                         cur_path, result_paths, visited)
                if cur_path.pop(-1) != f'{name}_item_{i}':
                    print(f'{name}_item_{i}')
        cur_path.pop(-1)

    @staticmethod
    def _reset_norm_running_stats(supernet):
        from torch.nn.modules.batchnorm import _NormBase

        for module in supernet.modules():
            if isinstance(module, _NormBase):
                module.reset_parameters()
    
    def gen_channel_mask(self):
        pass
    
    def get_channel_mask(self, space_id, out_mask):
        if isinstance(space_id, dict):
            if 'concat' in space_id:
                mask = torch.cat([self.get_channel_mask(parent_space_id,out_mask) for parent_space_id in space_id['concat']])
        elif space_id in self.same_out_channel_groups:
            mask = torch.zeros_like(out_mask)
            for parent_space_id in self.same_out_channel_groups[space_id]:
                mask = mask + self.get_channel_mask(parent_space_id,out_mask)
            mask = torch.clamp(mask,0,1)
        else:
            mask = self.name2module[space_id].out_mask
        return mask
                
    def sample_subnet(self):
        """Random sample subnet by random mask.
        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask(space_id, out_mask)
        return subnet_dict
    
    def set_subnet(self, subnet_dict):
        """Modify the in_mask and out_mask of modules in supernet according to
        subnet_dict.
        Args:
            subnet_dict (dict): the key is space_id and the value is the
                corresponding sampled out_mask.
        """
        # 剪module out_mask
        for module_name in self.modules_have_child:
            space_id = self.get_space_id(module_name)
            module = self.name2module[module_name]
            module.out_mask = subnet_dict[space_id].to(module.out_mask.device)  
        # 剪bn
        for norm, conv in self.norm_conv_links.items():
            module = self.name2module[norm]
            conv_space_id = self.get_space_id(conv)
            # conv_space_id is None means the conv layer in front of
            # this normalization module can not be pruned. So we should not set
            # the out_mask of this normalization layer
            if conv_space_id is not None:
                module.out_mask = subnet_dict[conv_space_id].to(
                    module.out_mask.device)
        # 剪module in_mask
        for module_name in self.modules_have_ancest:
            module = self.name2module[module_name]
            parents = self.node2parents[module_name]
            # To avoid ambiguity, we only allow the following two cases:
            # 1. all elements in parents are ``Conv2d``,
            # 2. there is only one element in parents, ``concat`` or ``chunk``
            # In case 1, all the ``Conv2d`` share the same space_id and
            # out_mask.
            # So in all cases, we only need the very first element in parents
            parent = parents[0]
            space_id = self.get_space_id(parent)

            if isinstance(space_id, dict):
                if 'concat' in space_id:
                    in_mask = []
                    for parent_space_id in space_id['concat']:
                        in_mask.append(subnet_dict[parent_space_id])
                    module.in_mask = torch.cat(
                        in_mask, dim=1).to(module.in_mask.device)
            else:
                module.in_mask = subnet_dict[space_id].to(
                    module.in_mask.device)
                
    def export_subnet(self):
        """Generate subnet configs according to the in_mask and out_mask of a
        module."""
        channel_cfg = dict()
        for name, module in self.name2module.items():

            channel_cfg[name] = dict()
            if hasattr(module, 'in_mask'):
                channel_cfg[name]['in_channels'] = int(
                    module.in_mask.sum())
                channel_cfg[name]['raw_in_channels'] = int(
                    module.in_mask.numel())
                channel_cfg[name]['in_mask'] = module.in_mask.cpu().numpy()

            if hasattr(module, 'out_mask'):
                channel_cfg[name]['out_channels'] = int(
                    module.out_mask.sum())
                channel_cfg[name]['raw_out_channels'] = int(
                    module.out_mask.numel())
                channel_cfg[name]['out_mask'] = module.out_mask.cpu().numpy()

        return channel_cfg
    
    def get_space_bias(self, space_id, out_mask):
        if isinstance(space_id, dict):
            if 'concat' in space_id:
                bias = torch.cat([self.get_space_bias(parent_space_id,out_mask) for parent_space_id in space_id['concat']])
        elif space_id in self.same_out_channel_groups:
            bias = torch.zeros_like(out_mask)
            for parent_space_id in self.same_out_channel_groups[space_id]:
                bias = bias + self.get_space_bias(parent_space_id,out_mask)
        else:
            if space_id in self.conv_norm_links:
                bn = self.conv_norm_links[space_id]
                bn_module = self.name2module[bn]
                bias = bn_module.bias.reshape(out_mask.shape)
            else:
                bias = torch.zeros_like(out_mask)
        return bias
    
    def get_subnet_bias(self):
        """Random sample subnet by random mask.
        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        bias_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            bias_dict[space_id] = self.get_space_bias(space_id, out_mask)
        return bias_dict
    
    def resize_subnet_bias(self, supernet, bias_dict):
        for name, module in supernet.named_modules():
            if name in self.modules_have_ancest:
                sub_module = self.name2module[name]
                parents = self.node2parents[name]
                parent = parents[0]
                space_id = self.get_space_id(parent)
                
                if isinstance(space_id, dict):
                    if 'concat' in space_id:
                        bias = []
                        for parent_space_id in space_id['concat']:
                            bias.append(bias_dict[parent_space_id])
                        bias = torch.cat(bias, dim=1)
                else:
                    bias = bias_dict[space_id]

                mask = sub_module.in_mask
                activation = torch.relu((1 - mask) * bias)

                conv_sum = module.weight.data.sum(dim=(2, 3)) 
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1) 
                
                if name in self.conv_norm_links:
                    bn = self.conv_norm_links[name]
                    bn_module = supernet.get_submodule(bn)
                    bn_module.running_mean.data.sub_(offset)
                else:
                    if hasattr(sub_module, 'bias'):
                        module.bias.data.add_(offset) 
                    else:
                        print(name)
                        module.bias = nn.Parameter(offset)
                
    def deploy_subnet(self, supernet, channel_cfg):
        """Deploy subnet according `channel_cfg`."""
        for name, module in supernet.named_modules():
            if name not in channel_cfg:
                continue
                
            sub_module = self.name2module[name]
            requires_grad = sub_module.weight.requires_grad
            temp_weight = sub_module.weight.data
            temp_weight = temp_weight[sub_module.out_mask.reshape(-1)==1].contiguous()
            
            out_channels = int(sub_module.out_mask.sum())
            if hasattr(module, 'out_channels'):
                module.out_channels = out_channels
            if hasattr(module, 'out_features'):
                module.out_features = out_channels
            if hasattr(module, 'num_features'):
                module.num_features = out_channels
                
            if hasattr(sub_module, 'in_mask'):
                temp_weight = temp_weight[:,sub_module.in_mask.reshape(-1)==1].contiguous()
                in_channels = int(sub_module.in_mask.sum())
                if hasattr(module, 'in_channels'):
                    module.in_channels = in_channels
                if hasattr(module, 'in_features'):
                    module.in_features = in_channels
                # TODO Seems not support GroupConv
                if getattr(module, 'groups', in_channels) > 1:
                    module.groups = in_channels
            
            module.weight = nn.Parameter(temp_weight.data)
            module.weight.requires_grad = requires_grad
            
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[sub_module.out_mask.reshape(-1)==1]).contiguous()
                module.bias.requires_grad = requires_grad

            if hasattr(module, 'running_mean'):
                module.running_mean = module.running_mean[sub_module.out_mask.reshape(-1)==1].contiguous()

            if hasattr(module, 'running_var'):
                module.running_var = module.running_var[sub_module.out_mask.reshape(-1)==1].contiguous()
                
    def get_except_layers(self, supernet):
        except_start_keys = list()
        for key in self.except_start_keys:
            except_start_keys.append(key)
            if key in self.norm_conv_links:
                except_start_keys.append(self.norm_conv_links[key])
            elif key in self.conv_norm_links:
                except_start_keys.append(self.conv_norm_links[key])
            
        self.except_layers = list()
        for name, module in supernet.named_modules():
            if hasattr(module, 'weight'):
                for key in except_start_keys:
                    if name.startswith(key):
                        self.except_layers.append(name)
                        break
    
    def prune_model(self, supernet, except_start_keys=None):
        model_copy = copy.deepcopy(supernet)
        if hasattr(model_copy, 'end_nodes'):
            self.end_nodes = model_copy.end_nodes
        else:
            self.end_nodes = []
        self.prepare_from_supernet(model_copy)
        
        if hasattr(model_copy, 'ignore_prune_layer'):
            self.except_start_keys = self.except_start_keys + model_copy.ignore_prune_layer
        if except_start_keys:
            self.except_start_keys = self.except_start_keys + except_start_keys
        self.get_except_layers(model_copy)
        
        self.gen_channel_mask()
        subnet_dict = self.sample_subnet()
        self.set_subnet(subnet_dict)
        
        bias_dict = self.get_subnet_bias()
        self.resize_subnet_bias(supernet,bias_dict)
        
        channel_cfg = self.export_subnet()
        self.deploy_subnet(supernet, channel_cfg)
        return supernet, channel_cfg
    