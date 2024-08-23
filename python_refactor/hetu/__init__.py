# TODO: output the library to `hetu` directory
import _hetu_core
from _hetu_core import *

from typing import Union, Tuple, List, Dict, Set, Any, Callable, Iterator, Optional, TypeVar
_tensor_type_t = TypeVar('T', bound='Tensor')

from hetu import nn as nn
from hetu import utils as utils
# from hetu import optim as optim

import builtins # bool is resovled as hetu.bool

cur_graph_contexts = []
cur_recompute_contexts = []

class _OpContext(object):
    def __init__(self, 
            eager_device=None, 
            device_group_hierarchy=None, 
            stream_index=None, 
            extra_deps=None
        ):
        self.eager_device = eager_device
        self.device_group_hierarchy = device_group_hierarchy
        self.stream_index = stream_index
        self.extra_deps = extra_deps
        if stream_index is not None:
            if ((not isinstance(stream_index, builtins.int)) or 
                isinstance(stream_index, builtins.bool)):
                raise ValueError(
                    "Expected 'int' type for stream index, got "
                    f"'{type(stream_index).__name__}'")
        if extra_deps is not None:
            if not isinstance(extra_deps, (list, tuple)):
                raise ValueError("Extra dependencies should be lists or tuples of tensors")
            for x in extra_deps:
                if not isinstance(x, Tensor):
                    raise ValueError(f"'{type(x).__name__}' object is not a tensor")
            self.extra_deps = [group(extra_deps)]

    def __enter__(self):
        _hetu_core._internal_context.push_op_ctx(
            eager_device=self.eager_device,
            device_group_hierarchy=self.device_group_hierarchy, 
            stream_index=self.stream_index, 
            extra_deps=self.extra_deps
        )
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_op_ctx(
            pop_eager_device=(self.eager_device is not None),
            pop_device_group_hierarchy=(self.device_group_hierarchy is not None), 
            pop_stream_index=(self.stream_index is not None), 
            pop_extra_deps=(self.extra_deps is not None)
        )

def context(eager_device=None, device_group_hierarchy=None, stream_index=None, extra_deps=None):
    return _OpContext(eager_device=eager_device, device_group_hierarchy=device_group_hierarchy, stream_index=stream_index, extra_deps=extra_deps)

def control_dependencies(control_inputs):
    return _OpContext(extra_deps=control_inputs)

class _GraphContext(object):
    def __init__(self, g, create_new=False, prefix="default", num_strategy=-1, tmp=False):
        if tmp == True:
            assert create_new == True, "Temporary graph must be a new graph"
        self.tmp = tmp
        if isinstance(g, Graph):
            assert create_new == False, f"Using type {type(g).__name__} to create a hetu.Graph is not allowed, please use a str instead."
            self.graph = g
        elif isinstance(g, builtins.str):
            if create_new:
                if g == "define_and_run":
                    self.graph = _hetu_core._internal_context.make_new_define_and_run_graph(prefix + '_' + g)
                elif g == "eager":
                    self.graph = _hetu_core._internal_context.make_new_eager_graph(prefix + '_' + g)
                else:
                    raise NotImplementedError("Can only create define_and_run or eager graph for now.")
            else:
                if prefix == "default":
                    if g == "eager":
                        self.graph = _hetu_core._internal_context.get_default_eager_graph()
                    elif g == "define_and_run":
                        self.graph = _hetu_core._internal_context.get_default_define_and_run_graph()
                    elif g == "define_by_run":
                        self.graph = _hetu_core._internal_context.get_default_define_by_run_graph()
                else:
                    self.graph = _hetu_core._internal_context.get_graph(prefix + '_' + g)
        elif isinstance(g, builtins.int):
            assert create_new == False, f"Using type {type(g).__name__} to create a hetu.Graph is not allowed, please use a str instead."
            self.graph = _hetu_core._internal_context.get_graph(g)
        else:
            raise ValueError(f"Cannot parse type '{type(g).__name__}' as hetu.Graph")
        if num_strategy >= 1:
            # print(f'before set num_startegy: {self.graph.num_strategy}')
            self.graph.set_num_strategy(num_strategy)
            # print(f'after set num_startegy: {self.graph.num_strategy}, cur_strategy_id: {self.graph.cur_strategy_id}')
        # print(f"Enter graph {self.graph}")

    def __enter__(self):
        if len(cur_graph_contexts) > 0:
            if str(self.graph) != str(cur_graph_contexts[0].graph):
                pass
                # print(f"Caution! You wrap {self.graph} graph in {cur_graph_contexts[0].graph} graph, we just return you the latter!")
            self.wrapped_context = True
            return cur_graph_contexts[0]
        self.wrapped_context = False
        _hetu_core._internal_context.push_graph_ctx(self.graph.id)
        cur_graph_contexts.append(self)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        if self.wrapped_context:
            return
        _hetu_core._internal_context.pop_graph_ctx()
        cur_graph_contexts.remove(self)
        if self.tmp:
            graph_id = self.graph.id
            self.graph = None
            _hetu_core._internal_context.delete_graph(graph_id)

def graph(g, create_new=False, prefix="default", num_strategy=-1, tmp=False):
    return _GraphContext(g, create_new=create_new, prefix=prefix, num_strategy=num_strategy, tmp=tmp)

class _AutocastContext(object):
    def __init__(self, dtype = None):
        if dtype is None:
            self.autocast = _hetu_core._internal_context.get_default_autocast()
        else:
            self.autocast = _hetu_core._internal_context.make_new_autocast(True, dtype)

    def __enter__(self):
        _hetu_core._internal_context.push_autocast_ctx(self.autocast.id)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_autocast_ctx()

def autocast():
    return _AutocastContext()

def autocast(dtype):
    return _AutocastContext(dtype)

class _RunLevel(object):
    def __init__(self, type):
        if type == "update":
            self.type = 0
        elif type == "local_grad":
            self.type = 1
        elif type == "grad":
            self.type = 2
        elif type == "alloc":
            self.type = 3
        elif type == "topo":
            self.type = 4
        else:
            raise ValueError(f"No level for '{type}' in hetu.Graph")
        
def run_level(type="update"):
    return _RunLevel(type).type

class _MergeStrategyContext(object):
    def __init__(self, target_graph="default", num_strategy=-1):
        self.graph = _hetu_core._internal_context.make_new_define_and_run_graph("tmp_define_and_run")
        self.graph.set_num_strategy(num_strategy)
        if target_graph == "default":
            self.target_graph = _hetu_core._internal_context.get_default_define_and_run_graph()
        else:
            self.target_graph = _hetu_core._internal_context.get_graph(target_graph + '_' + "define_and_run")
        
    def __enter__(self):
        if len(cur_graph_contexts) > 0:
            assert str(self.graph) != str(cur_graph_contexts[0].graph), \
                f"Caution! You wrap {self.tmp_graph} graph in {cur_graph_contexts[0].graph} graph"
        _hetu_core._internal_context.push_graph_ctx(self.graph.id)
        cur_graph_contexts.append(self)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_graph_ctx()
        cur_graph_contexts.remove(self)
        graph_id = self.graph.id
        self.target_graph.merge_strategy(graph_id)
        self.graph = None
        _hetu_core._internal_context.delete_graph(graph_id)
        
def merge_strategy(target_graph="default", num_strategy=-1):
    return _MergeStrategyContext(target_graph=target_graph, num_strategy=num_strategy)

class _RecomputeContext(object):
    def __init__(self, multi_recompute):
        self.multi_recompute = multi_recompute
        
    def __enter__(self):
        cur_recompute_contexts.append(self)
        # 可能会出现recompute module嵌套
        # 只要某一层的recompute是true那么最终就要recompute
        cur_multi_recompute = []
        self.multi_len = len(self.multi_recompute)
        for recompute_context in cur_recompute_contexts:
            assert self.multi_len == len(recompute_context.multi_recompute), "all multi len should be equal"
        for i in range(self.multi_len):
            pipeline_num = 1
            for recompute_context in cur_recompute_contexts:
                if pipeline_num == 1:
                    pipeline_num = len(recompute_context.multi_recompute[i])
                else:
                    assert len(recompute_context.multi_recompute[i]) == 1 or len(recompute_context.multi_recompute[i]) == pipeline_num \
                        , "recompute state len should be 1 or equal to the pipeline num"
            cur_recompute = [False] * pipeline_num
            for j in range(pipeline_num):
                for recompute_context in cur_recompute_contexts:
                    if len(recompute_context.multi_recompute[i]) == 1:
                        if recompute_context.multi_recompute[i][0] == True:
                            cur_recompute = [True] * pipeline_num
                            break
                    else:
                        if recompute_context.multi_recompute[i][j] == True:
                            cur_recompute[j] = True
                            break
            cur_multi_recompute.append(cur_recompute)
        _hetu_core._internal_context.push_recompute_ctx(cur_multi_recompute)
        return self

    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_recompute_ctx()
        cur_recompute_contexts.remove(self)

def recompute(multi_recompute):
    return _RecomputeContext(multi_recompute)

class _CPUOffloadContext(object):
    def __enter__(self):
        _hetu_core._internal_context.push_cpu_offload_ctx()
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_cpu_offload_ctx()

def cpu_offload():
    return _CPUOffloadContext()

class _ProfileContex(object):
    def __init__(self, enabled : bool = True, use_cpu : bool = False, use_cuda : bool = False,
                 record_shapes : bool = False , profile_memory : bool = False):
      self.profile = _hetu_core._internal_context.make_new_profile(enabled, use_cpu, use_cuda, record_shapes, profile_memory)

    def __enter__(self):
        _hetu_core._internal_context.push_profile_ctx(self.profile.id)
        return self.profile

    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_profile_ctx()

def profiler(enabled : bool = True, use_cpu : bool = False, use_cuda : bool = False,
             record_shapes : bool = False , profile_memory : bool = False):
    return _ProfileContex(enabled, use_cpu, use_cuda, record_shapes, profile_memory)

class _SubGraphContext(object):
    def __init__(self, subgraph_type = "", name = "global"):
        if name is None:
            self.subgraph = _hetu_core._internal_context.get_default_subgraph()
        else:
            self.subgraph = _hetu_core._internal_context.make_new_subgraph(subgraph_type=subgraph_type, name=name)
    def __enter__(self):
        _hetu_core._internal_context.push_subgraph_ctx(self.subgraph.name)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_subgraph_ctx()

def subgraph(subgraph_type="", name=""):
    return _SubGraphContext(subgraph_type=subgraph_type, name=name)

def add_to_subgraph(tensor, subgraph_name=""):
    _hetu_core._internal_context.add_op_to_subgraph(tensor, subgraph_name)
