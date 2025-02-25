import os
import gc
import re
import copy
import json
import hetu as ht
from packaging import version
from zipfile import is_zipfile
from collections import defaultdict
from hetu.models.utils.config_utils import PreTrainedConfig
from hetu.models.utils import SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, TORCH_WEIGHTS_NAME, TORCH_WEIGHTS_INDEX_NAME
from hetu.models.utils.hub import is_remote_url
from hetu.utils.checkpoint.ht_safetensors import load_file
from typing import Optional, Union

def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name

def get_state_dict_dtype(state_dict):
    for value in state_dict.values():
        if value.is_floating_point():
            return value.dtype
    return next(iter(state_dict.values())).dtype

def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    dtype: Optional[ht.dtype] = None,
    weights_only: bool = True,
):
    if checkpoint_file.endswith(".safetensors"):
        return load_file(checkpoint_file, dtype=dtype)
    else:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to load model weights in PyTorch format")

        def shared_pointers(tensors):
            ptrs = defaultdict(list)
            for k, v in tensors.items():
                ptrs[v.data_ptr()].append(k)
            return [names for names in ptrs.values() if len(names) > 1]

        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
            version.parse(torch.__version__) >= version.parse("2.1.0") and
            is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        weights_only_kwarg = {"weights_only": weights_only}
        try:
            loaded = torch.load(
                checkpoint_file,
                map_location="cpu",
                **weights_only_kwarg,
                **extra_args,
            )
            loaded = loaded.get("state_dict", loaded)            
            shared = shared_pointers(loaded)

            for shared_weights in shared:
                for name in shared_weights[1:]:
                    loaded.pop(name)

            # Convert the state dict to Hetu tensors
            state_dict = {}
            for key, value in loaded.items():
                dtype_str = "hetu." + str(value.dtype).split(".")[1]
                state_dict[key] = ht.numpy_to_NDArray(value.numpy(), eval(dtype_str))
            return state_dict
        except Exception as e:
            raise ValueError(f"Unable to load weights from checkpoint file {checkpoint_file}: {e}")

def _load_state_dict_into_model(
    model,
    state_dict,
    start_prefix,
    local_device,
):
    error_msgs = []
    state_dict = state_dict.copy()
    
    def load(module: ht.nn.Module, state_dict, prefix=""):
        module._load_from_state_dict(
            state_dict, local_device, prefix, True, [], [], error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")
    
    load(model, state_dict, prefix=start_prefix)
    del state_dict
    
    return error_msgs

def _get_tied_weight_keys(module: ht.nn.Module, prefix=""):
    tied_weight_keys = []
    if getattr(module, "_tied_weights_keys", None) is not None:
        names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]
        tied_weight_keys.extend(names)
    for name, submodule in module.named_children():
        local_prefix = f"{prefix}.{name}" if prefix else name
        tied_weight_keys.extend(_get_tied_weight_keys(submodule, prefix=local_prefix))
    return tied_weight_keys

class PreTrainedModel(ht.nn.Module):
    config_class = None
    base_model_prefix = ""
    
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None
    
    def __init__(self, config: PreTrainedConfig, **kwargs):
        super().__init__()
        self.config = config
        self.name_or_path = config.name_or_path
    
    @staticmethod
    def _fix_state_dict_key_on_load(key):
        is_changed = False
        fix_structure = {"language_model.":"",
                         "embedding.word_embeddings.weight":"transformer.wte.embedding_table",
                         "embedding.position_embeddings.weight":"transformer.wpe.embedding_table"}
        fix_name = {"encoder" : "transformer", "layers": "h", "self_attention" : "attn", "linear_proj" : "dense", 
                    "linear_qkv" : "qkv_dense", "final_layernorm":"ln_f", "input_norm":"ln_1",
                    "post_attention_norm":"ln_2", "query_key_value":"qkv_dense", "mlp":"mlp.parallel_mlp",
                    "final_norm": "ln_f"}
        for k, v in fix_structure.items():
            if k in key:
                key = key.replace(k, v)
                is_changed = True
        key_split = key.split(".")
        key_new = '.'.join([fix_name.get(k, k) for k in key_split])
        if key_new != key:
            is_changed = True
        return key_new, is_changed
    
    @classmethod
    def _fix_state_dict_keys_on_load(cls, state_dict):
        renamed_keys = {}
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            new_key, is_changed = cls._fix_state_dict_key_on_load(key)
            if is_changed:
                state_dict[new_key] = state_dict.pop(key)
                renamed_keys[key] = new_key
        
        if renamed_keys:
            warning_msg = f"A pretrained model of type `{cls.__name__}` "
            warning_msg += "contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
            for old_key, new_key in renamed_keys.values():
                warning_msg += f"* `{old_key}` -> `{new_key}`\n"
            warning_msg += "If you are using a model from the Hub, consider submitting a PR to adjust these weights and help future users."
            print(warning_msg)
        _tied_weights_keys = _get_tied_weight_keys(cls)
        # remove tied weights from state_dict and get them by comm ops
        for key in _tied_weights_keys:
            if key in state_dict:
                state_dict.pop(key)
        return state_dict
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        ds_parallel_configs,
        config: Optional[Union[PreTrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        weights_only: bool = True,
        use_safetensors: bool = True,
        **kwargs,
    ):
        model_dtype = kwargs.pop("model_dtype", None)
        subfolder = kwargs.pop("subfolder", "")
        state_dict = kwargs.pop("state_dict", None)
        variant = kwargs.pop("variant", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        # 1. read from model config file (cls.config_class.from_pretrained)
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = cls.config_class.from_pretrained(config_path, cache_dir=cache_dir, **kwargs)
        else:
            config = copy.deepcopy(config)
        # 2. 根据model_name_or_path判断是否从本地读取，以及是否为sharded存储（远端存储）
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            elif use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
                is_sharded = True
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_NAME, variant)
                )
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_INDEX_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif use_safetensors:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(TORCH_WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)}"
                    f" found in directory {pretrained_model_name_or_path}."
                )
        elif is_remote_url(pretrained_model_name_or_path):
            # TODO: support remote download
            raise NotImplementedError
        else:
            ValueError("Model name or path is not valid")
        if is_local:
            resolved_archive_file = archive_file
            print(f"loading weights file {resolved_archive_file}")
        else:
            # TODO: support remote download
            raise NotImplementedError
        # 3. 如果是sharded存储，需要读取metadata和sharded文件名
        if is_sharded:
            index_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            with open(index_file, "r") as f:
                index = json.loads(f.read())
            shard_filenames = sorted(set(index["weight_map"].values()))
            sharded_metadata = index["metadata"]
            sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
            sharded_metadata["weight_map"] = index["weight_map"].copy()
            
            if is_local:
                resolved_archive_file = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
            else:
                # TODO: download from remote
                raise NotImplementedError
        # 4. 如果是单文件存储，直接读取state_dict，否则只从metadata获取loaded key
        if not is_sharded and state_dict is None:
            state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only)
            loaded_state_dict_keys = list(state_dict.keys())
        else:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        
        # 5. 确定dtype
        if model_dtype is not None:
            if isinstance(model_dtype, str):
                if model_dtype == "auto":
                    # 从state_dict中获取dtype
                    if is_sharded and "dtype" in sharded_metadata:
                        model_dtype = sharded_metadata["dtype"]
                    elif not is_sharded:
                        model_dtype = get_state_dict_dtype(state_dict)
                    else:
                        one_state_dict = load_state_dict(resolved_archive_file[0], weights_only=weights_only)
                        model_dtype = get_state_dict_dtype(one_state_dict)
                        del one_state_dict  # free CPU memory
                    config.model_dtype = model_dtype
                    for sub_config_key in config.sub_configs.keys():
                        value = getattr(config, sub_config_key)
                        value.model_dtype = default_dtype
                    print(f"Model type is derived from model weights as {model_dtype}")
                elif hasattr(ht, model_dtype):
                    model_dtype = getattr(ht, model_dtype)
                for sub_config_key in config.sub_configs.keys():
                    sub_config = getattr(config, sub_config_key)
                    sub_config.model_dtype = model_dtype
            elif isinstance(model_dtype, ht.dtype):
                config.model_dtype = model_dtype
                for sub_config_key in config.sub_configs.keys():
                    value = getattr(config, sub_config_key)
                    value.model_dtype = default_dtype
            elif isinstance(model_dtype, dict):
                for key, dtype in model_dtype.items():
                    if hasattr(config, key):
                        value = getattr(config, key)
                        value.model_dtype = dtype
                model_dtype = model_dtype.get("")
                if isinstance(model_dtype, str) and hasattr(ht, model_dtype):
                    model_dtype = getattr(ht, model_dtype)
                elif model_dtype is None:
                    model_dtype = ht.float32
                config.model_dtype = model_dtype
            else:
                ValueError("model_dtype should be str, ht.dtype or dict")
        else:
            default_dtype = "float32"
            model_dtype = getattr(ht, default_dtype)
            config.model_dtype = model_dtype
            for sub_config_key in config.sub_configs.keys():
                value = getattr(config, sub_config_key)
                value.model_dtype = default_dtype
        
        config.name_or_path = pretrained_model_name_or_path

        # 6. 创建模型
        with ht.graph("define_and_run", num_strategy=len(ds_parallel_configs)):
            with ht.autocast(model_dtype):
                model = cls(config, ds_parallel_configs=ds_parallel_configs)
        
        # 7. 加载state_dict
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            state_dict,
            loaded_state_dict_keys,
            resolved_archive_file,
            pretrained_model_name_or_path,
            sharded_metadata=sharded_metadata,
            dtype=model_dtype,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            weights_only=weights_only,
        )
        
        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info
        
        return model
    
    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        sharded_metadata=None,
        dtype=None,
        ignore_mismatched_sizes=False,
        weights_only=True,
    ):
        is_sharded = sharded_metadata is not None
        
        # Retrieve missing and unexpected keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix
        
        if len(prefix) > 0:
            load_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expect_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            load_prefix_module = False
            expect_prefix_module = False
        
        original_loaded_keys = loaded_keys
        
        # re-name keys of the newly created model
        # instead of loaded keys
        remove_prefix_from_model = not load_prefix_module and expect_prefix_module
        add_prefix_to_model = load_prefix_module and not expect_prefix_module
        
        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [k for k in expected_keys if not k.startswith(_prefix)]
            expected_keys = [k[len(_prefix): ] if k.startswith(_prefix) else k for k in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, k]) for k in expected_keys]
        
        missing_keys = sorted(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)
        
        # Remove model buffer names from unexpected keys
        model_buffer_keys = {n for n, _ in model.named_buffers()}
        if remove_prefix_from_model:
            model_buffer_keys = {k[len(_prefix): ] if k.startswith(_prefix) else k for k in model_buffer_keys}
        elif add_prefix_to_model:
            model_buffer_keys = [".".join([prefix, k]) for k in model_buffer_keys]
        unexpected_keys = sorted(unexpected_keys - model_buffer_keys)
        
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and load_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        elif len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not load_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    f"Meet unexpected weights for base model {cls.base_model_prefix}. "
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
        
        def _find_mismatch_keys(
            ckpt_state_dict,
            model_state_dict,
            loaded_keys,
            original_loaded_keys,
            remove_prefix_from_model,
            add_prefix_to_model,
            ignore_mismatched_sizes,
        ):
            mismatch_keys = []
            if not ignore_mismatched_sizes:
                for ckpt_key, model_key in zip(original_loaded_keys, loaded_keys):
                    if ckpt_key not in ckpt_state_dict:
                        continue
                    if remove_prefix_from_model:
                        model_key = f"{prefix}.{model_key}"
                    elif add_prefix_to_model:
                        model_key = ".".join(model_key.split(".")[1:])
                    
                    if (
                        model_key in model_state_dict
                        and ckpt_state_dict[ckpt_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatch_keys.append(
                            (ckpt_key, ckpt_state_dict[ckpt_key].shape, model_state_dict[model_key].shape)
                        )
                        del ckpt_state_dict[ckpt_key]
            return mismatch_keys
        
        local_device = ht.local_device()
        
        if not is_sharded:
            mismatched_keys = _find_mismatch_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                original_loaded_keys,
                remove_prefix_from_model,
                add_prefix_to_model,
                ignore_mismatched_sizes,
            )
            fixed_state_dict = cls._fix_state_dict_keys_on_load(state_dict)
            error_msgs = _load_state_dict_into_model(
                model_to_load, fixed_state_dict, start_prefix, local_device
            )
        else:
            error_msgs = []
            mismatched_keys = []
            for shard_file in resolved_archive_file:
                state_dict = load_state_dict(shard_file, weights_only=weights_only, dtype=dtype)
                mismatched_keys += _find_mismatch_keys(
                    state_dict,
                    model_state_dict,
                    loaded_keys,
                    original_loaded_keys,
                    remove_prefix_from_model,
                    add_prefix_to_model,
                    ignore_mismatched_sizes,
                )
                fixed_state_dict = cls._fix_state_dict_keys_on_load(state_dict)
                error_msgs += _load_state_dict_into_model(
                    model_to_load, fixed_state_dict, start_prefix, local_device
                )
                del state_dict
                gc.collect()
        
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            print(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            print(f"All model checkpoint weights were used when initializing {model.__class__.__name__}")
        
        if len(missing_keys) > 0:
            print(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            print(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )
        else:
            print(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        
        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

__all__ = ["PreTrainedModel"]