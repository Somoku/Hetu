#pragma once

#include <Python.h>
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/core/dtype.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/core/stream.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/autograd/tensor.h"

namespace hetu {

using namespace hetu::autograd;

enum class ArgType : uint8_t {
  /* Python primitives */
  BOOL = 0, 
  INT64, 
  FLOAT64, 
  STRING, 
  
  /* Python sequences of primitives */
  INT64_LIST, 
  FLOAT64_LIST, 
  STRING_LIST, 
  
  /* NumPy arrays (NumPy scalars will be treated as primitives) */
  PY_ARRAY, 

  /* Python objects (for new tensors from lists or tuples, 
                     please use sparingly) */
  PY_OBJECT, 
  
  /* Hetu types */
  DATA_TYPE, 
  DEVICE, 
  DEVICE_GROUP, 
  STREAM, 
  ND_ARRAY, 
  ND_ARRAY_LIST, 
  TENSOR, 
  TENSOR_LIST
};

std::string ArgType2Str(ArgType);
ArgType Str2ArgType(const std::string&);
std::ostream& operator<<(std::ostream&, const ArgType&);

class FnArg;
class FnSignature;
class PyArgParser;
class ParsedPyArgs;

std::ostream& operator<<(std::ostream&, const FnArg&);
std::ostream& operator<<(std::ostream&, const FnSignature&);

class FnArg {
 public:
  FnArg(const std::string& fmt, size_t equal_sign_hint = std::string::npos);

  bool check_arg(PyObject* obj) const;

  inline ArgType type() const { return _arg_type; }
  inline const std::string& name() const { return _arg_name; }
  inline bool optional() const { return _optional; }
  inline bool allow_none() const { return _allow_none; }
  inline const std::string default_repr() const { return _default_repr; }

 private:
  friend class FnSignature;
  friend class ParsedPyArgs;

  ArgType _arg_type;
  std::string _arg_name;
  PyObject* _py_arg_name;
  bool _optional;
  bool _allow_none;
  bool _default_as_none;

  // default values
  std::string _default_repr;
  bool _default_bool;
  int64_t _default_int64;
  double _default_float64;
  std::string _default_string;
  std::vector<int64_t> _default_int64_list;
  std::vector<double> _default_float64_list;
};

class FnSignature {
 public:
  FnSignature(const std::string& fmt, size_t index);

  bool parse(PyObject* self, PyObject* args, PyObject* kwargs, 
             std::vector<PyObject*>& parsed, bool is_number_method, 
             std::ostream* const mismatch_reason = nullptr);

  inline const std::string& name() const { return _fn_name; }
  inline size_t index() const { return _index; }
  inline const FnArg& arg(size_t i) const { return _args.at(i); }
  inline size_t num_args() const { return _args.size(); }

 private:
  friend class ParsedPyArgs;
  std::string _fn_name;
  std::string _fn_body;
  size_t _index;
  std::vector<FnArg> _args;
  size_t _max_args;
  size_t _min_args;
};

class ParsedPyArgs {
 public:
  ParsedPyArgs(const FnSignature& signature, std::vector<PyObject*>&& args)
  : _signature(signature), _args(std::move(args)) {}

  inline const FnSignature& signature() const { return _signature; }

  inline const std::string& signature_name() const { 
    return _signature.name();
  }
  
  inline size_t signature_index() const { return _signature.index(); }

  inline const FnArg& signature_arg(size_t i) const {
    return _signature.arg(i);
  }

  inline bool has(size_t i) const { return _args[i] != nullptr; }

  // Explanation of the following getters:
  // - `get_<type>`: 
  //    Get the parsed arg without checking the existence.
  //    This is useful for non-optional arguments.
  // - `get_<type>_or_default`:
  //    Get the parsed arg if exists or return the default value if not.
  //    This is useful for optional arguments with non-"None" default values.
  // - `get_<type>_or_else`:
  //    Get the parsed arg if exists or return the "else" value if not.
  //    This is useful for optional arguments and the function arguments
  //    cannot be determined in advance.
  // - `get_<type>_or_peek`:
  //    Get the parsed arg if exists or peek from context libs. 
  //    If the context libs is empty, then `hetu::nullopt` will be returned.
  //    This is useful for optional arguments with "None" default values 
  //    and the function arguments are provided through context libs, 
  //    such as DataType, Device, and Stream. 
  // - `get_<type>_optional`:
  //    Get the parsed arg if exists or return an optional value. 
  //    It requires the `type` is derived from `hetu::shared_ptr_wrapper` 
  //    so that an empty constructor could work as an optional value. 
  //    This is useful for optional arguments with "None" default values 
  //    and the function arguments cannot be provided through context libs,
  //    such as NDArray and Tensor.
  
  inline bool get_bool(size_t i) const {
    return Bool_FromPyBool(_args[i]);
  }
  
  inline bool get_bool_or_default(size_t i) const {
    return has(i) ? get_bool(i) : signature_arg(i)._default_bool;
  }

  inline bool get_bool_or_else(size_t i, bool default_value) const {
    return has(i) ? get_bool(i) : default_value;
  }

  inline int64_t get_int64(size_t i) const {
    return Int64_FromPyLong(_args[i]);
  }
  
  inline int64_t get_int64_or_default(size_t i) const {
    return has(i) ? get_int64(i) : signature_arg(i)._default_int64;
  }

  inline int64_t get_int64_or_else(size_t i, int64_t default_value) const {
    return has(i) ? get_int64(i) : default_value;
  }

  inline double get_float64(size_t i) const {
    return Float64_FromPyFloat(_args[i]);
  }

  inline double get_float64_or_default(size_t i) const {
    return has(i) ? get_float64(i) : signature_arg(i)._default_float64;
  }

  inline double get_float64_or_else(size_t i, double default_value) const {
    return has(i) ? get_float64(i) : default_value;
  }

  inline std::string get_string(size_t i) const {
    return String_FromPyUnicode(_args[i]);
  }

  inline std::string get_string_or_default(size_t i) const {
    return has(i) ? get_string(i) : signature_arg(i)._default_string;
  }

  inline std::string get_string_or_else(size_t i,
                                        std::string&& default_value) const {
    return has(i) ? get_string(i) : default_value;
  }

  inline std::vector<int64_t> get_int64_list(size_t i) const {
    return Int64List_FromPyIntList(_args[i]);
  }

  inline std::vector<int64_t> get_int64_list_or_default(size_t i) const {
    return has(i) ? get_int64_list(i) : signature_arg(i)._default_int64_list;
  }

  inline std::vector<double> get_float64_list(size_t i) const {
    return Float64List_FromPyFloatList(_args[i]);
  }

  inline std::vector<double> get_float64_list_or_default(size_t i) const {
    return has(i) ? get_float64_list(i) 
                  : signature_arg(i)._default_float64_list;
  }

  inline std::vector<std::string> get_string_list(size_t i) const {
    return StringList_FromPyStringList(_args[i]);
  }

  inline optional<std::vector<std::string>> 
  get_string_list_optional(size_t i) const {
    return has(i) ? optional<std::vector<std::string>>(get_string_list(i)) 
                  : nullopt;
  }

  inline PyObject* get_numpy_array(size_t i) const {
    return _args[i];
  }

  inline PyObject* get_numpy_array_optional(size_t i) const {
    return has(i) ? get_numpy_array(i) : nullptr;
  }

  inline PyObject* get_py_obj(size_t i) const {
    return _args[i];
  }

  inline PyObject* get_py_obj_optional(size_t i) const {
    return has(i) ? get_py_obj(i) : nullptr;
  }

  inline DataType get_dtype(size_t i) const {
    return DataType_FromPyObject(_args[i]);
  }

  inline DataType get_dtype_or_else(size_t i, DataType default_value) const {
    return has(i) ? get_dtype(i) : default_value;
  }

  inline optional<DataType> get_dtype_or_peek(size_t i) const {
    return has(i) ? optional<DataType>(get_dtype(i)) : get_dtype_ctx().peek();
  }

  inline Device get_device(size_t i) const {
    return Device_FromPyObject(_args[i]);
  }

  inline optional<Device> get_device_or_peek(size_t i) const {
    return has(i) ? optional<Device>(get_device(i)) : get_device_ctx().peek();
  }

  inline DeviceGroup get_device_group(size_t i) const {
    return DeviceGroup_FromPyObject(_args[i]);
  }

  inline optional<DeviceGroup> get_device_group_or_peek(size_t i) const {
    return has(i) ? optional<DeviceGroup>(get_device_group(i)) \
                  : get_device_group_ctx().peek();
  }

  inline StreamIndex get_stream_index(size_t i) const {
    return static_cast<StreamIndex>(get_int64(i));
  }

  inline optional<StreamIndex> get_stream_index_or_peek(size_t i) const {
    return has(i) ? optional<StreamIndex>(get_stream_index(i))
                  : get_stream_index_ctx().peek();
  }

  inline Stream get_stream(size_t i) const {
    return Stream_FromPyObject(_args[i]);
  }

  inline optional<Stream> get_stream_or_peek(size_t i) const {
    return has(i) ? optional<Stream>(get_stream(i)) : get_stream_ctx().peek();
  }

  inline NDArray get_ndarray(size_t i) const {
    return NDArray_FromPyObject(_args[i]);
  }

  inline NDArray get_ndarray_optional(size_t i) const {
    return has(i) ? get_ndarray(i) : NDArray();
  }

  inline NDArrayList get_ndarray_list(size_t i) const {
    return NDArrayList_FromPyObject(_args[i]);
  }

  inline NDArrayList get_ndarray_list_or_empty(size_t i) const {
    return has(i) ? NDArrayList_FromPyObject(_args[i]) : NDArrayList();
  }

  inline Tensor get_tensor(size_t i) const {
    return Tensor_FromPyObject(_args[i]);
  }

  inline Tensor get_tensor_optional(size_t i) const {
    return has(i) ? get_tensor(i) : Tensor();
  }

  inline TensorList get_tensor_list(size_t i) const {
    return TensorList_FromPyObject(_args[i]);
  }

  inline TensorList get_tensor_list_or_empty(size_t i) const {
    return has(i) ? TensorList_FromPyObject(_args[i]) : TensorList();
  }

 private:
  const FnSignature& _signature;
  std::vector<PyObject*> _args;
};

class PyArgParser {
 public:
  PyArgParser(const std::vector<std::string>& fmts);
  
  ParsedPyArgs parse(PyObject* self, PyObject* args, PyObject* kwargs, 
                     bool is_number_method);

  inline ParsedPyArgs parse(PyObject* args, PyObject* kwargs) {
    return parse(nullptr, args, kwargs, false);
  }

  const std::vector<FnSignature>& signatures() const { return _signatures; }

  inline size_t num_signatures() const { return _signatures.size(); }

  const std::string& name() const { return _signatures.front().name(); }

 private:
  std::vector<FnSignature> _signatures;
};

#define OP_META_ARGS                                                           \
  "int stream_index=None, "                                                    \
  "DeviceGroup device_group=None, "                                            \
  "List[Tensor] extra_deps=None, "                                             \
  "OpName name=None"

inline OpMeta parse_op_meta(const ParsedPyArgs& parsed_args, size_t offset) {
  OpMeta ret;
  auto stream_index_opt = parsed_args.get_stream_index_or_peek(offset);
  if (stream_index_opt != nullopt)
    ret.set_stream_index(*stream_index_opt);
  auto device_group_opt = parsed_args.get_device_group_or_peek(offset + 1);
  if (device_group_opt != nullopt)
    ret.set_device_group(*device_group_opt);
  ret
    .set_extra_deps(parsed_args.get_tensor_list_or_empty(offset + 2))
    .set_name(parsed_args.get_string_or_else(offset + 3, OpName()));
  return ret;
}

} // namespace