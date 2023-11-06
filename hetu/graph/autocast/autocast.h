#pragma once

#include "hetu/graph/headers.h"
#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include <mutex>
#include <stack>

namespace hetu {
namespace graph {

class AutoCast {
public:
  AutoCast(bool enabled = true) : _enabled(enabled) {
    _id =  AutoCast::_next_autocast_id();
  }

  bool enabled() const {
    return _enabled;
  }

  AutoCastId id() const {
    return _id;
  }

  bool set_enabled(bool enabled) {
    bool cur_enabled = _enabled;
    _enabled = enabled;
    return cur_enabled;
  }

public:
  static std::shared_ptr<AutoCast>& MakeAutoCast(bool enabled = true) {
    auto autocast_ = std::make_shared<AutoCast>(enabled);
    AutoCast::_autocasts.push_back(autocast_);
    return reinterpret_cast<std::shared_ptr<AutoCast>&>(AutoCast::_autocasts.back());
  }

  static inline AutoCast& GetAutoCast(AutoCastId autocast_id) {
    HT_VALUE_ERROR_IF(autocast_id >= AutoCast::_autocasts.size())
      << "AutoCast with id " << autocast_id << " does not exist";
    return *AutoCast::_autocasts[autocast_id];
  }

  static AutoCast& get_default_autocast() {
    AutoCast::InitOnce();
    return *AutoCast::_default_autocast;
  }

  static void push_autocast_ctx(AutoCastId id) {
    HT_VALUE_ERROR_IF(id >= AutoCast::_autocasts.size())
      << "AutocastContext with id " << id << " does not exist";
    AutoCast::_cur_autocast_ctx.push(id);
  }

  static AutoCastId cur_autocast_ctx() {
    if (AutoCast::_cur_autocast_ctx.empty())
      return UINT64_MAX;
    return AutoCast::_cur_autocast_ctx.top();
  }

  static void pop_autocast_ctx() {
    AutoCast::_cur_autocast_ctx.pop();
  }

  static DataType WidestType(const TensorList& inputs);

  static void Tensor_AutoCast(TensorList& inputs, DataType datatype);

private:
  AutoCastId _id;

  bool _enabled;
//TODO:currently we only support autocast on GPU.
  bool _gpu_enabled;

  static AutoCastId _next_autocast_id() {
    static std::atomic<AutoCastId> _global_autocast_id{0};
    return _global_autocast_id++;
  }
  
protected:
  static void InitOnce() {
    std::call_once(AutoCast::_init_flag, AutoCast::Init);
  }

  static void Init();

  static std::once_flag _init_flag;
  static std::vector<std::shared_ptr<AutoCast>> _autocasts;
  static std::shared_ptr<AutoCast> _default_autocast;
  static thread_local std::stack<AutoCastId> _cur_autocast_ctx;
};

} // namespace graph
} // namespace hetu