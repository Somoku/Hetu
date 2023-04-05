#include "hetu/_binding/module.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/core/dtype.h"
#include "hetu/_binding/core/stream.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/graph/operator.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/graph/graph.h"

PYBIND11_MODULE(HT_CORE_PY_MODULE, m) {
  hetu::AddPyDeviceTypeToModule(m);
  hetu::AddPyDeviceGroupTypeToModule(m);
  hetu::AddPyDataTypeTypeToModule(m);
  hetu::AddPyStreamTypeToModule(m);
  hetu::AddPyNDArrayTypeToModule(m);
  hetu::graph::AddPyOperatorTypeToModule(m);
  hetu::graph::AddPyTensorTypeToModule(m);
  hetu::graph::AddPyGraphTypeToModule(m);
  auto internal_sub_module = m.def_submodule("_internal_context");
  hetu::graph::AddOpContextManagingFunctionsToModule(internal_sub_module);
  hetu::graph::AddGraphContextManagingFunctionsToModule(internal_sub_module);
}
