#include "hetu/graph/headers.h"
#include "hetu/graph/ops/op_headers.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/ops/matmul.h"
#include "hetu/graph/ops/binary_cross_entropy.h"
#include "hetu/graph/optim/optimizer.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/impl/communication/comm_group.h"
using namespace hetu;
using namespace hetu::graph;

void imperative_run(Graph& graph) {
  HT_LOG_INFO << "----------";
  Graph::push_graph_ctx(graph.id());
  auto w = MakeParameterOp(OnesInitializer(), {5,1}, kFloat32, true, OpMeta().set_name("w").set_eager_device(Device(kCUDA, 0)));
  SGDOptimizer optimizer(TensorList{w}, 0.1f);
  for (int i = 0; i < 5; i++) {
    auto x = MakeVariableOp(ConstantInitializer((i + 1) * 0.01), {10,5}, kFloat32, false, OpMeta().set_name("x").set_eager_device(Device(kCUDA, 0)));
    auto y = MakeVariableOp(ZerosInitializer(), {10, 1}, kFloat32, false, OpMeta().set_name("y").set_eager_device(Device(kCUDA, 0)));
    auto pred = MakeMatMulOp(x, w);
    auto loss = MakeBinaryCrossEntropyOp(pred, y, hetu::ReductionType::MEAN);
    // TODO: zero_grad --> backward --> step
    optimizer.Minimize(loss)->get_or_compute();
    HT_LOG_INFO << "loss = " << loss->get_or_compute();
    HT_LOG_INFO << "pred = " << pred->get_or_compute();
    HT_LOG_INFO << "w = " << w->get_or_compute();
  }
}

void static_run(Graph& graph) {
  HT_LOG_INFO << "----------";
  Graph::push_graph_ctx(graph.id());
  auto w = MakeParameterOp(OnesInitializer(), {5,1}, kFloat32, true, OpMeta().set_name("w"));
  auto x = MakePlaceholderOp(NDArrayMeta().set_shape({10,5}).set_dtype(kFloat32), OpMeta().set_name("x"));
  auto y = MakePlaceholderOp(NDArrayMeta().set_shape({10,1}).set_dtype(kFloat32), OpMeta().set_name("y"));
  auto pred = MakeMatMulOp(x, w);
  auto loss = MakeBinaryCrossEntropyOp(pred, y, hetu::ReductionType::MEAN);
  SGDOptimizer optimizer(0.1f);
  auto train_op = optimizer.Minimize(loss);
  for (size_t i = 0; i < 5; i++) {
    auto x_val = NDArray::full({10,5}, (i + 1) * 0.01, Device(kCUDA));
    auto y_val = NDArray::zeros({10,5}, Device(kCUDA));
    auto ret = graph.Run({loss, pred, w, train_op},
                         {{x->id(), x_val}, {y->id(), y_val}});
    HT_LOG_INFO << "loss = " << ret[0];
    HT_LOG_INFO << "pred = " << ret[1];
    HT_LOG_INFO << "w = " << ret[2];
  }
}


void static_run_dp_ds(Graph& graph) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto& all_devices = hetu::impl::comm::GetGlobalDeviceGroup();  
  HT_LOG_INFO << local_device << ": static_run_dp_ds...";
  Graph::push_graph_ctx(graph.id());
  HT_ASSERT(all_devices.num_devices() >= 4) << "device num must >= 4 !";
  DeviceGroup all_device_group({all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)});

  DistributedStates ds_dup(4, {{-1, 4}}, {-1});
  DistributedStates ds_split(4, {{0, 4}}, {0});

  int local_n = 2;
  int dim = 4;
  auto x = MakePlaceholderOp(NDArrayMeta().set_shape({local_n, dim}).set_dtype(kFloat32), OpMeta().set_name("x").set_device_group(all_device_group));
  x->set_distributed_states(ds_split);
  auto y = MakePlaceholderOp(NDArrayMeta().set_shape({local_n, dim}).set_dtype(kFloat32), OpMeta().set_name("y").set_device_group(all_device_group));
  y->set_distributed_states(ds_split);
  auto w = MakeParameterOp(OnesInitializer(), {dim, dim}, kFloat32, true, OpMeta().set_name("w").set_device_group(all_device_group));
  w->set_distributed_states(ds_dup);

  auto pred = MakeMatMulOp(x, w);
  auto loss = MakeBinaryCrossEntropyOp(pred, y, hetu::ReductionType::MEAN);
  SGDOptimizer optimizer(0.1f);
  auto train_op = optimizer.Minimize(loss);

  auto x_val = NDArray::randn({local_n, dim}, local_device, kFloat32, 0.0, 1.0, (2023 + all_device_group.get_index(local_device)));
  auto y_val = NDArray::zeros({local_n, dim}, local_device, kFloat32);
  auto ret = graph.Run({loss, pred, w, train_op},
                       {{x->id(), x_val}, {y->id(), y_val}});
  HT_LOG_INFO << local_device << "\nloss = " << ret[0] << "\npred = " << ret[1] << "\nw = " << ret[2];
}

void static_run_tp_ds(Graph& graph) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto& all_devices = hetu::impl::comm::GetGlobalDeviceGroup();  
  HT_LOG_INFO << local_device << ": static_run_tp_ds...";
  Graph::push_graph_ctx(graph.id());
  HT_ASSERT(all_devices.num_devices() >= 4) << "device num must >= 4 !";
  DeviceGroup all_device_group({all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)});

  DistributedStates ds_dup(4, {{-1, 4}}, {-1});
  DistributedStates ds_split(4, {{0, 4}}, {0});
  DistributedStates ds_split0_dup(4, {{-1, 2}, {0, 2}}, {0, -1});
  DistributedStates ds_dup_split1(4, {{-1, 2}, {1, 2}}, {-1, 1});
  DistributedStates ds_split01(4, {{0, 2}, {1, 2}}, {0, 1});  

  int local_n = 2;
  int dim = 4;
  auto x = MakePlaceholderOp(NDArrayMeta().set_shape({local_n, dim}).set_dtype(kFloat32), OpMeta().set_device_group(all_device_group).set_name("x"));
  x->set_distributed_states(ds_split);
  // auto y = MakePlaceholderOp(NDArrayMeta().set_shape({local_n*2, dim/2}).set_dtype(kFloat32), OpMeta().set_device_group(all_device_group).set_name("y"));
  // y->set_distributed_states(ds_split01);
  auto y = MakePlaceholderOp(NDArrayMeta().set_shape({local_n*2, dim}).set_dtype(kFloat32), OpMeta().set_device_group(all_device_group).set_name("y"));
  y->set_distributed_states(ds_split0_dup);  

  auto w_data = NDArray::rand({dim, dim}, Device(kCPU), kFloat32, 0.0, 1.0, 2023, kBlockingStream);
  auto w = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w"));
  w->set_distributed_states(ds_dup);
  auto w2_data = NDArray::rand({dim, dim/2}, Device(kCPU), kFloat32, 0.0, 1.0, 2023+1+local_device.index()%2, kBlockingStream);
  auto w2 = MakeParameterOp(w2_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w2"));
  w2->set_distributed_states(ds_dup_split1);

  auto x2 = MakeMatMulOp(x, w, false, false, OpMeta().set_name("mm1"));
  auto x3 = MakeCommOp(x2, ds_split0_dup, OpMeta().set_name("comm_op1"));
  // auto pred = MakeMatMulOp(x3, w2, false, false, OpMeta().set_name("mm2"));
  auto x4 = MakeMatMulOp(x3, w2, false, false, OpMeta().set_name("mm2"));
  auto pred = MakeCommOp(x4, ds_split0_dup, OpMeta().set_name("comm_op2")); 
  auto loss = MakeBinaryCrossEntropyOp(pred, y, hetu::ReductionType::MEAN);
  SGDOptimizer optimizer(0.1, 0.0);
  auto train_op = optimizer.Minimize(loss);

  NDArray data = NDArray::randn({local_n, dim}, local_device, kFloat32, 0.0, 1.0, (666 + all_device_group.get_index(local_device)), kBlockingStream);
  // NDArray labels = NDArray::zeros({local_n*2, dim/2}, local_device, kFloat32, kBlockingStream);
  NDArray labels = NDArray::zeros({local_n*2, dim}, local_device, kFloat32, kBlockingStream);

  auto ret = graph.Run({loss, w, w2, train_op},
                       {{x->id(), data}, {y->id(), labels}});
  HT_LOG_INFO << local_device << "\nw_init: " << w_data << "\nw_updated: " << ret[1]
                              << "\nw2_init: " << w2_data << "\nw2_updated: " << ret[2];                       
}

void static_run_tp_ds2(Graph& graph) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto& all_devices = hetu::impl::comm::GetGlobalDeviceGroup();  
  HT_LOG_INFO << local_device << ": static_run_tp_ds2...";
  Graph::push_graph_ctx(graph.id());
  HT_ASSERT(all_devices.num_devices() >= 4) << "device num must >= 4 !";
  DeviceGroup all_device_group({all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)});

  DistributedStates ds_dup(4, {{-1, 4}}, {-1});
  DistributedStates ds_split0(4, {{0, 4}}, {0});
  DistributedStates ds_split1(4, {{1, 4}}, {1});
  DistributedStates ds_split01(4, {{0, 2}, {1, 2}}, {0, 1});
  DistributedStates ds_split10(4, {{0, 2}, {1, 2}}, {1, 0});    
  DistributedStates ds_dup_split0(4, {{-1, 2}, {0, 2}}, {-1, 0});
  DistributedStates ds_split0_dup(4, {{-1, 2}, {0, 2}}, {0, -1}); 
  DistributedStates ds_dup_split1(4, {{-1, 2}, {1, 2}}, {-1, 1});
  DistributedStates ds_split1_dup(4, {{-1, 2}, {1, 2}}, {1, -1});

  int n = 4;
  int dim0 = 8; // n
  int dim1 = 4; // c
  int h = 4;
  int w = 4;

  auto x0 = MakePlaceholderOp(NDArrayMeta().set_shape({dim0/n, dim1}).set_dtype(kFloat32), OpMeta().set_device_group(all_device_group).set_name("x0"));
  x0->set_distributed_states(ds_split0);
  auto y = MakePlaceholderOp(NDArrayMeta().set_shape({dim0/n, dim1}).set_dtype(kFloat32), OpMeta().set_device_group(all_device_group).set_name("y"));
  y->set_distributed_states(ds_split0);  

  auto w_data = NDArray::rand({dim1, dim1}, Device(kCPU), kFloat32, 0.0, 1.0, 2023, kBlockingStream);
  
  auto w1 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w1"));
  w1->set_distributed_states(ds_dup);
  auto x1 = MakeMatMulOp(x0, w1, false, false, OpMeta().set_name("mm1"));

  auto w2 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w2"));
  w2->set_distributed_states(ds_dup);
  auto x2 = MakeMatMulOp(x0, w2, false, false, OpMeta().set_name("mm2"));

  auto w3 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w3"));
  w3->set_distributed_states(ds_dup);
  auto x3 = MakeMatMulOp(x0, w3, false, false, OpMeta().set_name("mm3"));

  auto w4 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w4"));
  w4->set_distributed_states(ds_dup);
  auto x4 = MakeMatMulOp(x0, w4, false, false, OpMeta().set_name("mm4"));

  auto w5 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w5"));
  w5->set_distributed_states(ds_dup);
  auto x5 = MakeMatMulOp(x0, w5, false, false, OpMeta().set_name("mm5"));

  auto w6 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w6"));
  w6->set_distributed_states(ds_dup);
  auto x6 = MakeMatMulOp(x0, w6, false, false, OpMeta().set_name("mm6"));

  auto w7 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w7"));
  w7->set_distributed_states(ds_dup);
  auto x7 = MakeMatMulOp(x0, w7, false, false, OpMeta().set_name("mm7"));

  auto w8 = MakeParameterOp(w_data, false, kFloat32, true, OpMeta().set_device_group(all_device_group).set_name("w8"));
  w8->set_distributed_states(ds_dup);
  auto x8 = MakeMatMulOp(x0, w8, false, false, OpMeta().set_name("mm8"));    

  auto x_split0 = MakeReluOp(MakeAddElewiseOp(x1, x1));

  x2 = MakeCommOp(x2, ds_dup);
  auto x_dup = MakeReluOp(MakeAddElewiseOp(x2, x2));
  
  x3 = MakeCommOp(x3, ds_split01);
  auto x_split01 = MakeReluOp(MakeAddElewiseOp(x3, x3));
  
  x4 = MakeCommOp(x4, ds_split10);
  auto x_split10 = MakeReluOp(MakeAddElewiseOp(x4, x4));
  
  x5 = MakeCommOp(x5, ds_dup_split0);
  auto x_dup_split0 = MakeReluOp(MakeAddElewiseOp(x5, x5));
  
  x6 = MakeCommOp(x6, ds_split0_dup);
  auto x_split0_dup = MakeReluOp(MakeAddElewiseOp(x6, x6));

  x7 = MakeCommOp(x7, ds_dup_split1);
  auto x_dup_split1 = MakeReluOp(MakeAddElewiseOp(x7, x7));

  x8 = MakeCommOp(x8, ds_split1_dup);
  auto x_split1_dup = MakeReluOp(MakeAddElewiseOp(x8, x8));


  auto loss_split0 = MakeBinaryCrossEntropyOp(x_split0, y, hetu::ReductionType::MEAN);
  x_dup = MakeCommOp(x_dup, ds_split0);
  auto loss_dup = MakeBinaryCrossEntropyOp(x_dup, y, hetu::ReductionType::MEAN);
  x_split01 = MakeCommOp(x_split01, ds_split0);
  auto loss_split01 = MakeBinaryCrossEntropyOp(x_split01, y, hetu::ReductionType::MEAN);
  x_split10 = MakeCommOp(x_split10, ds_split0);
  auto loss_split10 = MakeBinaryCrossEntropyOp(x_split10, y, hetu::ReductionType::MEAN);
  x_dup_split0 = MakeCommOp(x_dup_split0, ds_split0);
  auto loss_dup_split0 = MakeBinaryCrossEntropyOp(x_dup_split0, y, hetu::ReductionType::MEAN);
  x_split0_dup = MakeCommOp(x_split0_dup, ds_split0);
  auto loss_split0_dup = MakeBinaryCrossEntropyOp(x_split0_dup, y, hetu::ReductionType::MEAN);
  x_dup_split1 = MakeCommOp(x_dup_split1, ds_split0);
  auto loss_dup_split1 = MakeBinaryCrossEntropyOp(x_dup_split1, y, hetu::ReductionType::MEAN);
  x_split1_dup = MakeCommOp(x_split1_dup, ds_split0);
  auto loss_split1_dup = MakeBinaryCrossEntropyOp(x_split1_dup, y, hetu::ReductionType::MEAN);
    
  auto loss = MakeSumOp({loss_split0, loss_dup, 
                         loss_split01, loss_split10, 
                         loss_dup_split0, loss_split0_dup,
                         loss_dup_split1, loss_split1_dup});

  SGDOptimizer optimizer(0.1, 0.0);
  auto train_op = optimizer.Minimize(loss);

  NDArray data = NDArray::randn({dim0/n, dim1}, local_device, kFloat32, 0.0, 1.0, (666 + all_device_group.get_index(local_device)), kBlockingStream);
  NDArray labels = NDArray::zeros({dim0/n, dim1}, local_device, kFloat32, kBlockingStream);

  auto ret = graph.Run({loss, w1, w2, w3, w4, w5, w6, w7, w8, train_op},
                       {{x0->id(), data}, {y->id(), labels}});
  
  HT_LOG_INFO << local_device << "\nw_init: " << w_data
                              << "\nw1_updated: " << ret[1]
                              << "\nw2_updated: " << ret[2]
                              << "\nw3_updated: " << ret[3]
                              << "\nw4_updated: " << ret[4]
                              << "\nw5_updated: " << ret[5]
                              << "\nw6_updated: " << ret[6]
                              << "\nw7_updated: " << ret[7]
                              << "\nw8_updated: " << ret[8];
}


int main()
{
  hetu::impl::comm::SetUpDeviceMappingAndAssignLocalDeviceOnce();
  // imperative_run(Graph::get_default_eager_graph());
  // imperative_run(Graph::get_default_define_by_run_graph());
  // static_run(Graph::get_default_define_and_run_graph());

  static_run_dp_ds(Graph::get_default_define_and_run_graph());
  static_run_tp_ds(Graph::get_default_define_and_run_graph());
  static_run_tp_ds2(Graph::get_default_define_and_run_graph());
  return 0;
}

