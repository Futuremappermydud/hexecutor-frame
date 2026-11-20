// Build a small QNN graph from a text spec, finalize it with the HTP backend,
// optionally save the prepared context, then execute it once.
//
// Used to obtain bit-exact per-operator references from QAIRT's x86 HTP
// simulator, and to compare compiled constants with the Model context.
//
// Spec lines (whitespace separated, '#' comments):
//   tensor NAME KIND DTYPE SCALE OFFSET RANK D0 .. [FILE]
//       KIND  app_write | app_read | native | static
//       DTYPE ufxp8 sfxp8 ufxp16 sfxp16 sfxp32 f32 u32 i32
//       OFFSET is QNN's offset, i.e. -zero_point. SCALE 0 means unquantized.
//       FILE holds raw static data.
//   axis_scales NAME AXIS N S0 .. SN-1      per-axis scales, offsets 0
//   param_tensor NODE PNAME RANK D0 .. V0 ..   (uint32 values)
//   param_scalar NODE PNAME u32|i32|f32|bool VALUE
//   node NAME TYPE in N NAMES.. out M NAMES..
//
// Usage: qnn_op_harness BACKEND.so SPEC OUTDIR [--save-context FILE]
//            [--input NAME=FILE].. [--vtcm-mb N]

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "HTP/QnnHtpGraph.h"

struct TensorSpec {
  std::string name, kind, dtype;
  float scale = 0;
  int32_t offset = 0;
  std::vector<uint32_t> dims;
  std::vector<char> data;
  int axis = -1;
  std::vector<Qnn_ScaleOffset_t> axis_scales;
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
};

struct ParamSpec {
  std::string name, type;
  bool is_tensor = false;
  std::vector<uint32_t> dims, values;
  double scalar = 0;
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
};

static void die(const char* what, Qnn_ErrorHandle_t err = 0) {
  fprintf(stderr, "FAILED: %s (error %llu)\n", what, (unsigned long long)err);
  exit(1);
}

static Qnn_DataType_t dtype_of(const std::string& s) {
  if (s == "ufxp8") return QNN_DATATYPE_UFIXED_POINT_8;
  if (s == "sfxp8") return QNN_DATATYPE_SFIXED_POINT_8;
  if (s == "ufxp16") return QNN_DATATYPE_UFIXED_POINT_16;
  if (s == "sfxp16") return QNN_DATATYPE_SFIXED_POINT_16;
  if (s == "sfxp32") return QNN_DATATYPE_SFIXED_POINT_32;
  if (s == "f32") return QNN_DATATYPE_FLOAT_32;
  if (s == "u32") return QNN_DATATYPE_UINT_32;
  if (s == "i32") return QNN_DATATYPE_INT_32;
  die(("unknown dtype " + s).c_str());
  return QNN_DATATYPE_UNDEFINED;
}

static size_t dtype_bytes(Qnn_DataType_t d) {
  switch (d) {
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: return 1;
    case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16: return 2;
    default: return 4;
  }
}

static std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die(("cannot read " + path).c_str());
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void log_cb(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level > QNN_LOG_LEVEL_WARN) return;
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  if (argc < 4) die("usage: qnn_op_harness BACKEND.so SPEC OUTDIR [options]");
  std::string backend_path = argv[1], spec_path = argv[2], outdir = argv[3], save_context;
  std::map<std::string, std::string> inputs;
  uint32_t vtcm_mb = 8;
  for (int i = 4; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--save-context" && i + 1 < argc) save_context = argv[++i];
    else if (a == "--vtcm-mb" && i + 1 < argc) vtcm_mb = atoi(argv[++i]);
    else if (a == "--input" && i + 1 < argc) {
      std::string kv = argv[++i];
      size_t eq = kv.find('=');
      inputs[kv.substr(0, eq)] = kv.substr(eq + 1);
    } else die(("bad argument " + a).c_str());
  }

  // Parse the spec.
  std::vector<TensorSpec*> tensors;
  std::map<std::string, TensorSpec*> by_name;
  std::map<std::string, std::vector<ParamSpec*>> params;
  struct NodeSpec { std::string name, type; std::vector<std::string> in, out; };
  std::vector<NodeSpec> nodes;
  std::ifstream spec(spec_path);
  if (!spec) die("cannot open spec");
  std::string line;
  while (std::getline(spec, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string kw;
    ss >> kw;
    if (kw == "tensor") {
      auto* t = new TensorSpec;
      uint32_t rank;
      ss >> t->name >> t->kind >> t->dtype >> t->scale >> t->offset >> rank;
      t->dims.resize(rank);
      for (auto& d : t->dims) ss >> d;
      std::string file;
      if (ss >> file) t->data = read_file(file);
      tensors.push_back(t);
      by_name[t->name] = t;
    } else if (kw == "axis_scales") {
      std::string name;
      uint32_t n;
      ss >> name;
      TensorSpec* t = by_name.at(name);
      ss >> t->axis >> n;
      t->axis_scales.resize(n);
      for (auto& so : t->axis_scales) { ss >> so.scale; so.offset = 0; }
    } else if (kw == "param_tensor" || kw == "param_scalar") {
      auto* p = new ParamSpec;
      std::string node;
      ss >> node >> p->name;
      if (kw == "param_tensor") {
        p->is_tensor = true;
        uint32_t rank, count = 1;
        ss >> rank;
        p->dims.resize(rank);
        for (auto& d : p->dims) { ss >> d; count *= d; }
        p->values.resize(count);
        for (auto& v : p->values) ss >> v;
      } else {
        ss >> p->type >> p->scalar;
      }
      params[node].push_back(p);
    } else if (kw == "node") {
      NodeSpec n;
      std::string tok;
      uint32_t count;
      ss >> n.name >> n.type >> tok >> count;
      n.in.resize(count);
      for (auto& s : n.in) ss >> s;
      ss >> tok >> count;
      n.out.resize(count);
      for (auto& s : n.out) ss >> s;
      nodes.push_back(n);
    } else die(("unknown spec keyword " + kw).c_str());
  }

  // Load the backend interface.
  void* lib = dlopen(backend_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lib) die(dlerror());
  typedef Qnn_ErrorHandle_t (*GetProviders)(const QnnInterface_t***, uint32_t*);
  auto get_providers = (GetProviders)dlsym(lib, "QnnInterface_getProviders");
  const QnnInterface_t** providers = nullptr;
  uint32_t num_providers = 0;
  if (!get_providers || get_providers(&providers, &num_providers) || !num_providers)
    die("QnnInterface_getProviders");
  const QNN_INTERFACE_VER_TYPE& q = providers[0]->QNN_INTERFACE_VER_NAME;

  Qnn_LogHandle_t log = nullptr;
  q.logCreate(log_cb, QNN_LOG_LEVEL_WARN, &log);
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_ErrorHandle_t err;
  if ((err = q.backendCreate(log, nullptr, &backend))) die("backendCreate", err);
  Qnn_DeviceHandle_t device = nullptr;
  if (q.deviceCreate && (err = q.deviceCreate(log, nullptr, &device))) die("deviceCreate", err);
  Qnn_ContextHandle_t context = nullptr;
  if ((err = q.contextCreate(backend, device, nullptr, &context))) die("contextCreate", err);

  QnnHtpGraph_CustomConfig_t vtcm = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  vtcm.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB;
  vtcm.vtcmSizeInMB = vtcm_mb;
  QnnHtpGraph_CustomConfig_t opt = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  opt.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  opt.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  opt.optimizationOption.floatValue = 3;
  QnnGraph_Config_t gc_vtcm = QNN_GRAPH_CONFIG_INIT, gc_opt = QNN_GRAPH_CONFIG_INIT;
  gc_vtcm.option = gc_opt.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  gc_vtcm.customConfig = &vtcm;
  gc_opt.customConfig = &opt;
  const QnnGraph_Config_t* graph_cfg[] = {&gc_vtcm, &gc_opt, nullptr};
  Qnn_GraphHandle_t graph = nullptr;
  if ((err = q.graphCreate(context, "harness", graph_cfg, &graph))) die("graphCreate", err);

  // Register tensors.
  uint32_t next_id = 0;
  for (auto* t : tensors) {
    Qnn_TensorV1_t& v = t->t.v1;
    v.name = t->name.c_str();
    v.type = t->kind == "app_write" ? QNN_TENSOR_TYPE_APP_WRITE
           : t->kind == "app_read" ? QNN_TENSOR_TYPE_APP_READ
           : t->kind == "static" ? QNN_TENSOR_TYPE_STATIC : QNN_TENSOR_TYPE_NATIVE;
    v.dataType = dtype_of(t->dtype);
    if (!t->axis_scales.empty()) {
      v.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
      v.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
      v.quantizeParams.axisScaleOffsetEncoding.axis = t->axis;
      v.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets = t->axis_scales.size();
      v.quantizeParams.axisScaleOffsetEncoding.scaleOffset = t->axis_scales.data();
    } else if (t->scale != 0) {
      v.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
      v.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      v.quantizeParams.scaleOffsetEncoding.scale = t->scale;
      v.quantizeParams.scaleOffsetEncoding.offset = t->offset;
    }
    v.rank = t->dims.size();
    v.dimensions = t->dims.data();
    v.memType = QNN_TENSORMEMTYPE_RAW;
    if (v.type == QNN_TENSOR_TYPE_STATIC) {
      v.clientBuf.data = t->data.data();
      v.clientBuf.dataSize = t->data.size();
    }
    if ((err = q.tensorCreateGraphTensor(graph, &t->t))) die(("tensorCreateGraphTensor " + t->name).c_str(), err);
  }
  (void)next_id;

  for (auto& n : nodes) {
    std::vector<Qnn_Param_t> ps;
    for (auto* p : params[n.name]) {
      Qnn_Param_t qp = QNN_PARAM_INIT;
      qp.name = p->name.c_str();
      if (p->is_tensor) {
        qp.paramType = QNN_PARAMTYPE_TENSOR;
        Qnn_TensorV1_t& v = p->t.v1;
        std::string tname = n.name + "_" + p->name;
        v.name = strdup(tname.c_str());
        v.type = QNN_TENSOR_TYPE_STATIC;
        v.dataType = QNN_DATATYPE_UINT_32;
        v.rank = p->dims.size();
        v.dimensions = p->dims.data();
        v.memType = QNN_TENSORMEMTYPE_RAW;
        v.clientBuf.data = p->values.data();
        v.clientBuf.dataSize = p->values.size() * 4;
        if ((err = q.tensorCreateGraphTensor(graph, &p->t))) die(("param tensor " + tname).c_str(), err);
        qp.tensorParam = p->t;
      } else {
        qp.paramType = QNN_PARAMTYPE_SCALAR;
        if (p->type == "u32") { qp.scalarParam.dataType = QNN_DATATYPE_UINT_32; qp.scalarParam.uint32Value = (uint32_t)p->scalar; }
        else if (p->type == "i32") { qp.scalarParam.dataType = QNN_DATATYPE_INT_32; qp.scalarParam.int32Value = (int32_t)p->scalar; }
        else if (p->type == "f32") { qp.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; qp.scalarParam.floatValue = (float)p->scalar; }
        else { qp.scalarParam.dataType = QNN_DATATYPE_BOOL_8; qp.scalarParam.bool8Value = p->scalar != 0; }
      }
      ps.push_back(qp);
    }
    std::vector<Qnn_Tensor_t> ins, outs;
    for (auto& s : n.in) ins.push_back(by_name.at(s)->t);
    for (auto& s : n.out) outs.push_back(by_name.at(s)->t);
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = n.name.c_str();
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = n.type.c_str();
    op.v1.numOfParams = ps.size();
    op.v1.params = ps.data();
    op.v1.numOfInputs = ins.size();
    op.v1.inputTensors = ins.data();
    op.v1.numOfOutputs = outs.size();
    op.v1.outputTensors = outs.data();
    if ((err = q.graphAddNode(graph, op))) die(("graphAddNode " + n.name).c_str(), err);
  }
  if ((err = q.graphFinalize(graph, nullptr, nullptr))) die("graphFinalize", err);

  if (!save_context.empty()) {
    Qnn_ContextBinarySize_t size = 0, written = 0;
    if ((err = q.contextGetBinarySize(context, &size))) die("contextGetBinarySize", err);
    std::vector<char> blob(size);
    if ((err = q.contextGetBinary(context, blob.data(), size, &written))) die("contextGetBinary", err);
    std::ofstream(save_context, std::ios::binary).write(blob.data(), written);
    fprintf(stderr, "saved context %s (%llu bytes)\n", save_context.c_str(), (unsigned long long)written);
  }

  // Execute once if every app_write tensor has an input file.
  std::vector<Qnn_Tensor_t> in_t, out_t;
  std::vector<std::vector<char>> bufs;
  bufs.reserve(tensors.size());
  for (auto* t : tensors) {
    if (t->kind != "app_write" && t->kind != "app_read") continue;
    size_t bytes = dtype_bytes(t->t.v1.dataType);
    for (auto d : t->dims) bytes *= d;
    Qnn_Tensor_t qt = t->t;
    if (t->kind == "app_write") {
      if (!inputs.count(t->name)) { fprintf(stderr, "no input for %s; not executing\n", t->name.c_str()); return 0; }
      bufs.push_back(read_file(inputs[t->name]));
      if (bufs.back().size() != bytes) die(("input size mismatch " + t->name).c_str());
    } else {
      bufs.emplace_back(bytes);
    }
    qt.v1.clientBuf.data = bufs.back().data();
    qt.v1.clientBuf.dataSize = bytes;
    (t->kind == "app_write" ? in_t : out_t).push_back(qt);
  }
  if ((err = q.graphExecute(graph, in_t.data(), in_t.size(), out_t.data(), out_t.size(), nullptr, nullptr)))
    die("graphExecute", err);
  for (auto& o : out_t) {
    std::string path = outdir + "/" + o.v1.name + ".raw";
    std::ofstream(path, std::ios::binary).write((char*)o.v1.clientBuf.data, o.v1.clientBuf.dataSize);
  }
  fprintf(stderr, "executed; wrote %zu outputs\n", out_t.size());
  q.contextFree(context, nullptr);
  if (device) q.deviceFree(device);
  q.backendFree(backend);
  return 0;
}
