#include "mlvp/core.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>

namespace mlvp {

namespace {

constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();
constexpr double kEps = 1e-9;

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

void SortRanking(ExecutorRanking* ranking) {
  std::sort(ranking->begin(), ranking->end(),
            [](const RankedTask& lhs, const RankedTask& rhs) {
              if (std::fabs(lhs.score - rhs.score) > kEps) {
                return lhs.score > rhs.score;
              }
              return lhs.task_index < rhs.task_index;
            });
}

std::vector<double> NormalizePositive(const std::vector<double>& scores) {
  std::vector<double> weights(scores.size(), 0.0);
  double denom = 0.0;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    weights[i] = std::max(0.0, scores[i]);
    denom += weights[i];
  }

  if (scores.empty()) {
    return weights;
  }
  if (denom <= kEps) {
    const double uniform = 1.0 / static_cast<double>(scores.size());
    std::fill(weights.begin(), weights.end(), uniform);
    return weights;
  }

  for (double& weight : weights) {
    weight /= denom;
  }
  return weights;
}

std::string ShellQuote(const std::string& value) {
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

}  // namespace

bool OnlineSimulator::Event::operator<(const Event& other) const {
  if (std::fabs(time - other.time) > kEps) {
    return time > other.time;
  }
  return task_index > other.task_index;
}

std::string ToString(NodeType type) {
  switch (type) {
    case NodeType::kCpu:
      return "CPU";
    case NodeType::kGpu:
      return "GPU";
    case NodeType::kIo:
      return "IO";
    case NodeType::kUnknown:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

NodeType ParseNodeType(const std::string& value) {
  const std::string lowered = Lowercase(value);
  if (lowered == "cpu") {
    return NodeType::kCpu;
  }
  if (lowered == "gpu") {
    return NodeType::kGpu;
  }
  if (lowered == "io") {
    return NodeType::kIo;
  }
  return NodeType::kUnknown;
}

std::size_t Dag::AddTask(int id, double compute_cost, double alpha, NodeType type) {
  const std::size_t index = tasks_.size();
  tasks_.push_back(Task{id, compute_cost, alpha, type, {}, {}});
  id_to_index_[id] = index;
  return index;
}

void Dag::AddEdge(int src_id, int dst_id, double comm_cost) {
  const std::size_t src = IndexFromId(src_id);
  const std::size_t dst = IndexFromId(dst_id);
  tasks_[src].succs.push_back(SuccessorEdge{dst, comm_cost});
  tasks_[dst].preds.push_back(PredecessorEdge{src, comm_cost});
}

void Dag::SetTaskType(std::size_t index, NodeType type) {
  tasks_.at(index).type = type;
}

const Task& Dag::task(std::size_t index) const {
  return tasks_.at(index);
}

std::size_t Dag::size() const {
  return tasks_.size();
}

std::size_t Dag::IndexFromId(int node_id) const {
  const auto it = id_to_index_.find(node_id);
  if (it == id_to_index_.end()) {
    throw std::out_of_range("Unknown task id");
  }
  return it->second;
}

std::vector<std::size_t> Dag::EntryIndices() const {
  std::vector<std::size_t> entries;
  for (std::size_t i = 0; i < tasks_.size(); ++i) {
    if (tasks_[i].preds.empty()) {
      entries.push_back(i);
    }
  }
  return entries;
}

Dag ParseDotText(const std::string& text) {
  static const std::regex node_line(R"node(^\s*(\d+)\s*\[(.+)\]\s*$)node");
  static const std::regex edge_line(
      R"edge(^\s*(\d+)\s*->\s*(\d+)\s*\[size\s*=\s*"([^"]+)"\]\s*$)edge");
  static const std::regex attr(R"attr((\w+)\s*=\s*"([^"]+)")attr");

  struct NodeRecord {
    int id = 0;
    double size = 0.0;
    double alpha = 0.0;
    NodeType type = NodeType::kUnknown;
  };

  std::vector<NodeRecord> nodes;
  struct EdgeRecord {
    int src = 0;
    int dst = 0;
    double comm = 0.0;
  };
  std::vector<EdgeRecord> edges;

  std::stringstream input(text);
  std::string line;
  while (std::getline(input, line)) {
    std::smatch edge_match;
    if (std::regex_match(line, edge_match, edge_line)) {
      edges.push_back(EdgeRecord{
          std::stoi(edge_match[1].str()),
          std::stoi(edge_match[2].str()),
          std::stod(edge_match[3].str()),
      });
      continue;
    }

    if (line.find("->") != std::string::npos) {
      continue;
    }

    std::smatch node_match;
    if (!std::regex_match(line, node_match, node_line)) {
      continue;
    }

    std::unordered_map<std::string, std::string> attrs;
    const std::string raw_attrs = node_match[2].str();
    for (std::sregex_iterator it(raw_attrs.begin(), raw_attrs.end(), attr), end;
         it != end; ++it) {
      attrs[(*it)[1].str()] = (*it)[2].str();
    }
    if (attrs.count("size") == 0 || attrs.count("alpha") == 0) {
      continue;
    }
    nodes.push_back(NodeRecord{
        std::stoi(node_match[1].str()),
        std::stod(attrs["size"]),
        std::stod(attrs["alpha"]),
        attrs.count("node_type") ? ParseNodeType(attrs["node_type"]) : NodeType::kUnknown,
    });
  }

  Dag dag;
  for (const NodeRecord& node : nodes) {
    dag.AddTask(node.id, node.size, node.alpha, node.type);
  }
  for (const EdgeRecord& edge : edges) {
    dag.AddEdge(edge.src, edge.dst, edge.comm);
  }
  return dag;
}

Dag ParseDotFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to open dot file: " + path);
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  return ParseDotText(buffer.str());
}

std::string ToDotText(const Dag& dag) {
  std::ostringstream output;
  output << "digraph G {\n";
  for (std::size_t index = 0; index < dag.size(); ++index) {
    const Task& task = dag.task(index);
    output << task.id << " [size=\"" << task.compute_cost << "\", alpha=\"" << task.alpha << "\"";
    if (task.type != NodeType::kUnknown) {
      output << ", node_type=\"" << ToString(task.type) << "\"";
    }
    output << "]\n";
  }
  for (std::size_t index = 0; index < dag.size(); ++index) {
    const Task& task = dag.task(index);
    for (const SuccessorEdge& edge : task.succs) {
      output << task.id << " -> " << dag.task(edge.dst).id << " [size =\"" << edge.comm_cost
             << "\"]\n";
    }
  }
  output << "}\n";
  return output.str();
}

TypeAssignmentStrategy ParseTypeAssignmentStrategy(const std::string& value) {
  const std::string lowered = Lowercase(value);
  if (lowered == "random") {
    return TypeAssignmentStrategy::kRandom;
  }
  if (lowered == "alpha" || lowered == "alpha_based" || lowered == "alpha-based") {
    return TypeAssignmentStrategy::kAlphaBased;
  }
  throw std::invalid_argument("Unknown type assignment strategy: " + value);
}

bool HasUnknownNodeTypes(const Dag& dag) {
  for (std::size_t index = 0; index < dag.size(); ++index) {
    if (dag.task(index).type == NodeType::kUnknown) {
      return true;
    }
  }
  return false;
}

void AssignNodeTypes(Dag* dag, TypeAssignmentStrategy strategy, std::uint32_t seed) {
  if (dag == nullptr) {
    throw std::invalid_argument("dag must not be null");
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> node_type_dist(0, 2);
  for (std::size_t index = 0; index < dag->size(); ++index) {
    if (dag->task(index).type != NodeType::kUnknown) {
      continue;
    }

    NodeType type = NodeType::kUnknown;
    switch (strategy) {
      case TypeAssignmentStrategy::kRandom: {
        const int sampled = node_type_dist(rng);
        type = sampled == 0 ? NodeType::kCpu : sampled == 1 ? NodeType::kGpu : NodeType::kIo;
        break;
      }
      case TypeAssignmentStrategy::kAlphaBased: {
        const double alpha = dag->task(index).alpha;
        if (alpha > 0.15) {
          type = NodeType::kGpu;
        } else if (alpha < 0.05) {
          type = NodeType::kIo;
        } else {
          type = NodeType::kCpu;
        }
        break;
      }
    }
    dag->SetTaskType(index, type);
  }
}

int MlvpDagSizeForWorkspace(int workspace_id) {
  switch (workspace_id) {
    case 1:
      return 3000;
    case 2:
      return 6000;
    case 3:
      return 12000;
    case 4:
      return 24000;
    default:
      throw std::invalid_argument("workspace_id must be 1..4 for MLVP");
  }
}

std::vector<TopologyClass> MlvpTopologyClasses() {
  static const std::vector<double> fats = {0.2, 0.5};
  static const std::vector<double> densities = {0.1, 0.4, 0.8};
  static const std::vector<double> regularities = {0.2, 0.8};
  static const std::vector<int> jumps = {2, 4};
  static const std::vector<int> ccr_values = {2, 8};

  std::vector<TopologyClass> classes;
  classes.reserve(fats.size() * densities.size() * regularities.size() *
                  jumps.size() * ccr_values.size());
  for (double fat : fats) {
    for (double density : densities) {
      for (double regularity : regularities) {
        for (int jump : jumps) {
          for (int ccr : ccr_values) {
            classes.push_back(TopologyClass{fat, density, regularity, jump, ccr});
          }
        }
      }
    }
  }
  return classes;
}

std::string TopologyClassKey(const TopologyClass& topology) {
  auto code_tenths = [](double value) {
    std::ostringstream code;
    code << std::setw(2) << std::setfill('0')
         << static_cast<int>(std::lround(value * 10.0));
    return code.str();
  };

  std::ostringstream key;
  key << "f" << code_tenths(topology.fat)
      << "_d" << code_tenths(topology.density)
      << "_r" << code_tenths(topology.regularity)
      << "_j" << topology.jump
      << "_c" << topology.ccr;
  return key.str();
}

std::string GenerateDaggenDot(const DaggenParams& params) {
  char temp_path[] = "/tmp/mlvp_dagXXXXXX.dot";
  const int fd = mkstemps(temp_path, 4);
  if (fd == -1) {
    throw std::runtime_error("Unable to create temporary file for daggen output");
  }
  close(fd);

  std::ostringstream command;
  command << ShellQuote(params.binary)
          << " --dot"
          << " -o " << ShellQuote(temp_path)
          << " -n " << params.n
          << " --fat " << params.fat
          << " --regular " << params.regular
          << " --density " << params.density
          << " --jump " << params.jump
          << " --ccr " << params.ccr
          << " --minalpha " << params.minalpha
          << " --maxalpha " << params.maxalpha
          << " --mindata " << params.mindata
          << " --maxdata " << params.maxdata;
  command << " >/dev/null 2>&1";

  const int status = std::system(command.str().c_str());
  if (status != 0) {
    std::remove(temp_path);
    throw std::runtime_error("daggen exited with non-zero status");
  }

  std::ifstream input(temp_path);
  if (!input) {
    std::remove(temp_path);
    throw std::runtime_error("Unable to read daggen output");
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::remove(temp_path);
  return buffer.str();
}

Dag GenerateDaggenDag(const DaggenParams& params) {
  return ParseDotText(GenerateDaggenDot(params));
}

double Executor::ProcessingTime(double workload) const {
  return workload / (gflops * 1e9);
}

Platform::Platform(std::vector<Executor> executors) : executors_(std::move(executors)) {}

const std::vector<Executor>& Platform::executors() const {
  return executors_;
}

const Executor& Platform::executor(std::size_t index) const {
  return executors_.at(index);
}

std::vector<std::size_t> Platform::Compatible(NodeType type) const {
  std::vector<std::size_t> compatible;
  for (std::size_t i = 0; i < executors_.size(); ++i) {
    if (executors_[i].type == type) {
      compatible.push_back(i);
    }
  }
  return compatible;
}

std::size_t Platform::size() const {
  return executors_.size();
}

Platform MakeMlvpWorkspace(int workspace_id, std::uint32_t seed) {
  int per_type = 0;
  switch (workspace_id) {
    case 1:
      per_type = 1;
      break;
    case 2:
      per_type = 2;
      break;
    case 3:
      per_type = 4;
      break;
    case 4:
      per_type = 8;
      break;
    default:
      throw std::invalid_argument("workspace_id must be 1..4 for MLVP");
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> gflops_dist(10.0, 300.0);

  std::vector<Executor> executors;
  int next_id = 1;
  for (NodeType type : {NodeType::kCpu, NodeType::kGpu, NodeType::kIo}) {
    for (int i = 0; i < per_type; ++i) {
      executors.push_back(Executor{next_id++, type, gflops_dist(rng)});
    }
  }
  return Platform(std::move(executors));
}

void SavePlatformCsv(const Platform& platform, const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Unable to write platform CSV: " + path);
  }
  output << "id,type,gflops\n";
  for (const Executor& executor : platform.executors()) {
    output << executor.id << ','
           << ToString(executor.type) << ','
           << executor.gflops << '\n';
  }
}

Platform LoadPlatformCsv(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to read platform CSV: " + path);
  }

  std::string line;
  std::getline(input, line);  // header
  std::vector<Executor> executors;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream row(line);
    std::string id_str;
    std::string type_str;
    std::string gflops_str;
    if (!std::getline(row, id_str, ',') ||
        !std::getline(row, type_str, ',') ||
        !std::getline(row, gflops_str, ',')) {
      throw std::runtime_error("Malformed platform CSV row in " + path);
    }
    executors.push_back(Executor{
        std::stoi(id_str),
        ParseNodeType(type_str),
        std::stod(gflops_str),
    });
  }

  if (executors.empty()) {
    throw std::runtime_error("Platform CSV contains no executors: " + path);
  }
  return Platform(std::move(executors));
}

void SaveMlvpWeightsFile(const MlvpWeights& weights, const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Unable to write MLVP weights file: " + path);
  }
  output << "alpha_w=" << weights.alpha_w << '\n'
         << "alpha_q=" << weights.alpha_q << '\n'
         << "alpha_z=" << weights.alpha_z << '\n';
}

MlvpWeights LoadMlvpWeightsFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to read MLVP weights file: " + path);
  }

  MlvpWeights weights;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const std::size_t pos = line.find('=');
    if (pos == std::string::npos) {
      throw std::runtime_error("Malformed MLVP weights line in " + path);
    }
    const std::string key = Lowercase(line.substr(0, pos));
    const double value = std::stod(line.substr(pos + 1));
    if (key == "alpha_w") {
      weights.alpha_w = value;
    } else if (key == "alpha_q") {
      weights.alpha_q = value;
    } else if (key == "alpha_z") {
      weights.alpha_z = value;
    }
  }
  return weights;
}

std::string FifoPolicy::name() const {
  return "fifo";
}

std::vector<ExecutorRanking> FifoPolicy::Rank(const OnlineSimulator& simulator) const {
  std::vector<ExecutorRanking> rankings(simulator.platform().size());
  for (std::size_t executor = 0; executor < simulator.platform().size(); ++executor) {
    if (!simulator.executor_idle(executor)) {
      continue;
    }
    for (std::size_t task_index : simulator.CompatibleReadyTasks(executor)) {
      rankings[executor].push_back(
          RankedTask{task_index, -static_cast<double>(simulator.task_ready_sequence(task_index))});
    }
    SortRanking(&rankings[executor]);
  }
  return rankings;
}

std::string WindowedDonfPolicy::name() const {
  return "donf";
}

std::vector<ExecutorRanking> WindowedDonfPolicy::Rank(const OnlineSimulator& simulator) const {
  std::vector<ExecutorRanking> rankings(simulator.platform().size());
  for (std::size_t executor = 0; executor < simulator.platform().size(); ++executor) {
    if (!simulator.executor_idle(executor)) {
      continue;
    }
    for (std::size_t task_index : simulator.CompatibleReadyTasks(executor)) {
      double wod = 0.0;
      for (std::size_t succ : simulator.VisibleSuccessors(task_index)) {
        const int visible_preds = std::max(1, simulator.VisiblePredecessorCount(succ));
        wod += 1.0 / static_cast<double>(visible_preds);
      }
      rankings[executor].push_back(RankedTask{task_index, wod});
    }
    SortRanking(&rankings[executor]);
  }
  return rankings;
}

std::string LocalMinMinPolicy::name() const {
  return "minmin";
}

std::vector<ExecutorRanking> LocalMinMinPolicy::Rank(const OnlineSimulator& simulator) const {
  std::vector<ExecutorRanking> rankings(simulator.platform().size());
  for (std::size_t executor = 0; executor < simulator.platform().size(); ++executor) {
    if (!simulator.executor_idle(executor)) {
      continue;
    }
    for (std::size_t task_index : simulator.CompatibleReadyTasks(executor)) {
      rankings[executor].push_back(
          RankedTask{task_index, -simulator.EstimatedFinish(task_index, executor)});
    }
    SortRanking(&rankings[executor]);
  }
  return rankings;
}

std::string LocalMaxMinPolicy::name() const {
  return "maxmin";
}

std::vector<ExecutorRanking> LocalMaxMinPolicy::Rank(const OnlineSimulator& simulator) const {
  std::vector<ExecutorRanking> rankings(simulator.platform().size());
  for (std::size_t executor = 0; executor < simulator.platform().size(); ++executor) {
    if (!simulator.executor_idle(executor)) {
      continue;
    }
    for (std::size_t task_index : simulator.CompatibleReadyTasks(executor)) {
      rankings[executor].push_back(
          RankedTask{task_index, simulator.EstimatedFinish(task_index, executor)});
    }
    SortRanking(&rankings[executor]);
  }
  return rankings;
}

MlvpPolicy::MlvpPolicy(MlvpConfig config) : config_(config), rng_(config.seed) {}

std::string MlvpPolicy::name() const {
  return "mlvp";
}

std::vector<ExecutorRanking> MlvpPolicy::Rank(const OnlineSimulator& simulator) const {
  const std::size_t num_exec = simulator.platform().size();
  std::vector<std::vector<std::size_t>> candidates(num_exec);
  std::vector<std::vector<double>> prefs(num_exec);
  std::vector<double> load_state(num_exec, 0.0);
  std::vector<double> expected_work(num_exec, 0.0);

  for (std::size_t executor = 0; executor < num_exec; ++executor) {
    if (!simulator.executor_idle(executor)) {
      load_state[executor] = simulator.remaining_load(executor);
      continue;
    }

    candidates[executor] = simulator.SampledCompatibleReadyTasks(
        executor, config_.candidate_cap, rng_);
    std::vector<double> scores;
    scores.reserve(candidates[executor].size());
    for (std::size_t task_index : candidates[executor]) {
      const std::vector<std::size_t> succs = simulator.VisibleSuccessors(task_index);

      double w = 0.0;
      for (std::size_t succ : succs) {
        w += simulator.UnfinishedPredecessorCount(succ);
      }
      if (!succs.empty()) {
        w /= static_cast<double>(succs.size());
      }

      const int ready_total = static_cast<int>(simulator.ready_tasks().size());
      double q = 0.0;
      for (std::size_t succ : succs) {
        const NodeType succ_type = simulator.dag().task(succ).type;
        const int ready_of_type = std::max(1, simulator.ReadyCount(succ_type));
        q += static_cast<double>(ready_total) / static_cast<double>(ready_of_type);
      }

      double best_proc = std::numeric_limits<double>::infinity();
      for (std::size_t compatible_exec :
           simulator.platform().Compatible(simulator.dag().task(task_index).type)) {
        best_proc = std::min(best_proc, simulator.ProcessingTime(task_index, compatible_exec));
      }
      const double z = best_proc - simulator.ProcessingTime(task_index, executor);

      const double score = config_.weights.alpha_w * w + config_.weights.alpha_q * q +
                           config_.weights.alpha_z * z;
      scores.push_back(score);
    }
    prefs[executor] = NormalizePositive(scores);
    for (std::size_t i = 0; i < candidates[executor].size(); ++i) {
      expected_work[executor] +=
          prefs[executor][i] * simulator.ProcessingTime(candidates[executor][i], executor);
    }
    load_state[executor] = simulator.remaining_load(executor) + expected_work[executor];
  }

  auto neighbors = [&](std::size_t executor) {
    std::vector<std::size_t> result;
    const NodeType type = simulator.platform().executor(executor).type;
    for (std::size_t other = 0; other < num_exec; ++other) {
      if (other == executor) {
        continue;
      }
      if (!config_.fully_connected &&
          simulator.platform().executor(other).type != type) {
        continue;
      }
      result.push_back(other);
    }
    return result;
  };

  for (std::size_t iter = 0; iter < config_.max_iterations; ++iter) {
    std::vector<double> next_state = load_state;
    double max_gap = 0.0;
    for (std::size_t executor = 0; executor < num_exec; ++executor) {
      const std::vector<std::size_t> neigh = neighbors(executor);
      double update = 0.0;
      for (std::size_t other : neigh) {
        update += load_state[other] - load_state[executor];
        max_gap = std::max(max_gap, std::fabs(load_state[other] - load_state[executor]));
      }
      next_state[executor] = load_state[executor] + config_.gamma * update;
    }

    std::unordered_map<int, double> type_mean_load;
    std::unordered_map<int, int> type_counts;
    for (std::size_t executor = 0; executor < num_exec; ++executor) {
      const int key = static_cast<int>(simulator.platform().executor(executor).type);
      type_mean_load[key] += next_state[executor];
      type_counts[key] += 1;
    }
    for (auto& entry : type_mean_load) {
      entry.second /= static_cast<double>(type_counts[entry.first]);
    }

    std::unordered_map<int, std::unordered_map<std::size_t, double>> demand_by_type;
    for (std::size_t executor = 0; executor < num_exec; ++executor) {
      if (!simulator.executor_idle(executor)) {
        continue;
      }
      const int key = static_cast<int>(simulator.platform().executor(executor).type);
      for (std::size_t i = 0; i < candidates[executor].size(); ++i) {
        demand_by_type[key][candidates[executor][i]] += prefs[executor][i];
      }
    }

    std::vector<double> next_expected_work(num_exec, 0.0);
    for (std::size_t executor = 0; executor < num_exec; ++executor) {
      if (!simulator.executor_idle(executor) || candidates[executor].empty()) {
        continue;
      }
      const int key = static_cast<int>(simulator.platform().executor(executor).type);
      const double peer_mean = type_mean_load[key];
      double min_proc = std::numeric_limits<double>::infinity();
      double max_proc = 0.0;
      std::vector<double> proc_times(candidates[executor].size(), 0.0);
      for (std::size_t i = 0; i < candidates[executor].size(); ++i) {
        const double proc = simulator.ProcessingTime(candidates[executor][i], executor);
        proc_times[i] = proc;
        min_proc = std::min(min_proc, proc);
        max_proc = std::max(max_proc, proc);
      }
      const double proc_span = std::max(kEps, max_proc - min_proc);
      const double imbalance_scale =
          std::max({1.0, std::fabs(peer_mean), std::fabs(next_state[executor])});
      const double imbalance_ratio =
          std::clamp((peer_mean - next_state[executor]) / imbalance_scale, -1.0, 1.0);
      for (std::size_t i = 0; i < candidates[executor].size(); ++i) {
        const double demand = demand_by_type[key][candidates[executor][i]];
        double adjusted = prefs[executor][i] /
                          (1.0 + config_.collision_penalty * std::max(0.0, demand - 1.0));
        const double proc_norm = (proc_times[i] - min_proc) / proc_span;
        const double load_signal = 2.0 * proc_norm - 1.0;
        const double load_adjust =
            std::max(0.05, 1.0 + config_.load_balance_gain * imbalance_ratio * load_signal);
        adjusted *= load_adjust;
        prefs[executor][i] = adjusted;
      }
      prefs[executor] = NormalizePositive(prefs[executor]);
      for (std::size_t i = 0; i < candidates[executor].size(); ++i) {
        next_expected_work[executor] += prefs[executor][i] * proc_times[i];
      }
    }

    for (std::size_t executor = 0; executor < num_exec; ++executor) {
      if (!simulator.executor_idle(executor)) {
        load_state[executor] = next_state[executor];
        continue;
      }
      load_state[executor] =
          std::max(0.0, next_state[executor] - expected_work[executor] + next_expected_work[executor]);
    }
    expected_work = std::move(next_expected_work);
    if (max_gap <= config_.epsilon) {
      break;
    }
  }

  std::vector<ExecutorRanking> rankings(num_exec);
  for (std::size_t executor = 0; executor < num_exec; ++executor) {
    for (std::size_t i = 0; i < candidates[executor].size(); ++i) {
      rankings[executor].push_back(RankedTask{candidates[executor][i], prefs[executor][i]});
    }
    SortRanking(&rankings[executor]);
  }
  return rankings;
}

std::unique_ptr<SchedulingPolicy> MakePolicy(const std::string& name, const MlvpConfig& config) {
  const std::string lowered = Lowercase(name);
  if (lowered == "fifo") {
    return std::make_unique<FifoPolicy>();
  }
  if (lowered == "donf") {
    return std::make_unique<WindowedDonfPolicy>();
  }
  if (lowered == "minmin") {
    return std::make_unique<LocalMinMinPolicy>();
  }
  if (lowered == "maxmin") {
    return std::make_unique<LocalMaxMinPolicy>();
  }
  if (lowered == "mlvp") {
    return std::make_unique<MlvpPolicy>(config);
  }
  throw std::invalid_argument("Unknown policy: " + name);
}

ScheduleValidation ValidateSimulationResult(
    const Dag& dag, const Platform& platform, const SimulationResult& result) {
  ScheduleValidation validation;
  auto fail = [&](const std::string& message) {
    validation.valid = false;
    validation.errors.push_back(message);
  };

  if (result.completed_tasks != dag.size()) {
    fail("completed task count does not match DAG size");
  }
  if (result.task_executor_ids.size() != dag.size() ||
      result.task_start_times.size() != dag.size() ||
      result.task_finish_times.size() != dag.size()) {
    fail("result vector sizes do not match DAG size");
    return validation;
  }

  std::unordered_map<int, std::size_t> executor_by_id;
  for (std::size_t executor_index = 0; executor_index < platform.size(); ++executor_index) {
    executor_by_id[platform.executor(executor_index).id] = executor_index;
  }

  struct Interval {
    double start = 0.0;
    double finish = 0.0;
    std::size_t task_index = 0;
  };
  std::vector<std::vector<Interval>> executor_intervals(platform.size());
  std::vector<std::size_t> task_executor_indices(dag.size(), kInvalidIndex);

  for (std::size_t task_index = 0; task_index < dag.size(); ++task_index) {
    const int executor_id = result.task_executor_ids[task_index];
    const auto executor_it = executor_by_id.find(executor_id);
    if (executor_it == executor_by_id.end()) {
      fail("task " + std::to_string(dag.task(task_index).id) +
           " is assigned to an unknown executor");
      continue;
    }

    const std::size_t executor_index = executor_it->second;
    const Executor& executor = platform.executor(executor_index);
    task_executor_indices[task_index] = executor_index;
    if (dag.task(task_index).type != executor.type) {
      fail("task " + std::to_string(dag.task(task_index).id) +
           " is assigned to an incompatible executor");
    }

    const double start = result.task_start_times[task_index];
    const double finish = result.task_finish_times[task_index];
    if (!std::isfinite(start) || !std::isfinite(finish) || start < -kEps ||
        finish < start - kEps) {
      fail("task " + std::to_string(dag.task(task_index).id) +
           " has invalid start/finish times");
      continue;
    }

    const double expected_duration = executor.ProcessingTime(dag.task(task_index).compute_cost);
    if (finish + kEps < start + expected_duration) {
      fail("task " + std::to_string(dag.task(task_index).id) +
           " finishes before its processing time elapses");
    }
    executor_intervals[executor_index].push_back(Interval{start, finish, task_index});
  }

  for (std::size_t task_index = 0; task_index < dag.size(); ++task_index) {
    if (task_executor_indices[task_index] == kInvalidIndex) {
      continue;
    }
    const double start = result.task_start_times[task_index];
    for (const PredecessorEdge& edge : dag.task(task_index).preds) {
      if (task_executor_indices[edge.src] == kInvalidIndex) {
        continue;
      }
      double ready_time = result.task_finish_times[edge.src];
      if (task_executor_indices[edge.src] != task_executor_indices[task_index]) {
        ready_time += edge.comm_cost / kBandwidthBytesPerSecond;
      }
      if (start + kEps < ready_time) {
        fail("task " + std::to_string(dag.task(task_index).id) +
             " starts before predecessor communication readiness");
      }
    }
  }

  for (std::size_t executor_index = 0; executor_index < executor_intervals.size();
       ++executor_index) {
    std::vector<Interval>& intervals = executor_intervals[executor_index];
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& lhs, const Interval& rhs) {
                if (std::fabs(lhs.start - rhs.start) > kEps) {
                  return lhs.start < rhs.start;
                }
                return lhs.task_index < rhs.task_index;
              });
    for (std::size_t i = 1; i < intervals.size(); ++i) {
      if (intervals[i].start + kEps < intervals[i - 1].finish) {
        fail("executor " + std::to_string(platform.executor(executor_index).id) +
             " has overlapping task intervals");
      }
    }
  }

  return validation;
}

OnlineSimulator::OnlineSimulator(Dag dag, Platform platform)
    : dag_(std::move(dag)), platform_(std::move(platform)) {}

SimulationResult OnlineSimulator::Run(const SchedulingPolicy& policy) {
  ResetRuntime();

  while (finished_tasks_ < dag_.size()) {
    const bool has_idle = std::any_of(executor_available_.begin(), executor_available_.end(),
                                      [&](double available) {
                                        return available <= current_time_ + kEps;
                                      });
    if (ready_tasks_.empty() || !has_idle) {
      if (!AdvanceToNextEvent()) {
        throw std::runtime_error("MLVP simulation deadlocked before all tasks finished");
      }
      continue;
    }

    const std::vector<ExecutorRanking> rankings = policy.Rank(*this);
    const auto assignments = ResolveAssignments(rankings);
    ++cycles_;

    if (assignments.empty()) {
      if (!AdvanceToNextEvent()) {
        throw std::runtime_error("MLVP simulation deadlocked with no feasible assignment");
      }
      continue;
    }

    for (const auto& assignment : assignments) {
      CommitTask(assignment.first, assignment.second);
    }

    if (!AdvanceToNextEvent() && finished_tasks_ < dag_.size()) {
      throw std::runtime_error("MLVP simulation deadlocked after committing tasks");
    }
  }

  if (finished_tasks_ != dag_.size()) {
    throw std::runtime_error("MLVP simulation ended with unfinished tasks");
  }

  SimulationResult result;
  result.completed_tasks = finished_tasks_;
  result.cycles = cycles_;
  result.max_ready_width = max_ready_width_;
  result.max_visible_width = max_visible_width_;
  result.task_executor_ids.resize(dag_.size(), -1);
  result.task_start_times.resize(dag_.size(), 0.0);
  result.task_finish_times.resize(dag_.size(), 0.0);

  for (std::size_t i = 0; i < runtime_.size(); ++i) {
    if (runtime_[i].executor_index != kInvalidIndex) {
      result.task_executor_ids[i] = platform_.executor(runtime_[i].executor_index).id;
    }
    result.task_start_times[i] = runtime_[i].start_time;
    result.task_finish_times[i] = runtime_[i].finish_time;
    result.makespan = std::max(result.makespan, runtime_[i].finish_time);
  }

  return result;
}

double OnlineSimulator::current_time() const {
  return current_time_;
}

const Dag& OnlineSimulator::dag() const {
  return dag_;
}

const Platform& OnlineSimulator::platform() const {
  return platform_;
}

bool OnlineSimulator::executor_idle(std::size_t executor_index) const {
  return executor_available_.at(executor_index) <= current_time_ + kEps;
}

double OnlineSimulator::remaining_load(std::size_t executor_index) const {
  return std::max(0.0, executor_available_.at(executor_index) - current_time_);
}

const std::vector<std::size_t>& OnlineSimulator::ready_tasks() const {
  return ready_tasks_;
}

std::vector<std::size_t> OnlineSimulator::CompatibleReadyTasks(std::size_t executor_index) const {
  std::vector<std::size_t> candidates;
  const NodeType type = platform_.executor(executor_index).type;
  for (std::size_t task_index : ready_tasks_) {
    if (dag_.task(task_index).type == type) {
      candidates.push_back(task_index);
    }
  }
  return candidates;
}

std::vector<std::size_t> OnlineSimulator::SampledCompatibleReadyTasks(
    std::size_t executor_index, std::size_t cap, std::mt19937& rng) const {
  std::vector<std::size_t> candidates = CompatibleReadyTasks(executor_index);
  if (cap == 0 || candidates.size() <= cap) {
    return candidates;
  }
  std::shuffle(candidates.begin(), candidates.end(), rng);
  candidates.resize(cap);
  std::sort(candidates.begin(), candidates.end());
  return candidates;
}

std::size_t OnlineSimulator::task_ready_sequence(std::size_t task_index) const {
  return runtime_.at(task_index).ready_sequence;
}

double OnlineSimulator::task_ready_time(std::size_t task_index) const {
  return runtime_.at(task_index).ready_time;
}

double OnlineSimulator::ProcessingTime(std::size_t task_index, std::size_t executor_index) const {
  return platform_.executor(executor_index).ProcessingTime(dag_.task(task_index).compute_cost);
}

double OnlineSimulator::EstimatedStart(std::size_t task_index, std::size_t executor_index) const {
  return EstimateAssignment(task_index, executor_index).start_time;
}

double OnlineSimulator::EstimatedFinish(std::size_t task_index, std::size_t executor_index) const {
  return EstimateAssignment(task_index, executor_index).finish_time;
}

int OnlineSimulator::ReadyCount(NodeType type) const {
  int count = 0;
  for (std::size_t task_index : ready_tasks_) {
    if (dag_.task(task_index).type == type) {
      ++count;
    }
  }
  return count;
}

int OnlineSimulator::UnfinishedPredecessorCount(std::size_t task_index) const {
  int unfinished = 0;
  for (const PredecessorEdge& edge : dag_.task(task_index).preds) {
    if (!runtime_[edge.src].finished) {
      ++unfinished;
    }
  }
  return unfinished;
}

int OnlineSimulator::VisiblePredecessorCount(std::size_t task_index) const {
  RebuildWindowCache();
  return visible_predecessor_count_.at(task_index);
}

std::vector<std::size_t> OnlineSimulator::VisibleSuccessors(std::size_t task_index) const {
  RebuildWindowCache();
  return visible_successors_.at(task_index);
}

void OnlineSimulator::ResetRuntime() {
  for (std::size_t i = 0; i < dag_.size(); ++i) {
    if (dag_.task(i).type == NodeType::kUnknown) {
      throw std::invalid_argument("All tasks must have node_type for MLVP simulation");
    }
  }

  current_time_ = 0.0;
  finished_tasks_ = 0;
  cycles_ = 0;
  ready_sequence_ = 0;
  max_ready_width_ = 0;
  max_visible_width_ = 0;
  runtime_.assign(dag_.size(), TaskRuntimeState{});
  ready_tasks_.clear();
  executor_available_.assign(platform_.size(), 0.0);
  sender_available_.assign(platform_.size(), 0.0);
  events_ = std::priority_queue<Event>();
  InvalidateWindowCache();

  for (std::size_t task_index : dag_.EntryIndices()) {
    MarkTaskReady(task_index, 0.0);
  }
}

void OnlineSimulator::MarkTaskReady(std::size_t task_index, double ready_time) {
  TaskRuntimeState& state = runtime_.at(task_index);
  if (state.in_ready || state.scheduled || state.finished) {
    return;
  }
  state.in_ready = true;
  state.ready_time = ready_time;
  state.ready_sequence = ready_sequence_++;
  ready_tasks_.push_back(task_index);
  InvalidateWindowCache();
  RefreshWindowStats();
}

bool OnlineSimulator::AdvanceToNextEvent() {
  if (events_.empty()) {
    return false;
  }

  current_time_ = events_.top().time;
  std::vector<std::size_t> completed_tasks;
  while (!events_.empty() && std::fabs(events_.top().time - current_time_) <= kEps) {
    const Event event = events_.top();
    events_.pop();
    TaskRuntimeState& state = runtime_.at(event.task_index);
    if (state.finished) {
      continue;
    }
    state.finished = true;
    completed_tasks.push_back(event.task_index);
    ++finished_tasks_;
  }

  UpdateReadyTasks(completed_tasks);
  RefreshWindowStats();
  return true;
}

void OnlineSimulator::UpdateReadyTasks(const std::vector<std::size_t>& completed_tasks) {
  std::vector<std::size_t> candidates;
  for (std::size_t completed : completed_tasks) {
    for (const SuccessorEdge& edge : dag_.task(completed).succs) {
      candidates.push_back(edge.dst);
    }
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

  for (std::size_t task_index : candidates) {
    const TaskRuntimeState& state = runtime_.at(task_index);
    if (state.scheduled || state.finished || state.in_ready) {
      continue;
    }
    bool ready = true;
    for (const PredecessorEdge& edge : dag_.task(task_index).preds) {
      if (!runtime_[edge.src].finished) {
        ready = false;
        break;
      }
    }
    if (ready) {
      MarkTaskReady(task_index, current_time_);
    }
  }
}

void OnlineSimulator::RefreshWindowStats() {
  max_ready_width_ = std::max(max_ready_width_, ready_tasks_.size());
  RebuildWindowCache();
  const std::size_t visible_width =
      static_cast<std::size_t>(std::count(visible_mask_.begin(), visible_mask_.end(), true));
  max_visible_width_ = std::max(max_visible_width_, visible_width);
}

OnlineSimulator::AssignmentEstimate OnlineSimulator::EstimateAssignment(
    std::size_t task_index, std::size_t executor_index) const {
  const Task& task = dag_.task(task_index);
  const Executor& executor = platform_.executor(executor_index);
  if (task.type != executor.type) {
    throw std::invalid_argument("Task assigned to incompatible executor type");
  }

  AssignmentEstimate estimate;
  double data_ready = 0.0;
  std::vector<PredecessorEdge> preds = task.preds;
  std::sort(preds.begin(), preds.end(), [&](const PredecessorEdge& lhs, const PredecessorEdge& rhs) {
    if (std::fabs(runtime_[lhs.src].finish_time - runtime_[rhs.src].finish_time) > kEps) {
      return runtime_[lhs.src].finish_time < runtime_[rhs.src].finish_time;
    }
    return lhs.src < rhs.src;
  });

  for (const PredecessorEdge& edge : preds) {
    const TaskRuntimeState& pred = runtime_.at(edge.src);
    if (!pred.finished && !pred.scheduled) {
      return AssignmentEstimate{
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity(),
          {},
      };
    }

    double arrival = pred.finish_time;
    if (pred.executor_index != executor_index) {
      const double sender_free = estimate.sender_updates.count(pred.executor_index) != 0
                                     ? estimate.sender_updates[pred.executor_index]
                                     : sender_available_.at(pred.executor_index);
      const double start = std::max(pred.finish_time, sender_free);
      arrival = start + edge.comm_cost / kBandwidthBytesPerSecond;
      estimate.sender_updates[pred.executor_index] = arrival;
    }
    data_ready = std::max(data_ready, arrival);
  }

  estimate.start_time =
      std::max(current_time_, std::max(executor_available_.at(executor_index), data_ready));
  estimate.finish_time = estimate.start_time + executor.ProcessingTime(task.compute_cost);
  return estimate;
}

void OnlineSimulator::CommitTask(std::size_t executor_index, std::size_t task_index) {
  AssignmentEstimate estimate = EstimateAssignment(task_index, executor_index);
  TaskRuntimeState& state = runtime_.at(task_index);
  state.scheduled = true;
  state.in_ready = false;
  state.executor_index = executor_index;
  state.start_time = estimate.start_time;
  state.finish_time = estimate.finish_time;

  ready_tasks_.erase(std::remove(ready_tasks_.begin(), ready_tasks_.end(), task_index),
                     ready_tasks_.end());
  executor_available_[executor_index] = estimate.finish_time;
  for (const auto& entry : estimate.sender_updates) {
    sender_available_[entry.first] = entry.second;
  }

  events_.push(Event{estimate.finish_time, executor_index, task_index});
  InvalidateWindowCache();
  RefreshWindowStats();
}

std::vector<std::pair<std::size_t, std::size_t>> OnlineSimulator::ResolveAssignments(
    const std::vector<ExecutorRanking>& rankings) const {
  std::vector<std::pair<std::size_t, std::size_t>> assignments;
  std::vector<std::size_t> cursor(rankings.size(), 0);
  std::vector<char> resolved(rankings.size(), false);
  std::vector<char> task_taken(dag_.size(), false);

  std::size_t unresolved = 0;
  for (std::size_t executor = 0; executor < platform_.size(); ++executor) {
    if (executor_idle(executor)) {
      ++unresolved;
    }
  }

  while (unresolved > 0) {
    std::unordered_map<std::size_t, std::vector<std::size_t>> proposals;
    for (std::size_t executor = 0; executor < rankings.size(); ++executor) {
      if (!executor_idle(executor) || resolved[executor]) {
        continue;
      }
      while (cursor[executor] < rankings[executor].size() &&
             task_taken[rankings[executor][cursor[executor]].task_index]) {
        ++cursor[executor];
      }
      if (cursor[executor] >= rankings[executor].size()) {
        resolved[executor] = true;
        --unresolved;
        continue;
      }
      proposals[rankings[executor][cursor[executor]].task_index].push_back(executor);
    }

    bool progress = false;
    for (auto& entry : proposals) {
      const std::size_t task_index = entry.first;
      std::vector<std::size_t>& executors = entry.second;
      const std::size_t winner = *std::max_element(
          executors.begin(), executors.end(), [&](std::size_t lhs, std::size_t rhs) {
            const double lhs_score = rankings[lhs][cursor[lhs]].score;
            const double rhs_score = rankings[rhs][cursor[rhs]].score;
            if (std::fabs(lhs_score - rhs_score) > kEps) {
              return lhs_score < rhs_score;
            }
            return lhs > rhs;
          });

      task_taken[task_index] = true;
      assignments.push_back({winner, task_index});
      resolved[winner] = true;
      --unresolved;
      progress = true;

      for (std::size_t executor : executors) {
        if (executor != winner) {
          ++cursor[executor];
        }
      }
    }

    if (!progress) {
      break;
    }
  }

  std::sort(assignments.begin(), assignments.end());
  return assignments;
}

void OnlineSimulator::InvalidateWindowCache() const {
  window_cache_valid_ = false;
}

void OnlineSimulator::RebuildWindowCache() const {
  if (window_cache_valid_) {
    return;
  }

  visible_mask_.assign(dag_.size(), false);
  visible_predecessor_count_.assign(dag_.size(), 0);
  visible_successors_.assign(dag_.size(), {});

  for (std::size_t task_index : ready_tasks_) {
    visible_mask_[task_index] = true;
    for (const SuccessorEdge& edge : dag_.task(task_index).succs) {
      visible_mask_[edge.dst] = true;
    }
  }

  for (std::size_t task_index = 0; task_index < dag_.size(); ++task_index) {
    if (!visible_mask_[task_index]) {
      continue;
    }

    for (const PredecessorEdge& edge : dag_.task(task_index).preds) {
      if (visible_mask_[edge.src]) {
        ++visible_predecessor_count_[task_index];
      }
    }
    for (const SuccessorEdge& edge : dag_.task(task_index).succs) {
      if (visible_mask_[edge.dst]) {
        visible_successors_[task_index].push_back(edge.dst);
      }
    }
  }

  window_cache_valid_ = true;
}

}  // namespace mlvp
