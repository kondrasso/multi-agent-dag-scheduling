#ifndef MLVP_CORE_HPP
#define MLVP_CORE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlvp {

constexpr double kBandwidthBytesPerSecond = 1085e6;

enum class NodeType {
  kCpu,
  kGpu,
  kIo,
  kUnknown,
};

std::string ToString(NodeType type);
NodeType ParseNodeType(const std::string& value);

struct SuccessorEdge {
  std::size_t dst = 0;
  double comm_cost = 0.0;
};

struct PredecessorEdge {
  std::size_t src = 0;
  double comm_cost = 0.0;
};

struct Task {
  int id = 0;
  double compute_cost = 0.0;
  double alpha = 0.0;
  NodeType type = NodeType::kUnknown;
  std::vector<SuccessorEdge> succs;
  std::vector<PredecessorEdge> preds;
};

class Dag {
 public:
  std::size_t AddTask(int id, double compute_cost, double alpha, NodeType type);
  void AddEdge(int src_id, int dst_id, double comm_cost);
  void SetTaskType(std::size_t index, NodeType type);

  const Task& task(std::size_t index) const;
  std::size_t size() const;
  std::size_t IndexFromId(int node_id) const;
  std::vector<std::size_t> EntryIndices() const;

 private:
  std::vector<Task> tasks_;
  std::unordered_map<int, std::size_t> id_to_index_;
};

Dag ParseDotText(const std::string& text);
Dag ParseDotFile(const std::string& path);
std::string ToDotText(const Dag& dag);

enum class TypeAssignmentStrategy {
  kRandom,
  kAlphaBased,
};

TypeAssignmentStrategy ParseTypeAssignmentStrategy(const std::string& value);
bool HasUnknownNodeTypes(const Dag& dag);
void AssignNodeTypes(Dag* dag, TypeAssignmentStrategy strategy, std::uint32_t seed);

struct DaggenParams {
  std::string binary = "./daggen/daggen";
  int n = 100;
  double fat = 0.5;
  double regular = 0.9;
  double density = 0.5;
  int jump = 1;
  int ccr = 0;
  double minalpha = 0.0;
  double maxalpha = 0.2;
  int mindata = 2048;
  int maxdata = 11264;
  bool use_seed = false;
  std::uint32_t seed = 0;
};

struct TopologyClass {
  double fat = 0.0;
  double density = 0.0;
  double regularity = 0.0;
  int jump = 0;
  int ccr = 0;
};

int MlvpDagSizeForWorkspace(int workspace_id);
std::vector<TopologyClass> MlvpTopologyClasses();
std::string TopologyClassKey(const TopologyClass& topology);

std::string GenerateDaggenDot(const DaggenParams& params);
Dag GenerateDaggenDag(const DaggenParams& params);

struct Executor {
  int id = 0;
  NodeType type = NodeType::kUnknown;
  double gflops = 0.0;

  double ProcessingTime(double workload) const;
};

class Platform {
 public:
  explicit Platform(std::vector<Executor> executors);

  const std::vector<Executor>& executors() const;
  const Executor& executor(std::size_t index) const;
  std::vector<std::size_t> Compatible(NodeType type) const;
  std::size_t size() const;

 private:
  std::vector<Executor> executors_;
};

Platform MakeMlvpWorkspace(int workspace_id, std::uint32_t seed);
void SavePlatformCsv(const Platform& platform, const std::string& path);
Platform LoadPlatformCsv(const std::string& path);

class OnlineSimulator;

struct RankedTask {
  std::size_t task_index = 0;
  double score = 0.0;
};

using ExecutorRanking = std::vector<RankedTask>;

class SchedulingPolicy {
 public:
  virtual ~SchedulingPolicy() = default;

  virtual std::string name() const = 0;
  virtual std::vector<ExecutorRanking> Rank(const OnlineSimulator& simulator) const = 0;
};

class FifoPolicy final : public SchedulingPolicy {
 public:
  std::string name() const override;
  std::vector<ExecutorRanking> Rank(const OnlineSimulator& simulator) const override;
};

class WindowedDonfPolicy final : public SchedulingPolicy {
 public:
  std::string name() const override;
  std::vector<ExecutorRanking> Rank(const OnlineSimulator& simulator) const override;
};

class LocalMinMinPolicy final : public SchedulingPolicy {
 public:
  std::string name() const override;
  std::vector<ExecutorRanking> Rank(const OnlineSimulator& simulator) const override;
};

class LocalMaxMinPolicy final : public SchedulingPolicy {
 public:
  std::string name() const override;
  std::vector<ExecutorRanking> Rank(const OnlineSimulator& simulator) const override;
};

struct MlvpWeights {
  double alpha_w = 1.0;
  double alpha_q = 1.0;
  double alpha_z = 1.0;
};

void SaveMlvpWeightsFile(const MlvpWeights& weights, const std::string& path);
MlvpWeights LoadMlvpWeightsFile(const std::string& path);

struct MlvpConfig {
  MlvpWeights weights;
  double gamma = 0.2;
  double epsilon = 0.05;
  std::size_t max_iterations = 8;
  std::size_t candidate_cap = 8;
  double collision_penalty = 1.0;
  double load_balance_gain = 0.5;
  bool fully_connected = true;
  std::uint32_t seed = 0;
};

class MlvpPolicy final : public SchedulingPolicy {
 public:
  explicit MlvpPolicy(MlvpConfig config);

  std::string name() const override;
  std::vector<ExecutorRanking> Rank(const OnlineSimulator& simulator) const override;

 private:
  MlvpConfig config_;
  mutable std::mt19937 rng_;
};

std::unique_ptr<SchedulingPolicy> MakePolicy(const std::string& name, const MlvpConfig& config);

struct SimulationResult {
  double makespan = 0.0;
  std::size_t completed_tasks = 0;
  std::size_t cycles = 0;
  std::size_t max_ready_width = 0;
  std::size_t max_visible_width = 0;
  std::vector<int> task_executor_ids;
  std::vector<double> task_start_times;
  std::vector<double> task_finish_times;
};

struct ScheduleValidation {
  bool valid = true;
  std::vector<std::string> errors;
};

ScheduleValidation ValidateSimulationResult(
    const Dag& dag, const Platform& platform, const SimulationResult& result);

class OnlineSimulator {
 public:
  OnlineSimulator(Dag dag, Platform platform);

  SimulationResult Run(const SchedulingPolicy& policy);

  double current_time() const;
  const Dag& dag() const;
  const Platform& platform() const;

  bool executor_idle(std::size_t executor_index) const;
  double remaining_load(std::size_t executor_index) const;

  const std::vector<std::size_t>& ready_tasks() const;
  std::vector<std::size_t> CompatibleReadyTasks(std::size_t executor_index) const;
  std::vector<std::size_t> SampledCompatibleReadyTasks(
      std::size_t executor_index, std::size_t cap, std::mt19937& rng) const;

  std::size_t task_ready_sequence(std::size_t task_index) const;
  double task_ready_time(std::size_t task_index) const;
  double ProcessingTime(std::size_t task_index, std::size_t executor_index) const;
  double EstimatedStart(std::size_t task_index, std::size_t executor_index) const;
  double EstimatedFinish(std::size_t task_index, std::size_t executor_index) const;

  int ReadyCount(NodeType type) const;
  int UnfinishedPredecessorCount(std::size_t task_index) const;
  int VisiblePredecessorCount(std::size_t task_index) const;
  std::vector<std::size_t> VisibleSuccessors(std::size_t task_index) const;

 private:
  struct TaskRuntimeState {
    bool scheduled = false;
    bool finished = false;
    bool in_ready = false;
    std::size_t executor_index = std::numeric_limits<std::size_t>::max();
    double ready_time = 0.0;
    std::size_t ready_sequence = 0;
    double start_time = 0.0;
    double finish_time = 0.0;
  };

  struct AssignmentEstimate {
    double start_time = 0.0;
    double finish_time = 0.0;
    std::unordered_map<std::size_t, double> sender_updates;
  };

  struct Event {
    double time = 0.0;
    std::size_t executor_index = 0;
    std::size_t task_index = 0;

    bool operator<(const Event& other) const;
  };

  void ResetRuntime();
  void MarkTaskReady(std::size_t task_index, double ready_time);
  bool AdvanceToNextEvent();
  void UpdateReadyTasks(const std::vector<std::size_t>& completed_tasks);
  void RefreshWindowStats();
  AssignmentEstimate EstimateAssignment(std::size_t task_index, std::size_t executor_index) const;
  void CommitTask(std::size_t executor_index, std::size_t task_index);
  std::vector<std::pair<std::size_t, std::size_t>> ResolveAssignments(
      const std::vector<ExecutorRanking>& rankings) const;

  void InvalidateWindowCache() const;
  void RebuildWindowCache() const;

  Dag dag_;
  Platform platform_;

  double current_time_ = 0.0;
  std::size_t finished_tasks_ = 0;
  std::size_t cycles_ = 0;
  std::size_t ready_sequence_ = 0;
  std::size_t max_ready_width_ = 0;
  std::size_t max_visible_width_ = 0;

  std::vector<TaskRuntimeState> runtime_;
  std::vector<std::size_t> ready_tasks_;
  std::vector<double> executor_available_;
  std::vector<double> sender_available_;

  mutable bool window_cache_valid_ = false;
  mutable std::vector<char> visible_mask_;
  mutable std::vector<int> visible_predecessor_count_;
  mutable std::vector<std::vector<std::size_t>> visible_successors_;
  std::priority_queue<Event> events_;
};

}  // namespace mlvp

#endif  // MLVP_CORE_HPP
