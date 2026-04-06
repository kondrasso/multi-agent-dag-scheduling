#include "mlvp/core.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>

namespace {

const char* kSmallTypedDag = R"(digraph G {
1 [size="100", alpha="0.10", node_type="CPU"]
2 [size="110", alpha="0.20", node_type="CPU"]
3 [size="120", alpha="0.30", node_type="GPU"]
4 [size="130", alpha="0.40", node_type="GPU"]
5 [size="140", alpha="0.50", node_type="IO"]
1 -> 3 [size ="10"]
2 -> 4 [size ="10"]
3 -> 5 [size ="10"]
4 -> 5 [size ="10"]
}
)";

const char* kSmallUntypedDag = R"(digraph G {
1 [size="100", alpha="0.10"]
2 [size="110", alpha="0.20"]
1 -> 2 [size ="10"]
}
)";

void RunPolicy(const mlvp::Dag& dag, const mlvp::Platform& platform,
               const mlvp::SchedulingPolicy& policy) {
  mlvp::OnlineSimulator simulator(dag, platform);
  const mlvp::SimulationResult result = simulator.Run(policy);
  assert(result.completed_tasks == dag.size());
  assert(result.makespan > 0.0);
}

void CheckPersistenceRoundTrip() {
  const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  const std::filesystem::path platform_path = temp_dir / "mlvp_smoke_platform.csv";
  const std::filesystem::path weights_path = temp_dir / "mlvp_smoke_weights.txt";

  const mlvp::Platform platform({
      mlvp::Executor{1, mlvp::NodeType::kCpu, 12.0},
      mlvp::Executor{2, mlvp::NodeType::kGpu, 25.0},
      mlvp::Executor{3, mlvp::NodeType::kIo, 100.0},
  });
  const mlvp::MlvpWeights weights{1.25, -0.5, 3.75};

  mlvp::SavePlatformCsv(platform, platform_path.string());
  mlvp::SaveMlvpWeightsFile(weights, weights_path.string());

  const mlvp::Platform loaded_platform = mlvp::LoadPlatformCsv(platform_path.string());
  const mlvp::MlvpWeights loaded_weights = mlvp::LoadMlvpWeightsFile(weights_path.string());

  assert(loaded_platform.size() == platform.size());
  for (std::size_t i = 0; i < platform.size(); ++i) {
    assert(loaded_platform.executor(i).id == platform.executor(i).id);
    assert(loaded_platform.executor(i).type == platform.executor(i).type);
    assert(loaded_platform.executor(i).gflops == platform.executor(i).gflops);
  }
  assert(loaded_weights.alpha_w == weights.alpha_w);
  assert(loaded_weights.alpha_q == weights.alpha_q);
  assert(loaded_weights.alpha_z == weights.alpha_z);

  std::filesystem::remove(platform_path);
  std::filesystem::remove(weights_path);
}

}  // namespace

int main() {
  const mlvp::Dag dag = mlvp::ParseDotText(kSmallTypedDag);
  mlvp::Dag untyped = mlvp::ParseDotText(kSmallUntypedDag);

  assert(mlvp::MakeMlvpWorkspace(1, 7).size() == 3);
  assert(mlvp::MakeMlvpWorkspace(2, 7).size() == 6);
  assert(mlvp::MakeMlvpWorkspace(3, 7).size() == 12);
  assert(mlvp::MakeMlvpWorkspace(4, 7).size() == 24);

  assert(mlvp::HasUnknownNodeTypes(untyped));
  mlvp::AssignNodeTypes(&untyped, mlvp::TypeAssignmentStrategy::kAlphaBased, 0);
  assert(!mlvp::HasUnknownNodeTypes(untyped));
  const mlvp::Dag round_tripped = mlvp::ParseDotText(mlvp::ToDotText(untyped));
  assert(!mlvp::HasUnknownNodeTypes(round_tripped));
  CheckPersistenceRoundTrip();

  const mlvp::Platform platform = mlvp::MakeMlvpWorkspace(1, 11);
  RunPolicy(dag, platform, mlvp::FifoPolicy{});
  RunPolicy(dag, platform, mlvp::WindowedDonfPolicy{});
  RunPolicy(dag, platform, mlvp::LocalMinMinPolicy{});
  RunPolicy(dag, platform, mlvp::LocalMaxMinPolicy{});
  RunPolicy(dag, platform, mlvp::MlvpPolicy{mlvp::MlvpConfig{}});

  std::cout << "mlvp smoke test passed\n";
  return 0;
}
