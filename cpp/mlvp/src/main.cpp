#include "mlvp/core.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void PrintUsage() {
  std::cout
      << "Usage: mlvp_demo --dot <path> [--workspace 1..4] [--policy mlvp|fifo|donf|minmin|maxmin]\n"
      << "                 [--seed N] [--candidate-cap N] [--gamma X] [--epsilon X]\n"
      << "                 [--max-iterations N] [--alpha-w X] [--alpha-q X] [--alpha-z X]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string dot_path;
    int workspace = 1;
    std::string policy_name = "mlvp";
    std::string weights_file;
    std::uint32_t seed = 0;
    mlvp::MlvpConfig config;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&](const std::string& name) -> std::string {
        if (i + 1 >= argc) {
          throw std::invalid_argument("Missing value for " + name);
        }
        return argv[++i];
      };

      if (arg == "--dot") {
        dot_path = next(arg);
      } else if (arg == "--workspace") {
        workspace = std::stoi(next(arg));
      } else if (arg == "--policy") {
        policy_name = next(arg);
      } else if (arg == "--weights-file") {
        weights_file = next(arg);
      } else if (arg == "--seed") {
        seed = static_cast<std::uint32_t>(std::stoul(next(arg)));
      } else if (arg == "--candidate-cap") {
        config.candidate_cap = static_cast<std::size_t>(std::stoul(next(arg)));
      } else if (arg == "--gamma") {
        config.gamma = std::stod(next(arg));
      } else if (arg == "--epsilon") {
        config.epsilon = std::stod(next(arg));
      } else if (arg == "--max-iterations") {
        config.max_iterations = static_cast<std::size_t>(std::stoul(next(arg)));
      } else if (arg == "--alpha-w") {
        config.weights.alpha_w = std::stod(next(arg));
      } else if (arg == "--alpha-q") {
        config.weights.alpha_q = std::stod(next(arg));
      } else if (arg == "--alpha-z") {
        config.weights.alpha_z = std::stod(next(arg));
      } else if (arg == "--help" || arg == "-h") {
        PrintUsage();
        return 0;
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }

    if (dot_path.empty()) {
      PrintUsage();
      return 1;
    }

    config.seed = seed;
    if (!weights_file.empty()) {
      config.weights = mlvp::LoadMlvpWeightsFile(weights_file);
    }

    mlvp::Dag dag = mlvp::ParseDotFile(dot_path);
    mlvp::Platform platform = mlvp::MakeMlvpWorkspace(workspace, seed);
    std::unique_ptr<mlvp::SchedulingPolicy> policy = mlvp::MakePolicy(policy_name, config);

    mlvp::OnlineSimulator simulator(std::move(dag), std::move(platform));
    const mlvp::SimulationResult result = simulator.Run(*policy);

    std::cout << "policy=" << policy->name()
              << " tasks=" << result.completed_tasks
              << " cycles=" << result.cycles
              << " makespan=" << result.makespan << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
