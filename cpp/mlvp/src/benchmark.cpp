#include "mlvp/core.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

struct Options {
  std::vector<std::string> dot_paths;
  std::string dot_dir;
  std::size_t generate_count = 0;
  mlvp::DaggenParams daggen;
  int workspace = 1;
  std::uint32_t seed = 0;
  std::string weights_file;
  std::string platform_file;
  mlvp::TypeAssignmentStrategy type_strategy = mlvp::TypeAssignmentStrategy::kRandom;
  std::vector<std::string> policies = {"mlvp", "donf", "fifo", "minmin", "maxmin"};
  std::string save_dir;
  mlvp::MlvpConfig mlvp_config;
};

void PrintUsage() {
  std::cout
      << "Usage: mlvp_benchmark [--dot PATH ... | --dot-dir DIR | --generate N]\n"
      << "                      [--workspace 1..4] [--assign-types random|alpha]\n"
      << "                      [--policies csv] [--save-dir DIR] [--seed N]\n"
      << "                      [--weights-file PATH] [--platform-file PATH]\n"
      << "                      [--daggen-binary PATH] [--n N] [--fat X] [--regular X]\n"
      << "                      [--density X] [--jump N] [--ccr N]\n"
      << "                      [--candidate-cap N] [--gamma X] [--epsilon X]\n"
      << "                      [--max-iterations N] [--alpha-w X] [--alpha-q X] [--alpha-z X]\n";
}

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::vector<std::string> SplitCsv(const std::string& input) {
  std::vector<std::string> values;
  std::stringstream stream(input);
  std::string item;
  while (std::getline(stream, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(),
                              [](unsigned char ch) { return std::isspace(ch) != 0; }),
               item.end());
    if (!item.empty()) {
      values.push_back(Lowercase(item));
    }
  }
  return values;
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  options.mlvp_config.seed = options.seed;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--dot") {
      options.dot_paths.push_back(next(arg));
    } else if (arg == "--dot-dir") {
      options.dot_dir = next(arg);
    } else if (arg == "--generate") {
      options.generate_count = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--workspace") {
      options.workspace = std::stoi(next(arg));
    } else if (arg == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(next(arg)));
      options.mlvp_config.seed = options.seed;
    } else if (arg == "--weights-file") {
      options.weights_file = next(arg);
    } else if (arg == "--platform-file") {
      options.platform_file = next(arg);
    } else if (arg == "--assign-types") {
      options.type_strategy = mlvp::ParseTypeAssignmentStrategy(next(arg));
    } else if (arg == "--policies") {
      options.policies = SplitCsv(next(arg));
    } else if (arg == "--save-dir") {
      options.save_dir = next(arg);
    } else if (arg == "--daggen-binary") {
      options.daggen.binary = next(arg);
    } else if (arg == "--n") {
      options.daggen.n = std::stoi(next(arg));
    } else if (arg == "--fat") {
      options.daggen.fat = std::stod(next(arg));
    } else if (arg == "--regular") {
      options.daggen.regular = std::stod(next(arg));
    } else if (arg == "--density") {
      options.daggen.density = std::stod(next(arg));
    } else if (arg == "--jump") {
      options.daggen.jump = std::stoi(next(arg));
    } else if (arg == "--ccr") {
      options.daggen.ccr = std::stoi(next(arg));
    } else if (arg == "--candidate-cap") {
      options.mlvp_config.candidate_cap = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--gamma") {
      options.mlvp_config.gamma = std::stod(next(arg));
    } else if (arg == "--epsilon") {
      options.mlvp_config.epsilon = std::stod(next(arg));
    } else if (arg == "--max-iterations") {
      options.mlvp_config.max_iterations =
          static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--alpha-w") {
      options.mlvp_config.weights.alpha_w = std::stod(next(arg));
    } else if (arg == "--alpha-q") {
      options.mlvp_config.weights.alpha_q = std::stod(next(arg));
    } else if (arg == "--alpha-z") {
      options.mlvp_config.weights.alpha_z = std::stod(next(arg));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.dot_paths.empty() && options.dot_dir.empty() && options.generate_count == 0) {
    throw std::invalid_argument("Provide --dot, --dot-dir, or --generate");
  }
  if (options.policies.empty()) {
    throw std::invalid_argument("At least one policy must be selected");
  }

  return options;
}

std::vector<std::string> CollectDotPaths(const Options& options) {
  std::vector<std::string> paths = options.dot_paths;
  if (!options.dot_dir.empty()) {
    for (const auto& entry : std::filesystem::directory_iterator(options.dot_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".dot") {
        paths.push_back(entry.path().string());
      }
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = ParseArgs(argc, argv);
    std::vector<mlvp::Dag> corpus;

    for (const std::string& path : CollectDotPaths(options)) {
      corpus.push_back(mlvp::ParseDotFile(path));
    }
    for (std::size_t i = 0; i < options.generate_count; ++i) {
      corpus.push_back(mlvp::GenerateDaggenDag(options.daggen));
    }

    if (corpus.empty()) {
      throw std::runtime_error("No DAG instances were loaded");
    }

    for (std::size_t i = 0; i < corpus.size(); ++i) {
      if (mlvp::HasUnknownNodeTypes(corpus[i])) {
        mlvp::AssignNodeTypes(&corpus[i], options.type_strategy, options.seed + static_cast<std::uint32_t>(i));
      }
    }
    if (!options.weights_file.empty()) {
      options.mlvp_config.weights = mlvp::LoadMlvpWeightsFile(options.weights_file);
    }

    if (!options.save_dir.empty()) {
      std::filesystem::create_directories(options.save_dir);
      for (std::size_t i = 0; i < corpus.size(); ++i) {
        std::ostringstream path;
        path << options.save_dir << "/instance_" << std::setw(4) << std::setfill('0') << i
             << ".dot";
        std::ofstream output(path.str());
        output << mlvp::ToDotText(corpus[i]);
      }
    }

    const mlvp::Platform workspace = options.platform_file.empty()
                                         ? mlvp::MakeMlvpWorkspace(options.workspace, options.seed)
                                         : mlvp::LoadPlatformCsv(options.platform_file);
    std::map<std::string, std::vector<double>> results;

    for (std::size_t instance_idx = 0; instance_idx < corpus.size(); ++instance_idx) {
      for (std::size_t policy_idx = 0; policy_idx < options.policies.size(); ++policy_idx) {
        mlvp::MlvpConfig config = options.mlvp_config;
        config.seed = options.seed + static_cast<std::uint32_t>(instance_idx * 97 + policy_idx);
        std::unique_ptr<mlvp::SchedulingPolicy> policy =
            mlvp::MakePolicy(options.policies[policy_idx], config);
        mlvp::OnlineSimulator simulator(corpus[instance_idx], workspace);
        const mlvp::SimulationResult result = simulator.Run(*policy);
        results[policy->name()].push_back(result.makespan);
      }
    }

    std::cout << "instances=" << corpus.size()
              << " workspace=" << options.workspace
              << " executors=" << workspace.size()
              << " type_assignment="
              << (options.type_strategy == mlvp::TypeAssignmentStrategy::kRandom ? "random" : "alpha")
              << '\n';

    for (const std::string& policy_name : options.policies) {
      std::cout << policy_name << " mean_makespan=" << Mean(results[policy_name]) << '\n';
    }

    if (results.count("mlvp") != 0) {
      const std::vector<double>& mlvp_values = results["mlvp"];
      for (const std::string& policy_name : options.policies) {
        if (policy_name == "mlvp") {
          continue;
        }
        const std::vector<double>& baseline = results[policy_name];
        if (baseline.size() != mlvp_values.size()) {
          continue;
        }
        double mean_delta = 0.0;
        for (std::size_t i = 0; i < baseline.size(); ++i) {
          const double denom = baseline[i] > 0.0 ? baseline[i] : 1.0;
          mean_delta += (baseline[i] - mlvp_values[i]) / denom * 100.0;
        }
        mean_delta /= static_cast<double>(baseline.size());
        std::cout << "mlvp_vs_" << policy_name << "=" << mean_delta << "%\n";
      }
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
