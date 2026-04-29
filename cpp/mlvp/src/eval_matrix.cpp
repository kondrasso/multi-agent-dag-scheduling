#include "mlvp/core.hpp"
#include "mlvp/cli_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

struct WorkspaceSummary {
  int workspace = 0;
  int dag_size = 0;
  std::size_t instances = 0;
  std::map<std::string, std::vector<double>> makespans;
};

struct InstanceResultRow {
  int workspace = 0;
  int dag_size = 0;
  std::size_t instance = 0;
  std::string instance_name;
  std::string policy;
  double makespan = 0.0;
  std::size_t completed_tasks = 0;
  std::size_t cycles = 0;
  std::size_t max_ready_width = 0;
  std::size_t max_visible_width = 0;
  std::optional<double> mlvp_improvement_pct;
};

struct Options {
  std::string dot_root;
  std::size_t generate_per_class = 0;
  std::string save_root;
  std::string summary_csv;
  std::string instances_csv;
  std::string summary_json;
  std::string split = "eval";
  std::string weights_file;
  std::vector<int> workspaces = {1, 2, 3, 4};
  std::vector<std::string> policies = {"mlvp", "donf", "fifo", "minmin", "maxmin"};
  mlvp::TypeAssignmentStrategy type_strategy = mlvp::TypeAssignmentStrategy::kRandom;
  std::uint32_t seed = 0;
  std::string daggen_binary = "./daggen/daggen";
  mlvp::MlvpConfig mlvp_config;
};

void PrintUsage() {
  std::cout
      << "Usage: mlvp_eval_matrix [--dot-root DIR | --generate-per-class N]\n"
      << "                        [--workspace 1,2,3,4] [--assign-types random|alpha]\n"
      << "                        [--policies csv] [--save-root DIR]\n"
      << "                        [--split train|eval] [--weights-file PATH]\n"
      << "                        [--summary-csv PATH] [--instances-csv PATH]\n"
      << "                        [--summary-json PATH]\n"
      << "                        [--seed N] [--daggen-binary PATH]\n"
      << "                        [--candidate-cap N] [--gamma X] [--epsilon X]\n"
      << "                        [--max-iterations N] [--alpha-w X] [--alpha-q X] [--alpha-z X]\n";
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

    if (arg == "--dot-root") {
      options.dot_root = next(arg);
    } else if (arg == "--generate-per-class") {
      options.generate_per_class = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--split") {
      options.split = mlvp::Lowercase(next(arg));
    } else if (arg == "--save-root") {
      options.save_root = next(arg);
    } else if (arg == "--summary-csv") {
      options.summary_csv = next(arg);
    } else if (arg == "--instances-csv") {
      options.instances_csv = next(arg);
    } else if (arg == "--summary-json") {
      options.summary_json = next(arg);
    } else if (arg == "--weights-file") {
      options.weights_file = next(arg);
    } else if (arg == "--workspace") {
      options.workspaces = mlvp::SplitCsvInts(next(arg));
    } else if (arg == "--policies") {
      options.policies = mlvp::SplitCsvStrings(next(arg));
    } else if (arg == "--assign-types") {
      options.type_strategy = mlvp::ParseTypeAssignmentStrategy(next(arg));
    } else if (arg == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(next(arg)));
      options.mlvp_config.seed = options.seed;
    } else if (arg == "--daggen-binary") {
      options.daggen_binary = next(arg);
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

  if (options.dot_root.empty() == (options.generate_per_class == 0)) {
    throw std::invalid_argument("Provide exactly one of --dot-root or --generate-per-class");
  }
  if (options.workspaces.empty()) {
    throw std::invalid_argument("At least one workspace must be selected");
  }
  if (options.policies.empty()) {
    throw std::invalid_argument("At least one policy must be selected");
  }
  if (options.split != "train" && options.split != "eval") {
    throw std::invalid_argument("--split must be train or eval");
  }
  return options;
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

std::optional<double> MlvpImprovementPct(const WorkspaceSummary& summary,
                                         const std::string& baseline_name) {
  if (baseline_name == "mlvp") {
    return std::nullopt;
  }
  const auto mlvp_it = summary.makespans.find("mlvp");
  const auto base_it = summary.makespans.find(baseline_name);
  if (mlvp_it == summary.makespans.end() || base_it == summary.makespans.end() ||
      mlvp_it->second.size() != base_it->second.size()) {
    return std::nullopt;
  }

  double mean_delta = 0.0;
  for (std::size_t i = 0; i < mlvp_it->second.size(); ++i) {
    const double denom = base_it->second[i] > 0.0 ? base_it->second[i] : 1.0;
    mean_delta += (base_it->second[i] - mlvp_it->second[i]) / denom * 100.0;
  }
  mean_delta /= static_cast<double>(mlvp_it->second.size());
  return mean_delta;
}

std::vector<mlvp::Dag> GenerateWorkspaceCorpus(const Options& options, int workspace) {
  std::vector<mlvp::Dag> corpus;
  mlvp::DaggenParams params;
  params.binary = options.daggen_binary;
  params.n = mlvp::MlvpDagSizeForWorkspace(workspace);

  std::uint32_t seed_offset = 0;
  for (const mlvp::TopologyClass& topology : mlvp::MlvpTopologyClasses()) {
    params.fat = topology.fat;
    params.density = topology.density;
    params.regular = topology.regularity;
    params.jump = topology.jump;
    params.ccr = topology.ccr;
    for (std::size_t i = 0; i < options.generate_per_class; ++i) {
      params.use_seed = true;
      params.seed = options.seed + static_cast<std::uint32_t>(
                                       workspace * 100000 + seed_offset++);
      corpus.push_back(mlvp::GenerateDaggenDag(params));
    }
  }

  return corpus;
}

void MaybeSaveCorpus(const std::string& save_root, int workspace,
                     const std::vector<mlvp::Dag>& corpus) {
  if (save_root.empty()) {
    return;
  }

  const std::filesystem::path workspace_dir =
      mlvp::WorkspaceDir(save_root, workspace);
  std::filesystem::create_directories(workspace_dir);
  for (std::size_t i = 0; i < corpus.size(); ++i) {
    std::ostringstream name;
    name << "instance_" << std::setw(4) << std::setfill('0') << i << ".dot";
    std::ofstream output(workspace_dir / name.str());
    output << mlvp::ToDotText(corpus[i]);
  }
}

void WriteCsvSummary(const std::string& path, const std::vector<WorkspaceSummary>& summaries,
                     const std::vector<std::string>& policies) {
  if (path.empty()) {
    return;
  }

  std::ofstream output(path);
  output << "workspace,dag_size,instances,policy,mean_makespan,mlvp_improvement_pct\n";
  for (const WorkspaceSummary& summary : summaries) {
    for (const std::string& policy : policies) {
      output << summary.workspace << ','
             << summary.dag_size << ','
             << summary.instances << ','
             << policy << ','
             << Mean(summary.makespans.at(policy)) << ',';
      const std::optional<double> delta = MlvpImprovementPct(summary, policy);
      if (delta.has_value()) {
        output << *delta;
      }
      output << '\n';
    }
  }
}

void WriteInstanceCsv(const std::string& path, const std::vector<InstanceResultRow>& rows) {
  if (path.empty()) {
    return;
  }

  std::ofstream output(path);
  output << "workspace,dag_size,instance,instance_name,policy,makespan,completed_tasks,"
            "cycles,max_ready_width,max_visible_width,mlvp_improvement_pct\n";
  for (const InstanceResultRow& row : rows) {
    output << row.workspace << ','
           << row.dag_size << ','
           << row.instance << ','
           << row.instance_name << ','
           << row.policy << ','
           << row.makespan << ','
           << row.completed_tasks << ','
           << row.cycles << ','
           << row.max_ready_width << ','
           << row.max_visible_width << ',';
    if (row.mlvp_improvement_pct.has_value()) {
      output << *row.mlvp_improvement_pct;
    }
    output << '\n';
  }
}

void WriteJsonSummary(const std::string& path, const std::vector<WorkspaceSummary>& summaries,
                      const std::vector<std::string>& policies) {
  if (path.empty()) {
    return;
  }

  std::ofstream output(path);
  output << "[\n";
  for (std::size_t i = 0; i < summaries.size(); ++i) {
    const WorkspaceSummary& summary = summaries[i];
    output << "  {\n"
           << "    \"workspace\": " << summary.workspace << ",\n"
           << "    \"dag_size\": " << summary.dag_size << ",\n"
           << "    \"instances\": " << summary.instances << ",\n"
           << "    \"policies\": {\n";
    for (std::size_t j = 0; j < policies.size(); ++j) {
      const std::string& policy = policies[j];
      output << "      \"" << policy << "\": {\n"
             << "        \"mean_makespan\": " << Mean(summary.makespans.at(policy));
      const std::optional<double> delta = MlvpImprovementPct(summary, policy);
      if (delta.has_value()) {
        output << ",\n        \"mlvp_improvement_pct\": " << *delta << '\n';
      } else {
        output << '\n';
      }
      output << "      }";
      if (j + 1 != policies.size()) {
        output << ',';
      }
      output << '\n';
    }
    output << "    }\n"
           << "  }";
    if (i + 1 != summaries.size()) {
      output << ',';
    }
    output << '\n';
  }
  output << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = ParseArgs(argc, argv);
    if (!options.weights_file.empty()) {
      options.mlvp_config.weights = mlvp::LoadMlvpWeightsFile(options.weights_file);
    }
    std::vector<WorkspaceSummary> summaries;
    std::vector<InstanceResultRow> instance_rows;

    for (int workspace_id : options.workspaces) {
      WorkspaceSummary summary;
      summary.workspace = workspace_id;
      summary.dag_size = mlvp::MlvpDagSizeForWorkspace(workspace_id);

      std::vector<mlvp::Dag> corpus;
      std::vector<std::string> instance_names;
      if (!options.dot_root.empty()) {
        const std::vector<std::string> paths =
            mlvp::CollectWorkspaceDots(options.dot_root, workspace_id, options.split);
        for (const std::string& path : paths) {
          corpus.push_back(mlvp::ParseDotFile(path));
          instance_names.push_back(std::filesystem::path(path).filename().string());
        }
      } else {
        corpus = GenerateWorkspaceCorpus(options, workspace_id);
        for (std::size_t i = 0; i < corpus.size(); ++i) {
          std::ostringstream name;
          name << "generated_" << std::setw(4) << std::setfill('0') << i << ".dot";
          instance_names.push_back(name.str());
        }
      }

      for (std::size_t i = 0; i < corpus.size(); ++i) {
        if (mlvp::HasUnknownNodeTypes(corpus[i])) {
          mlvp::AssignNodeTypes(
              &corpus[i], options.type_strategy,
              options.seed + static_cast<std::uint32_t>(workspace_id * 100000 + i));
        }
      }

      MaybeSaveCorpus(options.save_root, workspace_id, corpus);

      const std::filesystem::path workspace_dir =
          options.dot_root.empty() ? std::filesystem::path()
                                   : mlvp::WorkspaceDir(options.dot_root, workspace_id);
      const std::filesystem::path platform_path = workspace_dir / "platform.csv";
      const mlvp::Platform platform =
          (!options.dot_root.empty() && std::filesystem::exists(platform_path))
              ? mlvp::LoadPlatformCsv(platform_path.string())
              : mlvp::MakeMlvpWorkspace(workspace_id,
                                        options.seed + static_cast<std::uint32_t>(workspace_id));

      for (std::size_t instance_idx = 0; instance_idx < corpus.size(); ++instance_idx) {
        std::map<std::string, mlvp::SimulationResult> per_instance_results;
        for (std::size_t policy_idx = 0; policy_idx < options.policies.size(); ++policy_idx) {
          mlvp::MlvpConfig config = options.mlvp_config;
          config.seed = options.seed + static_cast<std::uint32_t>(
                                           workspace_id * 100000 + instance_idx * 97 + policy_idx);
          std::unique_ptr<mlvp::SchedulingPolicy> policy =
              mlvp::MakePolicy(options.policies[policy_idx], config);
          mlvp::OnlineSimulator simulator(corpus[instance_idx], platform);
          const mlvp::SimulationResult result = simulator.Run(*policy);
          const mlvp::ScheduleValidation validation =
              mlvp::ValidateSimulationResult(corpus[instance_idx], platform, result);
          if (!validation.valid) {
            std::ostringstream error;
            error << "invalid schedule for WS" << workspace_id
                  << " instance=" << instance_names[instance_idx]
                  << " policy=" << policy->name() << ": ";
            for (std::size_t i = 0; i < validation.errors.size(); ++i) {
              if (i > 0) {
                error << "; ";
              }
              error << validation.errors[i];
            }
            throw std::runtime_error(error.str());
          }
          summary.makespans[policy->name()].push_back(result.makespan);
          per_instance_results[policy->name()] = result;
        }

        const auto mlvp_it = per_instance_results.find("mlvp");
        for (const std::string& policy_name : options.policies) {
          const mlvp::SimulationResult& result = per_instance_results.at(policy_name);
          InstanceResultRow row;
          row.workspace = workspace_id;
          row.dag_size = summary.dag_size;
          row.instance = instance_idx;
          row.instance_name = instance_names[instance_idx];
          row.policy = policy_name;
          row.makespan = result.makespan;
          row.completed_tasks = result.completed_tasks;
          row.cycles = result.cycles;
          row.max_ready_width = result.max_ready_width;
          row.max_visible_width = result.max_visible_width;
          if (policy_name != "mlvp" && mlvp_it != per_instance_results.end()) {
            const double denom = result.makespan > 0.0 ? result.makespan : 1.0;
            row.mlvp_improvement_pct =
                (result.makespan - mlvp_it->second.makespan) / denom * 100.0;
          }
          instance_rows.push_back(std::move(row));
        }
      }

      summary.instances = corpus.size();
      summaries.push_back(std::move(summary));
    }

    for (const WorkspaceSummary& summary : summaries) {
      std::cout << "WS" << summary.workspace
                << " n=" << summary.dag_size
                << " instances=" << summary.instances << '\n';
      for (const std::string& policy : options.policies) {
        std::cout << "  " << policy
                  << " mean_makespan=" << Mean(summary.makespans.at(policy));
        const std::optional<double> delta = MlvpImprovementPct(summary, policy);
        if (delta.has_value()) {
          std::cout << " mlvp_improvement_pct=" << *delta;
        }
        std::cout << '\n';
      }
    }

    WriteCsvSummary(options.summary_csv, summaries, options.policies);
    WriteInstanceCsv(options.instances_csv, instance_rows);
    WriteJsonSummary(options.summary_json, summaries, options.policies);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
