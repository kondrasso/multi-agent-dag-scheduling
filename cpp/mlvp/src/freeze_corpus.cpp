#include "mlvp/core.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

struct Options {
  std::string out_root;
  std::vector<int> workspaces = {1, 2, 3, 4};
  std::string daggen_binary = "./daggen/daggen";
  mlvp::TypeAssignmentStrategy type_strategy = mlvp::TypeAssignmentStrategy::kRandom;
  std::size_t train_per_class = 3;
  std::size_t eval_per_class = 10;
  std::uint32_t seed = 0;
  bool overwrite = false;
};

int DagSizeForWorkspace(int workspace) {
  switch (workspace) {
    case 1:
      return 3000;
    case 2:
      return 6000;
    case 3:
      return 12000;
    case 4:
      return 24000;
    default:
      throw std::invalid_argument("workspace must be in 1..4");
  }
}

std::vector<int> SplitCsvInts(const std::string& input) {
  std::vector<int> values;
  std::stringstream stream(input);
  std::string item;
  while (std::getline(stream, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(),
                              [](unsigned char ch) { return std::isspace(ch) != 0; }),
               item.end());
    if (!item.empty()) {
      values.push_back(std::stoi(item));
    }
  }
  return values;
}

void PrintUsage() {
  std::cout
      << "Usage: mlvp_freeze_corpus --out-root DIR [--workspace 1,2,3,4]\n"
      << "                         [--assign-types random|alpha] [--train-per-class N]\n"
      << "                         [--eval-per-class N] [--seed N] [--daggen-binary PATH]\n"
      << "                         [--overwrite]\n";
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--out-root") {
      options.out_root = next(arg);
    } else if (arg == "--workspace") {
      options.workspaces = SplitCsvInts(next(arg));
    } else if (arg == "--assign-types") {
      options.type_strategy = mlvp::ParseTypeAssignmentStrategy(next(arg));
    } else if (arg == "--train-per-class") {
      options.train_per_class = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--eval-per-class") {
      options.eval_per_class = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(next(arg)));
    } else if (arg == "--daggen-binary") {
      options.daggen_binary = next(arg);
    } else if (arg == "--overwrite") {
      options.overwrite = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.out_root.empty()) {
    throw std::invalid_argument("--out-root is required");
  }
  if (options.workspaces.empty()) {
    throw std::invalid_argument("At least one workspace must be selected");
  }
  return options;
}

std::string CodeTenths(double value) {
  std::ostringstream code;
  code << std::setw(2) << std::setfill('0') << static_cast<int>(std::lround(value * 10.0));
  return code.str();
}

std::string InstanceName(int dag_size, double fat, double density, double regularity,
                         int jump, int ccr, std::size_t index) {
  std::ostringstream name;
  name << "n" << dag_size
       << "_f" << CodeTenths(fat)
       << "_d" << CodeTenths(density)
       << "_r" << CodeTenths(regularity)
       << "_j" << jump
       << "_c" << ccr
       << '_' << std::setw(4) << std::setfill('0') << index
       << ".dot";
  return name.str();
}

void EnsureWritablePath(const std::filesystem::path& path, bool overwrite) {
  if (std::filesystem::exists(path) && !overwrite) {
    throw std::runtime_error("Refusing to overwrite existing file without --overwrite: " +
                             path.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);

    static const std::vector<double> fats = {0.2, 0.5};
    static const std::vector<double> densities = {0.1, 0.4, 0.8};
    static const std::vector<double> regularities = {0.2, 0.8};
    static const std::vector<int> jumps = {2, 4};
    static const std::vector<int> ccr_values = {2, 8};

    for (int workspace_id : options.workspaces) {
      const std::filesystem::path workspace_dir =
          std::filesystem::path(options.out_root) / ("ws" + std::to_string(workspace_id));
      const std::filesystem::path train_dir = workspace_dir / "train";
      const std::filesystem::path eval_dir = workspace_dir / "eval";
      std::filesystem::create_directories(train_dir);
      std::filesystem::create_directories(eval_dir);

      const mlvp::Platform platform =
          mlvp::MakeMlvpWorkspace(workspace_id,
                                  options.seed + static_cast<std::uint32_t>(workspace_id));
      const std::filesystem::path platform_path = workspace_dir / "platform.csv";
      EnsureWritablePath(platform_path, options.overwrite);
      mlvp::SavePlatformCsv(platform, platform_path.string());

      const std::filesystem::path manifest_path = workspace_dir / "manifest.csv";
      EnsureWritablePath(manifest_path, options.overwrite);
      std::ofstream manifest(manifest_path);
      manifest << "split,path,n,fat,density,regularity,jump,ccr,index\n";

      mlvp::DaggenParams params;
      params.binary = options.daggen_binary;
      params.n = DagSizeForWorkspace(workspace_id);

      std::size_t global_seed_offset = 0;
      for (double fat : fats) {
        params.fat = fat;
        for (double density : densities) {
          params.density = density;
          for (double regularity : regularities) {
            params.regular = regularity;
            for (int jump : jumps) {
              params.jump = jump;
              for (int ccr : ccr_values) {
                params.ccr = ccr;
                const std::size_t total = options.train_per_class + options.eval_per_class;
                for (std::size_t idx = 0; idx < total; ++idx) {
                  mlvp::Dag dag = mlvp::GenerateDaggenDag(params);
                  if (mlvp::HasUnknownNodeTypes(dag)) {
                    mlvp::AssignNodeTypes(
                        &dag,
                        options.type_strategy,
                        options.seed + static_cast<std::uint32_t>(
                                           workspace_id * 100000 + global_seed_offset));
                  }

                  const bool is_train = idx < options.train_per_class;
                  const std::string split = is_train ? "train" : "eval";
                  const std::filesystem::path split_dir = is_train ? train_dir : eval_dir;
                  const std::string file_name =
                      InstanceName(params.n, fat, density, regularity, jump, ccr, idx);
                  const std::filesystem::path out_path = split_dir / file_name;
                  EnsureWritablePath(out_path, options.overwrite);

                  std::ofstream output(out_path);
                  output << mlvp::ToDotText(dag);

                  manifest << split << ','
                           << out_path.filename().string() << ','
                           << params.n << ','
                           << fat << ','
                           << density << ','
                           << regularity << ','
                           << jump << ','
                           << ccr << ','
                           << idx << '\n';
                  ++global_seed_offset;
                }
              }
            }
          }
        }
      }

      std::cout << "froze WS" << workspace_id
                << " n=" << params.n
                << " train=" << (48 * options.train_per_class)
                << " eval=" << (48 * options.eval_per_class)
                << " -> " << workspace_dir << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
