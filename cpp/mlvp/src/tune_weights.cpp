#include "mlvp/core.hpp"
#include "mlvp/cli_utils.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

enum class ObjectiveMode {
  kMakespan,
  kMeanRatio,
  kBestBaselineRatio,
};

enum class SampleMode {
  kRandom,
  kStratified,
};

struct Options {
  std::string corpus_root;
  int workspace = 1;
  std::string split = "train";
  std::string out_weights;
  std::string history_csv;
  std::uint32_t seed = 0;
  std::size_t sample_size = 0;
  SampleMode sample_mode = SampleMode::kStratified;
  std::size_t jobs = 1;
  std::size_t population = 32;
  std::size_t generations = 40;
  double mutation_prob = 0.8;
  double mutation_sigma = 0.25;
  std::vector<std::string> baselines = {"donf"};
  ObjectiveMode objective = ObjectiveMode::kBestBaselineRatio;
  mlvp::TypeAssignmentStrategy type_strategy = mlvp::TypeAssignmentStrategy::kRandom;
  mlvp::MlvpConfig mlvp_config;
};

using Chromosome = std::array<double, 3>;

struct EvaluationMetrics {
  double objective = std::numeric_limits<double>::infinity();
  double mean_makespan = std::numeric_limits<double>::infinity();
  double mean_ratio = std::numeric_limits<double>::infinity();
  double best_baseline_ratio = std::numeric_limits<double>::infinity();
  bool has_baselines = false;
};

struct Individual {
  Chromosome genes{};
  EvaluationMetrics metrics;
};

struct BaselineCorpus {
  std::vector<std::string> names;
  std::vector<std::vector<double>> makespans;
};

ObjectiveMode ParseObjectiveMode(const std::string& value) {
  const std::string lowered = mlvp::Lowercase(value);
  if (lowered == "makespan") {
    return ObjectiveMode::kMakespan;
  }
  if (lowered == "mean-ratio") {
    return ObjectiveMode::kMeanRatio;
  }
  if (lowered == "best-baseline-ratio") {
    return ObjectiveMode::kBestBaselineRatio;
  }
  throw std::invalid_argument("Unknown objective mode: " + value);
}

const char* ObjectiveModeName(ObjectiveMode mode) {
  switch (mode) {
    case ObjectiveMode::kMakespan:
      return "makespan";
    case ObjectiveMode::kMeanRatio:
      return "mean-ratio";
    case ObjectiveMode::kBestBaselineRatio:
      return "best-baseline-ratio";
  }
  return "unknown";
}

SampleMode ParseSampleMode(const std::string& value) {
  const std::string lowered = mlvp::Lowercase(value);
  if (lowered == "random") {
    return SampleMode::kRandom;
  }
  if (lowered == "stratified") {
    return SampleMode::kStratified;
  }
  throw std::invalid_argument("Unknown sample mode: " + value);
}

const char* SampleModeName(SampleMode mode) {
  switch (mode) {
    case SampleMode::kRandom:
      return "random";
    case SampleMode::kStratified:
      return "stratified";
  }
  return "unknown";
}

void PrintUsage() {
  std::cout
      << "Usage: mlvp_tune_weights --corpus-root DIR --workspace 1..4 [--split train|eval]\n"
      << "                         [--out PATH] [--history-csv PATH] [--seed N]\n"
      << "                         [--sample-size N] [--sample-mode random|stratified]\n"
      << "                         [--jobs N]\n"
      << "                         [--population N] [--generations N]\n"
      << "                         [--mutation-prob X] [--mutation-sigma X]\n"
      << "                         [--objective makespan|mean-ratio|best-baseline-ratio]\n"
      << "                         [--baselines csv] [--assign-types random|alpha]\n"
      << "                         [--candidate-cap N] [--gamma X] [--epsilon X]\n"
      << "                         [--max-iterations N]\n";
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

    if (arg == "--corpus-root") {
      options.corpus_root = next(arg);
    } else if (arg == "--workspace") {
      options.workspace = std::stoi(next(arg));
    } else if (arg == "--split") {
      options.split = mlvp::Lowercase(next(arg));
    } else if (arg == "--out") {
      options.out_weights = next(arg);
    } else if (arg == "--history-csv") {
      options.history_csv = next(arg);
    } else if (arg == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(next(arg)));
      options.mlvp_config.seed = options.seed;
    } else if (arg == "--sample-size") {
      options.sample_size = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--sample-mode") {
      options.sample_mode = ParseSampleMode(next(arg));
    } else if (arg == "--jobs") {
      options.jobs = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--population") {
      options.population = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--generations") {
      options.generations = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--mutation-prob") {
      options.mutation_prob = std::stod(next(arg));
    } else if (arg == "--mutation-sigma") {
      options.mutation_sigma = std::stod(next(arg));
    } else if (arg == "--objective") {
      options.objective = ParseObjectiveMode(next(arg));
    } else if (arg == "--baselines") {
      options.baselines = mlvp::SplitCsvStrings(next(arg));
    } else if (arg == "--assign-types") {
      options.type_strategy = mlvp::ParseTypeAssignmentStrategy(next(arg));
    } else if (arg == "--candidate-cap") {
      options.mlvp_config.candidate_cap = static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--gamma") {
      options.mlvp_config.gamma = std::stod(next(arg));
    } else if (arg == "--epsilon") {
      options.mlvp_config.epsilon = std::stod(next(arg));
    } else if (arg == "--max-iterations") {
      options.mlvp_config.max_iterations =
          static_cast<std::size_t>(std::stoul(next(arg)));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.corpus_root.empty()) {
    throw std::invalid_argument("--corpus-root is required");
  }
  if (options.split != "train" && options.split != "eval") {
    throw std::invalid_argument("--split must be train or eval");
  }
  if (options.population == 0) {
    throw std::invalid_argument("--population must be >= 1");
  }
  if (options.jobs == 0) {
    throw std::invalid_argument("--jobs must be >= 1");
  }
  if (options.out_weights.empty()) {
    std::ostringstream path;
    path << options.corpus_root << "/ws" << options.workspace << "/weights.txt";
    options.out_weights = path.str();
  }

  std::vector<std::string> unique_baselines;
  for (const std::string& name : options.baselines) {
    const std::string lowered = mlvp::Lowercase(name);
    if (lowered == "mlvp") {
      throw std::invalid_argument("Baselines list must not include mlvp");
    }
    if (std::find(unique_baselines.begin(), unique_baselines.end(), lowered) ==
        unique_baselines.end()) {
      unique_baselines.push_back(lowered);
    }
  }
  options.baselines = std::move(unique_baselines);

  if (options.objective != ObjectiveMode::kMakespan && options.baselines.empty()) {
    throw std::invalid_argument("--baselines is required for the selected objective");
  }
  return options;
}

std::string DeriveClassKeyFromFilename(const std::string& filename) {
  const std::string suffix = std::filesystem::path(filename).extension().string();
  const std::string stem = std::filesystem::path(filename).stem().string();
  const std::size_t pos = stem.rfind('_');
  if (pos == std::string::npos || pos + 1 >= stem.size()) {
    return stem;
  }

  const bool numeric_suffix = std::all_of(
      stem.begin() + static_cast<std::ptrdiff_t>(pos + 1), stem.end(),
      [](unsigned char ch) { return std::isdigit(ch) != 0; });
  if (!numeric_suffix) {
    return stem;
  }
  return stem.substr(0, pos) + suffix;
}

std::map<std::string, std::string> LoadManifestClassKeys(const std::filesystem::path& workspace_dir,
                                                         const std::string& split) {
  const std::filesystem::path manifest_path = workspace_dir / "manifest.csv";
  std::map<std::string, std::string> class_keys;
  if (!std::filesystem::exists(manifest_path)) {
    return class_keys;
  }

  std::ifstream input(manifest_path);
  if (!input) {
    throw std::runtime_error("Unable to read manifest: " + manifest_path.string());
  }

  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream row(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(row, col, ',')) {
      cols.push_back(col);
    }
    if (cols.size() != 9) {
      throw std::runtime_error("Malformed manifest row in " + manifest_path.string());
    }
    if (mlvp::Lowercase(cols[0]) != mlvp::Lowercase(split)) {
      continue;
    }

    std::ostringstream key;
    key << "n=" << cols[2]
        << ",fat=" << cols[3]
        << ",density=" << cols[4]
        << ",regularity=" << cols[5]
        << ",jump=" << cols[6]
        << ",ccr=" << cols[7];
    class_keys[cols[1]] = key.str();
  }
  return class_keys;
}

void MaybeSamplePaths(std::vector<std::string>* paths, const std::filesystem::path& workspace_dir,
                      const std::string& split, std::size_t sample_size,
                      SampleMode sample_mode, std::uint32_t seed) {
  if (sample_size == 0 || paths->size() <= sample_size) {
    return;
  }

  std::mt19937 rng(seed);
  if (sample_mode == SampleMode::kRandom) {
    std::shuffle(paths->begin(), paths->end(), rng);
    paths->resize(sample_size);
    std::sort(paths->begin(), paths->end());
    return;
  }

  const std::map<std::string, std::string> manifest_class_keys =
      LoadManifestClassKeys(workspace_dir, split);
  std::map<std::string, std::vector<std::string>> groups;
  for (const std::string& path : *paths) {
    const std::string filename = mlvp::PathFilename(path);
    const auto manifest_it = manifest_class_keys.find(filename);
    const std::string class_key =
        manifest_it != manifest_class_keys.end()
            ? manifest_it->second
            : DeriveClassKeyFromFilename(filename);
    groups[class_key].push_back(path);
  }

  std::vector<std::string> group_keys;
  group_keys.reserve(groups.size());
  for (auto& entry : groups) {
    std::shuffle(entry.second.begin(), entry.second.end(), rng);
    group_keys.push_back(entry.first);
  }
  std::shuffle(group_keys.begin(), group_keys.end(), rng);

  std::map<std::string, std::size_t> cursor_by_group;
  std::vector<std::string> selected;
  selected.reserve(sample_size);
  while (selected.size() < sample_size) {
    bool progress = false;
    for (const std::string& group_key : group_keys) {
      std::size_t& cursor = cursor_by_group[group_key];
      const std::vector<std::string>& group = groups[group_key];
      if (cursor >= group.size()) {
        continue;
      }
      selected.push_back(group[cursor++]);
      progress = true;
      if (selected.size() >= sample_size) {
        break;
      }
    }
    if (!progress) {
      break;
    }
  }

  if (selected.empty()) {
    throw std::runtime_error("Stratified sampling selected no paths");
  }
  std::sort(selected.begin(), selected.end());
  *paths = std::move(selected);
}

template <typename Func>
void ParallelFor(std::size_t count, std::size_t jobs, Func func) {
  if (count == 0) {
    return;
  }
  const std::size_t worker_count = std::max<std::size_t>(1, std::min(jobs, count));
  if (worker_count == 1) {
    for (std::size_t i = 0; i < count; ++i) {
      func(i);
    }
    return;
  }

  std::atomic<std::size_t> next_index{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&]() {
      while (true) {
        const std::size_t index = next_index.fetch_add(1);
        if (index >= count) {
          break;
        }
        func(index);
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
}

Chromosome RandomChromosome(std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  return {dist(rng), dist(rng), dist(rng)};
}

mlvp::MlvpWeights ToWeights(const Chromosome& genes) {
  return mlvp::MlvpWeights{genes[0], genes[1], genes[2]};
}

double SafeRatio(double numerator, double denominator) {
  return numerator / (denominator > 0.0 ? denominator : 1.0);
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

BaselineCorpus PrecomputeBaselines(const Options& options,
                                   const std::vector<mlvp::Dag>& corpus,
                                   const mlvp::Platform& platform) {
  BaselineCorpus baseline_corpus;
  baseline_corpus.names = options.baselines;
  baseline_corpus.makespans.resize(options.baselines.size());

  for (std::size_t baseline_index = 0; baseline_index < options.baselines.size();
       ++baseline_index) {
    const std::string& baseline_name = options.baselines[baseline_index];
    std::unique_ptr<mlvp::SchedulingPolicy> policy =
        mlvp::MakePolicy(baseline_name, options.mlvp_config);
    std::vector<double>& makespans = baseline_corpus.makespans[baseline_index];
    makespans.reserve(corpus.size());

    for (const mlvp::Dag& dag : corpus) {
      mlvp::OnlineSimulator simulator(dag, platform);
      const mlvp::SimulationResult result = simulator.Run(*policy);
      makespans.push_back(result.makespan);
    }

    std::cout << "baseline " << baseline_name
              << " mean_makespan=" << Mean(makespans) << '\n';
  }

  return baseline_corpus;
}

EvaluationMetrics EvaluateChromosome(const Chromosome& genes,
                                     const std::vector<mlvp::Dag>& corpus,
                                     const mlvp::Platform& platform,
                                     const BaselineCorpus& baselines,
                                     const mlvp::MlvpConfig& base_config,
                                     ObjectiveMode objective,
                                     std::uint32_t seed_offset) {
  EvaluationMetrics metrics;
  metrics.has_baselines = !baselines.names.empty();

  double total_makespan = 0.0;
  double total_ratio = 0.0;
  double total_best_baseline_ratio = 0.0;

  for (std::size_t i = 0; i < corpus.size(); ++i) {
    mlvp::MlvpConfig config = base_config;
    config.weights = ToWeights(genes);
    config.seed = seed_offset + static_cast<std::uint32_t>(i);
    mlvp::MlvpPolicy policy(config);
    mlvp::OnlineSimulator simulator(corpus[i], platform);
    const mlvp::SimulationResult result = simulator.Run(policy);
    total_makespan += result.makespan;

    if (metrics.has_baselines) {
      double best_baseline = std::numeric_limits<double>::infinity();
      for (std::size_t baseline_index = 0; baseline_index < baselines.names.size();
           ++baseline_index) {
        const double baseline_makespan = baselines.makespans[baseline_index][i];
        total_ratio += SafeRatio(result.makespan, baseline_makespan);
        best_baseline = std::min(best_baseline, baseline_makespan);
      }
      total_best_baseline_ratio += SafeRatio(result.makespan, best_baseline);
    }
  }

  metrics.mean_makespan = total_makespan / static_cast<double>(corpus.size());
  if (metrics.has_baselines) {
    metrics.mean_ratio = total_ratio /
                         static_cast<double>(corpus.size() * baselines.names.size());
    metrics.best_baseline_ratio =
        total_best_baseline_ratio / static_cast<double>(corpus.size());
  }

  switch (objective) {
    case ObjectiveMode::kMakespan:
      metrics.objective = metrics.mean_makespan;
      break;
    case ObjectiveMode::kMeanRatio:
      metrics.objective = metrics.mean_ratio;
      break;
    case ObjectiveMode::kBestBaselineRatio:
      metrics.objective = metrics.best_baseline_ratio;
      break;
  }
  return metrics;
}

void EvaluatePopulation(std::vector<Individual>* population,
                        const std::vector<Chromosome>& genes,
                        const std::vector<std::uint32_t>& seed_offsets,
                        const std::vector<mlvp::Dag>& corpus,
                        const mlvp::Platform& platform,
                        const BaselineCorpus& baselines,
                        const mlvp::MlvpConfig& base_config,
                        ObjectiveMode objective,
                        std::size_t jobs) {
  if (genes.size() != seed_offsets.size()) {
    throw std::invalid_argument("genes and seed_offsets must have the same size");
  }
  population->resize(genes.size());
  ParallelFor(genes.size(), jobs, [&](std::size_t index) {
    (*population)[index].genes = genes[index];
    (*population)[index].metrics = EvaluateChromosome(
        genes[index], corpus, platform, baselines, base_config, objective,
        seed_offsets[index]);
  });
}

std::size_t TournamentSelect(const std::vector<Individual>& population,
                             std::mt19937& rng) {
  std::uniform_int_distribution<std::size_t> dist(0, population.size() - 1);
  std::size_t best = dist(rng);
  for (int k = 0; k < 2; ++k) {
    const std::size_t candidate = dist(rng);
    if (population[candidate].metrics.objective < population[best].metrics.objective) {
      best = candidate;
    }
  }
  return best;
}

Chromosome Crossover(const Chromosome& lhs, const Chromosome& rhs, std::mt19937& rng) {
  std::uniform_real_distribution<double> blend(0.0, 1.0);
  Chromosome child{};
  for (std::size_t i = 0; i < child.size(); ++i) {
    const double t = blend(rng);
    child[i] = t * lhs[i] + (1.0 - t) * rhs[i];
  }
  return child;
}

void Mutate(Chromosome* genes, std::mt19937& rng, double prob, double sigma) {
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, sigma);
  for (double& gene : *genes) {
    if (coin(rng) < prob) {
      gene += noise(rng);
      gene = std::clamp(gene, -10.0, 10.0);
    }
  }
}

void PrintMetrics(std::size_t generation, const Individual& individual) {
  std::cout << "gen " << generation
            << " mean_makespan=" << individual.metrics.mean_makespan;
  if (individual.metrics.has_baselines) {
    std::cout << " mean_ratio=" << individual.metrics.mean_ratio
              << " best_baseline_ratio=" << individual.metrics.best_baseline_ratio
              << " best_baseline_improvement_pct="
              << (1.0 - individual.metrics.best_baseline_ratio) * 100.0;
  }
  std::cout << " selected_objective=" << individual.metrics.objective
            << " weights=(" << individual.genes[0] << ", " << individual.genes[1]
            << ", " << individual.genes[2] << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);

    const std::filesystem::path workspace_dir =
        mlvp::WorkspaceDir(options.corpus_root, options.workspace);
    std::vector<std::string> corpus_paths =
        mlvp::CollectWorkspaceDots(options.corpus_root, options.workspace, options.split);
    MaybeSamplePaths(&corpus_paths, workspace_dir, options.split, options.sample_size,
                     options.sample_mode,
                     options.seed + static_cast<std::uint32_t>(options.workspace * 100));

    std::vector<mlvp::Dag> corpus;
    for (const std::string& path : corpus_paths) {
      corpus.push_back(mlvp::ParseDotFile(path));
    }
    for (std::size_t i = 0; i < corpus.size(); ++i) {
      if (mlvp::HasUnknownNodeTypes(corpus[i])) {
        mlvp::AssignNodeTypes(&corpus[i], options.type_strategy,
                              options.seed + static_cast<std::uint32_t>(i));
      }
    }

    const std::filesystem::path platform_path =
        mlvp::WorkspaceDir(options.corpus_root, options.workspace) / "platform.csv";
    const mlvp::Platform platform =
        std::filesystem::exists(platform_path)
            ? mlvp::LoadPlatformCsv(platform_path.string())
            : mlvp::MakeMlvpWorkspace(options.workspace,
                                      options.seed + static_cast<std::uint32_t>(options.workspace));

    std::cout << "objective=" << ObjectiveModeName(options.objective);
    std::cout << " sample_mode=" << SampleModeName(options.sample_mode);
    std::cout << " sample_instances=" << corpus.size();
    std::cout << " jobs=" << options.jobs;
    if (options.objective != ObjectiveMode::kMakespan && !options.baselines.empty()) {
      std::cout << " baselines=";
      for (std::size_t i = 0; i < options.baselines.size(); ++i) {
        if (i > 0) {
          std::cout << ',';
        }
        std::cout << options.baselines[i];
      }
    }
    std::cout << '\n';

    const BaselineCorpus baselines =
        options.objective == ObjectiveMode::kMakespan
            ? BaselineCorpus{}
            : PrecomputeBaselines(options, corpus, platform);

    std::mt19937 rng(options.seed);
    std::vector<Chromosome> initial_genes(options.population);
    std::vector<std::uint32_t> initial_seed_offsets(options.population,
                                                    options.seed + 1000U);
    for (Chromosome& genes : initial_genes) {
      genes = RandomChromosome(rng);
    }
    std::vector<Individual> population;
    std::vector<EvaluationMetrics> history;

    EvaluatePopulation(&population, initial_genes, initial_seed_offsets, corpus, platform,
                       baselines, options.mlvp_config, options.objective, options.jobs);

    auto best_it = std::min_element(population.begin(), population.end(),
                                    [](const Individual& lhs, const Individual& rhs) {
                                      return lhs.metrics.objective < rhs.metrics.objective;
                                    });
    Individual best = *best_it;
    history.push_back(best.metrics);
    PrintMetrics(0, best);

    for (std::size_t gen = 1; gen <= options.generations; ++gen) {
      std::vector<Chromosome> next_genes;
      std::vector<std::uint32_t> next_seed_offsets;
      next_genes.reserve(population.size());
      next_seed_offsets.reserve(population.size());
      next_genes.push_back(best.genes);
      next_seed_offsets.push_back(0);

      while (next_genes.size() < population.size()) {
        const Individual& parent_a = population[TournamentSelect(population, rng)];
        const Individual& parent_b = population[TournamentSelect(population, rng)];
        Chromosome child = Crossover(parent_a.genes, parent_b.genes, rng);
        Mutate(&child, rng, options.mutation_prob, options.mutation_sigma);
        next_genes.push_back(child);
        next_seed_offsets.push_back(
            options.seed + static_cast<std::uint32_t>(gen * 1000 + next_genes.size()));
      }

      std::vector<Individual> next_population;
      next_population.push_back(best);
      if (next_genes.size() > 1) {
        std::vector<Chromosome> child_genes(next_genes.begin() + 1, next_genes.end());
        std::vector<std::uint32_t> child_seed_offsets(next_seed_offsets.begin() + 1,
                                                      next_seed_offsets.end());
        std::vector<Individual> child_population;
        EvaluatePopulation(&child_population, child_genes, child_seed_offsets, corpus,
                           platform, baselines, options.mlvp_config, options.objective,
                           options.jobs);
        next_population.insert(next_population.end(),
                               std::make_move_iterator(child_population.begin()),
                               std::make_move_iterator(child_population.end()));
      }
      population = std::move(next_population);
      best_it = std::min_element(population.begin(), population.end(),
                                 [](const Individual& lhs, const Individual& rhs) {
                                   return lhs.metrics.objective < rhs.metrics.objective;
                                 });
      if (best_it->metrics.objective < best.metrics.objective) {
        best = *best_it;
      }
      history.push_back(best.metrics);
      PrintMetrics(gen, best);
    }

    mlvp::EnsureParentDirectory(options.out_weights);
    mlvp::SaveMlvpWeightsFile(ToWeights(best.genes), options.out_weights);
    std::cout << "saved weights -> " << options.out_weights << '\n';

    if (!options.history_csv.empty()) {
      mlvp::EnsureParentDirectory(options.history_csv);
      std::ofstream history_out(options.history_csv);
      history_out << "generation,best_objective,best_mean_makespan,mean_ratio,"
                     "best_baseline_ratio,best_baseline_improvement_pct\n";
      for (std::size_t gen = 0; gen < history.size(); ++gen) {
        history_out << gen << ','
                    << history[gen].objective << ','
                    << history[gen].mean_makespan << ',';
        if (history[gen].has_baselines) {
          history_out << history[gen].mean_ratio << ','
                      << history[gen].best_baseline_ratio << ','
                      << (1.0 - history[gen].best_baseline_ratio) * 100.0;
        } else {
          history_out << ",,";
        }
        history_out << '\n';
      }
      std::cout << "saved history -> " << options.history_csv << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
