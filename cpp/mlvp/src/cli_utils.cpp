#include "mlvp/cli_utils.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace mlvp {

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string PathFilename(const std::string& path) {
  return std::filesystem::path(path).filename().string();
}

std::vector<std::string> SplitCsvStrings(const std::string& input) {
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

std::filesystem::path WorkspaceDir(const std::string& root, int workspace) {
  return std::filesystem::path(root) / ("ws" + std::to_string(workspace));
}

std::vector<std::string> CollectDotFiles(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir)) {
    throw std::runtime_error("Directory not found: " + dir.string());
  }

  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".dot") {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::vector<std::string> CollectWorkspaceDots(const std::string& root, int workspace,
                                              const std::string& split) {
  const std::filesystem::path workspace_dir = WorkspaceDir(root, workspace);
  std::filesystem::path dots_dir = workspace_dir;
  if (std::filesystem::exists(workspace_dir / "train") ||
      std::filesystem::exists(workspace_dir / "eval")) {
    dots_dir = workspace_dir / split;
  }

  std::vector<std::string> paths = CollectDotFiles(dots_dir);
  if (paths.empty()) {
    throw std::runtime_error("No .dot files found in " + dots_dir.string());
  }
  return paths;
}

void EnsureParentDirectory(const std::string& path) {
  const std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

void EnsureWritablePath(const std::filesystem::path& path, bool overwrite) {
  if (std::filesystem::exists(path) && !overwrite) {
    throw std::runtime_error("Refusing to overwrite existing file without --overwrite: " +
                             path.string());
  }
}

}  // namespace mlvp
