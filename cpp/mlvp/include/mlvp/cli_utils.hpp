#ifndef MLVP_CLI_UTILS_HPP
#define MLVP_CLI_UTILS_HPP

#include <filesystem>
#include <string>
#include <vector>

namespace mlvp {

std::string Lowercase(std::string value);
std::string PathFilename(const std::string& path);

std::vector<std::string> SplitCsvStrings(const std::string& input);
std::vector<int> SplitCsvInts(const std::string& input);

std::filesystem::path WorkspaceDir(const std::string& root, int workspace);
std::vector<std::string> CollectDotFiles(const std::filesystem::path& dir);
std::vector<std::string> CollectWorkspaceDots(const std::string& root, int workspace,
                                              const std::string& split);

void EnsureParentDirectory(const std::string& path);
void EnsureWritablePath(const std::filesystem::path& path, bool overwrite);

}  // namespace mlvp

#endif  // MLVP_CLI_UTILS_HPP
