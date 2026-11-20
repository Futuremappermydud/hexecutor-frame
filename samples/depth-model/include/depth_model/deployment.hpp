#pragma once
#include "hexecutor/executor.hpp"
#include <filesystem>
#include <string>

namespace depth_model {
struct Deployment {
    hexecutor::Program program;
    hexecutor::Artifacts artifacts;
};
Deployment load_deployment(const std::filesystem::path &package,
                           const std::string &backend = "prepared-qnn",
                           const std::string &model = "primary");
} // namespace depth_model
