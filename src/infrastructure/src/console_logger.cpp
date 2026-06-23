#include "rtt/infrastructure/console_logger.hpp"

#include <iostream>

namespace rtt::infrastructure {

void ConsoleLogger::info(const std::string& message) {
  std::cout << "[info] " << message << '\n';
}

void ConsoleLogger::warn(const std::string& message) {
  std::cout << "[warn] " << message << '\n';
}

void ConsoleLogger::error(const std::string& message) {
  std::cerr << "[error] " << message << '\n';
}

}  // namespace rtt::infrastructure
