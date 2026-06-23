#pragma once

#include "rtt/domain/ports.hpp"

namespace rtt::infrastructure {

class ConsoleLogger : public domain::ILogger {
 public:
  void info(const std::string& message) override;
  void warn(const std::string& message) override;
  void error(const std::string& message) override;
};

}  // namespace rtt::infrastructure
