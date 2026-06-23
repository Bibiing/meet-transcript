#pragma once

#include "rtt/domain/ports.hpp"

namespace rtt::infrastructure {

class LibsoundioBackendProbe : public domain::IAudioBackendProbe {
 public:
  bool can_enumerate_devices() const override;
  bool can_open_input_stream() const override;
  std::string backend_name() const override;
};

}  // namespace rtt::infrastructure
