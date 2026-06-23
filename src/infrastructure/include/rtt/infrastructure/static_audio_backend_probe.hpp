#pragma once

#include "rtt/domain/ports.hpp"

namespace rtt::infrastructure {

class StaticAudioBackendProbe : public domain::IAudioBackendProbe {
 public:
  StaticAudioBackendProbe(
      bool can_enumerate_devices,
      bool can_open_input_stream,
      std::string backend_name);

  bool can_enumerate_devices() const override;
  bool can_open_input_stream() const override;
  std::string backend_name() const override;

 private:
  bool can_enumerate_devices_;
  bool can_open_input_stream_;
  std::string backend_name_;
};

}  // namespace rtt::infrastructure
