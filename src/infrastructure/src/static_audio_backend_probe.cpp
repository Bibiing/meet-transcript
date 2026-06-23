#include "rtt/infrastructure/static_audio_backend_probe.hpp"

namespace rtt::infrastructure {

StaticAudioBackendProbe::StaticAudioBackendProbe(
    bool can_enumerate_devices,
    bool can_open_input_stream,
    std::string backend_name)
    : can_enumerate_devices_(can_enumerate_devices),
      can_open_input_stream_(can_open_input_stream),
      backend_name_(std::move(backend_name)) {}

bool StaticAudioBackendProbe::can_enumerate_devices() const {
  return can_enumerate_devices_;
}

bool StaticAudioBackendProbe::can_open_input_stream() const {
  return can_open_input_stream_;
}

std::string StaticAudioBackendProbe::backend_name() const {
  return backend_name_;
}

}  // namespace rtt::infrastructure
