#pragma once

#include "rtt/domain/ports.hpp"

namespace rtt::infrastructure {

class NullAudioSource : public domain::IAudioSource {
 public:
  std::vector<domain::DeviceInfo> list_devices() const override;
  void start(domain::IAudioFrameSink& sink) override;
  void stop() override;

 private:
  bool running_ = false;
};

}  // namespace rtt::infrastructure
