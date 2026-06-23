#include "rtt/infrastructure/null_audio_source.hpp"

#include <chrono>
#include <thread>

namespace rtt::infrastructure {

std::vector<domain::DeviceInfo> NullAudioSource::list_devices() const {
  return {{
      .id = "null-device",
      .name = "Null Device",
      .target = domain::CaptureTarget::microphone,
      .is_default = true,
  }};
}

void NullAudioSource::start(domain::IAudioFrameSink& sink) {
  running_ = true;

  const auto now = std::chrono::steady_clock::now();
  for (int index = 0; index < 7 && running_; ++index) {
    domain::AudioFrame frame;
    frame.timestamp = now + std::chrono::seconds(index);
    frame.format = {};
    frame.samples.assign(16000, 0);
    sink.on_audio_frame(frame);
  }
}

void NullAudioSource::stop() {
  running_ = false;
}

}  // namespace rtt::infrastructure
