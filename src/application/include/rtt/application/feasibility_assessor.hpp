#pragma once

#include <memory>

#include "rtt/domain/ports.hpp"

namespace rtt::application {

class FeasibilityAssessor {
 public:
  FeasibilityAssessor(
      std::shared_ptr<domain::IPlatformInfoProvider> platform_info,
      std::shared_ptr<domain::IAudioBackendProbe> audio_backend_probe);

  domain::FeasibilityReport assess() const;

 private:
  domain::PlatformAudioCapabilities build_capabilities(domain::PlatformKind platform) const;

  std::shared_ptr<domain::IPlatformInfoProvider> platform_info_;
  std::shared_ptr<domain::IAudioBackendProbe> audio_backend_probe_;
};

}  // namespace rtt::application
