#include <cassert>
#include <memory>

#include "rtt/application/feasibility_assessor.hpp"
#include "rtt/domain/models.hpp"
#include "rtt/domain/ports.hpp"
#include "rtt/infrastructure/static_audio_backend_probe.hpp"

// test untuk FeasibilityAssessor, StaticAudioBackendProbe, dan IPlatformInfoProvider
namespace {
  class FakePlatformInfoProvider : public rtt::domain::IPlatformInfoProvider {
  public:
    explicit FakePlatformInfoProvider(rtt::domain::PlatformKind platform) : platform_(platform) {}

    rtt::domain::PlatformKind current_platform() const override {
      return platform_;
    }

  private:
    rtt::domain::PlatformKind platform_;
  };
}  // namespace

int main() {
  {
    auto platform = std::make_shared<FakePlatformInfoProvider>(rtt::domain::PlatformKind::windows);
    auto backend = std::make_shared<rtt::infrastructure::StaticAudioBackendProbe>(true, true, "libsoundio");
    rtt::application::FeasibilityAssessor assessor(platform, backend);

    const auto report = assessor.assess();
    assert(report.status == rtt::domain::FeasibilityStatus::ready);
    assert(report.capabilities.system_output_capture_supported);
    assert(!report.capabilities.requires_virtual_audio_device);
  }

  {
    auto platform = std::make_shared<FakePlatformInfoProvider>(rtt::domain::PlatformKind::macos);
    auto backend = std::make_shared<rtt::infrastructure::StaticAudioBackendProbe>(true, true, "libsoundio");
    rtt::application::FeasibilityAssessor assessor(platform, backend);

    const auto report = assessor.assess();
    assert(report.status == rtt::domain::FeasibilityStatus::partial);
    assert(!report.capabilities.system_output_capture_supported);
    assert(report.capabilities.requires_virtual_audio_device);
  }

  {
    auto platform = std::make_shared<FakePlatformInfoProvider>(rtt::domain::PlatformKind::windows);
    auto backend = std::make_shared<rtt::infrastructure::StaticAudioBackendProbe>(false, true, "libsoundio");
    rtt::application::FeasibilityAssessor assessor(platform, backend);

    const auto report = assessor.assess();
    assert(report.status == rtt::domain::FeasibilityStatus::blocked);
  }

  return 0;
}
