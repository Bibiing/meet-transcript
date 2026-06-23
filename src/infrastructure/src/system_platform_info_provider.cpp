#include "rtt/infrastructure/system_platform_info_provider.hpp"

namespace rtt::infrastructure {

domain::PlatformKind SystemPlatformInfoProvider::current_platform() const {
#if defined(_WIN32)
  return domain::PlatformKind::windows;
#elif defined(__APPLE__)
  return domain::PlatformKind::macos;
#elif defined(__linux__)
  return domain::PlatformKind::linux;
#else
  return domain::PlatformKind::unknown;
#endif
}

}  // namespace rtt::infrastructure
