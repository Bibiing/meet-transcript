#pragma once

#include "rtt/domain/ports.hpp"

namespace rtt::infrastructure {

class SystemPlatformInfoProvider : public domain::IPlatformInfoProvider {
 public:
  domain::PlatformKind current_platform() const override;
};

}  // namespace rtt::infrastructure
