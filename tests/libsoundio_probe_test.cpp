#include <memory>

#include "rtt/infrastructure/libsoundio_backend_probe.hpp"

int main() {
  rtt::infrastructure::LibsoundioBackendProbe probe;
  (void)probe.backend_name();
  (void)probe.can_enumerate_devices();
  (void)probe.can_open_input_stream();
  return 0;
}
