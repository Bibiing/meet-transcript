#include "rtt/infrastructure/libsoundio_backend_probe.hpp"

#include <memory>
#include <string>

#include <soundio/soundio.h>

namespace rtt::infrastructure {

namespace {

using SoundIoHandle = std::unique_ptr<SoundIo, decltype(&soundio_destroy)>;
using SoundIoDeviceHandle = std::unique_ptr<SoundIoDevice, decltype(&soundio_device_unref)>;
using SoundIoInStreamHandle = std::unique_ptr<SoundIoInStream, decltype(&soundio_instream_destroy)>;

void noop_read_callback(SoundIoInStream*, int, int) {}

SoundIoHandle connect_soundio() {
  SoundIoHandle soundio(soundio_create(), &soundio_destroy);
  if (!soundio) {
    return SoundIoHandle(nullptr, &soundio_destroy);
  }

  if (soundio_connect(soundio.get()) != SoundIoErrorNone) {
    return SoundIoHandle(nullptr, &soundio_destroy);
  }

  soundio_flush_events(soundio.get());
  return soundio;
}

std::string backend_to_string(SoundIoBackend backend) {
  switch (backend) {
    case SoundIoBackendJack:
      return "JACK";
    case SoundIoBackendPulseAudio:
      return "PulseAudio";
    case SoundIoBackendAlsa:
      return "ALSA";
    case SoundIoBackendCoreAudio:
      return "CoreAudio";
    case SoundIoBackendWasapi:
      return "WASAPI";
    case SoundIoBackendDummy:
      return "Dummy";
    case SoundIoBackendNone:
    default:
      return "None";
  }
}

}  // namespace

bool LibsoundioBackendProbe::can_enumerate_devices() const {
  auto soundio = connect_soundio();
  if (!soundio) {
    return false;
  }

  return soundio_input_device_count(soundio.get()) >= 0 &&
         soundio_output_device_count(soundio.get()) >= 0;
}

bool LibsoundioBackendProbe::can_open_input_stream() const {
  auto soundio = connect_soundio();
  if (!soundio) {
    return false;
  }

  const int device_index = soundio_default_input_device_index(soundio.get());
  if (device_index < 0) {
    return false;
  }

  SoundIoDeviceHandle device(soundio_get_input_device(soundio.get(), device_index), &soundio_device_unref);
  if (!device || device->probe_error != SoundIoErrorNone) {
    return false;
  }

  SoundIoInStreamHandle instream(soundio_instream_create(device.get()), &soundio_instream_destroy);
  if (!instream) {
    return false;
  }

  instream->read_callback = &noop_read_callback;
  return soundio_instream_open(instream.get()) == SoundIoErrorNone;
}

std::string LibsoundioBackendProbe::backend_name() const {
  auto soundio = connect_soundio();
  if (!soundio) {
    return "libsoundio(disconnected)";
  }

  return "libsoundio(" + backend_to_string(soundio->current_backend) + ")";
}

}  // namespace rtt::infrastructure
