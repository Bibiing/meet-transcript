#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rtt/domain/models.hpp"

namespace rtt::domain {

class IAudioFrameSink {
 public:
  virtual ~IAudioFrameSink() = default;
  virtual void on_audio_frame(const AudioFrame& frame) = 0;
};

class IAudioSource {
 public:
  virtual ~IAudioSource() = default;
  virtual std::vector<DeviceInfo> list_devices() const = 0;
  virtual void start(IAudioFrameSink& sink) = 0;
  virtual void stop() = 0;
};

class ITranscriptEngine {
 public:
  virtual ~ITranscriptEngine() = default;
  virtual TranscriptSegment transcribe(const AudioChunk& chunk) = 0;
};

class ITranscriptSink {
 public:
  virtual ~ITranscriptSink() = default;
  virtual void publish(const TranscriptSegment& segment) = 0;
};

class ILogger {
 public:
  virtual ~ILogger() = default;
  virtual void info(const std::string& message) = 0;
  virtual void warn(const std::string& message) = 0;
  virtual void error(const std::string& message) = 0;
};

class IPlatformInfoProvider {
 public:
  virtual ~IPlatformInfoProvider() = default;
  virtual PlatformKind current_platform() const = 0;
};

class IAudioBackendProbe {
 public:
  virtual ~IAudioBackendProbe() = default;
  virtual bool can_enumerate_devices() const = 0;
  virtual bool can_open_input_stream() const = 0;
  virtual std::string backend_name() const = 0;
};

}  // namespace rtt::domain
