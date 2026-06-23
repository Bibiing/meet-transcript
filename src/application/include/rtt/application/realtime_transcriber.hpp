#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "rtt/domain/ports.hpp"

namespace rtt::application {

struct TranscriberConfig {
  std::chrono::seconds window {6};
  std::chrono::seconds hop {2};
  std::chrono::seconds overlap {4};
};

class RealtimeTranscriber : public domain::IAudioFrameSink {
 public:
  RealtimeTranscriber(
      TranscriberConfig config,
      std::shared_ptr<domain::IAudioSource> audio_source,
      std::shared_ptr<domain::ITranscriptEngine> transcript_engine,
      std::shared_ptr<domain::ITranscriptSink> transcript_sink,
      std::shared_ptr<domain::ILogger> logger);

  void start();
  void stop();

  void on_audio_frame(const domain::AudioFrame& frame) override;

 private:
  void maybe_emit_chunk();

  TranscriberConfig config_;
  std::shared_ptr<domain::IAudioSource> audio_source_;
  std::shared_ptr<domain::ITranscriptEngine> transcript_engine_;
  std::shared_ptr<domain::ITranscriptSink> transcript_sink_;
  std::shared_ptr<domain::ILogger> logger_;
  std::vector<domain::AudioFrame> buffered_frames_;
  bool running_ = false;
};

}  // namespace rtt::application
