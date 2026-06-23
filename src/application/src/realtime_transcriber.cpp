#include "rtt/application/realtime_transcriber.hpp"

#include <numeric>
#include <stdexcept>

namespace rtt::application {

RealtimeTranscriber::RealtimeTranscriber(
    TranscriberConfig config,
    std::shared_ptr<domain::IAudioSource> audio_source,
    std::shared_ptr<domain::ITranscriptEngine> transcript_engine,
    std::shared_ptr<domain::ITranscriptSink> transcript_sink,
    std::shared_ptr<domain::ILogger> logger)
    : config_(config),
      audio_source_(std::move(audio_source)),
      transcript_engine_(std::move(transcript_engine)),
      transcript_sink_(std::move(transcript_sink)),
      logger_(std::move(logger)) {
  if (!audio_source_ || !transcript_engine_ || !transcript_sink_ || !logger_) {
    throw std::invalid_argument("RealtimeTranscriber requires all dependencies");
  }
}

void RealtimeTranscriber::start() {
  if (running_) {
    logger_->warn("transcriber already running");
    return;
  }

  buffered_frames_.clear();
  running_ = true;
  logger_->info("starting realtime transcriber");
  audio_source_->start(*this);
}

void RealtimeTranscriber::stop() {
  if (!running_) {
    return;
  }

  audio_source_->stop();
  running_ = false;
  logger_->info("stopping realtime transcriber");
}

void RealtimeTranscriber::on_audio_frame(const domain::AudioFrame& frame) {
  buffered_frames_.push_back(frame);
  maybe_emit_chunk();
}

void RealtimeTranscriber::maybe_emit_chunk() {
  if (buffered_frames_.empty()) {
    return;
  }

  const auto start = buffered_frames_.front().timestamp;
  const auto end = buffered_frames_.back().timestamp;
  if ((end - start) < config_.window) {
    return;
  }

  domain::AudioChunk chunk;
  chunk.start_time = start;
  chunk.end_time = end;
  chunk.format = buffered_frames_.front().format;

  for (const auto& frame : buffered_frames_) {
    chunk.samples.insert(chunk.samples.end(), frame.samples.begin(), frame.samples.end());
  }

  auto segment = transcript_engine_->transcribe(chunk);
  transcript_sink_->publish(segment);

  const auto hop_cutoff = end - config_.overlap;
  auto erase_until = buffered_frames_.begin();
  while (erase_until != buffered_frames_.end() && erase_until->timestamp < hop_cutoff) {
    ++erase_until;
  }
  buffered_frames_.erase(buffered_frames_.begin(), erase_until);
}

}  // namespace rtt::application
