#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace rtt::domain {

enum class CaptureTarget {
  microphone,
  system_output,
  mixed
};

enum class PlatformKind {
  windows,
  macos,
  linux,
  unknown
};

struct AudioFormat {
  std::uint32_t sample_rate = 16000;
  std::uint16_t channels = 1;
  std::uint16_t bits_per_sample = 16;
};

struct DeviceInfo {
  std::string id;
  std::string name;
  CaptureTarget target = CaptureTarget::microphone;
  bool is_default = false;
};

struct AudioFrame {
  std::chrono::steady_clock::time_point timestamp;
  AudioFormat format {};
  std::vector<std::int16_t> samples;
};

struct AudioChunk {
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
  AudioFormat format {};
  std::vector<std::int16_t> samples;
};

enum class TranscriptKind {
  partial,
  final
};

struct TranscriptSegment {
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
  TranscriptKind kind = TranscriptKind::partial;
  std::string text;
};

struct FeasibilityTarget {
  std::chrono::milliseconds partial_latency_target {2000};
  std::chrono::milliseconds final_latency_target_min {3000};
  std::chrono::milliseconds final_latency_target_max {6000};
};

struct PlatformAudioCapabilities {
  PlatformKind platform = PlatformKind::unknown;
  bool microphone_capture_supported = false;
  bool system_output_capture_supported = false;
  bool mixed_capture_supported = false;
  bool requires_virtual_audio_device = false;
  std::string preferred_backend;
  std::string system_output_strategy;
};

enum class FeasibilityStatus {
  ready,
  partial,
  blocked
};

struct FeasibilityReport {
  PlatformAudioCapabilities capabilities {};
  FeasibilityTarget latency_target {};
  FeasibilityStatus status = FeasibilityStatus::blocked;
  std::string mvp_scope;
  std::string risk_summary;
  std::vector<std::string> validation_notes;
  std::vector<std::string> next_actions;
};

}  // namespace rtt::domain
