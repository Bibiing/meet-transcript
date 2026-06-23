#include "rtt/application/feasibility_assessor.hpp"

#include <stdexcept>

namespace rtt::application {

namespace {

using domain::FeasibilityReport;
using domain::FeasibilityStatus;
using domain::PlatformAudioCapabilities;
using domain::PlatformKind;

}  // namespace

FeasibilityAssessor::FeasibilityAssessor(
    std::shared_ptr<domain::IPlatformInfoProvider> platform_info,
    std::shared_ptr<domain::IAudioBackendProbe> audio_backend_probe)
    : platform_info_(std::move(platform_info)),
      audio_backend_probe_(std::move(audio_backend_probe)) {
  if (!platform_info_ || !audio_backend_probe_) {
    throw std::invalid_argument("FeasibilityAssessor requires all dependencies");
  }
}

domain::FeasibilityReport FeasibilityAssessor::assess() const {
  const auto platform = platform_info_->current_platform();
  auto capabilities = build_capabilities(platform);
  capabilities.preferred_backend = audio_backend_probe_->backend_name();

  domain::FeasibilityReport report;
  report.capabilities = std::move(capabilities);

  if (!audio_backend_probe_->can_enumerate_devices()) {
    report.status = FeasibilityStatus::blocked;
    report.risk_summary = "audio backend cannot enumerate devices";
    report.mvp_scope = "do not start MVP implementation before backend enumeration works";
    report.validation_notes.push_back("device enumeration probe failed");
    report.next_actions.push_back("integrate libsoundio device enumeration and verify at runtime");
    return report;
  }

  report.validation_notes.push_back("device enumeration probe passed");

  if (!audio_backend_probe_->can_open_input_stream()) {
    report.status = FeasibilityStatus::blocked;
    report.risk_summary = "audio backend cannot open input stream";
    report.mvp_scope = "block realtime capture work until input stream opens successfully";
    report.validation_notes.push_back("input stream probe failed");
    report.next_actions.push_back("validate microphone input stream with libsoundio before pipeline work");
    return report;
  }

  report.validation_notes.push_back("input stream probe passed");

  switch (platform) {
    case PlatformKind::windows:
      report.status = FeasibilityStatus::ready;
      report.mvp_scope = "start with Windows microphone capture, then add WASAPI loopback";
      report.risk_summary = "Windows is the lowest-risk MVP path; system output depends on loopback plumbing";
      report.next_actions.push_back("implement Windows microphone capture first");
      report.next_actions.push_back("validate WASAPI loopback for system output as second step");
      break;
    case PlatformKind::macos:
      report.status = FeasibilityStatus::partial;
      report.mvp_scope = "microphone is feasible, system output requires virtual audio device";
      report.risk_summary = "macOS system output capture is gated by BlackHole or equivalent setup";
      report.next_actions.push_back("validate microphone capture path on macOS");
      report.next_actions.push_back("prepare BlackHole onboarding and permission flow");
      break;
    case PlatformKind::linux:
      report.status = FeasibilityStatus::partial;
      report.mvp_scope = "treat Linux as non-MVP until target backend strategy is fixed";
      report.risk_summary = "Linux capture path is outside current product scope";
      report.next_actions.push_back("defer Linux until Windows and macOS are stable");
      break;
    case PlatformKind::unknown:
    default:
      report.status = FeasibilityStatus::blocked;
      report.mvp_scope = "unsupported platform";
      report.risk_summary = "platform detection did not match a supported target";
      report.next_actions.push_back("verify platform detection wiring");
      break;
  }

  return report;
}

domain::PlatformAudioCapabilities FeasibilityAssessor::build_capabilities(domain::PlatformKind platform) const {
  switch (platform) {
    case PlatformKind::windows:
      return {
          .platform = platform,
          .microphone_capture_supported = true,
          .system_output_capture_supported = true,
          .mixed_capture_supported = true,
          .requires_virtual_audio_device = false,
          .preferred_backend = "libsoundio",
          .system_output_strategy = "WASAPI loopback",
      };
    case PlatformKind::macos:
      return {
          .platform = platform,
          .microphone_capture_supported = true,
          .system_output_capture_supported = false,
          .mixed_capture_supported = false,
          .requires_virtual_audio_device = true,
          .preferred_backend = "libsoundio",
          .system_output_strategy = "BlackHole virtual audio device",
      };
    case PlatformKind::linux:
      return {
          .platform = platform,
          .microphone_capture_supported = true,
          .system_output_capture_supported = false,
          .mixed_capture_supported = false,
          .requires_virtual_audio_device = false,
          .preferred_backend = "libsoundio",
          .system_output_strategy = "backend-specific and out of MVP scope",
      };
    case PlatformKind::unknown:
    default:
      return {
          .platform = platform,
          .microphone_capture_supported = false,
          .system_output_capture_supported = false,
          .mixed_capture_supported = false,
          .requires_virtual_audio_device = false,
          .preferred_backend = "unknown",
          .system_output_strategy = "unknown",
      };
  }
}

}  // namespace rtt::application
