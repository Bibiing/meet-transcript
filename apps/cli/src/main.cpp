#include <memory>

#include "rtt/application/feasibility_assessor.hpp"
#include "rtt/application/realtime_transcriber.hpp"
#include "rtt/infrastructure/console_logger.hpp"
#include "rtt/infrastructure/console_transcript_sink.hpp"
#include "rtt/infrastructure/libsoundio_backend_probe.hpp"
#include "rtt/infrastructure/null_audio_source.hpp"
#include "rtt/infrastructure/stub_transcript_engine.hpp"
#include "rtt/infrastructure/system_platform_info_provider.hpp"

int main() {
  auto logger = std::make_shared<rtt::infrastructure::ConsoleLogger>();
  auto platform_info = std::make_shared<rtt::infrastructure::SystemPlatformInfoProvider>();
  auto backend_probe = std::make_shared<rtt::infrastructure::LibsoundioBackendProbe>();

  rtt::application::FeasibilityAssessor assessor(platform_info, backend_probe);
  const auto report = assessor.assess();
  logger->info("phase 1 feasibility assessment complete");
  logger->info(report.mvp_scope);
  logger->info(report.risk_summary);

  auto audio_source = std::make_shared<rtt::infrastructure::NullAudioSource>();
  auto transcript_engine = std::make_shared<rtt::infrastructure::StubTranscriptEngine>();
  auto transcript_sink = std::make_shared<rtt::infrastructure::ConsoleTranscriptSink>();

  rtt::application::RealtimeTranscriber transcriber(
      {},
      audio_source,
      transcript_engine,
      transcript_sink,
      logger);

  transcriber.start();
  transcriber.stop();

  return 0;
}
