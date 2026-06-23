#include <memory>

#include "rtt/application/realtime_transcriber.hpp"
#include "rtt/infrastructure/console_logger.hpp"
#include "rtt/infrastructure/console_transcript_sink.hpp"
#include "rtt/infrastructure/null_audio_source.hpp"
#include "rtt/infrastructure/stub_transcript_engine.hpp"

int main() {
  auto logger = std::make_shared<rtt::infrastructure::ConsoleLogger>();
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
