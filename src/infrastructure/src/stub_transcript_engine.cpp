#include "rtt/infrastructure/stub_transcript_engine.hpp"

namespace rtt::infrastructure {

domain::TranscriptSegment StubTranscriptEngine::transcribe(const domain::AudioChunk& chunk) {
  return {
      .start_time = chunk.start_time,
      .end_time = chunk.end_time,
      .kind = domain::TranscriptKind::partial,
      .text = "stub transcript chunk",
  };
}

}  // namespace rtt::infrastructure
