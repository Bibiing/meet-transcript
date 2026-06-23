#include "rtt/infrastructure/console_transcript_sink.hpp"

#include <iostream>

namespace rtt::infrastructure {

void ConsoleTranscriptSink::publish(const domain::TranscriptSegment& segment) {
  const char* kind = segment.kind == domain::TranscriptKind::final ? "final" : "partial";
  std::cout << "[" << kind << "] " << segment.text << '\n';
}

}  // namespace rtt::infrastructure
