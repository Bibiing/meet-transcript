#pragma once

#include "rtt/domain/ports.hpp"

namespace rtt::infrastructure {

class StubTranscriptEngine : public domain::ITranscriptEngine {
 public:
  domain::TranscriptSegment transcribe(const domain::AudioChunk& chunk) override;
};

}  // namespace rtt::infrastructure
