#pragma once

#include "rtt/domain/ports.hpp"

namespace rtt::infrastructure {

class ConsoleTranscriptSink : public domain::ITranscriptSink {
 public:
  void publish(const domain::TranscriptSegment& segment) override;
};

}  // namespace rtt::infrastructure
