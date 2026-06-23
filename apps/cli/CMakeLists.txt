add_executable(rtt_cli
  src/main.cpp
)

target_link_libraries(rtt_cli
  PRIVATE
    rtt_application
    rtt_infrastructure
)
