add_library(rtt_application
  src/feasibility_assessor.cpp
  src/realtime_transcriber.cpp
)

target_include_directories(rtt_application
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rtt_application
  PUBLIC
    rtt_domain
)
