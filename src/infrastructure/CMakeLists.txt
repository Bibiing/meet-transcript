add_library(rtt_infrastructure
  src/console_logger.cpp
  src/libsoundio_backend_probe.cpp
  src/console_transcript_sink.cpp
  src/null_audio_source.cpp
  src/static_audio_backend_probe.cpp
  src/stub_transcript_engine.cpp
  src/system_platform_info_provider.cpp
)

target_include_directories(rtt_infrastructure
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(RTT_ENABLE_LIBSOUNDIO)
  target_include_directories(rtt_infrastructure
    PUBLIC
      ${PROJECT_SOURCE_DIR}/vendor/libsoundio
  )
  target_compile_definitions(rtt_infrastructure PUBLIC SOUNDIO_STATIC_LIBRARY)
endif()

target_link_libraries(rtt_infrastructure
  PUBLIC
    rtt_application
    rtt_domain
)

if(RTT_ENABLE_LIBSOUNDIO)
  target_link_libraries(rtt_infrastructure
    PUBLIC
      libsoundio_static
  )
endif()
