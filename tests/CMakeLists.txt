add_executable(rtt_smoke_test
  smoke_test.cpp
)

target_link_libraries(rtt_smoke_test
  PRIVATE
    rtt_application
    rtt_infrastructure
)

add_test(NAME rtt_smoke_test COMMAND rtt_smoke_test)

add_executable(rtt_feasibility_test
  feasibility_test.cpp
)

target_link_libraries(rtt_feasibility_test
  PRIVATE
    rtt_application
    rtt_infrastructure
)

add_test(NAME rtt_feasibility_test COMMAND rtt_feasibility_test)

add_executable(rtt_libsoundio_probe_test
  libsoundio_probe_test.cpp
)

target_link_libraries(rtt_libsoundio_probe_test
  PRIVATE
    rtt_infrastructure
)

add_test(NAME rtt_libsoundio_probe_test COMMAND rtt_libsoundio_probe_test)
