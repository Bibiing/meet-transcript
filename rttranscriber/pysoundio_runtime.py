from __future__ import annotations

from pathlib import Path
from typing import Final

from cffi import FFI


_DLL_PATH: Final = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "pysoundio"
    / "pysoundio"
    / "libraries"
    / "win64"
    / "libsoundio.dll"
)

_CDEF = """
typedef _Bool bool;
enum SoundIoError {
    SoundIoErrorNone,
    SoundIoErrorNoMem,
    SoundIoErrorInitAudioBackend,
    SoundIoErrorSystemResources,
    SoundIoErrorOpeningDevice,
    SoundIoErrorNoSuchDevice,
    SoundIoErrorInvalid,
    SoundIoErrorBackendUnavailable,
    SoundIoErrorStreaming,
    SoundIoErrorIncompatibleDevice,
    SoundIoErrorNoSuchClient,
    SoundIoErrorIncompatibleBackend,
    SoundIoErrorBackendDisconnected,
    SoundIoErrorInterrupted,
    SoundIoErrorUnderflow,
    SoundIoErrorEncodingString,
};
enum SoundIoBackend {
    SoundIoBackendNone,
    SoundIoBackendJack,
    SoundIoBackendPulseAudio,
    SoundIoBackendAlsa,
    SoundIoBackendCoreAudio,
    SoundIoBackendWasapi,
    SoundIoBackendDummy,
};
enum SoundIoFormat {
    SoundIoFormatInvalid,
    SoundIoFormatS8,
    SoundIoFormatU8,
    SoundIoFormatS16LE,
    SoundIoFormatS16BE,
    SoundIoFormatU16LE,
    SoundIoFormatU16BE,
    SoundIoFormatS24LE,
    SoundIoFormatS24BE,
    SoundIoFormatU24LE,
    SoundIoFormatU24BE,
    SoundIoFormatS32LE,
    SoundIoFormatS32BE,
    SoundIoFormatU32LE,
    SoundIoFormatU32BE,
    SoundIoFormatFloat32LE,
    SoundIoFormatFloat32BE,
    SoundIoFormatFloat64LE,
    SoundIoFormatFloat64BE,
};
struct SoundIoChannelLayout {
    const char *name;
    int channel_count;
    int channels[24];
};
struct SoundIoSampleRateRange {
    int min;
    int max;
};
struct SoundIoChannelArea {
    char *ptr;
    int step;
};
struct SoundIo {
    void *userdata;
    void (*on_devices_change)(struct SoundIo *);
    void (*on_backend_disconnect)(struct SoundIo *, int err);
    void (*on_events_signal)(struct SoundIo *);
    enum SoundIoBackend current_backend;
    const char *app_name;
    void (*emit_rtprio_warning)(void);
    void (*jack_info_callback)(const char *msg);
    void (*jack_error_callback)(const char *msg);
};
struct SoundIoDevice {
    struct SoundIo *soundio;
    char *id;
    char *name;
    int aim;
    struct SoundIoChannelLayout *layouts;
    int layout_count;
    struct SoundIoChannelLayout current_layout;
    enum SoundIoFormat *formats;
    int format_count;
    enum SoundIoFormat current_format;
    struct SoundIoSampleRateRange *sample_rates;
    int sample_rate_count;
    int sample_rate_current;
    double software_latency_min;
    double software_latency_max;
    double software_latency_current;
    bool is_raw;
    int ref_count;
    int probe_error;
};
struct SoundIoInStream {
    struct SoundIoDevice *device;
    enum SoundIoFormat format;
    int sample_rate;
    struct SoundIoChannelLayout layout;
    double software_latency;
    void *userdata;
    void (*read_callback)(struct SoundIoInStream *, int, int);
    void (*overflow_callback)(struct SoundIoInStream *);
    void (*error_callback)(struct SoundIoInStream *, int);
    const char *name;
    bool non_terminal_hint;
    int bytes_per_frame;
    int bytes_per_sample;
    int layout_error;
};
struct SoundIoRingBuffer;
struct SoundIo *soundio_create(void);
void soundio_destroy(struct SoundIo *soundio);
int soundio_connect(struct SoundIo *soundio);
int soundio_connect_backend(struct SoundIo *soundio, enum SoundIoBackend backend);
void soundio_disconnect(struct SoundIo *soundio);
void soundio_flush_events(struct SoundIo *soundio);
void soundio_wait_events(struct SoundIo *soundio);
void soundio_wakeup(struct SoundIo *soundio);
int soundio_input_device_count(struct SoundIo *soundio);
int soundio_output_device_count(struct SoundIo *soundio);
struct SoundIoDevice *soundio_get_input_device(struct SoundIo *soundio, int index);
int soundio_default_input_device_index(struct SoundIo *soundio);
int soundio_default_output_device_index(struct SoundIo *soundio);
void soundio_device_unref(struct SoundIoDevice *device);
void soundio_device_sort_channel_layouts(struct SoundIoDevice *device);
bool soundio_device_supports_format(struct SoundIoDevice *device, enum SoundIoFormat format);
bool soundio_device_supports_sample_rate(struct SoundIoDevice *device, int sample_rate);
int soundio_device_nearest_sample_rate(struct SoundIoDevice *device, int sample_rate);
const struct SoundIoChannelLayout *soundio_channel_layout_get_default(int channel_count);
int soundio_get_bytes_per_sample(enum SoundIoFormat format);
const char *soundio_strerror(int err);
const char *soundio_version_string(void);
struct SoundIoInStream *soundio_instream_create(struct SoundIoDevice *device);
void soundio_instream_destroy(struct SoundIoInStream *instream);
int soundio_instream_open(struct SoundIoInStream *instream);
int soundio_instream_start(struct SoundIoInStream *instream);
int soundio_instream_begin_read(struct SoundIoInStream *instream, struct SoundIoChannelArea **areas, int *frame_count);
int soundio_instream_end_read(struct SoundIoInStream *instream);
struct SoundIoRingBuffer *soundio_ring_buffer_create(struct SoundIo *soundio, int requested_capacity);
void soundio_ring_buffer_destroy(struct SoundIoRingBuffer *ring_buffer);
char *soundio_ring_buffer_write_ptr(struct SoundIoRingBuffer *ring_buffer);
void soundio_ring_buffer_advance_write_ptr(struct SoundIoRingBuffer *ring_buffer, int count);
char *soundio_ring_buffer_read_ptr(struct SoundIoRingBuffer *ring_buffer);
void soundio_ring_buffer_advance_read_ptr(struct SoundIoRingBuffer *ring_buffer, int count);
int soundio_ring_buffer_fill_count(struct SoundIoRingBuffer *ring_buffer);
int soundio_ring_buffer_free_count(struct SoundIoRingBuffer *ring_buffer);
"""

ffi = FFI()
ffi.cdef(_CDEF)
lib = ffi.dlopen(str(_DLL_PATH))
