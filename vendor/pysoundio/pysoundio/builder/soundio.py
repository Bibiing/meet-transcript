"""
soundio.py

CFFI builder for PySoundIO
"""

import os
import cffi
import subprocess
import platform

build_kwargs = {
    'libraries': [],
    'include_dirs': ['./pysoundio/libraries/include'],
    'library_dirs': [],
}
builder_path = os.path.dirname(os.path.realpath(__file__))
package_path = os.path.abspath(os.path.join(builder_path, os.pardir))

if platform.system() == 'Darwin':
    build_kwargs['libraries'].append('soundio.2.0.0')
    build_kwargs['library_dirs'].append('/usr/local/lib')
    build_kwargs['library_dirs'].append(os.path.join(package_path, 'libraries', 'darwin'))
    build_kwargs['extra_link_args'] = ['-Wl,-rpath,@loader_path/libraries/darwin']

elif platform.system() == 'Linux':
    if os.uname()[4].startswith('arm'):
        cpuinfo = subprocess.check_output(['cat', '/proc/cpuinfo']).decode()
        if 'neon' in cpuinfo:
            architecture = 'rpiv7'
        else:
            architecture = 'rpiv6'
    else:
        architecture = 'linux'

    build_kwargs['libraries'].append('soundio-2.0.0')
    build_kwargs['library_dirs'].append('/usr/local/lib')
    build_kwargs['library_dirs'].append(os.path.join(package_path, 'libraries', architecture))
    build_kwargs['extra_link_args'] = ['-Wl,-rpath,$ORIGIN/libraries/' + architecture]

elif platform.system() == 'Windows':
    build_kwargs['libraries'].append('soundio')
    if platform.architecture()[0] == '32bit':
        build_kwargs['library_dirs'].append(os.path.join(package_path, 'libraries', 'win32'))
    elif platform.architecture()[0] == '64bit':
        build_kwargs['library_dirs'].append(os.path.join(package_path, 'libraries', 'win64'))
    else:
        raise RuntimeError('Windows architecture %s is not supported' % platform.architecture()[0])
else:
    raise RuntimeError('%s platform is not supported' % platform.system())

ffibuilder = cffi.FFI()
ffibuilder.set_source(
    "pysoundio._soundio",
    """
    #include <soundio/soundio.h>
    """,
    **build_kwargs
)
ffibuilder.cdef("""

// ENUMS
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

enum SoundIoChannelId {
    SoundIoChannelIdInvalid,

    SoundIoChannelIdFrontLeft, ///< First of the more commonly supported ids.
    SoundIoChannelIdFrontRight,
    SoundIoChannelIdFrontCenter,
    SoundIoChannelIdLfe,
    SoundIoChannelIdBackLeft,
    SoundIoChannelIdBackRight,
    SoundIoChannelIdFrontLeftCenter,
    SoundIoChannelIdFrontRightCenter,
    SoundIoChannelIdBackCenter,
    SoundIoChannelIdSideLeft,
    SoundIoChannelIdSideRight,
    SoundIoChannelIdTopCenter,
    SoundIoChannelIdTopFrontLeft,
    SoundIoChannelIdTopFrontCenter,
    SoundIoChannelIdTopFrontRight,
    SoundIoChannelIdTopBackLeft,
    SoundIoChannelIdTopBackCenter,
    SoundIoChannelIdTopBackRight, ///< Last of the more commonly supported ids.

    SoundIoChannelIdBackLeftCenter, ///< First of the less commonly supported ids.
    SoundIoChannelIdBackRightCenter,
    SoundIoChannelIdFrontLeftWide,
    SoundIoChannelIdFrontRightWide,
    SoundIoChannelIdFrontLeftHigh,
    SoundIoChannelIdFrontCenterHigh,
    SoundIoChannelIdFrontRightHigh,
    SoundIoChannelIdTopFrontLeftCenter,
    SoundIoChannelIdTopFrontRightCenter,
    SoundIoChannelIdTopSideLeft,
    SoundIoChannelIdTopSideRight,
    SoundIoChannelIdLeftLfe,
    SoundIoChannelIdRightLfe,
    SoundIoChannelIdLfe2,
    SoundIoChannelIdBottomCenter,
    SoundIoChannelIdBottomLeftCenter,
    SoundIoChannelIdBottomRightCenter,

    /// Mid/side recording
    SoundIoChannelIdMsMid,
    SoundIoChannelIdMsSide,

    /// first order ambisonic channels
    SoundIoChannelIdAmbisonicW,
    SoundIoChannelIdAmbisonicX,
    SoundIoChannelIdAmbisonicY,
    SoundIoChannelIdAmbisonicZ,

    /// X-Y Recording
    SoundIoChannelIdXyX,
    SoundIoChannelIdXyY,

    SoundIoChannelIdHeadphonesLeft, ///< First of the "other" channel ids
    SoundIoChannelIdHeadphonesRight,
    SoundIoChannelIdClickTrack,
    SoundIoChannelIdForeignLanguage,
    SoundIoChannelIdHearingImpaired,
    SoundIoChannelIdNarration,
    SoundIoChannelIdHaptic,
    SoundIoChannelIdDialogCentricMix, ///< Last of the "other" channel ids

    SoundIoChannelIdAux,
    SoundIoChannelIdAux0,
    SoundIoChannelIdAux1,
    SoundIoChannelIdAux2,
    SoundIoChannelIdAux3,
    SoundIoChannelIdAux4,
    SoundIoChannelIdAux5,
    SoundIoChannelIdAux6,
    SoundIoChannelIdAux7,
    SoundIoChannelIdAux8,
    SoundIoChannelIdAux9,
    SoundIoChannelIdAux10,
    SoundIoChannelIdAux11,
    SoundIoChannelIdAux12,
    SoundIoChannelIdAux13,
    SoundIoChannelIdAux14,
    SoundIoChannelIdAux15,
};

/// Built-in channel layouts for convenience.
enum SoundIoChannelLayoutId {
    SoundIoChannelLayoutIdMono,
    SoundIoChannelLayoutIdStereo,
    SoundIoChannelLayoutId2Point1,
    SoundIoChannelLayoutId3Point0,
    SoundIoChannelLayoutId3Point0Back,
    SoundIoChannelLayoutId3Point1,
    SoundIoChannelLayoutId4Point0,
    SoundIoChannelLayoutIdQuad,
    SoundIoChannelLayoutIdQuadSide,
    SoundIoChannelLayoutId4Point1,
    SoundIoChannelLayoutId5Point0Back,
    SoundIoChannelLayoutId5Point0Side,
    SoundIoChannelLayoutId5Point1,
    SoundIoChannelLayoutId5Point1Back,
    SoundIoChannelLayoutId6Point0Side,
    SoundIoChannelLayoutId6Point0Front,
    SoundIoChannelLayoutIdHexagonal,
    SoundIoChannelLayoutId6Point1,
    SoundIoChannelLayoutId6Point1Back,
    SoundIoChannelLayoutId6Point1Front,
    SoundIoChannelLayoutId7Point0,
    SoundIoChannelLayoutId7Point0Front,
    SoundIoChannelLayoutId7Point1,
    SoundIoChannelLayoutId7Point1Wide,
    SoundIoChannelLayoutId7Point1WideBack,
    SoundIoChannelLayoutIdOctagonal,
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

enum SoundIoDeviceAim {
    SoundIoDeviceAimInput,  ///< capture / recording
    SoundIoDeviceAimOutput, ///< playback
};

enum SoundIoFormat {
    SoundIoFormatInvalid,
    SoundIoFormatS8,        ///< Signed 8 bit
    SoundIoFormatU8,        ///< Unsigned 8 bit
    SoundIoFormatS16LE,     ///< Signed 16 bit Little Endian
    SoundIoFormatS16BE,     ///< Signed 16 bit Big Endian
    SoundIoFormatU16LE,     ///< Unsigned 16 bit Little Endian
    SoundIoFormatU16BE,     ///< Unsigned 16 bit Big Endian
    SoundIoFormatS24LE,     ///< Signed 24 bit Little Endian using low three bytes in 32-bit word
    SoundIoFormatS24BE,     ///< Signed 24 bit Big Endian using low three bytes in 32-bit word
    SoundIoFormatU24LE,     ///< Unsigned 24 bit Little Endian using low three bytes in 32-bit word
    SoundIoFormatU24BE,     ///< Unsigned 24 bit Big Endian using low three bytes in 32-bit word
    SoundIoFormatS32LE,     ///< Signed 32 bit Little Endian
    SoundIoFormatS32BE,     ///< Signed 32 bit Big Endian
    SoundIoFormatU32LE,     ///< Unsigned 32 bit Little Endian
    SoundIoFormatU32BE,     ///< Unsigned 32 bit Big Endian
    SoundIoFormatFloat32LE, ///< Float 32 bit Little Endian, Range -1.0 to 1.0
    SoundIoFormatFloat32BE, ///< Float 32 bit Big Endian, Range -1.0 to 1.0
    SoundIoFormatFloat64LE, ///< Float 64 bit Little Endian, Range -1.0 to 1.0
    SoundIoFormatFloat64BE, ///< Float 64 bit Big Endian, Range -1.0 to 1.0
};

// STRUCTURES

struct SoundIoChannelLayout {
    const char *name;
    int channel_count;
    enum SoundIoChannelId channels[24];
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

    enum SoundIoDeviceAim aim;
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

struct SoundIoOutStream {
    struct SoundIoDevice *device;
    enum SoundIoFormat format;

    int sample_rate;
    struct SoundIoChannelLayout layout;

    double software_latency;
    float volume;
    void *userdata;
    void (*write_callback)(struct SoundIoOutStream *, int frame_count_min, int frame_count_max);
    void (*underflow_callback)(struct SoundIoOutStream *);
    void (*error_callback)(struct SoundIoOutStream *, int err);

    const char *name;
    bool non_terminal_hint;

    int bytes_per_frame;
    int bytes_per_sample;

    int layout_error;
};

struct SoundIoInStream {
    struct SoundIoDevice *device;
    enum SoundIoFormat format;

    int sample_rate;
    struct SoundIoChannelLayout layout;

    double software_latency;
    void *userdata;
    void (*read_callback)(struct SoundIoInStream *, int frame_count_min, int frame_count_max);
    void (*overflow_callback)(struct SoundIoInStream *);
    void (*error_callback)(struct SoundIoInStream *, int err);

    const char *name;
    bool non_terminal_hint;

    int bytes_per_frame;
    int bytes_per_sample;

    int layout_error;
};

// CALLBACKS
extern "Python" void _write_callback(struct SoundIoOutStream *, int, int);
extern "Python" void _underflow_callback(struct SoundIoOutStream *);
extern "Python" void _output_error_callback(struct SoundIoOutStream *, int);
extern "Python" void _read_callback(struct SoundIoInStream *, int, int);
extern "Python" void _overflow_callback(struct SoundIoInStream *);
extern "Python" void _input_error_callback(struct SoundIoInStream *, int);

// STATIC
const char *soundio_version_string(void);
int soundio_version_major(void);
int soundio_version_minor(void);
int soundio_version_patch(void);

// CONSTRUCTOR / DESTRUCTOR
struct SoundIo *soundio_create(void);
void soundio_destroy(struct SoundIo *soundio);

// CONFIGURATION
int soundio_connect(struct SoundIo *soundio);
int soundio_connect_backend(struct SoundIo *soundio, enum SoundIoBackend backend);
void soundio_disconnect(struct SoundIo *soundio);

const char *soundio_strerror(int error);
const char *soundio_backend_name(enum SoundIoBackend backend);

int soundio_backend_count(struct SoundIo *soundio);
enum SoundIoBackend soundio_get_backend(struct SoundIo *soundio, int index);

bool soundio_have_backend(enum SoundIoBackend backend);

void soundio_flush_events(struct SoundIo *soundio);
void soundio_wait_events(struct SoundIo *soundio);
void soundio_wakeup(struct SoundIo *soundio);

void soundio_force_device_scan(struct SoundIo *soundio);

// CHANNEL LAYOUTS
bool soundio_channel_layout_equal(
    const struct SoundIoChannelLayout *a,
    const struct SoundIoChannelLayout *b);

const char *soundio_get_channel_name(enum SoundIoChannelId id);
enum SoundIoChannelId soundio_parse_channel_id(const char *str, int str_len);

int soundio_channel_layout_builtin_count(void);
const struct SoundIoChannelLayout *soundio_channel_layout_get_builtin(int index);

const struct SoundIoChannelLayout *soundio_channel_layout_get_default(int channel_count);
int soundio_channel_layout_find_channel(
        const struct SoundIoChannelLayout *layout, enum SoundIoChannelId channel);

bool soundio_channel_layout_detect_builtin(struct SoundIoChannelLayout *layout);
const struct SoundIoChannelLayout *soundio_best_matching_channel_layout(
        const struct SoundIoChannelLayout *preferred_layouts, int preferred_layout_count,
        const struct SoundIoChannelLayout *available_layouts, int available_layout_count);

void soundio_sort_channel_layouts(struct SoundIoChannelLayout *layouts, int layout_count);

// SAMPLE FORMATS
int soundio_get_bytes_per_sample(enum SoundIoFormat format);
const char * soundio_format_string(enum SoundIoFormat format);

// DEVICES
int soundio_input_device_count(struct SoundIo *soundio);
int soundio_output_device_count(struct SoundIo *soundio);

struct SoundIoDevice *soundio_get_input_device(struct SoundIo *soundio, int index);
struct SoundIoDevice *soundio_get_output_device(struct SoundIo *soundio, int index);

int soundio_default_input_device_index(struct SoundIo *soundio);

int soundio_default_output_device_index(struct SoundIo *soundio);

void soundio_device_ref(struct SoundIoDevice *device);
void soundio_device_unref(struct SoundIoDevice *device);

bool soundio_device_equal(
        const struct SoundIoDevice *a,
        const struct SoundIoDevice *b);

void soundio_device_sort_channel_layouts(struct SoundIoDevice *device);

bool soundio_device_supports_format(struct SoundIoDevice *device,
        enum SoundIoFormat format);

bool soundio_device_supports_layout(struct SoundIoDevice *device,
        const struct SoundIoChannelLayout *layout);

bool soundio_device_supports_sample_rate(struct SoundIoDevice *device,
        int sample_rate);

int soundio_device_nearest_sample_rate(struct SoundIoDevice *device,
        int sample_rate);

// OUTPUT STREAMS
struct SoundIoOutStream *soundio_outstream_create(struct SoundIoDevice *device);
void soundio_outstream_destroy(struct SoundIoOutStream *outstream);

int soundio_outstream_open(struct SoundIoOutStream *outstream);
int soundio_outstream_start(struct SoundIoOutStream *outstream);

int soundio_outstream_begin_write(struct SoundIoOutStream *outstream,
        struct SoundIoChannelArea **areas, int *frame_count);

int soundio_outstream_end_write(struct SoundIoOutStream *outstream);
int soundio_outstream_clear_buffer(struct SoundIoOutStream *outstream);

int soundio_outstream_pause(struct SoundIoOutStream *outstream, bool pause);

int soundio_outstream_get_latency(struct SoundIoOutStream *outstream,
        double *out_latency);

int soundio_outstream_set_volume(struct SoundIoOutStream *outstream,
        double volume);

// INPUT STREAMS
struct SoundIoInStream *soundio_instream_create(struct SoundIoDevice *device);
void soundio_instream_destroy(struct SoundIoInStream *instream);

int soundio_instream_open(struct SoundIoInStream *instream);

int soundio_instream_start(struct SoundIoInStream *instream);

int soundio_instream_begin_read(struct SoundIoInStream *instream,
        struct SoundIoChannelArea **areas, int *frame_count);
int soundio_instream_end_read(struct SoundIoInStream *instream);

int soundio_instream_pause(struct SoundIoInStream *instream, bool pause);

int soundio_instream_get_latency(struct SoundIoInStream *instream,
        double *out_latency);

// RING BUFFERS
struct SoundIoRingBuffer;

struct SoundIoRingBuffer *soundio_ring_buffer_create(struct SoundIo *soundio, int requested_capacity);
void soundio_ring_buffer_destroy(struct SoundIoRingBuffer *ring_buffer);

int soundio_ring_buffer_capacity(struct SoundIoRingBuffer *ring_buffer);

char *soundio_ring_buffer_write_ptr(struct SoundIoRingBuffer *ring_buffer);
void soundio_ring_buffer_advance_write_ptr(struct SoundIoRingBuffer *ring_buffer, int count);

char *soundio_ring_buffer_read_ptr(struct SoundIoRingBuffer *ring_buffer);
void soundio_ring_buffer_advance_read_ptr(struct SoundIoRingBuffer *ring_buffer, int count);

int soundio_ring_buffer_fill_count(struct SoundIoRingBuffer *ring_buffer);

int soundio_ring_buffer_free_count(struct SoundIoRingBuffer *ring_buffer);

void soundio_ring_buffer_clear(struct SoundIoRingBuffer *ring_buffer);

""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True, debug=False)
