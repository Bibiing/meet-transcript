# -*- coding: utf-8 -*-
"""
pysoundio.py

Play and Record Sound in Python using libsoundio

libsoundio is a C library for cross-platform audio input and output.
It is suitable for real-time and consumer software.

"""
import logging
import threading
import time

from pysoundio._soundio import ffi as _ffi
from pysoundio._soundio import lib as _lib
from pysoundio import constants


class PySoundIoError(Exception):
    pass


class _InputProcessingThread(threading.Thread):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.buffer = parent.input['buffer']
        self.callback = parent.input['read_callback']
        self.bytes_per_frame = parent.input['bytes_per_frame']

        self.daemon = True
        self.running = True
        self.start()

    def run(self):
        """
        When there is data ready in the input buffer,
        pass it to the user callback.
        """
        while self.running:
            fill_bytes = _lib.soundio_ring_buffer_fill_count(self.buffer)
            if fill_bytes > 0:
                read_buf = _lib.soundio_ring_buffer_read_ptr(self.buffer)
                data = bytearray(fill_bytes)
                _ffi.memmove(data, read_buf, fill_bytes)
                if self.callback:
                    self.callback(data=data, length=int(fill_bytes / self.bytes_per_frame))
                _lib.soundio_ring_buffer_advance_read_ptr(self.buffer, fill_bytes)
            time.sleep(0.001)

    def stop(self):
        self.running = False


class _OutputProcessingThread(threading.Thread):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.buffer = parent.output['buffer']
        self.callback = parent.output['write_callback']
        self.bytes_per_frame = parent.output['bytes_per_frame']
        self.sample_rate = parent.output['sample_rate']
        self.block_size = parent.output['block_size']

        self.to_read = 0
        self.running = True
        self.daemon = True
        self.start()

    def run(self):
        """
        Request output data from user callback when there is
        free space in the buffer.
        """
        while self.running:
            if self.to_read > 0:
                data = bytearray(self.block_size * self.bytes_per_frame)
                free_bytes = _lib.soundio_ring_buffer_free_count(self.buffer)
                if free_bytes > len(data):
                    if self.callback:
                        self.callback(data=data, length=self.block_size)

                    write_buf = _lib.soundio_ring_buffer_write_ptr(self.buffer)
                    _ffi.memmove(write_buf, data, len(data))
                    _lib.soundio_ring_buffer_advance_write_ptr(self.buffer, len(data))

                with threading.Lock():
                    self.to_read -= 1
            time.sleep(0.001)

    def stop(self):
        self.running = False


class PySoundIo:

    def __init__(self, backend=None):
        """
        Initialise PySoundIo.
        Connect to a specific backend, or the default.

        Parameters
        ----------
        backend: (SoundIoBackend) see `Backends`_. (optional)
        """
        self.backend = backend

        self.input = {'device': None, 'stream': None, 'buffer': None, 'read_callback': None, 'thread': None}
        self.output = {'device': None, 'stream': None, 'buffer': None, 'write_callback': None, 'thread': None}

        self.logger = logging.getLogger(__name__)

        self._soundio = _lib.soundio_create()
        if not self._soundio:
            raise PySoundIoError('Out of memory')

        if backend:
            self._check(_lib.soundio_connect_backend(self._soundio, backend))
        else:
            self._check(_lib.soundio_connect(self._soundio))

        self._userdata = _ffi.new_handle(self)
        self.flush()

    def close(self):
        """
        Clean up allocated memory
        Close libsoundio connections
        """
        self.logger.info('Closing down threads...')

        if self.input['thread']:
            self.input['thread'].stop()
            while self.input['thread'].is_alive():
                time.sleep(0.001)
        if self.output['thread']:
            self.output['thread'].stop()
            while self.output['thread'].is_alive():
                time.sleep(0.001)

        self.logger.info('Closing down streams...')
        if self.input['stream']:
            _lib.soundio_instream_destroy(self.input['stream'])
            del self.input['stream']
        if self.output['stream']:
            _lib.soundio_outstream_destroy(self.output['stream'])
            del self.output['stream']

        if self.input['buffer']:
            _lib.soundio_ring_buffer_destroy(self.input['buffer'])
            del self.input['buffer']
        if self.output['buffer']:
            _lib.soundio_ring_buffer_destroy(self.output['buffer'])
            del self.output['buffer']

        if self.input['device']:
            _lib.soundio_device_unref(self.input['device'])
            del self.input['device']
        if self.output['device']:
            _lib.soundio_device_unref(self.output['device'])
            del self.output['device']

        if self._soundio:
            _lib.soundio_disconnect(self._soundio)
            _lib.soundio_destroy(self._soundio)
            del self._soundio

    def flush(self):
        """
        Atomically update information for all connected devices.
        """
        _lib.soundio_flush_events(self._soundio)

    @property
    def version(self):
        """
        Returns the current version of libsoundio
        """
        return _ffi.string(_lib.soundio_version_string()).decode()

    def _check(self, code):
        """
        Returns an error message associated with the return code
        """
        if code != _lib.SoundIoErrorNone:
            raise PySoundIoError(_ffi.string(_lib.soundio_strerror(code)).decode())

    @property
    def backend_count(self):
        """
        Returns the number of available backends.
        """
        return _lib.soundio_backend_count(self._soundio)

    def get_default_input_device(self):
        """
        Returns default input device

        Returns
        -------
        PySoundIoDevice input device

        Raises
        ------
        PySoundIoError if the input device is not available
        """
        device_id = _lib.soundio_default_input_device_index(self._soundio)
        return self.get_input_device(device_id)

    def get_input_device(self, device_id):
        """
        Return an input device by index

        Parameters
        ----------
        device_id: (int) input device index

        Returns
        -------
        PySoundIoDevice input device

        Raises
        ------
        PySoundIoError if an invalid device id is used, or device is unavailable
        """
        if device_id < 0 or device_id > _lib.soundio_input_device_count(self._soundio):
            raise PySoundIoError('Invalid input device id')
        self.input['device'] = _lib.soundio_get_input_device(self._soundio, device_id)
        return self.input['device']

    def get_default_output_device(self):
        """
        Returns default output device

        Returns
        -------
        PySoundIoDevice output device

        Raises
        ------
        PySoundIoError if the output device is not available
        """
        device_id = _lib.soundio_default_output_device_index(self._soundio)
        return self.get_output_device(device_id)

    def get_output_device(self, device_id):
        """
        Return an output device by index

        Parameters
        ----------
        device_id: (int) output device index

        Returns
        -------
        PySoundIoDevice output device

        Raises
        ------
        PySoundIoError if an invalid device id is used, or device is unavailable
        """
        if device_id < 0 or device_id > _lib.soundio_output_device_count(self._soundio):
            raise PySoundIoError('Invalid output device id')
        self.output['device'] = _lib.soundio_get_output_device(self._soundio, device_id)
        return self.output['device']

    def list_devices(self):
        """
        Return a list of available devices

        Returns
        -------
        (list)(dict) containing information on available input / output devices.
        """
        output_count = _lib.soundio_output_device_count(self._soundio)
        input_count = _lib.soundio_input_device_count(self._soundio)

        default_output = _lib.soundio_default_output_device_index(self._soundio)
        default_input = _lib.soundio_default_input_device_index(self._soundio)

        input_devices = []
        output_devices = []

        for i in range(0, input_count):
            device = _lib.soundio_get_input_device(self._soundio, i)
            input_devices.append({
                'id': _ffi.string(device.id).decode(),
                'name': _ffi.string(device.name).decode(),
                'is_raw': device.is_raw,
                'is_default': default_input == i,
                'sample_rates': self.get_sample_rates(device),
                'formats': self.get_formats(device),
                'layouts': self.get_layouts(device),
                'software_latency_min': device.software_latency_min,
                'software_latency_max': device.software_latency_max,
                'software_latency_current': device.software_latency_current,
                'probe_error': PySoundIoError(
                    _ffi.string(_lib.soundio_strerror(device.probe_error)).decode()
                    if device.probe_error else None)
            })
            _lib.soundio_device_unref(device)

        for i in range(0, output_count):
            device = _lib.soundio_get_output_device(self._soundio, i)
            output_devices.append({
                'id': _ffi.string(device.id).decode(),
                'name': _ffi.string(device.name).decode(),
                'is_raw': device.is_raw,
                'is_default': default_output == i,
                'sample_rates': self.get_sample_rates(device),
                'formats': self.get_formats(device),
                'layouts': self.get_layouts(device),
                'software_latency_min': device.software_latency_min,
                'software_latency_max': device.software_latency_max,
                'software_latency_current': device.software_latency_current,
                'probe_error': PySoundIoError(
                    _ffi.string(_lib.soundio_strerror(device.probe_error)).decode()
                    if device.probe_error else None)
            })
            _lib.soundio_device_unref(device)

        self.logger.info('%d devices found' % (input_count + output_count))
        return (input_devices, output_devices)

    def get_layouts(self, device):
        """
        Return a list of available layouts for a device

        Parameters
        ----------
        device: (SoundIoDevice) device object

        Returns
        -------
        (dict) Dictionary of available channel layouts for a device
        """
        current = device.current_layout
        layouts = {
            'current': {
                'name': _ffi.string(current.name).decode() if current.name else 'None'
            },
            'available': []
        }
        for idx in range(0, device.layout_count):
            layouts['available'].append({
                'name': (_ffi.string(device.layouts[idx].name).decode() if
                         device.layouts[idx].name else 'None'),
                'channel_count': device.layouts[idx].channel_count
            })
        return layouts

    def get_sample_rates(self, device):
        """
        Return a list of available sample rates for a device

        Parameters
        ----------
        device: (SoundIoDevice) device object

        Returns
        -------
        (dict) Dictionary of available sample rates for a device
        """
        sample_rates = {'current': device.sample_rate_current, 'available': []}
        for s in range(0, device.sample_rate_count):
            sample_rates['available'].append({
                'min': device.sample_rates[s].min,
                'max': device.sample_rates[s].max
            })
        return sample_rates

    def get_formats(self, device):
        """
        Return a list of available formats for a device

        Parameters
        ----------
        device: (SoundIoDevice) device object

        Returns
        -------
        (dict) Dictionary of available formats for a device
        """
        formats = {'current': device.current_format, 'available': []}
        for r in range(0, device.format_count):
            formats['available'].append(constants.FORMATS[device.formats[r]])
        return formats

    def supports_sample_rate(self, device, rate):
        """
        Check the sample rate is supported by the selected device.

        Parameters
        ----------
        device: (SoundIoDevice) device object
        rate (int): sample rate

        Returns
        -------
        (bool) True if sample rate is supported for this device
        """
        return bool(_lib.soundio_device_supports_sample_rate(device, rate))

    def get_default_sample_rate(self, device):
        """
        Get the best sample rate.

        Parameters
        ----------
        device: (SoundIoDevice) device object

        Returns
        -------
        (int) The best available sample rate
        """
        sample_rate = None
        for rate in constants.PRIORITISED_SAMPLE_RATES:
            if self.supports_sample_rate(device, rate):
                sample_rate = rate
                break
        if not sample_rate:
            sample_rate = device.sample_rates.max
        return sample_rate

    def supports_format(self, device, format):
        """
        Check the format is supported by the selected device.

        Parameters
        ----------
        device: (SoundIoDevice) device object
        format: (SoundIoFormat) see `Formats`_.

        Returns
        -------
        (bool) True if the format is supported for this device
        """
        return bool(_lib.soundio_device_supports_format(device, format))

    def get_default_format(self, device):
        """
        Get the best format value.

        Parameters
        ----------
        device: (SoundIoDevice) device object

        Returns
        ------
        (SoundIoFormat) The best available format
        """
        dtype = _lib.SoundIoFormatInvalid
        for fmt in constants.PRIORITISED_FORMATS:
            if self.supports_format(device, fmt):
                dtype = fmt
                break
        if dtype == _lib.SoundIoFormatInvalid:
            raise PySoundIoError('Incompatible sample formats')
        return dtype

    def sort_channel_layouts(self, device):
        """
        Sorts channel layouts by channel count, descending

        Parameters
        ----------
        device: (SoundIoDevice) device object
        """
        _lib.soundio_device_sort_channel_layouts(device)

    def _get_default_layout(self, channels):
        """
        Get default builtin channel layout for the given number of channels

        Parameters
        ----------
        channel_count: (int) desired number of channels
        """
        return _lib.soundio_channel_layout_get_default(channels)

    def get_bytes_per_frame(self, format, channels):
        """
        Get the number of bytes per frame

        Parameters
        ----------
        format: (SoundIoFormat) format
        channels: (int) number of channels

        Returns
        -------
        (int) number of bytes per frame
        """
        return _lib.soundio_get_bytes_per_sample(format) * channels

    def get_bytes_per_sample(self, format):
        """
        Get the number of bytes per sample

        Parameters
        ----------
        format: (SoundIoFormat) format

        Returns
        -------
        (int) number of bytes per sample
        """
        return _lib.soundio_get_bytes_per_sample(format)

    def get_bytes_per_second(self, format, channels, sample_rate):
        """
        Get the number of bytes per second

        Parameters
        ----------
        format: (SoundIoFormat) format
        channels (int) number of channels
        sample_rate (int) sample rate

        Returns
        -------
        (int) number of bytes per second
        """
        return self.get_bytes_per_frame(format, channels) * sample_rate

    def _create_input_ring_buffer(self, capacity):
        """
        Creates ring buffer with the capacity to hold 30 seconds of data,
        by default.
        """
        self.input['buffer'] = _lib.soundio_ring_buffer_create(self._soundio, capacity)
        return self.input['buffer']

    def _create_output_ring_buffer(self, capacity):
        """
        Creates ring buffer with the capacity to hold 30 seconds of data,
        by default.
        """
        self.output['buffer'] = _lib.soundio_ring_buffer_create(self._soundio, capacity)
        return self.output['buffer']

    def _create_input_stream(self):
        """
        Allocates memory and sets defaults for input stream
        """
        self.input['stream'] = _lib.soundio_instream_create(self.input['device'])
        if not self.input['stream']:
            raise PySoundIoError('Out of memory')

        self.input['stream'].userdata = self._userdata
        self.input['stream'].read_callback = _lib._read_callback
        self.input['stream'].overflow_callback = _lib._overflow_callback
        self.input['stream'].error_callback = _lib._input_error_callback

        layout = self._get_default_layout(self.input['channels'])
        if layout:
            self.input['stream'].layout = layout[0]
        else:
            raise RuntimeError('Failed to find a channel layout for %d channels' % self.input['channels'])

        self.input['stream'].format = self.input['format']
        self.input['stream'].sample_rate = self.input['sample_rate']
        if self.input['block_size']:
            self.input['stream'].software_latency = float(self.input['block_size']) / self.input['sample_rate']

        return self.input['stream']

    def _open_input_stream(self):
        """
        Open an input stream.
        """
        self._check(_lib.soundio_instream_open(self.input['stream']))

    def _start_input_stream(self):
        """
        Start an input stream running.
        """
        self._check(_lib.soundio_instream_start(self.input['stream']))

    def pause_input_stream(self, pause):
        """
        Pause input stream

        Parameters
        ----------
        pause: (bool) True to pause, False to unpause
        """
        self._check(_lib.soundio_instream_pause(self.input['stream'], pause))

    def get_input_latency(self, out_latency):
        """
        Obtain the number of seconds that the next frame of sound
        being captured will take to arrive in the buffer,
        plus the amount of time that is represented in the buffer.

        Parameters
        ----------
        out_latency: (float) output latency in seconds
        """
        c_latency = _ffi.new('double *', out_latency)
        return _lib.soundio_instream_get_latency(self.input['stream'], c_latency)

    def start_input_stream(self, device_id=None,
                           sample_rate=None, dtype=None,
                           block_size=None, channels=None,
                           read_callback=None, overflow_callback=None):
        """
        Creates input stream, and sets parameters. Then allocates
        a ring buffer and starts the stream.

        The read callback is called in an audio processing thread,
        when a block of data is read from the microphone. Data is
        passed from the ring buffer to the callback to process.

        Parameters
        ----------
        device_id: (int) input device id
        sample_rate: (int) desired sample rate (optional)
        dtype: (SoundIoFormat) desired format, see `Formats`_. (optional)
        block_size: (int) desired block size (optional)
        channels: (int) number of channels [1: mono, 2: stereo] (optional)
        read_callback: (fn) function to call with data, the function must have
                        the arguments data and length. See record example
        overflow_callback: (fn) function to call if data is not being read fast enough

        Raises
        ------
        PySoundIoError if any invalid parameters are used

        Notes
        -----
        An example read callback

        .. code-block:: python
            :linenos:

            # Note: `length` is the number of samples per channel
            def read_callback(data: bytearray, length: int):
                wav.write(data)

        Overflow callback example

        .. code-block:: python
            :linenos:

            def overflow_callback():
                print('buffer overflow')
        """
        self.input['sample_rate'] = sample_rate
        self.input['format'] = dtype
        self.input['block_size'] = block_size
        self.input['channels'] = channels
        self.input['read_callback'] = read_callback
        self.input['overflow_callback'] = overflow_callback

        if device_id is not None:
            self.input['device'] = self.get_input_device(device_id)
        else:
            self.input['device'] = self.get_default_input_device()

        self.logger.info('Input Device: %s' % _ffi.string(self.input['device'].name).decode())
        self.sort_channel_layouts(self.input['device'])

        if self.input['sample_rate']:
            if not self.supports_sample_rate(self.input['device'], self.input['sample_rate']):
                raise PySoundIoError('Invalid sample rate: %d' % self.input['sample_rate'])
        else:
            self.input['sample_rate'] = self.get_default_sample_rate(self.input['device'])

        if self.input['format']:
            if not self.supports_format(self.input['device'], self.input['format']):
                raise PySoundIoError('Invalid format: %s interleaved' %
                                     (_ffi.string(_lib.soundio_format_string(self.input['format'])).decode()))
        else:
            self.input['format'] = self.get_default_format(self.input['device'])

        self._create_input_stream()
        self._open_input_stream()
        self.input['bytes_per_frame'] = self.get_bytes_per_frame(self.input['format'], channels)
        capacity = int(constants.DEFAULT_RING_BUFFER_DURATION *
                       self.input['stream'].sample_rate * self.input['bytes_per_frame'])
        self._create_input_ring_buffer(capacity)

        if self.input['stream'].layout_error:
            raise RuntimeError('Layout error')

        layout_name = _ffi.string(self.input['stream'].layout.name).decode()
        self.logger.info('Created input stream with a %s layout', layout_name)

        self.input['thread'] = _InputProcessingThread(parent=self)
        self._start_input_stream()
        self.flush()

    def _create_output_stream(self):
        """
        Allocates memory and sets defaults for output stream
        """
        self.output['stream'] = _lib.soundio_outstream_create(self.output['device'])
        if not self.output['stream']:
            raise PySoundIoError('Out of memory')

        self.output['stream'].userdata = self._userdata
        self.output['stream'].write_callback = _lib._write_callback
        self.output['stream'].underflow_callback = _lib._underflow_callback
        self.output['stream'].error_callback = _lib._output_error_callback

        layout = self._get_default_layout(self.output['channels'])
        if layout:
            self.output['stream'].layout = layout[0]
        else:
            raise RuntimeError('Failed to find a channel layout for %d channels' % self.output['channels'])

        self.output['stream'].format = self.output['format']
        self.output['stream'].sample_rate = self.output['sample_rate']
        if self.output['block_size']:
            self.output['stream'].software_latency = float(self.output['block_size']) / self.output['sample_rate']

        return self.output['stream']

    def _open_output_stream(self):
        """
        Open an output stream.
        """
        self._check(_lib.soundio_outstream_open(self.output['stream']))
        self.output['block_size'] = int(self.output['stream'].software_latency / self.output['sample_rate'])

    def _start_output_stream(self):
        """
        Start an output stream running.
        """
        self._check(_lib.soundio_outstream_start(self.output['stream']))

    def pause_output_stream(self, pause):
        """
        Pause output stream

        Parameters
        ----------
        pause: (bool) True to pause, False to unpause
        """
        self._check(_lib.soundio_outstream_pause(self.output['stream'], pause))

    def _clear_output_buffer(self):
        """
        Clear the output buffer
        """
        if self.output['buffer']:
            _lib.soundio_ring_buffer_clear(self.output['buffer'])

    def get_output_latency(self, out_latency):
        """
        Obtain the total number of seconds that the next frame written
        will take to become audible.

        Parameters
        ----------
        out_latency: (float) output latency in seconds
        """
        c_latency = _ffi.new('double *', out_latency)
        return _lib.soundio_outstream_get_latency(self.output['stream'], c_latency)

    def start_output_stream(self, device_id=None,
                            sample_rate=None, dtype=None,
                            block_size=None, channels=None,
                            write_callback=None, underflow_callback=None):
        """
        Creates output stream, and sets parameters. Then allocates
        a ring buffer and starts the stream.

        The write callback is called in an audio processing thread,
        when a block of data should be passed to the speakers. Data is
        added to the ring buffer to process.

        Parameters
        ----------
        device_id: (int) output device id
        sample_rate: (int) desired sample rate (optional)
        dtype: (SoundIoFormat) desired format, see `Formats`_. (optional)
        block_size: (int) desired block size (optional)
        channels: (int) number of channels [1: mono, 2: stereo] (optional)
        write_callback: (fn) function to call with data, the function must have
                        the arguments data and length.
        underflow_callback: (fn) function to call if data is not being written fast enough

        Raises
        ------
        PySoundIoError if any invalid parameters are used

        Notes
        -----
        An example write callback

        .. code-block:: python
            :linenos:

            # Note: `length` is the number of samples per channel
            def write_callback(data: bytearray, length: int):
                outdata = ar.array('f', [0] * length)
                for value in outdata:
                    outdata = 1.0
                data[:] = outdata.tostring()

        Underflow callback example

        .. code-block:: python
            :linenos:

            def underflow_callback():
                print('buffer underflow')
        """
        self.output['sample_rate'] = sample_rate
        self.output['format'] = dtype
        self.output['block_size'] = block_size
        self.output['channels'] = channels
        self.output['write_callback'] = write_callback
        self.output['underflow_callback'] = underflow_callback

        if device_id is not None:
            self.output['device'] = self.get_output_device(device_id)
        else:
            self.output['device'] = self.get_default_output_device()

        self.logger.info('Input Device: %s' % _ffi.string(self.output['device'].name).decode())
        self.sort_channel_layouts(self.output['device'])

        if self.output['sample_rate']:
            if not self.supports_sample_rate(self.output['device'], self.output['sample_rate']):
                raise PySoundIoError('Invalid sample rate: %d' % self.output['sample_rate'])
        else:
            self.output['sample_rate'] = self.get_default_sample_rate(self.output['device'])

        if self.output['format']:
            if not self.supports_format(self.output['device'], self.output['format']):
                raise PySoundIoError('Invalid format: %s interleaved' %
                                     (_ffi.string(_lib.soundio_format_string(self.output['format'])).decode()))
        else:
            self.output['format'] = self.get_default_format(self.output['device'])

        self._create_output_stream()
        self._open_output_stream()
        self.output['bytes_per_frame'] = self.get_bytes_per_frame(self.output['format'], channels)
        capacity = int(constants.DEFAULT_RING_BUFFER_DURATION *
                       self.output['stream'].sample_rate * self.output['bytes_per_frame'])
        self._create_output_ring_buffer(capacity)
        self._clear_output_buffer()

        if self.output['stream'].layout_error:
            raise RuntimeError('Layout error')

        layout_name = _ffi.string(self.output['stream'].layout.name).decode()
        self.logger.info('Created output stream with a %s layout', layout_name)

        self.output['thread'] = _OutputProcessingThread(parent=self)
        self._start_output_stream()
        self.flush()


@_ffi.def_extern()
def _write_callback(output_stream,
                    frame_count_min,
                    frame_count_max):
    """
    Called internally when the output requires some data
    """
    self = _ffi.from_handle(output_stream.userdata)

    frame_count = 0
    read_ptr = _lib.soundio_ring_buffer_read_ptr(self.output['buffer'])
    fill_bytes = _lib.soundio_ring_buffer_fill_count(self.output['buffer'])
    fill_count = fill_bytes / output_stream.bytes_per_frame

    read_count = min(frame_count_max, fill_count)
    frames_left = read_count

    if frame_count_min > fill_count:
        frames_left = frame_count_min
        while frames_left > 0:
            frame_count = frames_left
            if frame_count <= 0:
                return

            frame_count_ptr = _ffi.new('int *', frame_count)
            areas_ptr = _ffi.new('struct SoundIoChannelArea **')
            self._check(
                _lib.soundio_outstream_begin_write(output_stream,
                                                   areas_ptr,
                                                   frame_count_ptr)
            )
            if frame_count_ptr[0] <= 0:
                return

            num_bytes = output_stream.bytes_per_sample * output_stream.layout.channel_count * frame_count_ptr[0]
            fill_bytes = bytearray(b'\x00' * num_bytes)
            _ffi.memmove(areas_ptr[0][0].ptr, fill_bytes, num_bytes)

            self._check(_lib.soundio_outstream_end_write(output_stream))
            frames_left -= frame_count_ptr[0]

    while frames_left > 0:
        frame_count = int(frames_left)
        frame_count_ptr = _ffi.new('int *', frame_count)
        areas_ptr = _ffi.new('struct SoundIoChannelArea **')
        self._check(
            _lib.soundio_outstream_begin_write(output_stream, areas_ptr, frame_count_ptr)
        )
        if frame_count_ptr[0] <= 0:
            break

        num_bytes = output_stream.bytes_per_sample * output_stream.layout.channel_count * frame_count_ptr[0]
        _ffi.memmove(areas_ptr[0][0].ptr, read_ptr, num_bytes)
        read_ptr += num_bytes

        self._check(_lib.soundio_outstream_end_write(output_stream))
        frames_left -= frame_count_ptr[0]

    _lib.soundio_ring_buffer_advance_read_ptr(
        self.output['buffer'],
        int(read_count * output_stream.bytes_per_frame)
    )

    with threading.Lock():
        self.output['thread'].block_size = frame_count_max
        self.output['thread'].to_read += 1


@_ffi.def_extern()
def _underflow_callback(output_stream):
    """
    Called internally when the sound device runs out of
    buffered audio data to play.
    """
    logger = logging.getLogger(__name__)
    logger.error('Output underflow')

    self = _ffi.from_handle(output_stream.userdata)
    if self.output['underflow_callback']:
        self.output['underflow_callback']()


@_ffi.def_extern()
def _output_error_callback(output_stream,
                           error_code):
    """
    Called internally when an error occurs in the
    output stream.
    """
    logger = logging.getLogger(__name__)
    logger.error(_ffi.string(_lib.soundio_strerror(error_code)).decode())


@_ffi.def_extern()
def _read_callback(input_stream,
                   frame_count_min,
                   frame_count_max):
    """
    Called internally when there is input data available.
    """
    self = _ffi.from_handle(input_stream.userdata)

    write_ptr = _lib.soundio_ring_buffer_write_ptr(self.input['buffer'])
    free_bytes = _lib.soundio_ring_buffer_free_count(self.input['buffer'])

    free_count = free_bytes / input_stream.bytes_per_frame

    if free_count < frame_count_min:
        logger = logging.getLogger(__name__)
        logger.critical('Ring buffer overflow')

    write_frames = min(free_count, frame_count_max)
    frames_left = write_frames

    while True:
        frame_count = frames_left
        frame_count_ptr = _ffi.new('int *', int(frame_count))
        areas_ptr = _ffi.new('struct SoundIoChannelArea **')

        self._check(
            _lib.soundio_instream_begin_read(input_stream,
                                             areas_ptr,
                                             frame_count_ptr)
        )
        if not frame_count_ptr[0]:
            break
        if not areas_ptr[0]:
            # Due to an overflow there is a hole.
            # Fill the ring buffer with silence for the size of the hole.
            fill = bytearray(b'\x00' * frame_count_ptr[0] * input_stream.bytes_per_frame)
            _ffi.memmove(write_ptr, fill, len(fill))
        else:
            num_bytes = input_stream.bytes_per_sample * input_stream.layout.channel_count * frame_count_ptr[0]
            _ffi.memmove(write_ptr, areas_ptr[0][0].ptr, num_bytes)
            write_ptr += num_bytes

        self._check(_lib.soundio_instream_end_read(input_stream))
        frames_left -= frame_count_ptr[0]
        if frames_left <= 0:
            break

    advance_bytes = int(write_frames * input_stream.bytes_per_frame)
    _lib.soundio_ring_buffer_advance_write_ptr(self.input['buffer'], advance_bytes)


@_ffi.def_extern()
def _overflow_callback(input_stream):
    """
    Called internally when the sound device buffer is full,
    yet there is more captured audio to put in it.
    """
    logger = logging.getLogger(__name__)
    logger.error('Input overflow')

    self = _ffi.from_handle(input_stream.userdata)
    if self.input['overflow_callback']:
        self.input['overflow_callback']()


@_ffi.def_extern()
def _input_error_callback(input_stream,
                          error_code):
    """
    Called internally when an error occurs in the
    input stream.
    """
    logger = logging.getLogger(__name__)
    logger.error(_ffi.string(_lib.soundio_strerror(error_code)).decode())
