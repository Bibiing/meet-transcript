"""
constants.py

Constants and enumerations.
"""

from pysoundio._soundio import lib as _lib

DEFAULT_RING_BUFFER_DURATION = 30  # secs

BACKENDS = {
    _lib.SoundIoBackendNone: 'SoundIoBackendNone',
    _lib.SoundIoBackendPulseAudio: 'SoundIoBackendPulseAudio',
    _lib.SoundIoBackendJack: 'SoundIoBackendJack',
    _lib.SoundIoBackendAlsa: 'SoundIoBackendAlsa',
    _lib.SoundIoBackendCoreAudio: 'SoundIoBackendCoreAudio',
    _lib.SoundIoBackendWasapi: 'SoundIoBackendWasapi',
    _lib.SoundIoBackendDummy: 'SoundIoBackendDummy',
}

FORMATS = {
    _lib.SoundIoFormatS8: 'SoundIoFormatS8',
    _lib.SoundIoFormatU8: 'SoundIoFormatU8',
    _lib.SoundIoFormatS16LE: 'SoundIoFormatS16LE',
    _lib.SoundIoFormatS16BE: 'SoundIoFormatS16BE',
    _lib.SoundIoFormatU16LE: 'SoundIoFormatU16LE',
    _lib.SoundIoFormatU16BE: 'SoundIoFormatU16BE',
    _lib.SoundIoFormatS24LE: 'SoundIoFormatS24LE',
    _lib.SoundIoFormatS24BE: 'SoundIoFormatS24BE',
    _lib.SoundIoFormatU24LE: 'SoundIoFormatU24LE',
    _lib.SoundIoFormatU24BE: 'SoundIoFormatU24BE',
    _lib.SoundIoFormatS32LE: 'SoundIoFormatS32LE',
    _lib.SoundIoFormatS32BE: 'SoundIoFormatS32BE',
    _lib.SoundIoFormatU32LE: 'SoundIoFormatU32LE',
    _lib.SoundIoFormatU32BE: 'SoundIoFormatU32BE',
    _lib.SoundIoFormatFloat32LE: 'SoundIoFormatFloat32LE',
    _lib.SoundIoFormatFloat32BE: 'SoundIoFormatFloat32BE',
    _lib.SoundIoFormatFloat64LE: 'SoundIoFormatFloat64LE',
    _lib.SoundIoFormatFloat64BE: 'SoundIoFormatFloat64BE',
    _lib.SoundIoFormatInvalid: 'SoundIoFormatInvalid'
}

PRIORITISED_FORMATS = [
    _lib.SoundIoFormatFloat32LE,
    _lib.SoundIoFormatFloat32BE,
    _lib.SoundIoFormatS32LE,
    _lib.SoundIoFormatS32BE,
    _lib.SoundIoFormatS24LE,
    _lib.SoundIoFormatS24BE,
    _lib.SoundIoFormatS16LE,
    _lib.SoundIoFormatS16BE,
    _lib.SoundIoFormatFloat64LE,
    _lib.SoundIoFormatFloat64BE,
    _lib.SoundIoFormatU32LE,
    _lib.SoundIoFormatU32BE,
    _lib.SoundIoFormatU24LE,
    _lib.SoundIoFormatU24BE,
    _lib.SoundIoFormatU16LE,
    _lib.SoundIoFormatU16BE,
    _lib.SoundIoFormatS8,
    _lib.SoundIoFormatU8,
    _lib.SoundIoFormatInvalid,
]

PRIORITISED_SAMPLE_RATES = [
    48000,
    44100,
    96000,
    24000,
    0,
]

ARRAY_FORMATS = {
    _lib.SoundIoFormatS8: 'b',
    _lib.SoundIoFormatU8: 'B',
    _lib.SoundIoFormatS16LE: 'h',
    _lib.SoundIoFormatS16BE: 'h',
    _lib.SoundIoFormatU16LE: 'H',
    _lib.SoundIoFormatU16BE: 'H',
    _lib.SoundIoFormatS24LE: None,
    _lib.SoundIoFormatS24BE: None,
    _lib.SoundIoFormatU24LE: None,
    _lib.SoundIoFormatU24BE: None,
    _lib.SoundIoFormatS32LE: 'l',
    _lib.SoundIoFormatS32BE: 'l',
    _lib.SoundIoFormatU32LE: 'L',
    _lib.SoundIoFormatU32BE: 'L',
    _lib.SoundIoFormatFloat32LE: 'f',
    _lib.SoundIoFormatFloat32BE: 'f',
    _lib.SoundIoFormatFloat64LE: 'd',
    _lib.SoundIoFormatFloat64BE: 'd'
}

SOUNDFILE_FORMATS = {
    _lib.SoundIoFormatS8: None,
    _lib.SoundIoFormatU8: None,
    _lib.SoundIoFormatS16LE: 'int16',
    _lib.SoundIoFormatS16BE: 'int16',
    _lib.SoundIoFormatU16LE: None,
    _lib.SoundIoFormatU16BE: None,
    _lib.SoundIoFormatS24LE: None,
    _lib.SoundIoFormatS24BE: None,
    _lib.SoundIoFormatU24LE: None,
    _lib.SoundIoFormatU24BE: None,
    _lib.SoundIoFormatS32LE: 'int32',
    _lib.SoundIoFormatS32BE: 'int32',
    _lib.SoundIoFormatU32LE: None,
    _lib.SoundIoFormatU32BE: None,
    _lib.SoundIoFormatFloat32LE: 'float32',
    _lib.SoundIoFormatFloat32BE: 'float32',
    _lib.SoundIoFormatFloat64LE: 'float64',
    _lib.SoundIoFormatFloat64BE: 'float64'
}
