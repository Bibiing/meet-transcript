PySoundIo
=========

.. image:: https://github.com/joextodd/pysoundio/workflows/tests/badge.svg
    :target: https://github.com/joextodd/pysoundio/workflows/tests
.. image:: https://coveralls.io/repos/github/joextodd/pysoundio/badge.svg
    :target: https://coveralls.io/github/joextodd/pysoundio
.. image:: https://readthedocs.org/projects/pysoundio/badge/?version=latest
    :target: http://pysoundio.readthedocs.io/en/latest/?badge=latest


A simple Pythonic interface for `libsoundio <http://libsound.io>`_.

libsoundio is a robust, cross-platform solution for real-time audio. It performs
no buffering or processing on your behalf, instead exposing the raw power of the
underlying backend.


Installation
------------

You can use pip to download and install the latest release with a single command. ::

    pip3 install pysoundio


Examples
--------

See examples directory.

Some of the examples require `PySoundFile <https://pysoundfile.readthedocs.io/en/latest/>`_ ::

    pip3 install soundfile

On Windows and OS X, this will also install the library libsndfile. On Linux you will need
to install the library as well.

* Ubuntu / Debian ::

    apt-get install libsndfile1


:download:`devices.py <../examples/devices.py>`

List the available input and output devices on the system and their properties. ::

    python devices.py


:download:`record.py <../examples/record.py>`

Records data from microphone and saves to a wav file.
Supports specifying backend, device, sample rate, block size. ::

    python record.py out.wav --device 0 --rate 44100


:download:`play.py <../examples/play.py>`

Plays a wav file through the speakers.
Supports specifying backend, device, block size. ::

    python play.py in.wav --device 0


:download:`sine.py <../examples/sine.py>`

Plays a sine wave through the speakers.
Supports specifying backend, device, sample rate, block size. ::

    python sine.py --freq 442


Testing
-------

To run the test suite. ::

    tox -r


Advanced
--------

If you wish to use your own build of libsoundio (perhaps you want Jack enabled)
then build from source and install it globally and reinstall pysoundio.

Note: PySoundIo only works with libsoundio versions >= 1.1.0
