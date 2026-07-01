import json
import logging
import threading
import time
import queue
import numpy as np

from whisper_live import metrics as wl_metrics
from whisper_live.local_agreement import LocalAgreementStabilizer


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    MAX_BUFFER_DURATION_S = 45
    """Maximum audio buffer duration in seconds before trimming."""
    BUFFER_TRIM_DURATION_S = 30
    """Duration in seconds to trim from the buffer when it exceeds MAX_BUFFER_DURATION_S."""
    CLIP_THRESHOLD_DURATION_S = 25
    """Duration threshold in seconds for clipping audio with no valid segments."""
    CLIP_TAIL_DURATION_S = 5
    """Duration in seconds of audio to keep after clipping."""

    client_uid: str
    """A unique identifier for the client."""
    websocket: object
    """The WebSocket connection for the client."""
    send_last_n_segments: int
    """Number of most recent segments to send to the client."""
    no_speech_thresh: float
    """Segments with no speech probability above this threshold will be discarded."""
    clip_audio: bool
    """Whether to clip audio with no valid segments."""
    same_output_threshold: int
    """Number of repeated outputs before considering it as a valid segment."""

    MAX_TRANSCRIPT_LENGTH = 500
    MAX_TRANSLATION_QUEUE_SIZE = 100

    def __init__(
        self,
        client_uid,
        websocket,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
        diarization=None,
        word_timestamps=False,
        local_agreement=False,
        local_agreement_window_seconds=15.0,
        local_agreement_hop_seconds=2.0,
        local_agreement_trailing_guard_seconds=0.6,
        local_agreement_retain_seconds=1.0,
        dynamic_prompt=True,
        dynamic_prompt_max_chars=700,
    ):
        self.client_uid = client_uid
        self.websocket = websocket
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        self.diarization = diarization
        self.word_timestamps = word_timestamps

        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ""
        self.prev_out = ""
        self.exit = False
        self.same_output_count = 0
        self.transcript = []
        self.end_time_for_same_output = None
        self.translation_queue = translation_queue
        self.local_agreement = local_agreement
        self.local_agreement_window_seconds = max(2.0, float(local_agreement_window_seconds))
        self.local_agreement_hop_seconds = max(0.2, float(local_agreement_hop_seconds))
        self.local_agreement_retain_seconds = max(0.0, float(local_agreement_retain_seconds))
        self.dynamic_prompt = dynamic_prompt
        self.dynamic_prompt_max_chars = max(0, int(dynamic_prompt_max_chars))
        self.last_transcription_at = 0.0
        self.processing_offset = 0.0
        self.local_agreement_stabilizer = (
            LocalAgreementStabilizer(trailing_guard_seconds=local_agreement_trailing_guard_seconds)
            if local_agreement
            else None
        )

        # Optional post-processing callable for segments.
        # If set, called with a segment dict and must return a segment dict.
        # Allows external projects to plug in custom post-processing
        # (e.g. PII redaction, formatting, diarization) without modifying
        # WhisperLive's core code.
        self.segment_post_processor = None

        # threading
        self.lock = threading.Lock()

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            if self.local_agreement:
                now = time.time()
                if now - self.last_transcription_at < self.local_agreement_hop_seconds:
                    time.sleep(0.05)
                    continue
                self.last_transcription_at = now
            try:
                input_sample = input_bytes.copy()
                t0 = time.time()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    continue
                wl_metrics.track_transcription_latency(time.time() - t0)
                wl_metrics.track_audio_processed(duration)
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                wl_metrics.track_error("transcription")
                time.sleep(0.01)

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self, result, duration):
        raise NotImplementedError
    
    def format_segment(
        self,
        start,
        end,
        text,
        completed=False,
        speaker=None,
        words=None,
        no_speech_prob=None,
        avg_logprob=None,
        compression_ratio=None,
    ):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.
            speaker (str, optional): Speaker label from diarization.
            words (list, optional): Word-level timestamps and probabilities.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places and the 'text'
                of the transcription.
        """
        seg = {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text,
            'completed': completed,
        }
        if speaker is not None:
            seg['speaker'] = speaker
        if words is not None:
            seg['words'] = words
        if no_speech_prob is not None:
            seg['no_speech_prob'] = no_speech_prob
        if avg_logprob is not None:
            seg['avg_logprob'] = avg_logprob
        if compression_ratio is not None:
            seg['compression_ratio'] = compression_ratio
        return seg

    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        self.lock.acquire()
        if self.frames_np is not None and self.frames_np.shape[0] > self.MAX_BUFFER_DURATION_S*self.RATE:
            self.frames_offset += float(self.BUFFER_TRIM_DURATION_S)
            self.frames_np = self.frames_np[int(self.BUFFER_TRIM_DURATION_S*self.RATE):]
            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)
        self.lock.release()

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > self.CLIP_THRESHOLD_DURATION_S * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - self.CLIP_TAIL_DURATION_S

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            if self.local_agreement:
                total_samples = self.frames_np.shape[0]
                max_window_samples = int(self.local_agreement_window_seconds * self.RATE)
                if total_samples - samples_take > max_window_samples:
                    samples_take = total_samples - max_window_samples
                    self.timestamp_offset = self.frames_offset + (samples_take / self.RATE)
            self.processing_offset = self.frames_offset + (samples_take / self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        If a ``segment_post_processor`` callable is set, each segment is passed through it
        before sending. The callable receives a segment dict and must return a segment dict.

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        if self.segment_post_processor is not None:
            processed = []
            for seg in segments:
                try:
                    result = self.segment_post_processor(seg)
                    if result is not None:
                        processed.append(result)
                except Exception as e:
                    logging.error(f"[ERROR]: segment_post_processor failed: {e}")
                    processed.append(seg)
            segments = processed

        if not segments:
            return segments

        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments,
                })
            )
            for seg in segments:
                wl_metrics.track_segment_emitted(completed=seg.get("completed", False))
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True
    
    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_start(self, segment):
        return getattr(segment, "start", getattr(segment, "start_ts", 0))

    def get_segment_end(self, segment):
        return getattr(segment, "end", getattr(segment, "end_ts", 0))

    def _identify_speaker(self, segment):
        """Run diarization on a segment's audio slice if diarization is enabled.

        Returns:
            str or None: Speaker label, or None if diarization is disabled or audio unavailable.
        """
        if self.diarization is None or self.frames_np is None:
            return None
        try:
            seg_start = self.get_segment_start(segment)
            seg_end = self.get_segment_end(segment)
            start_sample = int(seg_start * self.RATE)
            end_sample = int(seg_end * self.RATE)
            samples_offset = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            audio_slice = self.frames_np[samples_offset + start_sample:samples_offset + end_sample]
            if len(audio_slice) < self.RATE * 0.3:
                return None
            return self.diarization.identify_speaker(audio_slice, self.RATE)
        except Exception as e:
            logging.error(f"Diarization error: {e}")
            return None

    def _extract_words(self, segment, time_offset):
        """Extracts word-level timestamps from a segment if word_timestamps is enabled."""
        if not self.word_timestamps:
            return None
        words = getattr(segment, "words", None)
        if not words:
            return None
        return [
            {
                "word": w.word,
                "start": "{:.3f}".format(time_offset + w.start),
                "end": "{:.3f}".format(time_offset + w.end),
                "probability": round(w.probability, 4),
            }
            for w in words
        ]

    def build_initial_prompt(self):
        """Build static glossary plus recent confirmed text as decode context."""
        static_prompt = getattr(self, "initial_prompt", None)
        if not self.dynamic_prompt or self.dynamic_prompt_max_chars <= 0 or not self.transcript:
            return static_prompt

        recent_text = " ".join(
            str(segment.get("text", "")).strip()
            for segment in self.transcript[-12:]
            if str(segment.get("text", "")).strip()
        )
        if not recent_text:
            return static_prompt
        recent_text = recent_text[-self.dynamic_prompt_max_chars :]
        if static_prompt:
            return (
                f"{static_prompt}\n"
                f"Previous confirmed transcript for context only; do not repeat it: {recent_text}"
            )
        return f"Previous confirmed transcript for context only; do not repeat it: {recent_text}"

    def update_segments(self, segments, duration):
        """
        Processes the segments from Whisper and updates the transcript.
        Uses helper methods to account for differences between backends.
        
        Args:
            segments (list): List of segments returned by the transcriber.
            duration (float): Duration of the current audio chunk.
        
        Returns:
            dict or None: The last processed segment (if any).
        """
        if self.local_agreement:
            return self.update_segments_with_local_agreement(segments, duration)

        offset = None
        self.current_out = ''
        last_segment = None

        # Process complete segments only if there are more than one
        # and if the last segment's no_speech_prob is below the threshold.
        if len(segments) > 1 and self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            for s in segments[:-1]:
                text_ = s.text
                self.text.append(text_)
                with self.lock:
                    start = self.timestamp_offset + self.get_segment_start(s)
                    end = self.timestamp_offset + min(duration, self.get_segment_end(s))
                if start >= end:
                    continue
                if self.get_segment_no_speech_prob(s) > self.no_speech_thresh:
                    continue
                speaker = self._identify_speaker(s)
                words = self._extract_words(s, self.timestamp_offset)
                completed_segment = self.format_segment(
                    start,
                    end,
                    text_,
                    completed=True,
                    speaker=speaker,
                    words=words,
                    no_speech_prob=getattr(s, "no_speech_prob", None),
                    avg_logprob=getattr(s, "avg_logprob", None),
                    compression_ratio=getattr(s, "compression_ratio", None),
                )
                self.transcript.append(completed_segment)

                if self.translation_queue:
                    try:
                        self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                    except queue.Full:
                        logging.warning("Translation queue is full, skipping segment")
                offset = min(duration, self.get_segment_end(s))

        # Process the last segment if its no_speech_prob is acceptable.
        if self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            self.current_out += segments[-1].text
            words = self._extract_words(segments[-1], self.timestamp_offset)
            with self.lock:
                last_segment = self.format_segment(
                    self.timestamp_offset + self.get_segment_start(segments[-1]),
                    self.timestamp_offset + min(duration, self.get_segment_end(segments[-1])),
                    self.current_out,
                    completed=False,
                    words=words,
                    no_speech_prob=getattr(segments[-1], "no_speech_prob", None),
                    avg_logprob=getattr(segments[-1], "avg_logprob", None),
                    compression_ratio=getattr(segments[-1], "compression_ratio", None),
                )

        # Handle repeated output logic.
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1

            # if we remove the audio because of same output on the nth reptition we might remove the 
            # audio thats not yet transcribed so, capturing the time when it was repeated for the first time
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(segments[-1])
            time.sleep(0.1)  # wait briefly for any new voice activity
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # If the same incomplete segment is repeated too many times,
        # append it to the transcript and update the offset.
        if self.same_output_count > self.same_output_threshold:
            if not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                with self.lock:
                    completed_segment = self.format_segment(
                        self.timestamp_offset,
                        self.timestamp_offset + min(duration, self.end_time_for_same_output),
                        self.current_out,
                        completed=True,
                        no_speech_prob=getattr(segments[-1], "no_speech_prob", None),
                        avg_logprob=getattr(segments[-1], "avg_logprob", None),
                        compression_ratio=getattr(segments[-1], "compression_ratio", None),
                    )
                    self.transcript.append(completed_segment)

                    if self.translation_queue:
                        try:
                            self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                        except queue.Full:
                            logging.warning("Translation queue is full, skipping segment")

            self.current_out = ''
            offset = min(duration, self.end_time_for_same_output)
            self.same_output_count = 0
            last_segment = None
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        self._trim_transcript()
        return last_segment

    def update_segments_with_local_agreement(self, segments, duration):
        """Finalize only locally stable hypothesis prefixes."""
        if self.local_agreement_stabilizer is None:
            return None

        result = self.local_agreement_stabilizer.update(
            segments,
            offset_seconds=self.processing_offset,
            window_duration_seconds=duration,
            no_speech_threshold=self.no_speech_thresh,
        )

        for completed in result.completed:
            segment = self.format_segment(
                completed["start"],
                completed["end"],
                completed["text"],
                completed=True,
            )
            self.transcript.append(segment)
            self.text.append(completed["text"])
            if self.translation_queue:
                try:
                    self.translation_queue.put(segment.copy(), timeout=0.1)
                except queue.Full:
                    logging.warning("Translation queue is full, skipping segment")

        if result.confirmed_until > 0:
            with self.lock:
                retained_offset = result.confirmed_until - self.local_agreement_retain_seconds
                self.timestamp_offset = max(self.timestamp_offset, retained_offset)

        last_segment = None
        if result.partial is not None:
            partial = result.partial
            last_segment = self.format_segment(
                partial["start"],
                partial["end"],
                partial["text"],
                completed=False,
            )

        self._trim_transcript()
        return last_segment

    def _trim_transcript(self):
        """Trims transcript and text lists to prevent unbounded memory growth."""
        if len(self.transcript) > self.MAX_TRANSCRIPT_LENGTH:
            self.transcript = self.transcript[-self.MAX_TRANSCRIPT_LENGTH:]
        if len(self.text) > self.MAX_TRANSCRIPT_LENGTH:
            self.text = self.text[-self.MAX_TRANSCRIPT_LENGTH:]
