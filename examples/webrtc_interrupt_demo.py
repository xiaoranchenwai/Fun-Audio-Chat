from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
import threading
import time
from typing import Callable, TypeAlias

import fastrtc
import gradio as gr
import librosa
import numpy as np
import sphn
from fastrtc import AdditionalOutputs, WebRTC
from ten_vad import TenVad
import websockets.sync.client


class MessageType:
    HANDSHAKE = 0x00
    AUDIO = 0x01
    TEXT = 0x02
    CONTROL = 0x03
    ERROR = 0x05


class ControlMessage:
    START = 0x00
    END_TURN = 0x01
    PAUSE = 0x02


@dataclass(frozen=True)
class StreamConfig:
    server_url: str = "ws://10.250.2.27:11235/api/chat"
    transport_sample_rate: int = 16000
    output_sample_rate: int = 16000
    frame_duration_ms: int = 60
    response_idle_timeout_s: float = 30.0
    response_max_wait_s: float = 120.0


StreamerGenerator = Generator[tuple[int, np.ndarray] | AdditionalOutputs, None, None]
StreamerFn: TypeAlias = Callable[[tuple[int, np.ndarray], str, list[dict[str, str]]], StreamerGenerator]


@dataclass
class VADEvent:
    interrupt_signal: bool | None = None
    full_audio: tuple[int, np.ndarray] | None = None


@dataclass
class WSInterruptController:
    _lock: threading.Lock
    _ws: websockets.sync.client.ClientConnection | None = None
    _interrupt_in_flight: bool = False

    def set_ws(self, ws: websockets.sync.client.ClientConnection | None) -> None:
        with self._lock:
            self._ws = ws
            if ws is None:
                self._interrupt_in_flight = False

    def send_interrupt(self) -> bool:
        with self._lock:
            if self._ws is None or self._interrupt_in_flight:
                return False
            self._interrupt_in_flight = True
            self._ws.send(bytes([MessageType.CONTROL, ControlMessage.START]))
            return True

    def clear_interrupt(self) -> None:
        with self._lock:
            self._interrupt_in_flight = False


global_ten_vad: TenVad | None = None
global_vad_lock = threading.Lock()


def global_vad_process(audio_data: np.ndarray) -> float:
    global global_ten_vad

    with global_vad_lock:
        if global_ten_vad is None:
            global_ten_vad = TenVad()

        prob, _ = global_ten_vad.process(audio_data)
        return prob


class RealtimeVAD:
    def __init__(
        self,
        *,
        src_sr: int = 24000,
        start_threshold: float = 0.8,
        end_threshold: float = 0.7,
        pad_start_s: float = 0.6,
        min_positive_s: float = 0.4,
        min_silence_s: float = 1.2,
        max_segment_s: float = 8.0,
    ) -> None:
        self.src_sr = src_sr
        self.vad_sr = 16000
        self.hop_size = 256
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.pad_start_s = pad_start_s
        self.min_positive_s = min_positive_s
        self.min_silence_s = min_silence_s
        self.max_segment_s = max_segment_s

        self.vad_buffer = np.array([], dtype=np.int16)
        self.src_buffer = np.array([], dtype=np.int16)

        self.vad_buffer_offset = 0
        self.src_buffer_offset = 0

        self.active = False
        self.interrupt_signal = False
        self.interrupt_dispatched = False
        self.sum_positive_s = 0.0
        self.silence_start_s: float | None = None
        self.active_start_s: float | None = None

    def process(self, audio_data: np.ndarray) -> Generator[VADEvent, None, None]:
        if audio_data.ndim == 2:
            audio_data = audio_data[0]

        self.src_buffer = np.concatenate((self.src_buffer, audio_data))

        vad_audio_data = librosa.resample(
            audio_data.astype(np.float32) / 32768.0,
            orig_sr=self.src_sr,
            target_sr=self.vad_sr,
        )
        vad_audio_data = (vad_audio_data * 32767.0).round().astype(np.int16)
        self.vad_buffer = np.concatenate((self.vad_buffer, vad_audio_data))
        vad_buffer_size = self.vad_buffer.shape[0]

        def process_chunk(chunk_offset_s: float, vad_chunk: np.ndarray) -> Generator[VADEvent, None, None]:
            speech_prob = global_vad_process(vad_chunk)
            hop_s = self.hop_size / self.vad_sr

            if not self.active:
                if speech_prob >= self.start_threshold:
                    self.active = True
                    self.active_start_s = chunk_offset_s
                    self.sum_positive_s = hop_s
                else:
                    new_src_offset = int((chunk_offset_s - self.pad_start_s) * self.src_sr)
                    cut_pos = new_src_offset - self.src_buffer_offset
                    if cut_pos > 0:
                        self.src_buffer = self.src_buffer[cut_pos:]
                        self.src_buffer_offset = new_src_offset
                return

            chunk_src_pos = int(chunk_offset_s * self.src_sr)

            if speech_prob >= self.end_threshold:
                self.silence_start_s = None
                self.sum_positive_s += hop_s
                if not self.interrupt_signal and self.sum_positive_s >= self.min_positive_s:
                    self.interrupt_signal = True
                    if not self.interrupt_dispatched:
                        self.interrupt_dispatched = True
                        yield VADEvent(interrupt_signal=True)
            elif self.silence_start_s is None:
                self.silence_start_s = chunk_offset_s

            if self.silence_start_s is not None and chunk_offset_s - self.silence_start_s >= self.min_silence_s:
                cut_pos = chunk_src_pos - self.src_buffer_offset
                if self.interrupt_signal:
                    webrtc_audio = self.src_buffer[np.newaxis, :cut_pos]
                    yield VADEvent(full_audio=(self.src_sr, webrtc_audio))
                self.src_buffer = self.src_buffer[cut_pos:]
                self.src_buffer_offset = chunk_src_pos

                self.active = False
                self.interrupt_signal = False
                self.interrupt_dispatched = False
                self.sum_positive_s = 0.0
                self.silence_start_s = None
                self.active_start_s = None
                return

            if (
                self.active_start_s is not None
                and chunk_offset_s - self.active_start_s >= self.max_segment_s
            ):
                cut_pos = chunk_src_pos - self.src_buffer_offset
                if self.interrupt_signal:
                    webrtc_audio = self.src_buffer[np.newaxis, :cut_pos]
                    yield VADEvent(full_audio=(self.src_sr, webrtc_audio))
                self.src_buffer = self.src_buffer[cut_pos:]
                self.src_buffer_offset = chunk_src_pos

                self.active = False
                self.interrupt_signal = False
                self.interrupt_dispatched = False
                self.sum_positive_s = 0.0
                self.silence_start_s = None
                self.active_start_s = None

        processed_samples = 0
        for chunk_pos in range(0, vad_buffer_size - self.hop_size, self.hop_size):
            processed_samples = chunk_pos + self.hop_size
            chunk_offset_s = (self.vad_buffer_offset + chunk_pos) / self.vad_sr
            vad_chunk = self.vad_buffer[chunk_pos : chunk_pos + self.hop_size]
            yield from process_chunk(chunk_offset_s, vad_chunk)

        self.vad_buffer = self.vad_buffer[processed_samples:]
        self.vad_buffer_offset += processed_samples


def init_global_ten_vad(input_sample_rate: int = 24000) -> None:
    global global_ten_vad

    require_warmup = False
    with global_vad_lock:
        if global_ten_vad is None:
            global_ten_vad = TenVad()
            require_warmup = True

    if require_warmup:
        realtime_vad = RealtimeVAD(src_sr=input_sample_rate)
        for _ in range(25):
            for _ in realtime_vad.process(np.zeros(960, dtype=np.int16)):
                pass


class VADStreamHandler(fastrtc.StreamHandler):
    def __init__(
        self,
        streamer_fn: StreamerFn,
        interrupt_controller: WSInterruptController,
        *,
        input_sample_rate: int = 24000,
    ) -> None:
        super().__init__(
            "mono",
            24000,
            None,
            input_sample_rate,
            30,
        )
        self.streamer_fn = streamer_fn
        self.interrupt_controller = interrupt_controller
        self.realtime_vad = RealtimeVAD(src_sr=input_sample_rate)
        self.generator: StreamerGenerator | None = None

        init_global_ten_vad(input_sample_rate)

    def emit(self) -> fastrtc.tracks.EmitType:
        if self.generator is None:
            return None

        try:
            return next(self.generator)
        except StopIteration:
            self.generator = None
            return None

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, audio_data = frame
        for event in self.realtime_vad.process(audio_data):
            if event.interrupt_signal:
                if not self.interrupt_controller.send_interrupt():
                    print("[VAD] Interrupt suppressed (no active session or already sent).")
                self.generator = None
                self.clear_queue()
            if event.full_audio is not None:
                self.wait_for_args_sync()
                self.latest_args[0] = event.full_audio
                self.generator = self.streamer_fn(*self.latest_args)
                self.interrupt_controller.clear_interrupt()

    def copy(self) -> fastrtc.StreamHandler:
        return VADStreamHandler(
            self.streamer_fn,
            self.interrupt_controller,
            input_sample_rate=self.input_sample_rate,
        )


def _normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=0)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32768.0
    return audio_data


def _resample_if_needed(audio_data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio_data
    return librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)


def chat_generator(
    audio_input: tuple[int, np.ndarray],
    history: list[dict[str, str]],
    config: StreamConfig,
    interrupt_controller: WSInterruptController,
) -> StreamerGenerator:
    sr, audio_data = audio_input
    history = list(history)
    audio_data = _normalize_audio(audio_data)

    user_msg = "🎤 [语音消息]"
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ""})
    yield AdditionalOutputs(history)

    opus_writer = sphn.OpusStreamWriter(config.transport_sample_rate)
    opus_reader = sphn.OpusStreamReader(config.transport_sample_rate)

    audio_resampled = _resample_if_needed(
        audio_data,
        sr,
        config.transport_sample_rate,
    )

    frame_size = int(config.transport_sample_rate * (config.frame_duration_ms / 1000.0))

    full_text_response = ""
    ws = None

    try:
        with websockets.sync.client.connect(config.server_url) as ws_conn:
            ws = ws_conn
            interrupt_controller.set_ws(ws)
            _ = ws.recv()

            ws.send(bytes([MessageType.CONTROL, ControlMessage.START]))

            cursor = 0
            total_len = len(audio_resampled)
            while cursor < total_len:
                end = min(cursor + frame_size, total_len)
                chunk = audio_resampled[cursor:end]
                opus_bytes = opus_writer.append_pcm(chunk)

                if opus_bytes:
                    ws.send(bytes([MessageType.AUDIO]) + opus_bytes)
                cursor = end

            ws.send(bytes([MessageType.CONTROL, ControlMessage.END_TURN]))

            last_message_time = time.monotonic()
            response_start_time = last_message_time
            while True:
                try:
                    message = ws.recv(timeout=5.0)
                except TimeoutError:
                    now = time.monotonic()
                    if now - last_message_time > config.response_idle_timeout_s:
                        break
                    if now - response_start_time > config.response_max_wait_s:
                        break
                    continue

                if not message:
                    continue

                last_message_time = time.monotonic()
                msg_type = message[0]
                payload = message[1:]

                if msg_type == MessageType.AUDIO:
                    pcm = opus_reader.append_bytes(payload)

                    if pcm.shape[-1] > 0:
                        pcm_int16 = (pcm * 32767.0).astype(np.int16)
                        yield (config.output_sample_rate, pcm_int16)
                elif msg_type == MessageType.TEXT:
                    text_chunk = payload.decode("utf-8", errors="ignore")
                    if text_chunk.startswith("\x02"):
                        text_chunk = text_chunk[1:]
                    full_text_response += text_chunk
                    history[-1]["content"] = full_text_response
                    yield AdditionalOutputs(history)
                elif msg_type == MessageType.ERROR:
                    err = payload.decode("utf-8", errors="ignore")
                    history[-1]["content"] += f"\n[Error: {err}]"
                    yield AdditionalOutputs(history)
                    break
    except GeneratorExit:
        if ws is not None:
            try:
                ws.send(bytes([MessageType.CONTROL, ControlMessage.PAUSE]))
            except Exception:
                pass
        raise
    except Exception as exc:
        history[-1]["content"] += f"\n[连接断开: {exc}]"
        yield AdditionalOutputs(history)
    finally:
        interrupt_controller.set_ws(None)


def clear_history() -> tuple[list[dict[str, str]], None]:
    gr.Info("Cleared chat history", duration=3)
    return [], None


def main() -> None:
    config = StreamConfig()
    interrupt_controller = WSInterruptController(threading.Lock())

    with gr.Blocks() as demo:
        gr.Markdown("# Local Speech Service Chat (Interrupt Enabled)")

        chat_state = gr.State([])
        webrtc = WebRTC(
            modality="audio",
            mode="send-receive",
            full_screen=False,
        )
        text_out = gr.Textbox(
            lines=6,
            label="Output",
        )
        clear_btn = gr.Button("Reset chat")

        webrtc.stream(
            VADStreamHandler(
                lambda audio, _id, history: chat_generator(
                    audio,
                    history,
                    config,
                    interrupt_controller,
                ),
                interrupt_controller,
                input_sample_rate=config.output_sample_rate,
            ),
            inputs=[webrtc, chat_state],
            outputs=[webrtc],
        )
        webrtc.on_additional_outputs(
            lambda s: s,
            outputs=[text_out],
        )
        clear_btn.click(clear_history, outputs=[chat_state, text_out])

    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        ssl_certfile="/egova/cx/liquid-audio/cert.pem",
        ssl_keyfile="/egova/cx/liquid-audio/key.pem",
        ssl_keyfile_password=None,
        ssl_verify=False,
    )


if __name__ == "__main__":
    main()
