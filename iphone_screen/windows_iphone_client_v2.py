"""
windows_iphone_client.py

Windows-side client for the Mac DeviceKit bridge.

Public interface:
    iphone = IPhoneRemote(mac_ip="192.168.1.50")

    frame = iphone.get_screen()
    iphone.send_tap(x, y)          # x/y are pixels in frame returned by get_screen()

    iphone.send_tap_device(x, y)   # logical iOS coordinates, if needed
    iphone.swipe(...)
    iphone.type_text(...)
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
import socket
import threading
import time
from typing import Any, Optional, Tuple

import av
import numpy as np
import requests
import websocket


class IPhoneRemote:
    def __init__(
        self,
        mac_ip: str,
        control_port: int = 22004,
        video_port: int = 22005,
        expected_fps: float = 30.0,
        rpc_timeout: float = 3.0,
        reconnect_delay: float = 0.1,
        auto_start: bool = True,
    ):
        self.mac_ip = mac_ip
        self.control_port = int(control_port)
        self.video_port = int(video_port)
        self.expected_fps = float(expected_fps)
        self.rpc_timeout = float(rpc_timeout)
        self.reconnect_delay = float(reconnect_delay)

        self.ws_url = f"ws://{mac_ip}:{control_port}/ws"
        self.http_rpc_url = f"http://{mac_ip}:{control_port}/rpc"

        self._rpc_lock = threading.Lock()
        self._rpc_id = 0
        self._ws = None
        self._http = requests.Session()

        # Keep input RPC off the OpenCV/video thread. DeviceKit waits for
        # XCTest event completion, which can otherwise visibly freeze the UI.
        self._input_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="iphone-input",
        )

        self._frame_cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_frame_time = 0.0

        self._screen_width: Optional[float] = None
        self._screen_height: Optional[float] = None

        self._stop = threading.Event()
        self._video_thread: Optional[threading.Thread] = None
        self._video_socket: Optional[socket.socket] = None
        self._video_error: Optional[str] = None

        self._connect_ws()
        self._refresh_device_info()

        if auto_start:
            self.start_video()

    # ---------------------------------------------------------------
    # JSON-RPC control
    # ---------------------------------------------------------------

    def _connect_ws(self) -> None:
        with self._rpc_lock:
            self._close_ws_unlocked()

            self._ws = websocket.create_connection(
                self.ws_url,
                timeout=self.rpc_timeout,
                enable_multithread=True,
            )
            self._ws.settimeout(self.rpc_timeout)

    def _close_ws_unlocked(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _next_rpc_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _rpc_ws(self, method: str, params: dict | None = None) -> Any:
        with self._rpc_lock:
            if self._ws is None:
                self._ws = websocket.create_connection(
                    self.ws_url,
                    timeout=self.rpc_timeout,
                    enable_multithread=True,
                )
                self._ws.settimeout(self.rpc_timeout)

            rpc_id = self._next_rpc_id()

            request = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": method,
                "params": params or {},
            }

            self._ws.send(
                json.dumps(
                    request,
                    separators=(",", ":"),
                )
            )

            deadline = time.perf_counter() + self.rpc_timeout

            while True:
                left = deadline - time.perf_counter()

                if left <= 0:
                    raise TimeoutError(f"RPC timeout: {method}")

                self._ws.settimeout(left)

                response = json.loads(self._ws.recv())

                if response.get("id") != rpc_id:
                    continue

                if "error" in response:
                    raise RuntimeError(response["error"])

                return response.get("result")

    def _rpc_http(self, method: str, params: dict | None = None) -> Any:
        rpc_id = self._next_rpc_id()

        response = self._http.post(
            self.http_rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": method,
                "params": params or {},
            },
            timeout=self.rpc_timeout,
        )

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        return data.get("result")

    def rpc(self, method: str, params: dict | None = None) -> Any:
        try:
            return self._rpc_ws(method, params)

        except Exception:
            with self._rpc_lock:
                self._close_ws_unlocked()

            try:
                self._connect_ws()
                return self._rpc_ws(method, params)

            except Exception:
                return self._rpc_http(method, params)

    def _refresh_device_info(self) -> None:
        info = self.rpc("device.info")
        size = info["screenSize"]
        self._screen_width = float(size["width"])
        self._screen_height = float(size["height"])

    # ---------------------------------------------------------------
    # Public control API
    # ---------------------------------------------------------------

    def send_tap(self, x: float, y: float) -> Any:
        """
        Tap using PIXEL coordinates from the frame returned by get_screen().
        """
        frame = self.get_screen(wait_new=False, copy=False)

        frame_h, frame_w = frame.shape[:2]

        screen_w = self._screen_width
        screen_h = self._screen_height

        if screen_w is None or screen_h is None:
            self._refresh_device_info()
            screen_w = self._screen_width
            screen_h = self._screen_height

        assert screen_w is not None
        assert screen_h is not None

        # Handle portrait/landscape.
        if (frame_w > frame_h) != (screen_w > screen_h):
            screen_w, screen_h = screen_h, screen_w

        device_x = float(x) * screen_w / frame_w
        device_y = float(y) * screen_h / frame_h

        return self.send_tap_device(device_x, device_y)

    def send_tap_async(self, x: float, y: float) -> Future:
        """
        Non-blocking tap using PIXEL coordinates from get_screen().

        Returns a Future immediately. Use this from OpenCV callbacks and
        latency-sensitive automation loops so video rendering never waits for
        XCTest tap completion.
        """
        return self._input_executor.submit(
            self.send_tap,
            float(x),
            float(y),
        )

    def send_tap_device(self, x: float, y: float) -> Any:
        """Tap using DeviceKit logical iOS coordinates."""
        return self.rpc(
            "device.io.tap",
            {
                "x": float(x),
                "y": float(y),
            },
        )

    def send_tap_device_async(self, x: float, y: float) -> Future:
        """Non-blocking logical-coordinate tap."""
        return self._input_executor.submit(
            self.send_tap_device,
            float(x),
            float(y),
        )

    def swipe(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        duration: float = 0.1,
    ) -> Any:
        """
        Swipe using logical iOS coordinates.
        """
        return self.rpc(
            "device.io.swipe",
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "duration": float(duration),
            },
        )

    def type_text(self, text: str) -> Any:
        return self.rpc(
            "device.io.text",
            {"text": str(text)},
        )

    # ---------------------------------------------------------------
    # Video
    # ---------------------------------------------------------------

    def start_video(self) -> None:
        if self._video_thread and self._video_thread.is_alive():
            return

        self._stop.clear()

        self._video_thread = threading.Thread(
            target=self._video_worker,
            name="iphone-remote-h264",
            daemon=True,
        )
        self._video_thread.start()

    def _video_worker(self) -> None:
        while not self._stop.is_set():
            sock = None
            file_obj = None
            container = None

            try:
                sock = socket.create_connection(
                    (self.mac_ip, self.video_port),
                    timeout=3.0,
                )

                sock.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )

                sock.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_RCVBUF,
                    256 * 1024,
                )

                sock.settimeout(3.0)
                self._video_socket = sock
                self._video_error = None

                file_obj = sock.makefile("rb", buffering=0)

                container = av.open(
                    file_obj,
                    mode="r",
                    format="h264",
                    options={
                        "fflags": "nobuffer",
                        "probesize": "32768",
                        "analyzeduration": "0",
                    },
                )

                stream = container.streams.video[0]
                stream.thread_type = "NONE"
                stream.thread_count = 1

                # Once decoding has started, allow blocking reads.
                sock.settimeout(None)

                for packet in container.demux(stream):
                    if self._stop.is_set():
                        return

                    for frame in packet.decode():
                        if self._stop.is_set():
                            return

                        image = frame.to_ndarray(format="bgr24")

                        with self._frame_cond:
                            # No queue: replace old frame with newest frame.
                            self._latest_frame = image
                            self._frame_id += 1
                            self._last_frame_time = time.perf_counter()
                            self._frame_cond.notify_all()

                raise ConnectionError("H264 stream ended")

            except Exception as exc:
                self._video_error = f"{type(exc).__name__}: {exc}"

                if not self._stop.is_set():
                    time.sleep(self.reconnect_delay)

            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass

                if file_obj is not None:
                    try:
                        file_obj.close()
                    except Exception:
                        pass

                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

                if self._video_socket is sock:
                    self._video_socket = None

    # ---------------------------------------------------------------
    # Public screen API
    # ---------------------------------------------------------------

    def get_screen(
        self,
        wait_new: bool = True,
        timeout: float = 2.0,
        copy: bool = False,
    ) -> np.ndarray:
        """
        Return an OpenCV BGR ndarray.

        wait_new=True:
            wait for a frame newer than the one seen when this method was
            called. Useful for synchronous automation loops.

        wait_new=False:
            return the newest frame immediately (waiting only if no frame
            has ever arrived).
        """
        deadline = time.perf_counter() + timeout

        with self._frame_cond:
            starting_id = self._frame_id

            while True:
                if self._latest_frame is not None:
                    if not wait_new or self._frame_id > starting_id:
                        return (
                            self._latest_frame.copy()
                            if copy
                            else self._latest_frame
                        )

                left = deadline - time.perf_counter()

                if left <= 0:
                    raise TimeoutError(
                        "No H264 frame received from Mac bridge. "
                        f"Last video error: {self._video_error}"
                    )

                self._frame_cond.wait(left)

    def get_latest_screen(self, copy: bool = False) -> np.ndarray:
        """Return newest available frame without waiting for a newer one."""
        return self.get_screen(
            wait_new=False,
            timeout=2.0,
            copy=copy,
        )

    @property
    def frame_age_ms(self) -> Optional[float]:
        with self._frame_cond:
            t = self._last_frame_time

        if not t:
            return None

        return (time.perf_counter() - t) * 1000.0

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def close(self) -> None:
        self._input_executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        self._stop.set()

        sock = self._video_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass

            try:
                sock.close()
            except Exception:
                pass

        if self._video_thread is not None:
            self._video_thread.join(timeout=2.0)

        with self._rpc_lock:
            self._close_ws_unlocked()

        self._http.close()

    def __enter__(self) -> "IPhoneRemote":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
