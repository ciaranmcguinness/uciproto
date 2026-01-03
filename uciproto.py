#!/usr/bin/env python3
"""
chess_uci.py — simple UCI interface for python-chess

Requirements:
    pip install python-chess

How to use as a standalone engine (stdin/stdout UCI):
    python chess_uci.py

How to use by importing and supplying a search function:

    import chess
    from chess_uci import UCIEngine

    def my_search(board, limits, stop_event, info_callback, options=None):
        # board: chess.Board()
        # limits: dict with keys like 'wtime','btime','winc','binc','movetime','depth','nodes','mate'
        # stop_event: threading.Event that is set when the engine should stop thinking
        # info_callback: function(info_dict) -> None for sending UCI "info" lines (optional)
        # options: dict of engine options (set via 'setoption')
        #
        # Return a chess.Move or UCI string for the best move.
        return chess.Move.from_uci("e2e4")

    engine = UCIEngine(search_fn=my_search, name="MyEngine", author="Me")
    engine.run()  # blocking; reads UCI commands from stdin/stdout

Notes:
- You must provide your own search / evaluation code. The default search_fn raises NotImplementedError.
- info_callback can be used to send periodic "info" updates (score, depth, nodes, nps) via UCI.
"""

from __future__ import annotations
import sys
import threading
import time
import traceback
from typing import Callable, Optional, Any, Dict

import chess

# Types
# search_fn signature: (board, limits, stop_event, info_callback, options) -> chess.Move or UCI string
SearchFnType = Callable[[chess.Board, Dict[str, Any], threading.Event, Optional[Callable[[Dict[str, Any]], None]], Dict[str, Any]], Any]


class UCIOption:
    """Represents a UCI engine option with proper type handling and validation."""
    
    def __init__(self, name: str, opt_type: str, default=None, min_val=None, max_val=None, var_list=None):
        """
        Create a UCI option.
        
        Args:
            name: Option name
            opt_type: Type of option: 'check', 'spin', 'combo', 'string', or 'button'
            default: Default value (will be converted to proper type)
            min_val: Minimum value (for 'spin' type)
            max_val: Maximum value (for 'spin' type)
            var_list: List of valid values (for 'combo' type)
        """
        self.name = name
        self.opt_type = opt_type
        self.min_val = min_val
        self.max_val = max_val
        self.var_list = var_list or []
        
        # Set and validate default
        self.default = self._convert_and_validate(default, is_default=True)
        # Current value starts as the default
        self.current = self.default
    
    def _convert_and_validate(self, value, is_default=False):
        """Convert value to proper type and validate."""
        if value is None:
            if self.opt_type == 'button':
                return None
            elif self.opt_type == 'check':
                return False
            elif self.opt_type == 'spin':
                return 0 if self.min_val is None else self.min_val
            elif self.opt_type in ('combo', 'string'):
                return ""
        
        if self.opt_type == 'button':
            return None
        elif self.opt_type == 'check':
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return bool(value)
        elif self.opt_type == 'spin':
            try:
                val = int(value)
                if self.min_val is not None and val < self.min_val:
                    if is_default:
                        return self.min_val
                    raise ValueError(f"Value {val} below minimum {self.min_val}")
                if self.max_val is not None and val > self.max_val:
                    if is_default:
                        return self.max_val
                    raise ValueError(f"Value {val} above maximum {self.max_val}")
                return val
            except (ValueError, TypeError):
                raise ValueError(f"Expected integer for spin option, got {value}")
        elif self.opt_type == 'combo':
            val = str(value)
            if self.var_list and val not in self.var_list:
                raise ValueError(f"Value '{val}' not in combo options: {self.var_list}")
            return val
        else:  # string
            return str(value) if value is not None else ""
    
    def set_value(self, value):
        """Set the current value with validation."""
        self.current = self._convert_and_validate(value, is_default=False)
    
    def to_option_command(self) -> str:
        """Generate the UCI option command for this option."""
        parts = [f"option name {self.name} type {self.opt_type}"]
        
        if self.opt_type != 'button':
            parts.append(f"default {self.default}")
        
        if self.opt_type == 'spin':
            if self.min_val is not None:
                parts.append(f"min {self.min_val}")
            if self.max_val is not None:
                parts.append(f"max {self.max_val}")
        elif self.opt_type == 'combo':
            for var in self.var_list:
                parts.append(f"var {var}")
        
        return " ".join(parts)


class UCIEngine:
    def __init__(self, search_fn: Optional[SearchFnType] = None, name: str = "PythonUCIEngine", author: str = "Author", logger: Optional[Callable[[str], None]] = None, FrontendTimer = True):
        """
        Create a UCIEngine.

        search_fn signature: (board, limits, stop_event, info_callback, options) -> chess.Move or UCI string
            - board: chess.Board (current position)
            - limits: dict describing search limits (see parse_go)
            - stop_event: threading.Event set when the search should stop
            - info_callback: function(info_dict) to send periodic info lines; optional
            - options: dict of engine options (as set via `setoption`)

        If search_fn is None, the engine will raise NotImplementedError on 'go'.
        """
        self.name = name
        self.author = author
        self.search_fn = search_fn or self._default_search_fn
        self.logger = logger or (lambda s: None)
        self.board = chess.Board()
        self.options: Dict[str, UCIOption] = {}
        self.frontend_timer = False
        self._timer_thread: Optional[threading.Thread] = None
        self._timer_cancel_event: Optional[threading.Event] = None
        self.search_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.ponder = False
        self.ponder_move = None
        self._search_lock = threading.Lock()
        self._last_info_time = 0.0

    def _default_search_fn(self, board, limits, stop_event, info_callback=None, options: Optional[Dict[str, Any]] = None):
        raise NotImplementedError("No search function provided. Set search_fn when creating UCIEngine.")

    def _send(self, line: str):
        # send line to stdout (UCI expects newline-terminated lines)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.logger(f"-> {line}")

    def _send_info(self, info: Dict[str, Any]):
        """
        Send an 'info' line to the GUI. info is a dict (e.g. {'score': {'cp': 12}, 'depth': 5, 'nodes': 12345})
        This helper converts it into a simple UCI info string. You can extend it as needed.
        """
        parts = ["info"]
        # depth
        if "depth" in info:
            parts += ["depth", str(info["depth"])]
        if "seldepth" in info:
            parts += ["seldepth", str(info["seldepth"])]
        if "score" in info:
            sc = info["score"]
            if isinstance(sc, dict):
                if "cp" in sc:
                    parts += ["score", "cp", str(sc["cp"])]
                elif "mate" in sc:
                    parts += ["score", "mate", str(sc["mate"])]
            else:
                parts += ["score", str(sc)]
        if "nodes" in info:
            parts += ["nodes", str(info["nodes"])]
        if "nps" in info:
            parts += ["nps", str(info["nps"])]
        if "pv" in info:
            # pv as list of moves or string
            pv = info["pv"]
            if isinstance(pv, (list, tuple)):
                pvstr = " ".join(m.uci() if isinstance(m, chess.Move) else str(m) for m in pv)
            else:
                pvstr = str(pv)
            parts += ["pv", pvstr]
        # time
        if "time" in info:
            parts += ["time", str(int(info["time"] * 1000))]
        if "debug" in info:
            parts += ["string", str(info["debug"])]
        self._send(" ".join(parts))

    def run(self):
        """
        Main loop: read UCI commands from stdin and respond.
        Blocks until 'quit' is received.
        """
        try:
            while True:
                line = sys.stdin.readline()
                if line == "":
                    # EOF
                    break
                line = line.strip()
                if not line:
                    continue
                self.logger(f"<- {line}")
                try:
                    stop = self._handle_line(line)
                    if stop:
                        break
                except Exception:
                    traceback.print_exc(file=sys.stderr)
        finally:
            # Ensure any running search thread is stopped
            self._stop_search(wait=True)

    def _handle_line(self, line: str) -> bool:
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "isready":
            self._cmd_isready()
        elif cmd == "setoption":
            self._cmd_setoption(tokens[1:])
        elif cmd == "ucinewgame":
            self._cmd_ucinewgame()
        elif cmd == "position":
            self._cmd_position(tokens[1:])
        elif cmd == "go":
            self._cmd_go(tokens[1:])
        elif cmd == "stop":
            self._cmd_stop()
        elif cmd == "ponderhit":
            self._cmd_ponderhit()
        elif cmd == "quit":
            self._cmd_quit()
            return True
        else:
            # Unknown or ignored commands (e.g., debug, perft, register)
            self.logger(f"Unknown command: {line}")
        return False

    def _cmd_uci(self):
        self._send(f"id name {self.name}")
        self._send(f"id author {self.author}")
        # Send all registered options
        for opt_name, opt_obj in self.options.items():
            self._send(opt_obj.to_option_command())
        self._send("uciok")

    def _cmd_isready(self):
        # If engine requires setup, do it here.
        self._send("readyok")

    def _cmd_setoption(self, tokens):
        # tokens like: ['name', 'Hash', 'value', '128']
        # Parse option name and value
        name = None
        value = None
        i = 0
        while i < len(tokens):
            if tokens[i] == "name":
                i += 1
                start = i
                while i < len(tokens) and tokens[i] != "value":
                    i += 1
                name = " ".join(tokens[start:i])
            elif tokens[i] == "value":
                i += 1
                value = " ".join(tokens[i:])
                break
            else:
                i += 1
        
        if name is None:
            return
        
        # If option is registered, use its validation; otherwise create a string option
        if name in self.options:
            try:
                self.options[name].set_value(value)
            except ValueError as e:
                self.logger(f"Invalid value for option {name}: {e}")
        else:
            # Create a default string option if not registered
            opt = UCIOption(name, "string", default=value)
            self.options[name] = opt
        
        # Call set_option hook for custom handling
        try:
            self.options[name].set_value(value)
            self.logger(f"Set option {name} to {self.options[name].current}")
        except Exception:
            traceback.print_exc(file=sys.stderr)
    
    def register_option(self, name: str, opt_type: str, default=None, min_val=None, max_val=None, var_list=None):
        """Register a UCI option with the engine before uci command is sent."""
        self.options[name] = UCIOption(name, opt_type, default, min_val, max_val, var_list)
    
    def get_option(self, name: str):
        """Get the current value of an option, or None if not registered."""
        if name in self.options:
            return self.options[name].current
        return None

    def _maybe_request_frontend_timer(self, limits: Dict[str, Any]):
        """If `FrontendTimerOverride` is enabled and the limits are
        compatible, start an internal timer that will set `stop_event`
        when the allocated time elapses.

        Compatibility: must not specify `mate`, `depth` or `nodes`.
        Time allocation policy (simple):
          - if `movetime` present, use it (milliseconds)
          - else if side has `wtime`/`btime`, allocate base_time/movestogo + increment
            (uses `movestogo` if present, otherwise assumes 40 moves remaining)
        """
        if not self.frontend_timer:
            return
        # incompatible limit types
        if any(k in limits for k in ("searchmoves", "mate", "depth", "nodes")):
            return
        # determine milliseconds to wait
        ms = None
        if "movetime" in limits:
            try:
                ms = int(limits["movetime"])
            except Exception:
                ms = None
        else:
            side_key = "wtime" if self.board.turn == chess.WHITE else "btime"
            inc_key = "winc" if self.board.turn == chess.WHITE else "binc"
            if side_key in limits:
                try:
                    base = int(limits[side_key])
                    inc = int(limits.get(inc_key, 0))
                    movestogo = int(limits.get("movestogo", 40) or 40)
                    ms = max(1, int(base / movestogo) + int(inc))
                except Exception:
                    ms = None
        if ms is None:
            return

        # Cancel any existing timer
        try:
            if self._timer_cancel_event is not None:
                self._timer_cancel_event.set()
            # Create a new cancel event and start the timer thread
            cancel_evt = threading.Event()
            self._timer_cancel_event = cancel_evt

            def _timer_worker(wait_ms: int, cancel_event: threading.Event):
                # Wait until either cancelled or time elapses
                if not cancel_event.wait(wait_ms / 1000.0):
                    # Time expired; signal the search to stop
                    try:
                        self.logger(f"engine timer expired after {wait_ms} ms")
                        self.stop_event.set()
                        self._send("info string engine_timer_expired")
                    except Exception:
                        traceback.print_exc(file=sys.stderr)

            t = threading.Thread(target=_timer_worker, args=(ms, cancel_evt), daemon=True)
            self._timer_thread = t
            t.start()
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def _cmd_ucinewgame(self):
        # Reset internal state if needed
        self.board.reset()
        self.stop_event.set()
        self._stop_search(wait=True)
        self.stop_event.clear()

    def _cmd_position(self, tokens):
        # tokens: ["startpos", "moves", ...] or ["fen", ...]
        if not tokens:
            return
        i = 0
        if tokens[0] == "startpos":
            self.board = chess.Board()
            i = 1
        elif tokens[0] == "fen":
            # fen is next 6 tokens
            fen = " ".join(tokens[1:7])
            self.board = chess.Board(fen)
            i = 7
        # optional "moves"
        if i < len(tokens) and tokens[i] == "moves":
            i += 1
            while i < len(tokens):
                mv = tokens[i]
                try:
                    move = chess.Move.from_uci(mv)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        # try SAN? but UCI 'position' uses uci moves
                        self.logger(f"Illegal move in position: {mv}")
                        # still apply to allow continued input (some GUIs may send illegal)
                        self.board.push(move)
                except Exception:
                    self.logger(f"Failed to parse move in position: {mv}")
                i += 1

    def _parse_go(self, tokens):
        """
        Parse tokens of 'go' and return a limits dict.
        Supported tokens: wtime btime winc binc movestogo depth nodes mate movetime ponder
        """
        limits: Dict[str, Any] = {}
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ("wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "mate", "movetime"):
                if i + 1 < len(tokens):
                    try:
                        limits[t] = int(tokens[i + 1])
                    except ValueError:
                        limits[t] = tokens[i + 1]
                i += 2
            elif t == "infinite":
                limits["infinite"] = True
                i += 1
            elif t == "ponder":
                limits["ponder"] = True
                i += 1
            else:
                i += 1
        # Time-based fields (e.g. 'wtime', 'btime', 'movetime') are provided
        # as raw milliseconds per the UCI protocol. We do not convert them
        # here so that either the engine's search function or an external
        # frontend timer (if requested) can handle allocation policies.
        return limits

    def _cmd_go(self, tokens):
        limits = self._parse_go(tokens)
        self.ponder = bool(limits.get("ponder", False))
        # Start search thread (internal timer will be started from _start_search)
        self._start_search(limits)

    def _start_search(self, limits: Dict[str, Any]):
        with self._search_lock:
            # stop any existing search
            self._stop_search(wait=True)
            self.stop_event.clear()
            self.search_thread = threading.Thread(target=self._search_worker, args=(limits,), daemon=True)
            self.search_thread.start()
            # Start internal engine timer if requested and limits are compatible
            try:
                self._maybe_request_frontend_timer(limits)
            except Exception:
                traceback.print_exc(file=sys.stderr)

    def _stop_search(self, wait: bool):
        # Signal stop and optionally wait for thread to finish
        self.stop_event.set()
        # Cancel any running internal timer
        try:
            if self._timer_cancel_event is not None:
                self._timer_cancel_event.set()
            if self._timer_thread is not None:
                tt = self._timer_thread
                self._timer_thread = None
                tt.join(timeout=1.0)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        th = self.search_thread
        self.search_thread = None
        if th is not None and wait:
            th.join(timeout=5.0)

    def _cmd_stop(self):
        self._stop_search(wait=True)

    def _cmd_ponderhit(self):
        # When GUI says ponderhit, continue with previously pondered move
        # We'll just clear ponder flag in this simple implementation
        self.ponder = False

    def _cmd_quit(self):
        self._stop_search(wait=True)
        # exit run() loop by returning True in handler

    def _search_worker(self, limits: Dict[str, Any]):
        """
        Runs in a separate thread. Calls self.search_fn and sends bestmove line when done.
        """
        try:
            # Provide info_callback to allow search function to send periodic info lines
            def info_callback(info: Dict[str, Any]):
                # Throttle info messages to avoid spamming (e.g., at most 10 per second)
                now = time.time()
                if now - self._last_info_time >= 0.05:  # ~20Hz max
                    try:
                        self._send_info(info)
                    finally:
                        self._last_info_time = now

            # Build a dict of current option values (not UCIOption objects)
            current_options = {name: opt.current for name, opt in self.options.items()}
            
            # Call the search function, passing current engine options as a dict
            result = self.search_fn(self.board.copy(stack=False), limits, self.stop_event, info_callback, current_options)
            if result is None:
                # No move found (should not happen), return a legal move if any
                try:
                    move = next(iter(self.board.legal_moves))
                    best = move.uci()
                except StopIteration:
                    best = "(none)"
            else:
                if isinstance(result, chess.Move):
                    best = result.uci()
                else:
                    best = str(result)
            # If ponder requested, search_fn may return a tuple or set self.ponder_move
            ponder = None
            if isinstance(result, tuple) and len(result) >= 2:
                best = str(result[0])
                ponder = str(result[1])
            elif self.ponder_move:
                ponder = self.ponder_move

            if ponder:
                self._send(f"bestmove {best} ponder {ponder}")
            else:
                self._send(f"bestmove {best}")
        except Exception as e:
            # If an exception happens in the search, notify the GUI (debug via stderr) and send a bestmove if possible
            traceback.print_exc(file=sys.stderr)
            try:
                move = next(iter(self.board.legal_moves))
                self._send(f"bestmove {move.uci()}")
            except StopIteration:
                self._send("bestmove (none)")

    # Convenience alias
    start = run
