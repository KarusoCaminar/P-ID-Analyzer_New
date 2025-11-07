"""
Live Log Monitor - Monitors log files in real-time during test execution.

This module provides functionality to monitor log files and output them
to the console in real-time, allowing for live monitoring of test progress.
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Callable, List
from queue import Queue, Empty


class LiveLogMonitor:
    """
    Monitors a log file and outputs new lines in real-time.
    
    Uses a background thread to read the log file and a queue to
    communicate with the main thread for thread-safe output.
    """
    
    def __init__(
        self,
        log_file: Path,
        output_callback: Optional[Callable[[str], None]] = None,
        poll_interval: float = 0.1
    ):
        """
        Initialize Live Log Monitor.
        
        Args:
            log_file: Path to log file to monitor
            output_callback: Optional callback function to call for each new line
                             (default: print to stdout)
            poll_interval: Interval in seconds to check for new lines (default: 0.1s)
        """
        self.log_file = Path(log_file)
        self.output_callback = output_callback or self._default_output
        self.poll_interval = poll_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.line_queue = Queue()
        self.last_position = 0
        
    def _default_output(self, line: str) -> None:
        """Default output callback - prints to stdout."""
        print(line, end='', flush=True)
    
    def start(self) -> None:
        """Start monitoring the log file in a background thread."""
        if self.running:
            return
        
        if not self.log_file.exists():
            # Create empty file if it doesn't exist
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.touch()
        
        self.running = True
        self.last_position = 0
        
        # Start background thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="LiveLogMonitor"
        )
        self.monitor_thread.start()
        
    def stop(self) -> None:
        """Stop monitoring the log file."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self) -> None:
        """Background thread loop that monitors the log file."""
        while self.running:
            try:
                if not self.log_file.exists():
                    time.sleep(self.poll_interval)
                    continue
                
                # Read new lines from file
                with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                    # Seek to last known position
                    f.seek(self.last_position)
                    
                    # Read all new lines
                    new_lines = f.readlines()
                    if new_lines:
                        # Update position
                        self.last_position = f.tell()
                        
                        # Output each new line
                        for line in new_lines:
                            self.output_callback(line)
                
                # Sleep before next check
                time.sleep(self.poll_interval)
                
            except Exception as e:
                # Log error but continue monitoring
                print(f"[LiveLogMonitor] Error: {e}", file=sys.stderr, flush=True)
                time.sleep(self.poll_interval)
    
    def flush(self) -> None:
        """Flush any remaining log lines."""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(self.last_position)
                remaining_lines = f.readlines()
                if remaining_lines:
                    for line in remaining_lines:
                        self.output_callback(line)
                    self.last_position = f.tell()
        except Exception as e:
            print(f"[LiveLogMonitor] Error flushing: {e}", file=sys.stderr, flush=True)


def monitor_test_logs(
    log_file: Path,
    test_name: str,
    output_callback: Optional[Callable[[str], None]] = None
) -> LiveLogMonitor:
    """
    Convenience function to start monitoring a test log file.
    
    Args:
        log_file: Path to log file to monitor
        test_name: Name of the test (for display)
        output_callback: Optional callback function (default: print with prefix)
        
    Returns:
        LiveLogMonitor instance (call .stop() when done)
    """
    if output_callback is None:
        def prefixed_output(line: str) -> None:
            # Add test name prefix for clarity
            print(f"[{test_name}] {line}", end='', flush=True)
        output_callback = prefixed_output
    
    monitor = LiveLogMonitor(log_file, output_callback)
    monitor.start()
    
    return monitor

