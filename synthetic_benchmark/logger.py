import os
import errno
import fcntl
import threading
from multiprocessing import current_process

class FlytLogger:
    def __init__(self, pid=None):
        self.pid = pid or os.getpid()
        self.fifo_name = "/tmp/flyt_shmem.log"
        self.send = 0
        self.function_names = set()
        
        # Open and initialize shared memory
        self.init_shared_memory()
        print("Initializing thread")
        
        # Register the application
        command = self.format_start_command(self.pid, 1)
        self.write_message(command)
    
    def __del__(self):
        # Unmap and close the shared memory
        command = self.format_stop_command(self.pid)
        self.write_message(command)
        self.close_shared_memory()

    # Log a function's metrics
    def writelog(self, function_name, metrics, upperlimit, lowerlimit, spike):
        # Register a function if needed
        if self.fifo_fd == -1:
            self.pid = os.getpid()
            self.init_shared_memory()
            if self.send == 0:
                command = self.format_start_command(self.pid, 1)
                self.write_message(command)
                self.send = 1

        print("logging")
        if self.add_function_name(function_name):
            self.log_function_name(function_name, upperlimit, lowerlimit, spike)

        command = self.format_log_command(self.pid, function_name, metrics)
        print("writing")
        self.write_message(command)
        self.close_shared_memory()

    def init_shared_memory(self):
        try:
            self.fifo_fd = os.open(self.fifo_name, os.O_WRONLY)
        except OSError as e:
            if e.errno == errno.ENOENT:
                print(f"Shared memory file not found: {self.fifo_name}")
            else:
                raise RuntimeError("Failed to open shared memory") from e

    def close_shared_memory(self):
        if hasattr(self, 'fifo_fd'):
            os.close(self.fifo_fd)
            self.fifo_fd = -1

    # Method to add a function name if it's unique
    def add_function_name(self, function_name):
        if function_name in self.function_names:
            return False
        self.function_names.add(function_name)
        return True

    def log_function_name(self, function_name, upperlimit, lowerlimit, count):
        command = self.format_fn_init_command(self.pid, function_name, upperlimit, lowerlimit, count)
        self.write_message(command)

    def write_message(self, message):
        try:
            os.write(self.fifo_fd, message.encode('utf-8'))
        except OSError as e:
            print(f"Failed to write to fifo: {e}")
            raise RuntimeError("Failed to write to fifo") from e

    @staticmethod
    def format_start_command(pid, count):
        return f"START {pid} {count}\n"

    @staticmethod
    def format_fn_init_command(pid, function_name, upper_threshold, lower_threshold, spike_count):
        return f"FN_INIT {pid} {function_name} {upper_threshold} {lower_threshold} {spike_count}\n"

    @staticmethod
    def format_log_command(pid, function_name, metrics):
        return f"LOG {pid} {function_name} {metrics}\n"

    @staticmethod
    def format_stop_command(pid):
        return f"STOP {pid}\n"


# Example usage
if __name__ == "__main__":
    logger = FlytLogger()

    # Logging a function's details
    logger.writelog("test_function", metrics=100, upperlimit=200, lowerlimit=50, spike=10)

