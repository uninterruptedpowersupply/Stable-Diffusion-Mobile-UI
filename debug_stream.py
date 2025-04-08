# debug_stream.py
import sys

class DebugStream:
    def __init__(self, original):
        self.original = original
        self.buffer = []
    def write(self, message):
        # Avoid recursive write if original is also self
        if self.original is not self:
            self.buffer.append(str(message)) # Ensure message is string
            self.original.write(str(message))
    def flush(self):
         if self.original is not self:
            self.original.flush()
    def get_output(self):
        return ''.join(self.buffer)

# Initialize only once
if not isinstance(sys.stdout, DebugStream):
    debug_stdout = DebugStream(sys.stdout)
    sys.stdout = debug_stdout
if not isinstance(sys.stderr, DebugStream):
    debug_stderr = DebugStream(sys.stderr)
    sys.stderr = debug_stderr

def get_debug_output():
    """Safely get output from stdout buffer if it's our stream."""
    if isinstance(sys.stdout, DebugStream):
        return sys.stdout.get_output()
    return "Debug stream not initialized correctly."