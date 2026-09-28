"""Skip mock DSP execution while compile3 captures target-specific source and binaries."""
import os

if os.getenv("MOCKDSP_CAPTURE_ONLY") == "1":
  import tinygrad.runtime.ops_dsp as dsp

  def _capture_without_simulator(self, *bufs, **kwargs):
    return 0.0

  dsp.MockDSPProgram.__call__ = _capture_without_simulator
