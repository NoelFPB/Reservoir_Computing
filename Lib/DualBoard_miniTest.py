# set_multi_channels.py
import time
import DualBoard
import numpy as np

bus = DualBoard.DualAD5380Controller()

# small batch of channels (keeps the device happy)
#voltages = np.linspace(1.2, 4.6, 40)   # example ramp
#channels = list(range(40))
#payload = {ch: float(v) for ch, v in zip(channels, voltages)}

bus.set(1,3.5)
time.sleep(0.5)
print("Done.")
bus.set(1,2.5)
time.sleep(0.1)
print("Done.")

