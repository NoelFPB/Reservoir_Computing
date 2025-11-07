# set_one_channel.py
import time
import DualBoard

# ================== EDIT THESE ==================
CHANNEL = 72      # logical channel (1–80)
VOLTAGE = 2.50    # volts (0.0–5.0 recommended)
# ================================================

# Init DAC
sdp = DualBoard.DualAD5380Controller()

# Set the voltage
print(f"Setting channel {CHANNEL} -> {VOLTAGE:.3f} V ...")
sdp.set(CHANNEL, VOLTAGE)

# small settle delay (optional)
time.sleep(0.1)
print("Done.")
