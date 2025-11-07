# sweep_all_heaters.py
import time
import numpy as np
from Lib.heater_bus import HeaterBus

def main():
    bus = HeaterBus()

    heaters = list(range(30))             # modify if your board has fewer heaters
    v_values = np.arange(0.0, 5.0, 0.1)


    for v in v_values:
        cmd = {h: float(v) for h in heaters}
        bus.send(cmd)
        print(f"Set all heaters → {v:.2f} V")
        time.sleep(0.2)               # settle time, adjust if needed

    cmd = {h: 4.8 for h in heaters}
    bus.send(cmd)
    time.sleep(1)
    cmd = {h: 4.95 for h in heaters}
    bus.send(cmd)
    
if __name__ == "__main__":
    main()
