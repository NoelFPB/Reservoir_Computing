import time, serial

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200


class HeaterBus:
    """Serial sender for 'heater,value;...\\n' strings."""
    def __init__(self):
        
        print(f"Connecting to serial port {SERIAL_PORT}...")
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    # def send(self, config):
    #       # Create a new dictionary with standard Python floats for printing
    #     # printable_config = {
    #     #     heater: float(value) for heater, value in config.items()
    #     # }
    #     # print(printable_config)
        
    #     voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
    #     self.ser.write(voltage_message.encode())
    #     self.ser.flush()
    #     time.sleep(0.01)
    #     self.ser.reset_input_buffer()
    #     self.ser.reset_output_buffer()

    def send(self, config):
        # Accept dict OR (heaters, values) tuple
        if isinstance(config, dict):
            items = config.items()
        elif isinstance(config, tuple) and len(config) == 2:
            heaters, values = config
            items = zip(heaters, values)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        voltage_message = "".join(f"{heater},{float(value):.3f};" for heater, value in items) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        # NOTE: the sleeps and buffer resets here may be slowing you down
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()


    def close(self):
        try: 
            self.ser.close()
        except: 
            pass
