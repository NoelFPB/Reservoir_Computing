import time
import pyvisa


class LaserSource:
    def __init__(
        self,
        address: str,
        timeout_ms: int = 5000,
        write_termination: str = "",
        read_termination: str = "",
        auto_idn: bool = False,
    ):

        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(address)

        # Basic VISA configuration
        self.inst.timeout = timeout_ms
        self.inst.write_termination = write_termination
        self.inst.read_termination = read_termination

        time.sleep(0.2)

        if auto_idn:
            try:
                idn = self.query("*IDN?")
                print(f"[LaserSource] Connected to: {idn.strip()}")
            except Exception as e:
                print(f"[LaserSource] Warning: could not query IDN: {e}")

    # ---------- Low-level helpers ----------

    def write(self, cmd: str) -> None:
        """Send a raw command string to the laser."""
        # print(f"[LaserSource] >> {cmd}")  # uncomment for debugging
        self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        """Send a command and read back response."""
        # print(f"[LaserSource] ?? {cmd}")  # uncomment for debugging
        return self.inst.query(cmd)

    # ---------- High-level operations ----------

    def set_wavelength(self, wavelength_nm: float, settle: float = 0.5) -> None:
        cmd = f"LW{wavelength_nm}nm"
        self.write(cmd)
        if settle > 0:
            time.sleep(settle)

    def turn_on(self, settle: float = 0.5) -> None:
        """
        Turn the laser emission ON.
        """
        self.write("LE1")
        if settle > 0:
            time.sleep(settle)

    def turn_off(self, settle: float = 0.5) -> None:
        """
        Turn the laser emission OFF.
        """
        self.write("LE0")
        if settle > 0:
            time.sleep(settle)

    def close(self) -> None:
        """
        Close the VISA resource and resource manager.
        """
        try:
            # Ensure laser is off before closing, if you want:
            # self.turn_off(settle=0.0)
            self.inst.close()
        finally:
            self.rm.close()

    # ---------- Context manager support ----------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
