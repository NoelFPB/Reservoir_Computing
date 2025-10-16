"""
Simple Rigol Scope Interface
============================

Minimal, no-frills interface for Rigol oscilloscopes.
"""
from concurrent.futures import ThreadPoolExecutor
import time, re
import numpy as np
import pyvisa

class RigolScope:
    """Simple Rigol scope interface."""
    
    def __init__(self, channels):
        self.channels = channels
        self._connect()
    
    def _connect(self):
        """Connect to scope."""
        self.rm = pyvisa.ResourceManager()
        addr = self.rm.list_resources()[0]
        print(f"[SCOPE] Connecting to {addr}")
        
        self.scope = self.rm.open_resource(addr)
        self.scope.timeout = 5000
        self.scope.read_termination = '\n'
        self.scope.write_termination = '\n'
        
        for ch in self.channels:
            self.scope.write(f':CHANnel{ch}:DISPlay ON')
            self.scope.write(f':CHANnel{ch}:SCALe 2')
            self.scope.write(f':CHANnel{ch}:OFFSet -6')
        
        time.sleep(0.1)
    
    def read_channel(self, ch):
        """Read single channel."""
        try:
            query = f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{ch}'
            value = float(self.scope.query(query))
            if not np.isnan(value) and -10 <= value <= 15:
                return round(value, 5)
            return np.nan
        except:
            return np.nan
    
    def read_many(self, avg=1):
        """Read all channels with averaging."""
        vals = []
        for ch in self.channels:
            samples = []
            for _ in range(max(1, avg)):
                v = self.read_channel(ch)
                if np.isfinite(v):
                    samples.append(v)
                #time.sleep(0.002)
            
            if samples:
                vals.append(float(np.mean(samples)))
            else:
                vals.append(np.nan)
        
        return np.array(vals, float)
    
    def close(self):
        """Close connections."""
        try:
            if self.scope: self.scope.close()
        except: pass
        try:
            if self.rm: self.rm.close()
        except: pass


_MEAS_QUERY = ':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{ch}'


def _parse_serial(idn: str) -> str:
    parts = [p.strip() for p in idn.split(',')]
    if len(parts) >= 3 and parts[2]:
        return parts[2]
    m = re.search(r'(\d{5,})', idn)
    return m.group(1) if m else idn


class RigolDualScopes:
    """
    Minimal dual-scope version:
      - scope1 is selected by its serial number (serial_scope1)
      - scope2 is whichever other Rigol scope is found
    The readout order is always: scope1 channels → scope2 channels
    """

    def __init__(self, channels_scope1, channels_scope2, serial_scope1=None, timeout_ms=5000):
        self.channels1 = list(channels_scope1)
        self.channels2 = list(channels_scope2)
        self.channels = self.channels1 + self.channels2
        self.serial_scope1 = serial_scope1
        self.timeout_ms = timeout_ms
        self._connect()

    def _connect(self):
        rm = pyvisa.ResourceManager()
        self.rm = rm
        candidates = []

        for addr in rm.list_resources():
            try:
                inst = rm.open_resource(addr)
                inst.timeout = self.timeout_ms
                inst.read_termination = '\n'
                inst.write_termination = '\n'
                idn = inst.query("*IDN?").strip()
                if "RIGOL" in idn.upper() or "HDO" in idn.upper():
                    candidates.append({
                        "addr": addr,
                        "idn": idn,
                        "serial": _parse_serial(idn),
                        "inst": inst
                    })
                else:
                    inst.close()
            except Exception:
                try:
                    if inst:
                        inst.close()
                except Exception:
                    pass

        if len(candidates) < 2:
            raise RuntimeError(f"[SCOPE] Need 2 Rigol scopes; found {len(candidates)}")

        # --- determine scope1 and scope2 ---
        scope1 = None
        if self.serial_scope1:
            for c in candidates:
                if c["serial"] == self.serial_scope1:
                    scope1 = c
                    break
            if not scope1:
                raise RuntimeError(f"[SCOPE] serial_scope1 '{self.serial_scope1}' not found.")
        else:
            # Default: pick the first alphabetically by serial
            candidates.sort(key=lambda x: x["serial"])
            scope1 = candidates[0]

        # The other becomes scope2
        scope2 = [c for c in candidates if c is not scope1][0]

        self.scope1 = scope1["inst"]
        self.scope2 = scope2["inst"]
        self._idn1, self._idn2 = scope1["idn"], scope2["idn"]
        self._ser1, self._ser2 = scope1["serial"], scope2["serial"]

        print(f"[SCOPE1] {scope1['addr']} | {self._idn1} | serial={self._ser1}")
        print(f"[SCOPE2] {scope2['addr']} | {self._idn2} | serial={self._ser2}")

        # Configure channels
        for ch in self.channels1:
            self.scope1.write(f":CHANnel{ch}:DISPlay ON")
            self.scope1.write(f":CHANnel{ch}:SCALe 2")
            self.scope1.write(f":CHANnel{ch}:OFFSet -6")
        for ch in self.channels2:
            self.scope2.write(f":CHANnel{ch}:DISPlay ON")
            self.scope2.write(f":CHANnel{ch}:SCALe 2")
            self.scope2.write(f":CHANnel{ch}:OFFSet -6")

        time.sleep(0.1)

    def _read_channel(self, scope, ch):
        try:
            v = float(scope.query(_MEAS_QUERY.format(ch=ch)))
            return v if -10 <= v <= 15 else np.nan
        except Exception:
            return np.nan

    def _read_scope_channels(self, scope, channels, avg):
            avg = max(1, int(avg))
            out = []
            # Remove per-sample sleep; VISA I/O already blocks until reply
            for ch in channels:
                samples = [self._read_channel(scope, ch) for _ in range(avg)]
                samples = [v for v in samples if np.isfinite(v)]
                out.append(float(np.mean(samples)) if samples else np.nan)
            return out

    def read_many(self, avg=1):
        """
        Returns values in GUARANTEED order:
          [scope1 channels ... , scope2 channels ...]
        Reads both scopes in parallel to cut wall time roughly in half.
        """
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(self._read_scope_channels, self.scope1, self.channels1, avg)
            f2 = ex.submit(self._read_scope_channels, self.scope2, self.channels2, avg)
            v1 = f1.result()
            v2 = f2.result()
        return np.array(v1 + v2, dtype=float)
        
    def close(self):
        for s in (self.scope1, self.scope2):
            try:
                s.close()
            except Exception:
                pass
        try:
            self.rm.close()
        except Exception:
            pass
