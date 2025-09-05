"""
Simple Rigol Scope Interface
============================

Minimal, no-frills interface for Rigol oscilloscopes.
"""

import time
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
                time.sleep(0.002)
            
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