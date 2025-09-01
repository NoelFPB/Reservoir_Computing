# Sanity test for Rigol HDO1074
import pyvisa, time

rm = pyvisa.ResourceManager()
addr = rm.list_resources()[0]
scope = rm.open_resource(addr)
scope.read_termination = '\n'
scope.write_termination = '\n'
print("IDN:", scope.query('*IDN?'))
scope.write('*CLS')
scope.write(':RUN')
for ch in (1,2,3,4):
    scope.write(f':CHANnel{ch}:DISPlay ON')
    scope.write(f':CHANnel{ch}:SCALe 2')
    scope.write(f':CHANnel{ch}:OFFSet 0')
    try:
        v = float(scope.query(f':MEASure:ITEM? VAVG,CHANnel{ch}'))
    except Exception:
        v = float(scope.query(f':MEASure:VAVerage? CHANnel{ch}'))
    print(f"CH{ch} VAVG = {v:.3f} V")
scope.close()
rm.close()
