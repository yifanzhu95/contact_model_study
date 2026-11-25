"""Concise fault labels for the experiment terminal; full tracebacks stay in logs."""
from pathlib import Path
import re

# Native motor IDs, not MuJoCo q-vector order.
NATIVE_JOINTS = ('index rotation', 'index MCP', 'index PIP', 'index DIP',
                'middle rotation', 'middle MCP', 'middle PIP', 'middle DIP',
                'ring rotation', 'ring MCP', 'ring PIP', 'ring DIP',
                'thumb CMC', 'thumb axial', 'thumb MCP', 'thumb IPL')


def tail(path, length=32768):
    try:
        with Path(path).open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-length))
            return f.read().decode(errors='replace')
    except OSError:
        return ''


def short_failure(directory, reason):
    directory=Path(directory)
    bridge=tail(directory/'bridge.log')
    # Motor/USB failures are the root cause when the controller subsequently
    # reports stale feedback. Never replace them with that secondary traceback.
    current=re.findall(r'Measured current exceeds hardware limit: ([^\n;]+) raw; allowed \+/-([\d.]+)',bridge)
    if current:
        readings,limit=current[-1]
        details=[]
        for motor,value in re.findall(r'motor (\d+)=(-?[\d.]+)',readings):
            i=int(motor)
            name=NATIVE_JOINTS[i] if 0<=i<16 else 'unknown joint'
            details.append(f'motor {i} ({name}) current {value}')
        cleanup=('torque disabled and verified.' if 'Torque disabled and verified on all 16 motors.' in bridge
                 else 'torque-off requested.')
        return f'MOTOR FAULT: {", ".join(details)} exceeds +/-{limit} raw; '+cleanup
    serial=re.findall(r'(?:OSError|RuntimeError): ([^\n]*(?:Dynamixel failure|Fresh feedback missing|feedback device error|temperature limit|hardware error|torque disable unconfirmed)[^\n]*)',bridge)
    if serial:
        return 'MOTOR/USB FAULT: '+serial[-1][:180]
    sources=sorted(['bridge','controller','tracker','viewer','mujoco','mask','sam','container'],key=lambda name:name not in reason.lower())
    for name in sources:
        if name not in reason and name!='controller':continue
        text=tail(directory/(name+'.log'))
        errors=re.findall(r'^(\w*(?:Error|Exception)): (.+)$',text,re.MULTILINE)
        if errors:
            kind, message=errors[-1]
            return f'{name.upper()} FAILURE: {kind}: {message[:180]}'
    return 'EXPERIMENT FAILURE: '+str(reason)[:200]
