"""Gradual, screened empty-hand zero return for the Method 5 hold process."""
import time
import numpy as np
from articulated_motion import screen
from ContactModelStudy.Hardware.Episodes import HardwareEpisodes


def return_to_zero(controller):
    if not (controller.run/'object_removed.trigger').exists():
        raise RuntimeError('Cube-removal confirmation is required before returning to zero')
    q,_=controller.cycle();start=controller.command.copy();zero=np.zeros(16)
    command_screen=screen(controller.m,[start,zero],controller.pairs,controller.config['collision_margin_m'])
    measured_screen=screen(controller.m,[q,zero],controller.pairs,controller.config['collision_margin_m'])
    controller.record['zero_return']=dict(command_path=command_screen,measured_path=measured_screen,settled=False)
    if not command_screen['passed'] or not measured_screen['passed']:
        raise RuntimeError('Zero-return path failed the fresh collision screen; refusing motion')
    controller.status('returning_zero','Cube removed. Closing viewer and gradually returning the hand to zero before torque off.')
    controller.move(start,zero)
    settings=HardwareEpisodes(None,controller.hardware,.3);end=time.monotonic()+20;quiet=None
    while time.monotonic()<end:
        q,v=controller.cycle();now=time.monotonic()
        reached=np.all(abs(q)<=settings.position_tolerances()) and max(abs(v))<=settings.settle_velocity
        quiet=now if reached and quiet is None else quiet if reached else None
        if quiet is not None and now-quiet>=settings.settle_time:
            controller.record['zero_return'].update(settled=True,final_measured_q=q.tolist());return
    controller.record['zero_return']['final_measured_q']=q.tolist()
    raise RuntimeError('Zero was commanded but did not settle within tolerance in 20 seconds')
