#!/usr/bin/env python3
"""Recover an unattended LEAP hand: gradual zero return, then verified torque off."""
import sys
from pathlib import Path

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'camera_hand_calibration'))
from shutdown_hand import main

if __name__=='__main__':raise SystemExit(main())
