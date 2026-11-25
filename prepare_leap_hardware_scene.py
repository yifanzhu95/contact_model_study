#!/usr/bin/env python3
"""Rebuild calibrated hand XML and its cube scene from the supplied archive."""
import argparse

from ContactModelStudy.Hardware.Scene import (
    DEFAULT_ARCHIVE, DEFAULT_HAND_SOURCE, DEFAULT_SCENE, prepare_scene,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--hand-source", default=DEFAULT_HAND_SOURCE)
    parser.add_argument("--output", default=DEFAULT_SCENE)
    parser.add_argument("--cube-size", type=float, default=0.07, help="cube side length in meters")
    parser.add_argument("--cube-mass", type=float, default=0.07, help="cube mass in kilograms")
    args = parser.parse_args()
    print(prepare_scene(args.archive, args.hand_source, args.output, args.cube_size, args.cube_mass))


if __name__ == "__main__":
    main()
