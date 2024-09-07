#!/usr/bin/env python3

def recommend_md_timestep(T, hmu, volume, factor=80):
    """
    Recommend a timestep (fs) for molecular dynamics simulation.
    """
    kB = 8.617330337217213e-05
    v2 = T * kB * 3 / hmu
    v2 = v2 * 1.60217662 / 1.66053906660 / 100
    v = v2**0.5
    radius = (volume * 3 / 4 / 3.14159265359)**(1/3)
    t = 2 * radius / v
    return t / factor
