#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""This module defines some helper functions to get timestamps file names
from a set of simulation parameters.

You can define the max rate for donor and acceptor, background rates,
simulation IDs, etc ..., and get as a result the filename of donor and
acceptor timestamps.

For an example of all the needed parameters see `pybromo_ts_params_example`.

"""

# This dict defines a unique set of timestamps
# It can be passed to get_bromo_fnames_da() as **kwargs
pybromo_ts_params_example = {
    "d_em_kHz": 20.0,
    "d_bg_kHz": 6,
    "a_em_kHz": 180.0,
    "a_bg_kHz": 3,
    "ID": "1+2+3+4+5+6",
    "t_tot": "480",
    "num_p": "30",
    "pM": "64",
    "t_step": 0.5e-6,
    "D": 1.2e-11,
    "dir_": "sim_timestamps_folder",
}


def get_bromo_fnames_da(
    d_em_kHz,
    d_bg_kHz,
    a_em_kHz,
    a_bg_kHz,
    ID="1+2+3+4+5+6",
    t_tot="480",
    num_p="30",
    pM="64",
    t_step=0.5e-6,
    D=1.2e-11,
    dir_="",
):
    """Get filenames for donor and acceptor timestamps for the given parameters."""
    clk_p = t_step / 32.0  # with t_step=0.5us -> 156.25 ns
    E_sim = 1.0 * a_em_kHz / (a_em_kHz + d_em_kHz)

    FRET_val = 100.0 * E_sim
    print(f"Simulated FRET value: {FRET_val:.1f}%")

    d_em_kHz_str = "%04d" % d_em_kHz
    a_em_kHz_str = "%04d" % a_em_kHz
    d_bg_kHz_str = f"{d_bg_kHz:04.1f}"
    a_bg_kHz_str = f"{a_bg_kHz:04.1f}"

    print(f"D: EM {d_em_kHz_str} BG {d_bg_kHz_str} ")
    print(f"A: EM {a_em_kHz_str} BG {a_bg_kHz_str} ")

    fname_d = (
        f"ph_times_{t_tot}s_D{D}_{num_p}P_{pM}pM_"
        f"step{t_step * 1e6}us_ID{ID}_EM{d_em_kHz_str}kHz_BG{d_bg_kHz_str}kHz.npy"
    )

    fname_a = (
        f"ph_times_{t_tot}s_D{D}_{num_p}P_{pM}pM_"
        f"step{t_step * 1e6}us_ID{ID}_EM{a_em_kHz_str}kHz_BG{a_bg_kHz_str}kHz.npy"
    )
    print(fname_d)
    print(fname_a)

    name = f"BroSim_E{FRET_val:.1f}_dBG{d_bg_kHz:.1f}k_aBG{a_bg_kHz:.1f}k_dEM{d_em_kHz:.0f}k"

    return dir_ + fname_d, dir_ + fname_a, name, clk_p, E_sim
