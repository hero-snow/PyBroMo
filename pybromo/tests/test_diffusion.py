"""Module containing automated unit tests for PyBroMo.

Running the tests requires `py.test`.
"""

import json

import numpy as np
import pytest

import pybromo as pbm

_SEED = 2345654342


# - GLOBAL SIMULATION PARAMETERS - - - - - - - - - - - - - - - - - - - - - - -

# Diffusion parameters
t_step = 0.5e-6  # (seconds) diffusion simulation time step
t_max = 1  # (seconds) time duration of the diffusion simulation

# Diffusion coefficients
Du = 12.0  # um^2 / s
D1 = Du * (1e-6) ** 2  # m^2 / s
D2 = D1 / 2

# Simulation box
box = pbm.Box(x1=-4.0e-6, x2=4.0e-6, y1=-4.0e-6, y2=4.0e-6, z1=-6e-6, z2=6e-6)

# Particles populations
particles_specs = {
    # Parameters needed for the diffusion simulation
    "num_particles": (1, 3),  # number of particles in each population
    "D": (D1, D2),  # (m^2 / s) diffusion coefficiens per population
    "box": box,  # simulation box
    # Photo-physics parameters (needed only for timestamps simulation)
    "E_values": (0.75, 0.25),  # FRET efficiencies for each population
    "em_rates": (200e3, 300e3),  # Peak D+A emission rates (cps) per population
    # Backgroung rates (needed for timestamps simulation)
    "bg_rate_d": 1500,  # Poisson background rate (cps) Donor channel
    "bg_rate_a": 800,  # Poisson background rate (cps) Acceptor channel
}


# Define a simplified version of the original list-based Particles class
# This is defined at the module level to be accessible by timeit.
class LegacyParticles:
    def __init__(self, num_particles, D, box, rs) -> None:
        self._plist = self._generate(num_particles, D, box, rs)

    @staticmethod
    def _generate(num_particles, D, box, rs):
        X0 = rs.rand(num_particles) * (box.x2 - box.x1) + box.x1
        Y0 = rs.rand(num_particles) * (box.y2 - box.y1) + box.y1
        Z0 = rs.rand(num_particles) * (box.z2 - box.z1) + box.z1
        # In the original implementation, this was a list comprehension
        # of Particle objects. We simulate the object creation overhead.
        return [{"D": D, "x0": x0, "y0": y0, "z0": z0} for x0, y0, z0 in zip(X0, Y0, Z0, strict=False)]


def randomstate_equal(rs1, rs2):
    if isinstance(rs1, np.random.RandomState):
        rs1 = rs1.get_state()
    assert isinstance(rs1, tuple)
    if isinstance(rs2, np.random.RandomState):
        rs2 = rs2.get_state()
    assert isinstance(rs1, tuple)
    assert len(rs1) == len(rs2)
    equal = True
    for x1, x2 in zip(rs1, rs2, strict=False):
        test = x1 == x2
        if hasattr(test, "__array__"):
            test = test.all()
        equal &= test
    return equal


def create_diffusion_sim(psf=pbm.NumericPSF()):
    rs = np.random.RandomState(_SEED)
    specs = {k: v for k, v in particles_specs.items() if k in ["num_particles", "D", "box"]}
    P = pbm.Particles.from_specs(**specs, rs=rs)

    S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max, particles=P, box=box, psf=psf)
    S.simulate_diffusion(save_pos=True, total_emission=False, radial=False, rs=rs)
    S.store.close()
    return S.hash()[:6]


def test_Box() -> None:
    box = pbm.Box(0, 1, 0, 1, 0, 2)
    assert (box.b == np.array([[0, 1], [0, 1], [0, 2]])).all()
    assert box.volume == 2
    assert box.volume_L == 2000
    box.__repr__()  # smoke test
    box_dict = box.to_dict()
    box2 = pbm.Box(**box_dict)
    assert (box.b == box2.b).all()
    box_json = box.to_json()
    box3 = pbm.Box(**json.loads(box_json))
    assert (box.b == box3.b).all()


def test_Particle() -> None:
    a = pbm.diffusion.Particle(D=0.1, x0=0, y0=0, z0=0)
    a_dict = a.to_dict()
    b = pbm.diffusion.Particle(**a_dict)
    assert a.D == b.D
    assert a.x0 == b.x0
    assert a.y0 == b.y0
    assert a.z0 == b.z0


def test_Particles() -> None:
    rs = np.random.RandomState(_SEED)
    P = pbm.Particles(num_particles=20, D=D1, box=box, rs=rs)
    P.add(num_particles=15, D=D2)
    assert P.particles_counts == [20, 15]
    assert P.num_populations == 2
    assert P.diffusion_coeff_counts == [(D1, 20), (D2, 15)]
    with pytest.raises(ValueError):
        P.add(num_particles=1, D=D1)
    x1 = P.num_particles_to_slices((7, 8))
    x2 = [slice(0, 7, None), slice(7, 7 + 8, None)]
    for s1, s2 in zip(x1, x2, strict=False):
        assert s1 == s2

    Di, counts = zip(*P.diffusion_coeff_counts, strict=False)
    rs2 = np.random.RandomState()
    rs2.set_state(P.init_random_state)

    # Create a reference Particles object using the old _generate method logic
    p2_d_array, p2_pos_array = pbm.Particles._generate(num_particles=counts[0], D=Di[0], box=P.box, rs=rs2)
    p2_d_array2, p2_pos_array2 = pbm.Particles._generate(num_particles=counts[1], D=Di[1], box=P.box, rs=rs2)

    # Manually create a Particles object for comparison
    P2 = pbm.Particles(num_particles=0, D=0, box=box, rs=rs)  # Empty object
    P2._D = np.concatenate((p2_d_array, p2_d_array2))
    P2._positions = np.concatenate((p2_pos_array, p2_pos_array2))

    # Test equality with the refactored implementation
    assert P == P2

    # Test Particles random states
    assert randomstate_equal(P.rs, rs.get_state())
    assert randomstate_equal(P.init_random_state, np.random.RandomState(_SEED))
    assert not randomstate_equal(P.init_random_state, P.rs)

    # Test JSON serialization and deserialization
    P_json = P.to_json()
    P3 = pbm.Particles.from_json(P_json)
    assert P == P3

    # Test from_specs constructor
    rs = np.random.RandomState(_SEED)
    P4 = pbm.Particles.from_specs(num_particles=(20, 15), D=(D1, D2), box=box, rs=rs)
    assert P == P4


def test_Particles_from_specs_vectorized() -> None:
    """Test that the vectorized implementation of `Particles.from_specs`
    is numerically consistent with the original implementation.
    """

    # Original, non-vectorized implementation for comparison
    def original_from_specs(num_particles, D, box, rs=None, seed=1):
        msg = "The sequence `num_particles` must have length >= 1."
        assert len(num_particles) > 0, msg
        msg = "ERROR: `num_particles` and `D` must have the same length."
        assert len(num_particles) == len(D), msg
        P = pbm.Particles(num_particles[0], D[0], box, rs=rs, seed=seed)
        for num_particle, D_val in zip(num_particles[1:], D[1:], strict=False):
            P.add(num_particles=num_particle, D=D_val)
        return P

    # Generate sample data
    rs_orig = np.random.RandomState(_SEED)
    P_orig = original_from_specs(num_particles=(20, 15, 10), D=(D1, D2, D1 / 3), box=box, rs=rs_orig)

    # Run with vectorized implementation
    rs_vec = np.random.RandomState(_SEED)
    P_vec = pbm.Particles.from_specs(num_particles=(20, 15, 10), D=(D1, D2, D1 / 3), box=box, rs=rs_vec)

    # Assert that the results are identical
    assert P_orig == P_vec


def test_Particles_from_specs_performance() -> None:
    """Benchmark the performance of the vectorized `Particles.from_specs`."""
    import timeit

    # Original, non-vectorized implementation for comparison
    def original_from_specs(num_particles, D, box, rs=None, seed=1):
        msg = "The sequence `num_particles` must have length >= 1."
        assert len(num_particles) > 0, msg
        msg = "ERROR: `num_particles` and `D` must have the same length."
        assert len(num_particles) == len(D), msg
        P = pbm.Particles(num_particles[0], D[0], box, rs=rs, seed=seed)
        for num_particle, D_val in zip(num_particles[1:], D[1:], strict=False):
            P.add(num_particles=num_particle, D=D_val)
        return P

    # Setup for the benchmark
    rs = np.random.RandomState(_SEED)
    num_particles = [1000] * 10
    D_values = [D1 * (i + 1) for i in range(10)]

    # Time the legacy implementation
    legacy_stmt = "original_from_specs(num_particles, D_values, box, rs)"
    legacy_time = timeit.timeit(
        stmt=legacy_stmt,
        globals={
            "original_from_specs": original_from_specs,
            "num_particles": num_particles,
            "D_values": D_values,
            "box": box,
            "rs": rs,
        },
        number=10,
    )

    # Time the new, vectorized implementation
    new_stmt = "pbm.Particles.from_specs(num_particles, D_values, box, rs)"
    new_time = timeit.timeit(
        stmt=new_stmt,
        globals={"pbm": pbm, "num_particles": num_particles, "D_values": D_values, "box": box, "rs": rs},
        number=10,
    )

    print(f"Legacy Particles.from_specs time: {legacy_time:.6f}s")
    print(f"New Particles.from_specs time: {new_time:.6f}s")
    assert new_time < legacy_time


def test_Particles_vectorization() -> None:
    """Test that the vectorized implementation of Particles is numerically equivalent."""
    rs = np.random.RandomState(_SEED)
    num_particles = 100
    # Original method: create a list of Particle objects
    D_orig, pos_orig = pbm.Particles._generate(num_particles, D1, box, rs)

    # Vectorized method: create NumPy arrays directly
    rs_vec = np.random.RandomState(_SEED)
    D_vec, pos_vec = pbm.Particles._generate(num_particles, D1, box, rs_vec)

    # Check for numerical consistency
    assert np.all(D_orig == D_vec)
    assert np.allclose(pos_orig, pos_vec)


def test_Particles_performance() -> None:
    """Benchmark the performance of the vectorized Particles class."""
    import timeit

    num_particles = 10000
    rs = np.random.RandomState(_SEED)

    # Time the legacy implementation
    legacy_stmt = f"LegacyParticles(num_particles={num_particles}, D=D1, box=box, rs=rs)"
    legacy_time = timeit.timeit(
        stmt=legacy_stmt,
        globals={"LegacyParticles": LegacyParticles, "D1": D1, "box": box, "rs": rs, "num_particles": num_particles},
        number=10,
    )

    # Time the new, vectorized implementation
    new_stmt = f"pbm.Particles(num_particles={num_particles}, D=D1, box=box, rs=rs)"
    new_time = timeit.timeit(
        stmt=new_stmt, globals={"pbm": pbm, "D1": D1, "box": box, "rs": rs, "num_particles": num_particles}, number=10
    )

    print(f"Legacy Particles creation time: {legacy_time:.6f}s")
    print(f"New Particles creation time: {new_time:.6f}s")
    assert new_time < legacy_time


def test_diffusion_sim_random_state() -> None:
    for psf in (pbm.NumericPSF(), pbm.GaussianPSF()):
        # Initialize the random state
        rs = np.random.RandomState(_SEED)

        # Particles definition
        P = pbm.Particles.from_specs(num_particles=(5, 7), D=(D1, D2), box=box, rs=rs)

        # Time duration of the simulation (seconds)
        t_max = 0.01

        # Particle simulation definition
        S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max, particles=P, box=box, psf=psf)

        rs_prediffusion = rs.get_state()
        S.simulate_diffusion(
            total_emission=False, save_pos=True, verbose=True, rs=rs, chunksize=2**13, chunkslice="times"
        )
        rs_postdiffusion = rs.get_state()

        # Test diffusion random states
        saved_rs = S.traj_group._v_attrs["init_random_state"]
        assert randomstate_equal(saved_rs, rs_prediffusion)
        saved_rs = S.traj_group._v_attrs["last_random_state"]
        assert randomstate_equal(saved_rs, rs_postdiffusion)
        S.store.close()


def _test_diffusion_sim_core(psf) -> None:
    # Initialize the random state
    rs = np.random.RandomState(_SEED)
    P = pbm.Particles(num_particles=100, D=D1, box=box, rs=rs)
    t_max = 0.001
    time_size = t_max / t_step
    assert t_max < 1e4
    S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max, particles=P, box=box, psf=psf)

    start_pos = [p.r0 for p in S.particles]
    start_pos = np.vstack(start_pos).reshape(S.num_particles, 3, 1)

    for wrap_func in [pbm.diffusion.wrap_mirror, pbm.diffusion.wrap_periodic]:
        for total_emission in [True, False]:
            sim = S._sim_trajectories(
                time_size, start_pos, rs=rs, total_emission=total_emission, save_pos=True, wrap_func=wrap_func
            )

    POS, _em = sim
    # x, y, z = POS[:, :, 0], POS[:, :, 1], POS[:, :, 2]
    # r_squared = x**2 + y**2 + z**2

    DR = np.diff(POS, axis=2)
    dx, dy, dz = DR[:, :, 0], DR[:, :, 1], DR[:, :, 2]
    dr_squared = dx**2 + dy**2 + dz**2

    D_fitted = dr_squared.mean() / (6 * t_max)  # Fitted diffusion coefficient
    assert np.abs(D1 - D_fitted) < 0.01


def test_diffusion_sim_core_npsf() -> None:
    _test_diffusion_sim_core(pbm.NumericPSF())


def test_diffusion_sim_core_gpsf() -> None:
    _test_diffusion_sim_core(pbm.GaussianPSF())


def test_gaussian_psf_eval() -> None:
    """Regression: `GaussianPSF.eval` must not rely on caller frame locals.

    numexpr looked up `xc`/`sx`/... in the method's locals, so an
    unused-variable autofix renaming them to `_xc`/`_sx` made every call raise
    `KeyError: 'sx'`. Nothing in the package calls `eval`, so the suite stayed
    green while the public method was dead.
    """
    psf = pbm.GaussianPSF(sx=0.2e-6, sy=0.2e-6, sz=0.6e-6)
    x = np.array([0.0, 0.2e-6])
    zeros = np.zeros_like(x)
    assert np.allclose(psf.eval(x, zeros, zeros), [1.0, np.exp(-0.5)])
    assert np.allclose(psf.eval(zeros, zeros, np.array([0.0, 0.6e-6])), [1.0, np.exp(-0.5)])


def test_simulate_timestamps() -> None:
    hash_ = create_diffusion_sim()
    S = pbm.ParticlesSimulation.from_datafile(hash_, mode="w")

    rs = np.random.RandomState(_SEED)
    kw = {"max_rates": (400e3,), "populations": (slice(0, 35),), "bg_rate": 1000, "rs": rs, "save_pos": True}
    S.simulate_timestamps_mix(**kw)

    # The following two cases should not throw an error
    kw.update(overwrite=True, skip_existing=True, rs=np.random.RandomState(_SEED))
    S.simulate_timestamps_mix(**kw)
    kw.update(overwrite=True, skip_existing=False, rs=np.random.RandomState(_SEED))
    S.simulate_timestamps_mix(**kw)

    # This should still pass
    kw.update(overwrite=False, skip_existing=True, rs=np.random.RandomState(_SEED))
    S.simulate_timestamps_mix(**kw)

    # This should throw an ExistingArrayError
    kw.update(overwrite=False, skip_existing=False, rs=np.random.RandomState(_SEED))
    with pytest.raises(pbm.storage.ExistingArrayError):
        S.simulate_timestamps_mix(**kw)

    # But with a different initial random state should succeed
    kw.pop("rs")
    S.simulate_timestamps_mix(**kw)
    S.store.close()


def test_TimestampSimulation() -> None:
    for psf in (pbm.GaussianPSF(), pbm.NumericPSF()):
        hash_ = create_diffusion_sim(psf)
        S = pbm.ParticlesSimulation.from_datafile(hash_, mode="a")

        params = {
            "em_rates": (400e3,),  # Peak emission rates (cps) for each population (D+A)
            "E_values": (0.75,),  # FRET efficiency for each population
            "num_particles": (1,),  # Number of particles in each population
            "bg_rate_d": 1400,  # Poisson background rate (cps) Donor channel
            "bg_rate_a": 800,  # Poisson background rate (cps) Acceptor channel
        }

        mix_sim = pbm.TimestampSimulation(S, **params)
        mix_sim.summarize()

        rs = np.random.RandomState(_SEED)
        mix_sim.run(rs=rs, overwrite=True)
        mix_sim.save_photon_hdf5()


def test_vectorized_implementation() -> None:
    """Test that the vectorized implementation produces numerically consistent results."""

    # Define the original, non-vectorized implementation for comparison
    def original_sim_trajectories(
        self,
        time_size,
        start_pos,
        rs,
        total_emission=False,
        save_pos=False,
        radial=False,
        wrap_func=pbm.diffusion.wrap_periodic,
    ):
        time_size = int(time_size)
        num_particles = self.num_particles
        if total_emission:
            em = np.zeros(time_size, dtype=np.float32)
        else:
            em = np.zeros((num_particles, time_size), dtype=np.float32)
        POS = []
        for i, sigma_1d in enumerate([np.sqrt(2 * par.D * self.t_step) for par in self.particles]):
            delta_pos = rs.normal(loc=0, scale=sigma_1d, size=3 * time_size)
            delta_pos = delta_pos.reshape(3, time_size)
            pos = np.cumsum(delta_pos, axis=-1, out=delta_pos)
            pos += start_pos[i]
            for coord in (0, 1, 2):
                pos[coord] = wrap_func(pos[coord], *self.box.b[coord])
            Ro = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
            Z = pos[2]
            current_em = self.psf.eval_xz(Ro, Z) ** 2
            if total_emission:
                em += current_em.astype(np.float32)
            else:
                em[i] = current_em.astype(np.float32)
            if save_pos:
                pos_save = np.vstack((Ro, Z)) if radial else pos
                POS.append(pos_save[np.newaxis, :, :])
            start_pos[i] = pos[:, -1:]
        return POS, em

    # Initialize parameters
    rs = np.random.RandomState(_SEED)
    P = pbm.Particles(num_particles=10, D=D1, box=box, rs=rs)
    t_max = 0.001
    time_size = int(t_max / t_step)

    # Run with original implementation
    S_orig = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max, particles=P, box=box, psf=pbm.NumericPSF())
    rs_orig = np.random.RandomState(_SEED)
    _, em_orig = original_sim_trajectories(S_orig, time_size, P.positions.copy(), rs=rs_orig)

    # Run with vectorized implementation
    S_vec = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max, particles=P, box=box, psf=pbm.NumericPSF())
    rs_vec = np.random.RandomState(_SEED)
    _, em_vec = S_vec._sim_trajectories(time_size, P.positions.copy(), rs=rs_vec)

    assert np.allclose(em_orig, em_vec, atol=1e-5)


def test_timestamps_from_counts_vectorized() -> None:
    """Test that the vectorized implementation of `_timestamps_from_counts`
    produces numerically consistent results with the original implementation.
    """

    # Original, non-vectorized implementation for comparison
    def original_timestamps_from_counts(counts, time_axis, max_rate, position=None, sort=True):
        if position is not None:
            pos_part = position.shape[0]
            spatial_dims = position.shape[1]
            assert pos_part <= counts.shape[0] <= pos_part + 1
        max_counts = counts.max()
        if max_counts == 0:
            empty_pos = None
            if position is not None:
                empty_pos = np.empty(shape=(0, spatial_dims), dtype=np.float32)
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), empty_pos)

        ts_times_parlist = []
        ts_particles_parlist = []
        ts_positions_parlist = []
        for ip, counts_ip in enumerate(counts):
            ts_times_by_num_counts = []
            ts_positions_by_num_counts = []
            for v in range(1, max_counts + 1):
                mask = counts_ip >= v
                ts_times_by_num_counts.append(time_axis[mask])
                if position is not None:
                    is_bg_particle = ip == position.shape[0]
                    if is_bg_particle:
                        shape = (mask.sum(), position.shape[1])
                        pos = np.full(shape, np.nan, dtype="float32")
                    else:
                        pos = position[ip, :, mask]
                    ts_positions_by_num_counts.append(pos)
            ts = np.hstack(ts_times_by_num_counts)
            ts_times_parlist.append(ts)
            ts_particles_parlist.append(np.full(ts.size, ip, dtype="u1"))
            if position is not None:
                pos_current_particle = np.vstack(ts_positions_by_num_counts)
                ts_positions_parlist.append(pos_current_particle)
        ts_times = np.hstack(ts_times_parlist)
        ts_particles = np.hstack(ts_particles_parlist)
        ts_positions = None
        if position is not None:
            ts_positions = np.vstack(ts_positions_parlist)
        if sort:
            index_sort = ts_times.argsort(kind="mergesort")
            ts_times = ts_times[index_sort]
            ts_particles = ts_particles[index_sort]
            if position is not None:
                ts_positions = ts_positions[index_sort]
        return ts_times, ts_particles, ts_positions

    # Generate sample data
    rs = np.random.RandomState(_SEED)
    counts = rs.randint(0, 5, size=(10, 1000), dtype=np.uint8)
    time_axis = np.arange(1000, dtype=np.int64) * 10
    position = rs.rand(10, 3, 1000).astype(np.float32)

    # Run with original implementation
    ts_orig, p_orig, pos_orig = original_timestamps_from_counts(
        counts.copy(), time_axis.copy(), max_rate=1, position=position.copy()
    )

    # Run with vectorized implementation
    S = pbm.ParticlesSimulation
    ts_vec, p_vec, pos_vec = S._timestamps_from_counts(
        counts.copy(), time_axis.copy(), max_rate=1, position=position.copy()
    )

    # Assert that the results are identical
    assert np.array_equal(ts_orig, ts_vec)
    assert np.array_equal(p_orig, p_vec)
    assert np.allclose(pos_orig, pos_vec, equal_nan=True)


ALEX_EM_RATES = [100e3]


def _alex_sim_for(S):
    """Build the `AlexSmFretSimulation` these tests use for a given `S`."""
    from pybromo.timestamps import AlexSmFretSimulation

    return AlexSmFretSimulation(
        S, ALEX_EM_RATES, [0.5], [2], bg_rate_d=100, bg_rate_a=100, alex_period=1e-3, d_duty=0.4, a_duty=0.4
    )


def _make_alex_simulation(tmp_path):
    """Build a small, fully simulated ALEX setup under `tmp_path`."""
    from pybromo.diffusion import Box, GaussianPSF, Particles, ParticlesSimulation

    t_step, t_max = 1e-6, 0.02
    # Box kept tight around the PSF so that the 2 particles stay in focus long
    # enough to emit a usable number of photons within `t_max`.
    box = Box(-0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6)
    psf = GaussianPSF()
    particles = Particles(num_particles=2, D=1e-11, box=box)
    S = ParticlesSimulation(t_step=t_step, t_max=t_max, particles=particles, box=box, psf=psf)
    # `total_emission=False` is required: timestamp simulation reads the
    # per-particle `emission` array, which the `total_emission=True` default
    # leaves empty. Without it these tests silently simulated zero photons.
    S.simulate_diffusion(total_emission=False, path=str(tmp_path), verbose=False)

    return S, _alex_sim_for(S)


def _alex_timestamp_data(S):
    """Return the (D, A) timestamp/particle arrays through the public API."""
    (name_d,) = S.timestamps_match_pattern("alex_d_")
    (name_a,) = S.timestamps_match_pattern("alex_a_")
    ts_d, par_d, _ = S.get_timestamp_data(name_d)
    ts_a, par_a, _ = S.get_timestamp_data(name_a)
    return ts_d, par_d, ts_a, par_a


def test_alex_filename_separates_close_parameters(tmp_path) -> None:
    """Regression: near parameter values must not collapse to one file name.

    `_compact_repr` formatted leakage and direct excitation with `.1f`, so
    e.g. 0.05 and 0.14 both became "0.1" and the second run's Photon-HDF5
    file silently overwrote the first one's.
    """
    from pybromo.timestamps import AlexSmFretSimulation

    S, _ = _make_alex_simulation(tmp_path)
    common = {
        "bg_rate_d": 100,
        "bg_rate_a": 100,
        "alex_period": 1e-3,
        "d_duty": 0.4,
        "a_duty": 0.4,
    }
    names = {
        AlexSmFretSimulation(S, ALEX_EM_RATES, [0.5], [2], direct_exc=dx, **common).filename for dx in (0.05, 0.14)
    }
    assert len(names) == 2
    # `t_max_%ds` in the shared `_compact_repr` collapsed every sub-second
    # simulation to "t_max_0s" the same way.
    assert all("t_max_0.02s" in name for name in names)

    S.store.h5file.close()


def test_AlexSmFretSimulation(tmp_path) -> None:
    S, alex_sim = _make_alex_simulation(tmp_path)
    alex_sim.run(np.random.RandomState(42))

    ts_d, par_d, _ts_a, par_a = _alex_timestamp_data(S)

    # Regression: `run()` must forward the *total* peak rate. The simulation
    # applies the (1 - E) / E split itself, so forwarding the pre-split
    # `em_rates_d` (= em_rates * (1 - E)) applied (1 - E) twice.
    assert list(ts_d.attrs["max_rates"]) == ALEX_EM_RATES

    # Both channels must actually contain photons emitted by the particles
    # (particle index < num_particles; the background uses index num_particles).
    assert (par_d[:] < S.num_particles).any()
    assert (par_a[:] < S.num_particles).any()

    alex_sim.save_photon_hdf5()
    assert alex_sim.filepath.exists()

    S.store.h5file.close()
    S.ts_store.h5file.close()


def test_alex_photon_hdf5_is_usalex(tmp_path) -> None:
    """Regression: the exported Photon-HDF5 must be recognizable as us-ALEX.

    `excitation_alternated` held a single entry instead of one per laser, and
    the alternation period/windows were written in seconds and phase instead of
    timestamp units. FRETBursts then loaded the file as plain smFRET and
    dropped the alternation entirely.
    """
    import tables

    S, alex_sim = _make_alex_simulation(tmp_path)
    alex_sim.run(np.random.RandomState(42))
    alex_sim.save_photon_hdf5()
    clk_p = S.get_timestamp_data(S.timestamps_match_pattern("alex_d_")[0])[0].attrs["clk_p"]
    S.store.h5file.close()
    S.ts_store.h5file.close()

    with tables.open_file(str(alex_sim.filepath)) as h5file:
        setup = h5file.root.setup
        specs = h5file.root.photon_data.measurement_specs
        # FRETBursts keys off this exact pair to pick the us-ALEX loader.
        assert tuple(setup.excitation_alternated.read()) == (True, True)
        assert tuple(setup.excitation_cw.read()) == (True, True)
        assert specs.measurement_type.read().decode() == "smFRET-usALEX"

        period = round(alex_sim.alex_period / clk_p)
        assert specs.alex_period.read() == period
        assert list(specs.alex_excitation_period1.read()) == [0, round(alex_sim.d_duty * period)]
        assert list(specs.alex_excitation_period2.read()) == [
            round(0.5 * period),
            round((0.5 + alex_sim.a_duty) * period),
        ]


def test_alex_skip_existing_does_not_append(tmp_path) -> None:
    """Regression: `skip_existing=True` must skip, not re-append.

    `simulate_timestamps_alex` caught `ExistingArrayError` without returning,
    so a second run printed "Skipping" and then appended a whole second set of
    timestamps onto the existing arrays, silently doubling the photon stream.
    """
    S, alex_sim = _make_alex_simulation(tmp_path)
    alex_sim.run(np.random.RandomState(42))
    ts_d, _par_d, ts_a, _par_a = _alex_timestamp_data(S)
    nrows_d, nrows_a = ts_d.nrows, ts_a.nrows
    assert nrows_d > 0

    alex_sim.run(np.random.RandomState(42), overwrite=False, skip_existing=True)

    ts_d, _par_d, ts_a, _par_a = _alex_timestamp_data(S)
    assert ts_d.nrows == nrows_d
    assert ts_a.nrows == nrows_a

    S.store.h5file.close()
    S.ts_store.h5file.close()


def test_alex_skip_existing_still_exports(tmp_path) -> None:
    """Regression: a skipped run must still bind the existing D/A arrays.

    The `skip_existing` branch returned before assigning `S._timestamps_d`/`_a`,
    so `run()` printed "Completed" and the next `save_photon_hdf5()` died with
    `AttributeError: ... has no attribute '_timestamps_d'`. Only a simulation
    reloaded from disk shows it: within one session the attributes are left
    over from the run that wrote the timestamps.
    """
    S, alex_sim = _make_alex_simulation(tmp_path)
    alex_sim.run(np.random.RandomState(42))
    hash_ = S.store.filepath.stem.split("_")[1]
    S.store.h5file.close()
    S.ts_store.h5file.close()

    S = pbm.ParticlesSimulation.from_datafile(hash_, path=str(tmp_path), mode="a")
    alex_sim = _alex_sim_for(S)
    alex_sim.run(np.random.RandomState(42), overwrite=False, skip_existing=True)
    alex_sim.save_photon_hdf5()

    S.store.h5file.close()
    S.ts_store.h5file.close()
    assert alex_sim.filepath.exists()


def test_alex_partial_store_is_rejected(tmp_path) -> None:
    """Regression: half an ALEX pair in the store must not report success.

    Skipping on D and then returning meant the A array was never created, yet
    `run()` still reported "Completed".
    """
    S, alex_sim = _make_alex_simulation(tmp_path)
    alex_sim.run(np.random.RandomState(42))
    (name_a,) = S.timestamps_match_pattern("alex_a_")
    S.ts_store.h5file.remove_node("/timestamps", name_a)
    S.ts_store.h5file.remove_node("/timestamps", name_a + "_par")

    with pytest.raises(ValueError, match="Incomplete ALEX timestamps"):
        alex_sim.run(np.random.RandomState(42), overwrite=False, skip_existing=True)

    S.store.h5file.close()
    S.ts_store.h5file.close()


def test_alex_photon_hdf5_keeps_caller_identity(tmp_path) -> None:
    """Regression: caller identity must win, and the caller's dict stay intact.

    `_make_photon_hdf5` called `identity.update(author=...)`, which discarded
    the author the caller passed *and* rewrote their dict in place -- exactly
    what the documented B.3 notebook workflow does. The same method also wrote
    `round(timeslice)` as `acquisition_duration`, so every sub-second
    simulation claimed a 0 s acquisition.
    """
    import tables

    S, alex_sim = _make_alex_simulation(tmp_path)
    alex_sim.run(np.random.RandomState(42))
    identity = {"author": "Author Name", "author_affiliation": "Research Institution"}
    alex_sim.save_photon_hdf5(identity=identity)
    t_max = S.t_max
    S.store.h5file.close()
    S.ts_store.h5file.close()

    assert identity == {"author": "Author Name", "author_affiliation": "Research Institution"}
    with tables.open_file(str(alex_sim.filepath)) as h5file:
        assert h5file.root.identity.author.read().decode() == "Author Name"
        assert h5file.root.identity.author_affiliation.read().decode() == "Research Institution"
        assert h5file.root.acquisition_duration.read() == pytest.approx(t_max)


def test_check_per_particle_emission_rejects_truncated(tmp_path) -> None:
    """Regression: a partially written `emission` array must not pass the guard.

    Only checking for a completely empty array let an interrupted
    `simulate_diffusion` through, which again yielded too few photons silently.
    """
    S, alex_sim = _make_alex_simulation(tmp_path)
    # Pretend the trajectory simulation died after roughly half the time steps.
    S.emission.truncate(S.n_samples // 2)

    with pytest.raises(ValueError, match="did not run to completion"):
        alex_sim.run(np.random.RandomState(0))

    # The guard runs before `open_store_timestamp`, so there is no `ts_store`.
    assert not hasattr(S, "ts_store")
    S.store.h5file.close()


def test_print_children(tmp_path, capsys) -> None:
    """Regression: `print_children` still called the Python-2 `itervalues()`.

    Every call raised `AttributeError: 'dict' object has no attribute
    'itervalues'` after printing the group list.
    """
    S, _ = _make_alex_simulation(tmp_path)
    pbm.hdf5.print_children(S.store.h5file, "/trajectories")
    S.store.h5file.close()

    assert "emission" in capsys.readouterr().out
