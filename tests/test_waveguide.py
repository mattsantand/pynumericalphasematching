import pytest
from pynumpm import waveguide
import numpy as np


def test_creation_ideal_waveguide():
    thiswaveguide = waveguide.Waveguide(length=20e-3, poling_period=10e-6)

    # test that the values are passed with the correct units
    assert thiswaveguide.length == 20e-3
    assert thiswaveguide.poling_period_um == 10


def test_creation_real_waveguide():
    import numpy as np
    z = np.linspace(0, 20, 1000) * 1e-3
    thisrealwaveguide = waveguide.RealisticWaveguide(z=z, poling_period=10e-3, nominal_parameter_name=r"$\delta\beta$")
    assert len(thisrealwaveguide.z) == 1000


def test_waveguide_profiles():
    z = np.linspace(0, 20, 1000) * 1e-3
    thisrealwaveguide = waveguide.RealisticWaveguide(z=z, poling_period=10e-3, nominal_parameter=0,
                                                     nominal_parameter_name=r"$\delta\beta$")

    profile = (thisrealwaveguide.z - thisrealwaveguide.z.mean()) ** 2
    profile /= profile.max() / 10
    thisrealwaveguide.load_waveguide_profile(profile)
    assert np.all(np.equal(thisrealwaveguide.profile, profile))

    np.random.seed(42)
    reference_noise = np.load("pink_noise_wg_profile.npy")
    thisrealwaveguide = waveguide.RealisticWaveguide(z=z, poling_period=10e-3, nominal_parameter=0,
                                                     nominal_parameter_name=r"$\delta\beta$")
    thisrealwaveguide.create_noisy_waveguide()
    assert np.all(np.isclose(thisrealwaveguide.profile, reference_noise))

    out = thisrealwaveguide.plot_waveguide_properties()
    assert len(out) == 2
    assert len(out[1]) == 3


if __name__ == '__main__':
    pytest.main([__file__])
