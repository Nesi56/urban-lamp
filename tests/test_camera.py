import numpy as np
import pytest

from phasescope.camera import (ManualShifter, SerialPiezoShifter, SimulatedInterferometer,
                               SimulatedShifter, StaticImageSource, _to_gray)


def test_simulator_frames_and_controls():
    cam = SimulatedInterferometer(width=64, height=48, fps=0, seed=0)
    frame = cam.grab()
    assert frame.shape == (48, 64) and frame.dtype == np.uint16 and frame.max() <= 4095
    cam.set_exposure_ms(1000)  # clamped to the allowed range -> saturates
    assert cam.get_exposure_ms() == cam.exposure_range_ms[1]
    assert (cam.grab() == 4095).mean() > 0.3


def test_simulated_shifter_moves_fringes():
    cam = SimulatedInterferometer(width=64, height=48, fps=0, animate=False, noise=0, seed=0)
    a = cam.grab().astype(float)
    SimulatedShifter(cam).set_phase(np.pi)
    b = cam.grab().astype(float)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] < -0.5  # half a fringe -> contrast inverted


def test_static_source_and_gray_conversion():
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 1] = 200
    src = StaticImageSource(rgb, fps=1000)
    assert src.grab().shape == (10, 12)
    assert _to_gray(np.zeros((5, 6, 4), np.uint16)).shape == (5, 6)


def test_piezo_voltage_mapping_and_command():
    sh = SerialPiezoShifter(volts_per_2pi=4.0, offset_v=10.0, max_v=12.0)
    assert sh.voltage_for(0.0) == pytest.approx(10.0)
    assert sh.voltage_for(np.pi) == pytest.approx(12.0)
    assert sh.voltage_for(2 * np.pi) == pytest.approx(12.0)  # clipped to max_v
    assert sh.command.format(volts=sh.voltage_for(np.pi / 2)) == "xvoltage=11.000\r"
    assert not ManualShifter.automatic


def test_piezo_over_loopback_port():
    pytest.importorskip("serial")
    sh = SerialPiezoShifter(port="loop://")
    sh.connect()
    sh.set_phase(np.pi)
    sh.disconnect()
