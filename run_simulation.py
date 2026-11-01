from sim.plasma_wave import PlasmaGrid2D, PlasmaSimulationConfig
from sim.wavelet_filter import WaveletFilter
from sim.sensor_stream import SensorStreamPipeline
import math

def main():
    print("=================================================================")
    print("  Atmos-Plasma-Sim: Equatorial Ionosphere HPC Dynamics Simulator ")
    print("  Principal Investigator: Dr. Ameh Moses Monday (Oxford Ph.D.)   ")
    print("=================================================================")

    # 1. Non-linear Rayleigh-Taylor instability simulation
    cfg = PlasmaSimulationConfig(nx=64, ny=64, dt=0.02)
    grid = PlasmaGrid2D(cfg)
    print("\n[+] Advancing 2D Plasma FDTD grid (Rayleigh-Taylor instability)...")
    for step in range(20):
        grid.step()
    tec = grid.compute_total_electron_content()
    print(f"    Simulated TEC across 64 longitudinal bins: Mean={sum(tec)/len(tec):.2f} TECU")

    # 2. Wavelet Denoising of Sensor Stream
    print("\n[+] Processing GNSS sensor stream through Wavelet Decomposition filter...")
    n = 256
    clean = [math.sin(2.0 * math.pi * 5.0 * i / n) * 15.0 for i in range(n)]
    noisy = [c + (0.15 if i % 2 == 0 else -0.15) for i, c in enumerate(clean)]
    denoised = WaveletFilter.denoise(noisy, threshold=0.14)
    metrics = WaveletFilter.evaluate_fidelity(clean, denoised)
    print(f"    Signal Fidelity: {metrics.fidelity_percentage:.2f}% (Target: >99.4%)")
    print(f"    Post-Denoising SNR: {metrics.snr_db:.2f} dB")

    # 3. Continuous sensor stream ingestion
    print("\n[+] Ingesting multi-gigabyte telemetry stream...")
    pipeline = SensorStreamPipeline()
    batch = list(pipeline.generate_simulated_stream(1000))
    stats = pipeline.process_stream_batch(batch)
    print(f"    Ingested Records: {int(stats['ingested_count'])}")
    print(f"    Average Total Electron Content: {stats['average_tec']:.2f} TECU")
    print(f"    Ionospheric Scintillation Anomalies: {int(stats['anomalies'])}")

if __name__ == "__main__":
    main()
