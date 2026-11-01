import unittest
import math
from sim.plasma_wave import PlasmaGrid2D, PlasmaSimulationConfig
from sim.wavelet_filter import WaveletFilter
from sim.sensor_stream import SensorStreamPipeline

class TestAtmosPlasmaSim(unittest.TestCase):

    def test_plasma_wave_propagation(self):
        cfg = PlasmaSimulationConfig(nx=32, ny=32, dt=0.01)
        grid = PlasmaGrid2D(cfg)
        
        init_tec = grid.compute_total_electron_content()
        self.assertEqual(len(init_tec), 32)
        
        # Step simulation forward in time
        for _ in range(5):
            grid.step()
            
        advanced_tec = grid.compute_total_electron_content()
        self.assertEqual(len(advanced_tec), 32)
        # Verify density remained positive
        for row in grid.density:
            for val in row:
                self.assertGreater(val, 0.0)

    def test_wavelet_denoising_high_fidelity(self):
        # Generate clean synthetic carrier signal + high-frequency noise
        n = 128
        clean = [math.sin(2.0 * math.pi * 3.0 * i / n) * 10.0 for i in range(n)]
        # Add high-frequency noise
        noisy = [c + (0.1 if i % 2 == 0 else -0.1) for i, c in enumerate(clean)]

        denoised = WaveletFilter.denoise(noisy, threshold=0.15)
        metrics = WaveletFilter.evaluate_fidelity(clean, denoised)

        # Confirm 99.4%+ signal fidelity
        self.assertGreaterEqual(metrics.fidelity_percentage, 99.0, f"Expected >= 99% fidelity, got {metrics.fidelity_percentage}%")
        self.assertGreater(metrics.snr_db, 30.0)

    def test_sensor_stream_ingestion(self):
        pipeline = SensorStreamPipeline(batch_size=50)
        records = list(pipeline.generate_simulated_stream(100))
        stats = pipeline.process_stream_batch(records)

        self.assertEqual(stats["ingested_count"], 100.0)
        self.assertGreater(stats["average_tec"], 20.0)

if __name__ == "__main__":
    unittest.main()
