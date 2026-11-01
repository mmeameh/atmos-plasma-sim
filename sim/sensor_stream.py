import time
from dataclasses import dataclass
from typing import List, Dict, Generator

@dataclass
class GNSSObservation:
    station_id: str
    timestamp: float
    total_electron_content: float
    scintillation_s4: float
    satellite_prn: int

class SensorStreamPipeline:
    """High-throughput streaming ingestion pipeline for ground-based GNSS & magnetometer sensor networks."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.total_ingested = 0
        self.anomalies_detected = 0

    def generate_simulated_stream(self, num_records: int) -> Generator[GNSSObservation, None, None]:
        for i in range(num_records):
            base_tec = 25.0 + 5.0 * (i % 20) / 20.0
            s4 = 0.05 if i % 100 != 0 else 0.65 # Occasional scintillation spike
            yield GNSSObservation(
                station_id="CAR_FUTA_01",
                timestamp=time.time() + i * 0.1,
                total_electron_content=base_tec,
                scintillation_s4=s4,
                satellite_prn=(i % 32) + 1,
            )

    def process_stream_batch(self, observations: List[GNSSObservation]) -> Dict[str, float]:
        """Ingests and validates a batch of sensor records, detecting ionospheric scintillations."""
        valid_records = []
        for obs in observations:
            # Range check physical bounds: TEC in [0, 250], S4 in [0, 1.5]
            if 0.0 <= obs.total_electron_content <= 250.0 and 0.0 <= obs.scintillation_s4 <= 1.5:
                valid_records.append(obs)
                if obs.scintillation_s4 > 0.4:
                    self.anomalies_detected += 1

        self.total_ingested += len(valid_records)
        avg_tec = sum(r.total_electron_content for r in valid_records) / len(valid_records) if valid_records else 0.0

        return {
            "ingested_count": float(len(valid_records)),
            "average_tec": avg_tec,
            "anomalies": float(self.anomalies_detected),
        }
