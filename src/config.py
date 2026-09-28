"""Central configuration, loaded from environment variables / .env."""
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "predictive_maintenance"
    postgres_user: str = "pm_user"
    postgres_password: str = "pm_password"

    mlflow_tracking_uri: str = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_dir: str = str(BASE_DIR / "models_store")

    class Config:
        env_file = ".env"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()

# --- Domain constants shared across modules --------------------------------

SENSOR_COLUMNS = [
    "vibration_mm_s",
    "temperature_c",
    "pressure_bar",
    "rpm",
    "current_amp",
]

ROLLING_WINDOWS = [5, 20, 60]        # in readings, not wall-clock time
FFT_WINDOW = 60                       # readings used per FFT segment
FAILURE_WINDOW_CYCLES = 30            # "will it fail within N cycles?" label