from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = 1
IMAGE_SIZE = (256, 256)
PROBABILITY_TOLERANCE = 1e-5
Unit = Annotated[float, Field(ge=0, le=1)]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = PROTOCOL_VERSION
    session: str = Field(min_length=1)
    episode: int = Field(ge=0)
    frame: int = Field(ge=0)
    captured: float = Field(ge=0)
    sim_time: float = Field(ge=0)


class Reply(Observation):
    model: str = Field(min_length=1)
    mode: Literal["stub", "cnn", "classical"]
    position: tuple[Unit, Unit]
    eyelid: tuple[Unit, Unit, Unit]
    visible: Unit
    service_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_probabilities(self):
        if abs(sum(self.eyelid) - 1) > PROBABILITY_TOLERANCE:
            raise ValueError("開閉確率の合計が1ではありません")
        return self
