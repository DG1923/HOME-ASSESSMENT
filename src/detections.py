from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    scale: float
    angle: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["confidence"] = round(float(self.confidence), 4)
        data["scale"] = round(float(self.scale), 4)
        return data

