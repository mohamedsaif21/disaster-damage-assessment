from pydantic import BaseModel
from typing import List


class ImageInfo(BaseModel):
    filename: str
    content_type: str
    format: str
    width: int
    height: int
    size_bytes: int


class PredictionInfo(BaseModel):
    mask_shape: List[int]
    predicted_classes: List[int]


class ClassStatistics(BaseModel):
    pixels: int
    percentage: float


class DamageClasses(BaseModel):
    background: ClassStatistics
    no_damage: ClassStatistics
    minor_damage: ClassStatistics
    major_damage: ClassStatistics
    destroyed: ClassStatistics


class Statistics(BaseModel):
    total_pixels: int
    damage_pixels: int
    damage_percentage: float
    damage_level: str
    classes: DamageClasses


class ModelInfo(BaseModel):
    name: str
    input_channels: int
    output_classes: int
    checkpoint: str


class AssessmentResponse(BaseModel):
    status: str
    message: str

    before_image: ImageInfo
    after_image: ImageInfo

    prediction: PredictionInfo

    statistics: Statistics

    model: ModelInfo