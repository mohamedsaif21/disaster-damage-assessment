from pydantic import BaseModel
from typing import List, Optional


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


# ============================================================
# RETRIEVAL RESPONSE MODELS
#
# These mirror the rows returned by
# backend/services/assessment_repository.py.
# They are only used by the GET retrieval endpoints and
# do not affect the POST /analyze response above.
# ============================================================

class RegisteredModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    checkpoint: str
    input_channels: int
    output_classes: int
    epoch: Optional[int]
    validation_loss: Optional[float]


class AssessmentImageRecord(BaseModel):
    id: str
    assessment_id: str
    image_type: str
    filename: Optional[str]
    storage_path: Optional[str]
    content_type: Optional[str]
    format: Optional[str]
    width: Optional[int]
    height: Optional[int]
    size_bytes: Optional[int]
    created_at: str


class AssessmentPredictionRecord(BaseModel):
    id: str
    assessment_id: str
    mask_width: int
    mask_height: int
    predicted_classes: List[int]
    mask_storage_path: Optional[str]
    created_at: str


class AssessmentClassStatistic(BaseModel):
    id: str
    assessment_id: str
    class_id: int
    class_name: str
    pixel_count: int
    percentage: float
    created_at: str


class AssessmentHistoryItem(BaseModel):
    id: str
    user_id: Optional[str]
    model_id: Optional[str]
    status: Optional[str]
    damage_level: Optional[str]
    damage_percentage: float
    total_pixels: Optional[int]
    damage_pixels: Optional[int]
    created_at: str
    updated_at: str
    assessment_models: Optional[RegisteredModelInfo]


class AssessmentHistoryResponse(BaseModel):
    status: str
    count: int
    limit: Optional[int]
    assessments: List[AssessmentHistoryItem]


class AssessmentDetail(BaseModel):
    id: str
    user_id: Optional[str]
    model_id: Optional[str]
    status: Optional[str]
    damage_level: Optional[str]
    damage_percentage: float
    total_pixels: Optional[int]
    damage_pixels: Optional[int]
    created_at: str
    updated_at: str
    assessment_models: Optional[RegisteredModelInfo]
    assessment_images: List[AssessmentImageRecord]
    assessment_predictions: Optional[AssessmentPredictionRecord]
    assessment_class_statistics: List[AssessmentClassStatistic]