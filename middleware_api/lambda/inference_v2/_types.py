import dataclasses
from typing import Optional, Any


@dataclasses.dataclass
class InferenceJob:
    InferenceJobId: str
    startTime: str
    status: str
    taskType: str
    inference_info_name: Optional[Any] = None
    image_names: Optional[Any] = None
    sagemakerRaw: Optional[Any] = None
    params: Optional[dict[str, Any]] = None


# a copy of aws_extensions.models.InvocationsRequest
@dataclasses.dataclass
class InvocationsRequest:
    task: str
    username: Optional[str]
    param_s3: str
    # checkpoint_info:Optional[dict]
    models: Optional[dict]
    # txt2img_payload: Optional[StableDiffusionTxt2ImgProcessingAPI]
    # img2img_payload: Optional[StableDiffusionImg2ImgProcessingAPI]
    # extras_single_payload: Optional[ExtrasSingleImageRequest]
    # extras_batch_payload: Optional[ExtrasBatchImagesRequest]
    # interrogate_payload: Optional[InterrogateRequest]
    # db_create_model_payload: Optional[str]
    # merge_checkpoint_payload: Optional[dict]
