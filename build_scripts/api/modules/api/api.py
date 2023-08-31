import base64
import io
import os
import time
import datetime
import uvicorn
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import threading
import json
from secrets import compare_digest
import traceback

from modules import shared, scripts, pipeline, errors
# from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, errors, restart
# from modules.api import models
from modules.shared import opts
# from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
# from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
# from modules.textual_inversion.preprocess import preprocess
# from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin,Image
# from modules.sd_models import checkpoints_list, unload_model_weights, reload_model_weights, checkpoint_aliases
# from modules.sd_vae import vae_dict
# from modules.sd_models_config import find_checkpoint_config_near_filename
# from modules.realesrgan_model import get_realesrgan_models
# from modules import devices
# from typing import Dict, List, Any
import piexif
import piexif.helper
from contextlib import closing

from modules.api import models

import logging

if os.environ.get("DEBUG_API", False):
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# def script_name_to_index(name, scripts):
#     try:
#         return [script.title().lower() for script in scripts].index(name.lower())
#     except Exception as e:
#         raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e


# def validate_sampler_name(name):
#     config = sd_samplers.all_samplers_map.get(name, None)
#     if config is None:
#         raise HTTPException(status_code=404, detail="Sampler not found")

#     return name


# def setUpscalers(req: dict):
#     reqDict = vars(req)
#     reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
#     reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
#     return reqDict


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            if image.mode == "RGBA":
                image = image.convert("RGB")
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get('WEBUI_RICH_EXCEPTIONS', None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console
            console = Console()
            rich_available = True
    except Exception:
        pass

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        if shared.cmd_opts.api_log and endpoint.startswith('/sdapi'):
            print('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
                t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code=res.status_code,
                ver=req.scope.get('http_version', '0.0'),
                cli=req.scope.get('client', ('0:0.0.0', 0))[0],
                prot=req.scope.get('scheme', 'err'),
                method=req.scope.get('method', 'err'),
                endpoint=endpoint,
                duration=duration,
            ))
        return res

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        if not isinstance(e, HTTPException):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[anyio, starlette], word_wrap=False, width=min([console.width, 200]))
            else:
                errors.report(message, exc_info=True)
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        # # TODO: do we need api_middleware? Xiujuan
        api_middleware(self.app)
        self.add_api_route("/invocations", self.invocations, methods=["POST"], response_model=[])
        self.add_api_route("/ping", self.ping, methods=["GET"], response_model=models.PingResponse)

        # if shared.cmd_opts.api_server_stop:
        #     self.add_api_route("/sdapi/v1/server-kill", self.kill_webui, methods=["POST"])
        #     self.add_api_route("/sdapi/v1/server-restart", self.restart_webui, methods=["POST"])
        #     self.add_api_route("/sdapi/v1/server-stop", self.stop_webui, methods=["POST"])

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def add_api_route(self, path: str, endpoint, **kwargs):
        # TODO: do we need api_auth? Xiujuan
        # if shared.cmd_opts.api_auth:
        #     return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    # def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    #     if credentials.username in self.credentials:
    #         if compare_digest(credentials.password, self.credentials[credentials.username]):
    #             return True

    #     raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def txt2img_pipeline(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        args = vars(txt2imgreq)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)
        send_images = args.pop('send_images', True)
        args.pop('save_images', None)
        with closing(pipeline.StableDiffusionPipelineTxt2Img(sd_model=None, **args)) as p:
            processed = pipeline.process_images(p)
            b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
            return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def invocations(self, req: models.InvocationsRequest):
        """
        @return:
        """
        logger.info('-------invocation------')
        logger.info("Loading Sagemaker API Endpoints.")

        def show_slim_dict(payload):
            pay_type = type(payload)
            if pay_type is dict:
                for k, v in payload.items():
                    logger.info(f"{k}")
                    show_slim_dict(v)
            elif pay_type is list:
                for v in payload:
                    logger.info(f"list")
                    show_slim_dict(v)
            elif pay_type is str:
                if len(payload) > 50:
                    logger.info(f" : {len(payload)} contents")
                else:
                    logger.info(f" : {payload}")
            else:
                logger.info(f" : {payload}")

        logger.info(f"{threading.current_thread().ident}_{threading.current_thread().name}")
        logger.info(f"task is {req.task}")
        logger.info(f"checkpoint_info is {req.checkpoint_info}")
        logger.info(f"models is {req.models}")
        logger.info(f"txt2img_payload is: ")
        txt2img_payload = {} if req.txt2img_payload is None else json.loads(req.txt2img_payload.json())
        show_slim_dict(txt2img_payload)
        logger.info(f"img2img_payload is: ")
        img2img_payload = {} if req.img2img_payload is None else json.loads(req.img2img_payload.json())
        show_slim_dict(img2img_payload)
        logger.info(f"extra_single_payload is: ")
        extra_single_payload = {} if req.extras_single_payload is None else json.loads(
            req.extras_single_payload.json())
        show_slim_dict(extra_single_payload)
        logger.info(f"extra_batch_payload is: ")
        extra_batch_payload = {} if req.extras_batch_payload is None else json.loads(
            req.extras_batch_payload.json())
        show_slim_dict(extra_batch_payload)
        logger.info(f"interrogate_payload is: ")
        interrogate_payload = {} if req.interrogate_payload is None else json.loads(
            req.interrogate_payload.json())
        show_slim_dict(interrogate_payload)
        # logger.info(f"db_create_model_payload is: ")
        # logger.info(f"{req.db_create_model_payload}")
        # logger.info(f"merge_checkpoint_payload is: ")
        # logger.info(f"{req.merge_checkpoint_payload}")
        # logger.info(f"json is {json.loads(req.json())}")
        try:
            if req.task == 'txt2img':
                with self.queue_lock:
                    # logger.info(
                    #     f"{threading.current_thread().ident}_{threading.current_thread().name}_______ txt2img start !!!!!!!!")
                    # selected_models = req.models
                    # checkpoint_info = req.checkpoint_info
                    # checkspace_and_update_models(selected_models, checkpoint_info)
                    # logger.info(
                    #     f"{threading.current_thread().ident}_{threading.current_thread().name}_______ txt2img models update !!!!!!!!")
                    # logger.info(json.loads(req.txt2img_payload.json()))
                    # # response = requests.post(url=f'http://0.0.0.0:8080/sdapi/v1/txt2img',
                    # #                             json=json.loads(req.txt2img_payload.json()))
                    response = self.txt2img_pipeline(req.txt2img_payload)
                    logger.info(
                        f"{threading.current_thread().ident}_{threading.current_thread().name}_______ txt2img end !!!!!!!! {len(response.json())}")
                    return response
            elif req.task == 'img2img':
                logger.info("img2img not implemented!")
                return 0
                # with self.queue_lock:
                #     logger.info(
                #         f"{threading.current_thread().ident}_{threading.current_thread().name}_______ img2img start!!!!!!!!")
                #     selected_models = req.models
                #     checkpoint_info = req.checkpoint_info
                #     checkspace_and_update_models(selected_models, checkpoint_info)
                #     logger.info(
                #         f"{threading.current_thread().ident}_{threading.current_thread().name}_______ txt2img models update !!!!!!!!")
                #     response = requests.post(url=f'http://0.0.0.0:8080/sdapi/v1/img2img',
                #                                 json=json.loads(req.img2img_payload.json()))
                #     logger.info(
                #         f"{threading.current_thread().ident}_{threading.current_thread().name}_______ img2img end !!!!!!!!{len(response.json())}")
                #     return response.json()
            elif req.task == 'interrogate_clip' or req.task == 'interrogate_deepbooru':
                logger.info("interrogate not implemented!")
                return 0
                # response = requests.post(url=f'http://0.0.0.0:8080/sdapi/v1/interrogate',
                #                             json=json.loads(req.interrogate_payload.json()))
                # return response.json()
            elif req.task == 'db-create-model':
                logger.info("db-create-model not implemented!")
                return 0
                r"""
                task: db-create-model
                db_create_model_payload:
                    :s3_input_path: S3 path for download src model.
                    :s3_output_path: S3 path for upload generated model.
                    :ckpt_from_cloud: Whether to get ckpt from cloud or local.
                    :job_id: job id.
                    :param
                        :new_model_name: generated model name.
                        :ckpt_path: S3 path for download src model.
                        :db_new_model_shared_src="",
                        :from_hub=False,
                        :new_model_url="",
                        :new_model_token="",
                        :extract_ema=False,
                        :train_unfrozen=False,
                        :is_512=True,
                """
                # try:
                #     db_create_model_payload = json.loads(req.db_create_model_payload)
                #     job_id = db_create_model_payload["job_id"]
                #     s3_output_path = db_create_model_payload["s3_output_path"]
                #     output_bucket_name = get_bucket_name_from_s3_path(s3_output_path)
                #     output_path = get_path_from_s3_path(s3_output_path)
                #     db_create_model_params = db_create_model_payload["param"]["create_model_params"]
                #     if "ckpt_from_cloud" in db_create_model_payload["param"]:
                #         ckpt_from_s3 = db_create_model_payload["param"]["ckpt_from_cloud"]
                #     else:
                #         ckpt_from_s3 = False
                #     if not db_create_model_params['from_hub']:
                #         if ckpt_from_s3:
                #             s3_input_path = db_create_model_payload["param"]["s3_ckpt_path"]
                #             local_model_path = db_create_model_params["ckpt_path"]
                #             input_path = get_path_from_s3_path(s3_input_path)
                #             logger.info(f"ckpt from s3 {input_path} {local_model_path}")
                #         else:
                #             s3_input_path = db_create_model_payload["s3_input_path"]
                #             local_model_path = db_create_model_params["ckpt_path"]
                #             input_path = os.path.join(get_path_from_s3_path(s3_input_path), local_model_path)
                #             logger.info(f"ckpt from local {input_path} {local_model_path}")
                #         input_bucket_name = get_bucket_name_from_s3_path(s3_input_path)
                #         logging.info("Check disk usage before download.")
                #         os.system("df -h")
                #         logger.info(
                #             f"Download src model from s3 {input_bucket_name} {input_path} {local_model_path}")
                #         download_folder_from_s3_by_tar(input_bucket_name, input_path, local_model_path)
                #         # Refresh the ckpt list.
                #         sd_models.list_models()
                #         logger.info("Check disk usage after download.")
                #         os.system("df -h")
                #     logger.info("Start creating model.")
                #     # local_response = requests.post(url=f'http://0.0.0.0:8080/dreambooth/createModel',
                #     #                         params=db_create_model_params)
                #     create_model_func_args = copy.deepcopy(db_create_model_params)
                #     # ckpt_path = create_model_func_args.pop("new_model_src")
                #     # create_model_func_args["ckpt_path"] = ckpt_path
                #     local_response = create_model(**create_model_func_args)
                #     target_local_model_dir = f'models/dreambooth/{db_create_model_params["new_model_name"]}'
                #     logging.info(
                #         f"Upload tgt model to s3 {target_local_model_dir} {output_bucket_name} {output_path}")
                #     upload_folder_to_s3_by_tar(target_local_model_dir, output_bucket_name, output_path)
                #     config_file = os.path.join(target_local_model_dir, "db_config.json")
                #     with open(config_file, 'r') as openfile:
                #         config_dict = json.load(openfile)
                #     message = {
                #         "response": local_response,
                #         "config_dict": config_dict
                #     }
                #     response = {
                #         "id": job_id,
                #         "statusCode": 200,
                #         "message": message,
                #         "outputLocation": [f'{s3_output_path}/db_create_model_params["new_model_name"]']
                #     }
                #     return response
                # except Exception as e:
                #     response = {
                #         "id": job_id,
                #         "statusCode": 500,
                #         "message": traceback.format_exc(),
                #     }
                #     logger.error(traceback.format_exc())
                #     return response
                # finally:
                #     # Clean up
                #     logger.info("Delete src model.")
                #     delete_src_command = f"rm -rf models/Stable-diffusion/{db_create_model_params['ckpt_path']}"
                #     logger.info(delete_src_command)
                #     os.system(delete_src_command)
                #     logging.info("Delete tgt model.")
                #     delete_tgt_command = f"rm -rf models/dreambooth/{db_create_model_params['new_model_name']}"
                #     logger.info(delete_tgt_command)
                #     os.system(delete_tgt_command)
                #     logging.info("Check disk usage after request.")
                #     os.system("df -h")
            else:
                raise NotImplementedError
        except Exception as e:
            traceback.print_exc()

    def ping(self):
        return {'status': 'Healthy'}

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port, timeout_keep_alive=shared.cmd_opts.timeout_keep_alive)
