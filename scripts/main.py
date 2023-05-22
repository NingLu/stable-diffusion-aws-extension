#import sagemaker
import re
import time
import math
import json
import threading
import requests
import copy
import os
import logging
import gradio as gr
import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images, sd_models
from modules.ui import create_refresh_button
from utils import upload_file_to_s3_by_presign_url, upload_multipart_files_to_s3_by_signed_url
from utils import get_variable_from_json
from utils import save_variable_to_json

import sys
import pickle
import html

# TODO: Automaticly append the dependent module path.
sys.path.append("extensions/sd_dreambooth_extension")
sys.path.append("extensions/stable-diffusion-aws-extension")
sys.path.append("extensions/stable-diffusion-aws-extension/scripts")
# TODO: Do not use the dreambooth status module.
from dreambooth.shared import status
from dreambooth import shared as dreambooth_shared
# from extensions.sd_dreambooth_extension.scripts.main import get_sd_models
from dreambooth.ui_functions import load_model_params, load_params
from dreambooth.dataclasses.db_config import save_config, from_file
from urllib.parse import urljoin
import sagemaker_ui

db_model_name = None
cloud_db_model_name = None
db_use_txt2img = None
db_sagemaker_train = None
db_save_config = None
txt2img_show_hook = None
txt2img_gallery = None
txt2img_generation_info = None
txt2img_html_info = None
modelmerger_merge_hook = None
modelmerger_merge_component = None
job_link_list = []
ckpt_dict = {}

base_model_folder = "models/sagemaker_dreambooth/"

class SageMakerUI(scripts.Script):
    def title(self):
        return "SageMaker embeddings"

    def show(self, is_txt2img):
        return scripts.AlwaysVisible

    def ui(self, is_txt2img):
        sagemaker_endpoint, sd_checkpoint, sd_checkpoint_refresh_button, textual_inversion_dropdown, lora_dropdown, hyperNetwork_dropdown, controlnet_dropdown, inference_job_dropdown, txt2img_inference_job_ids_refresh_button, primary_model_name, secondary_model_name, tertiary_model_name, modelmerger_merge_on_cloud= sagemaker_ui.create_ui()
        return [sagemaker_endpoint, sd_checkpoint, sd_checkpoint_refresh_button, textual_inversion_dropdown, lora_dropdown, hyperNetwork_dropdown, controlnet_dropdown, inference_job_dropdown, txt2img_inference_job_ids_refresh_button, primary_model_name, secondary_model_name, tertiary_model_name, modelmerger_merge_on_cloud]
    def process(self, p, sagemaker_endpoint, sd_checkpoint, sd_checkpoint_refresh_button, textual_inversion_dropdown, lora_dropdown, hyperNetwork_dropdown, controlnet_dropdown, choose_txt2img_inference_job_id, txt2img_inference_job_ids_refresh_button, primary_model_name, secondary_model_name, tertiary_model_name, modelmerger_on_cloud):
        pass

def on_after_component_callback(component, **_kwargs):
    global db_model_name, db_use_txt2img, db_sagemaker_train, db_save_config, cloud_db_model_name
    # if getattr(component, 'elem_id', None) is not None:
    #     print(getattr(component, 'elem_id', None))
    is_dreambooth_train = type(component) is gr.Button and getattr(component, 'elem_id', None) == 'db_train'
    is_dreambooth_model_name = type(component) is gr.Dropdown and \
                               (getattr(component, 'elem_id', None) == 'model_name' or \
                                (getattr(component, 'label', None) == 'Model' and getattr(component.parent.parent.parent.parent, 'elem_id', None) == 'ModelPanel'))
    is_cloud_dreambooth_model_name = type(component) is gr.Dropdown and \
                                     getattr(component, 'elem_id', None) == 'cloud_db_model_name'
    is_dreambooth_use_txt2img = type(component) is gr.Checkbox and getattr(component, 'label', None) == 'Use txt2img'
    is_db_save_config = getattr(component, 'elem_id', None) == 'db_save_config'
    if is_dreambooth_train:
        print('Add SageMaker button')
        db_sagemaker_train = gr.Button(value="SageMaker Train", elem_id = "db_sagemaker_train", variant='primary')
    if is_dreambooth_model_name:
        print('Get local model name')
        db_model_name = component
    if is_cloud_dreambooth_model_name:
        print('Get cloud model name')
        cloud_db_model_name = component
    if is_dreambooth_use_txt2img:
        print('Get use tet2img')
        db_use_txt2img = component
    if is_db_save_config:
        print('Get save config button')
        db_save_config = component
    # After all requiment comment is loaded, add the SageMaker training button click callback function.
    if cloud_db_model_name is not None and db_model_name is not None and \
            db_use_txt2img is not None and db_sagemaker_train is not None and \
            (is_dreambooth_train or is_dreambooth_model_name or is_dreambooth_use_txt2img or is_cloud_dreambooth_model_name):
        print('Create click callback')
        db_model_name.value = "dummy_local_model"
        db_sagemaker_train.click(
            fn=cloud_train,
            _js="db_start_sagemaker_train",
            inputs=[
                cloud_db_model_name,
                db_use_txt2img,
            ],
            outputs=[]
        )
    # Hook image display logic
    global txt2img_gallery, txt2img_generation_info, txt2img_html_info, txt2img_show_hook
    is_txt2img_gallery = type(component) is gr.Gallery and getattr(component, 'elem_id', None) == 'txt2img_gallery'
    is_txt2img_generation_info = type(component) is gr.Textbox and getattr(component, 'elem_id', None) == 'generation_info_txt2img'
    is_txt2img_html_info = type(component) is gr.HTML and getattr(component, 'elem_id', None) == 'html_info_txt2img'
    if is_txt2img_gallery:
        print("create txt2img gallery")
        txt2img_gallery = component
    if is_txt2img_generation_info:
        print("create txt2img generation info")
        txt2img_generation_info = component
    if is_txt2img_html_info:
        print("create txt2img html info")
        txt2img_html_info = component
        # return test
    if sagemaker_ui.inference_job_dropdown is not None and txt2img_gallery is not None and txt2img_generation_info is not None and txt2img_html_info is not None and txt2img_show_hook is None:
        txt2img_show_hook = "finish"
        sagemaker_ui.inference_job_dropdown.change(
            fn=lambda selected_value: sagemaker_ui.fake_gan(selected_value),
            inputs=[sagemaker_ui.inference_job_dropdown],
            outputs=[txt2img_gallery, txt2img_generation_info, txt2img_html_info]
        )

        sagemaker_ui.sagemaker_endpoint.change(
            fn=lambda selected_value: sagemaker_ui.displayEndpointInfo(selected_value),
            inputs=[sagemaker_ui.sagemaker_endpoint],
            outputs=[txt2img_html_info]
        )

        sagemaker_ui.generate_on_cloud_button_with_js.click(
            fn=sagemaker_ui.generate_on_cloud_no_input,
            _js="txt2img_config_save",
            inputs=[sagemaker_ui.sagemaker_endpoint],
            outputs=[txt2img_gallery, txt2img_generation_info, txt2img_html_info]
        )
        sagemaker_ui.modelmerger_merge_on_cloud.click(
            fn=sagemaker_ui.modelmerger_on_cloud_func,
            # fn=None,
            _js="txt2img_config_save",
            inputs=[sagemaker_ui.sagemaker_endpoint],
            # inputs=[
            #     sagemaker_ui.primary_model_name,
            #     sagemaker_ui.secondary_model_name,
            #     sagemaker_ui.tertiary_model_name,
            # ],
            outputs=[
            ])


def update_connect_config(api_url, api_token):
    # Check if api_url ends with '/', if not append it
    if not api_url.endswith('/'):
        api_url += '/'

    save_variable_to_json('api_gateway_url', api_url)
    save_variable_to_json('api_token', api_token)
    global api_gateway_url
    api_gateway_url = get_variable_from_json('api_gateway_url')
    global api_key
    api_key = get_variable_from_json('api_token')
    print(f"update the api_url:{api_gateway_url} and token: {api_key}............")
    return "config updated to local config!"

def test_aws_connect_config(api_url, api_token):
    update_connect_config(api_url, api_token)
    api_url = get_variable_from_json('api_gateway_url')
    api_token = get_variable_from_json('api_token')
    if not api_url.endswith('/'):
        api_url += '/'
    print(f"get the api_url:{api_url} and token: {api_token}............")
    target_url = f'{api_url}inference/test-connection'
    headers = {
        "x-api-key": api_token,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(target_url,headers=headers)  # Assuming sagemaker_ui.server_request is a wrapper around requests
        response.raise_for_status()  # Raise an exception if the HTTP request resulted in an error
        r = response.json()
        return "Successfully Connected"
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to get server request. Details: {e}")
        return "failed to connect to backend server, please check the url and token"

def on_ui_tabs():
    buildin_model_list = ['AWS JumpStart Model','AWS BedRock Model','Hugging Face Model']
    with gr.Blocks() as sagemaker_interface:
        with gr.Row(equal_height=True, elem_id="aws_sagemaker_ui_row"):
            sm_load_params = gr.Button(value="Load Settings", elem_id="aws_load_params")
            sm_save_params = gr.Button(value="Save Settings", elem_id="aws_save_params")
            sm_train_model = gr.Button(value="Train", variant="primary", elem_id="aws_train_model")
            sm_generate_checkpoint = gr.Button(value="Generate Ckpt", elem_id="aws_gen_ckpt")
        with gr.Row():
            gr.HTML(value="Enter your API URL & Token to start the connection.", elem_id="hint_row")
        with gr.Row():
            with gr.Column(variant="panel", scale=1):
                gr.HTML(value="AWS Connection Setting")
                api_url_textbox = gr.Textbox(value=get_variable_from_json('api_gateway_url'), lines=1, placeholder="Please enter API Url", label="API Url",elem_id="aws_middleware_api")
                api_token_textbox = gr.Textbox(value=get_variable_from_json('api_token'), lines=1, placeholder="Please enter API Token", label="API Token", elem_id="aws_middleware_token")
                aws_connect_button = gr.Button(value="Update Setting", variant='primary',elem_id="aws_config_save")
                aws_connect_button.click(_js="update_auth_settings",
                                         fn=update_connect_config,
                                         inputs = [api_url_textbox, api_token_textbox],
                                         outputs= [])
                aws_test_button = gr.Button(value="Test Connection", variant='primary',elem_id="aws_config_test")
                test_connection_result = gr.Label();
                aws_test_button.click(test_aws_connect_config, inputs = [api_url_textbox, api_token_textbox], outputs=[test_connection_result])
            with gr.Column(variant="panel", scale=1.5):
                gr.HTML(value="<u><b>Cloud Assets Management</b></u>") 
                sagemaker_html_log = gr.HTML(elem_id=f'html_log_sagemaker')
                with gr.Blocks(title="Upload Model to S3", variant="panel"):
                    gr.HTML(value="Upload Model to S3") 
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            sd_checkpoints_path = gr.Textbox(value="", lines=1, placeholder="Please input absolute path", label="Stable Diffusion Checkpoints",elem_id="sd_checkpoints_path_textbox")
                        with gr.Column(variant="panel"):
                            textual_inversion_path = gr.Textbox(value="", lines=1, placeholder="Please input absolute path", label="Textual Inversion",elem_id="sd_textual_inversion_path_textbox")
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            lora_path = gr.Textbox(value="", lines=1, placeholder="Please input absolute path", label="LoRA",elem_id="sd_lora_path_textbox")
                            controlnet_model_path = gr.Textbox(value="", lines=1, placeholder="Please input absolute path", label="ControlNet-Model",elem_id="sd_controlnet_model_path_textbox")
                        with gr.Column(variant="panel"):
                            hypernetwork_path = gr.Textbox(value="", lines=1, placeholder="Please input absolute path", label="HyperNetwork",elem_id="sd_hypernetwork_path_textbox")

                    with gr.Row():
                        model_update_button = gr.Button(value="Upload Models to Cloud", variant="primary",elem_id="sagemaker_model_update_button", size=(200, 50))
                        model_update_button.click(_js="model_update",
                                          fn=sagemaker_ui.sagemaker_upload_model_s3,
                                          inputs=[sd_checkpoints_path, textual_inversion_path, lora_path, hypernetwork_path, controlnet_model_path],
                                          outputs=[test_connection_result])
                

                with gr.Blocks(title="Deploy New SageMaker Endpoint", variant='panel'):
                    gr.HTML(value="<u><b>Deploy New SageMaker Endpoint</b></u>") 
                    with gr.Row():
                        # instance_type_textbox = gr.Textbox(value="", lines=1, placeholder="Please enter Instance type, e.g. ml.g4dn.xlarge", label="SageMaker Instance Type",elem_id="sagemaker_inference_instance_type_textbox")
                        instance_type_dropdown = gr.Dropdown(label="SageMaker Instance Type", choices=["ml.g4dn.xlarge","ml.g4dn.2xlarge","ml.g4dn.4xlarge","ml.g4dn.8xlarge","ml.g4dn.12xlarge"], elem_id="sagemaker_inference_instance_type_textbox", value="ml.g4dn.xlarge")
                        # instance_count_textbox = gr.Textbox(value="", lines=1, placeholder="Please enter Instance count, e.g. 1,2", label="SageMaker Instance Count",elem_id="sagemaker_inference_instance_count_textbox", default=1)
                        instance_count_dropdown = gr.Dropdown(label="Please select Instance count", choices=["1","2","3","4"], elem_id="sagemaker_inference_instance_count_textbox", value="1")

                    with gr.Row():
                        sagemaker_deploy_button = gr.Button(value="Deploy", variant='primary',elem_id="sagemaker_deploy_endpoint_buttion")
                        sagemaker_deploy_button.click(sagemaker_ui.sagemaker_deploy,
                                              _js="deploy_endpoint", \
                                              inputs = [instance_type_dropdown, instance_count_dropdown],
                                              outputs=[test_connection_result])
                
                with gr.Blocks(title="Delete SageMaker Endpoint", variant='panel'):
                    gr.HTML(value="<u><b>Delete SageMaker Endpoint</b>(Work In Progress)</u>") 
                    sagemaker_endpoint_delete_dropdown = gr.Dropdown(choices=["endpoint1", "endpoint2", "endpoint3", "endpoint4"], value=["endpoint1", "endpoint2"], multiselect=True, interactive=False, label="")
                    sagemaker_endpoint_delete_button = gr.Button(value="Delete", variant='primary',interactive=False, elem_id="sagemaker_endpoint_delete_button")

                
                    
            with gr.Column(variant="panel", scale=1):
                gr.HTML(value="AWS Model Setting")
                with gr.Tab("Select"):
                    gr.HTML(value="AWS Built-in Model")
                    model_select_dropdown = gr.Dropdown(buildin_model_list, label="Select Built-In Model", elem_id="aws_select_model")
                with gr.Tab("Create"):
                    gr.HTML(value="AWS Custom Model")
                    model_name_textbox = gr.Textbox(value="", lines=1, placeholder="Please enter model name", label="Model Name")
                    model_create_button = gr.Button(value="Create Model", variant='primary',elem_id="aws_create_model")
                    # model_create_button.click(_js="create_model",
                    #                           fn=create_model,
                    #                           inputs = [model_name_textbox],
                    #                           outputs= [])

    return (sagemaker_interface, "Amazon SageMaker", "sagemaker_interface"),


script_callbacks.on_after_component(on_after_component_callback)
script_callbacks.on_ui_tabs(on_ui_tabs)
# create new tabs for create Model
origin_callback = script_callbacks.ui_tabs_callback

def avoid_duplicate_from_restart_ui(res):
    for extension_ui in res:
        if extension_ui[1] == 'Dreambooth':
            for key in list(extension_ui[0].blocks):
                val = extension_ui[0].blocks[key]
                if type(val) is gr.Tab:
                    if val.label == 'Select From Cloud':
                        return True

    return False


def ui_tabs_callback():
    res = origin_callback()
    if avoid_duplicate_from_restart_ui(res):
        return res
    for extension_ui in res:
        if extension_ui[1] == 'Dreambooth':
            for key in list(extension_ui[0].blocks):
                val = extension_ui[0].blocks[key]
                if type(val) is gr.Tab:
                    if val.label == 'Select':
                        with extension_ui[0]:
                            with val.parent:
                                with gr.Tab('Select From Cloud'):
                                    with gr.Row():
                                        cloud_db_model_name = gr.Dropdown(
                                            label="Model", choices=sorted(get_cloud_db_model_name_list()),
                                            elem_id="cloud_db_model_name"
                                        )
                                        create_refresh_button(
                                            cloud_db_model_name,
                                            get_cloud_db_model_name_list,
                                            lambda: {"choices": sorted(get_cloud_db_model_name_list())},
                                            "refresh_db_models",
                                        )
                                    with gr.Row():
                                        cloud_db_snapshot = gr.Dropdown(
                                            label="Cloud Snapshot to Resume",
                                            choices=sorted(get_cloud_model_snapshots()),
                                            elem_id="cloud_snapshot_to_resume_dropdown"
                                        )
                                        create_refresh_button(
                                            cloud_db_snapshot,
                                            get_cloud_model_snapshots,
                                            lambda: {"choices": sorted(get_cloud_model_snapshots())},
                                            "refresh_db_snapshots",
                                        )
                                    with gr.Row(visible=False) as lora_model_row:
                                        cloud_db_lora_model_name = gr.Dropdown(
                                            label="Lora Model", choices=get_sorted_lora_cloud_models(),
                                            elem_id="cloud_lora_model_dropdown"
                                        )
                                        create_refresh_button(
                                            cloud_db_lora_model_name,
                                            get_sorted_lora_cloud_models,
                                            lambda: {"choices": get_sorted_lora_cloud_models()},
                                            "refresh_lora_models",
                                        )
                                    with gr.Row():
                                        gr.HTML(value="Loaded Model from Cloud:")
                                        cloud_db_model_path = gr.HTML()
                                    with gr.Row():
                                        gr.HTML(value="Cloud Model Revision:")
                                        cloud_db_revision = gr.HTML(elem_id="cloud_db_revision")
                                    with gr.Row():
                                        gr.HTML(value="Cloud Model Epoch:")
                                        cloud_db_epochs = gr.HTML(elem_id="cloud_db_epochs")
                                    with gr.Row():
                                        gr.HTML(value="V2 Model From Cloud:")
                                        cloud_db_v2 = gr.HTML(elem_id="cloud_db_v2")
                                    with gr.Row():
                                        gr.HTML(value="Has EMA:")
                                        cloud_db_has_ema = gr.HTML(elem_id="cloud_db_has_ema")
                                    with gr.Row():
                                        gr.HTML(value="Source Checkpoint From Cloud:")
                                        cloud_db_src = gr.HTML()
                                    with gr.Row():
                                        gr.HTML(value="Cloud DB Status:")
                                        cloud_db_status = gr.HTML(elem_id="db_status", value="")
                                    with gr.Row():
                                        gr.HTML(value="<b>Training Jobs Details:<b/>")
                                    with gr.Row():
                                        gr.Dataframe(
                                            headers=["id", "model name", "status", "sagemaker train name"],
                                            datatype=["str", "str", "str", "str"],
                                            col_count=(4, "fixed"),
                                            value=get_train_job_list,
                                            interactive=False,
                                            every=3
                                            # show_progress=True
                                        )
                                with gr.Tab('Create From Cloud'):
                                    with gr.Column():
                                        cloud_db_create_model = gr.Button(
                                            value="Create Model From Cloud", variant="primary"
                                        )
                                    cloud_db_new_model_name = gr.Textbox(label="Name", placeholder="Model names can only contain alphanumeric and -")
                                    with gr.Row():
                                        cloud_db_create_from_hub = gr.Checkbox(
                                            label="Create From Hub", value=False
                                        )
                                        cloud_db_512_model = gr.Checkbox(label="512x Model", value=True)
                                    with gr.Column(visible=False) as hub_row:
                                        cloud_db_new_model_url = gr.Textbox(
                                            label="Model Path",
                                            placeholder="runwayml/stable-diffusion-v1-5",
                                            elem_id="cloud_db_model_path_text_box"
                                        )
                                        cloud_db_new_model_token = gr.Textbox(
                                            label="HuggingFace Token", value=""
                                        )
                                    with gr.Column(visible=True) as local_row:
                                        with gr.Row():
                                            cloud_db_new_model_src = gr.Dropdown(
                                                label="Source Checkpoint",
                                                choices=sorted(get_sd_cloud_models()),
                                                elem_id="cloud_db_source_checkpoint_dropdown"
                                            )
                                            create_refresh_button(
                                                cloud_db_new_model_src,
                                                get_sd_cloud_models,
                                                lambda: {"choices": sorted(get_sd_cloud_models())},
                                                "refresh_sd_models",
                                            )
                                    cloud_db_new_model_extract_ema = gr.Checkbox(
                                        label="Extract EMA Weights", value=False
                                    )
                                    cloud_db_train_unfrozen = gr.Checkbox(label="Unfreeze Model", value=False, elem_id="cloud_db_unfreeze_model_checkbox")
                                    with gr.Row():
                                        gr.HTML(value="<b>Model Creation Jobs Details:<b/>")
                                    with gr.Row():
                                        gr.Dataframe(
                                            headers=["id", "model name", "status"],
                                            datatype=["str", "str", "str"],
                                            col_count=(3, "fixed"),
                                            value=get_create_model_job_list,
                                            interactive=False,
                                            every=3
                                            # show_progress=True
                                        )

                                def toggle_new_rows(create_from):
                                    return gr.update(visible=create_from), gr.update(visible=not create_from)

                                cloud_db_create_from_hub.change(
                                    fn=toggle_new_rows,
                                    inputs=[cloud_db_create_from_hub],
                                    outputs=[hub_row, local_row],
                                )

                                cloud_db_model_name.change(
                                    _js="clear_loaded",
                                    fn=wrap_load_model_params,
                                    inputs=[cloud_db_model_name],
                                    outputs=[
                                        cloud_db_model_path,
                                        cloud_db_revision,
                                        cloud_db_epochs,
                                        cloud_db_v2,
                                        cloud_db_has_ema,
                                        cloud_db_src,
                                        cloud_db_snapshot,
                                        cloud_db_lora_model_name,
                                        cloud_db_status,
                                    ],
                                )
                                cloud_db_create_model.click(
                                    fn=cloud_create_model,
                                    _js="check_create_model_params",
                                    inputs=[
                                        cloud_db_new_model_name,
                                        cloud_db_new_model_src,
                                        cloud_db_create_from_hub,
                                        cloud_db_new_model_url,
                                        cloud_db_new_model_token,
                                        cloud_db_new_model_extract_ema,
                                        cloud_db_train_unfrozen,
                                        cloud_db_512_model,
                                    ],
                                    outputs=[
                                        # cloud_db_model_name,
                                        # cloud_db_model_path,
                                        # cloud_db_revision,
                                        # cloud_db_epochs,
                                        # cloud_db_src,
                                        # cloud_db_has_ema,
                                        # cloud_db_v2,
                                        # cloud_db_resolution,
                                        # cloud_db_status,
                                    ]
                                )
                    break
    return res

script_callbacks.ui_tabs_callback = ui_tabs_callback

def get_sorted_lora_cloud_models():
    return ["ran", "ate", "slept"]

def get_cloud_model_snapshots():
    return ["ran", "swam", "slept"]

def get_cloud_db_models(types="Stable-diffusion", status="Complete"):
    try:
        api_gateway_url = get_variable_from_json('api_gateway_url')
        print("Get request for model list.")
        if api_gateway_url is None:
            print(f"failed to get the api_gateway_url, can not fetch date from remote")
            return []
        url = f"{api_gateway_url}models?"
        if types:
            url = f"{url}types={types}&"
        if status:
            url = f"{url}status={status}&"
        url = url.strip("&")
        response = requests.get(url=url, headers={'x-api-key': get_variable_from_json('api_token')}).json()
        model_list = []
        if "models" not in response:
            return []
        for model in response["models"]:
            model_list.append(model)
            params = model['params']
            if 'resp' in params:
                db_config = params['resp']['config_dict']
                # TODO:
                model_dir = f"{base_model_folder}/{model['model_name']}"
                for k in db_config:
                    if type(db_config[k]) is str:
                        db_config[k] = db_config[k].replace("/opt/ml/code/", "")
                        db_config[k] = db_config[k].replace("models/dreambooth/", base_model_folder)

                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                with open(f"{model_dir}/db_config.json", "w") as db_config_file:
                    json.dump(db_config, db_config_file)
        return model_list
    except Exception as e:
        print('Failed to get cloud models.')
        print(e)
        return []

def get_cloud_ckpts():
    try:
        api_gateway_url = get_variable_from_json('api_gateway_url')
        print("Get request for model list.")
        if api_gateway_url is None:
            print(f"failed to get the api_gateway_url, can not fetch date from remote")
            return []

        url = api_gateway_url + "checkpoints?status=Active&types=Stable-diffusion"
        response = requests.get(url=url, headers={'x-api-key': get_variable_from_json('api_token')}).json()
        if "checkpoints" not in response:
            return []
        global ckpt_dict
        for ckpt in response["checkpoints"]:
            # Only get ckpts whose name is not empty.
            if len(ckpt['name']) > 0:
                ckpt_key = f"cloud-{ckpt['name'][0]}-{ckpt['id']}"
                ckpt_dict[ckpt_key] = ckpt
    except Exception as e:
        print('Failed to get cloud ckpts.')
        print(e)
        return []

def get_cloud_ckpt_name_list():
    get_cloud_ckpts()
    return ckpt_dict.keys()

def get_cloud_db_model_name_list():
    model_list = get_cloud_db_models()
    if model_list is None:
        model_name_list = []
    else:
        model_name_list = [model['model_name'] for model in model_list]
    return model_name_list

# get local and cloud checkpoints.
def get_sd_cloud_models():
    sd_models.list_models()
    local_sd_list = sd_models.checkpoints_list
    names = []
    for key in local_sd_list:
        names.append(f"local-{key}")
    names += get_cloud_ckpt_name_list()
    return names

def async_prepare_for_training_on_sagemaker(
        model_id: str,
        model_name: str,
        s3_model_path: str,
        data_path_list: list,
        class_data_path_list: list,
        db_config_path: str,
):
    url = get_variable_from_json('api_gateway_url')
    api_key = get_variable_from_json('api_token')
    if url is None or api_key is None:
        logging.error("Url or API-Key is not setting.")
        return
    url += "train"
    upload_files = []
    db_config_tar = f"db_config.tar"
    os.system(f"tar cvf {db_config_tar} {db_config_path}")
    upload_files.append(db_config_tar)
    data_tar_list = []
    for data_path in data_path_list:
        data_tar = f'data_{data_path.replace("/", "-")}.tar'
        data_tar_list.append(data_tar)
        print("Pack the data file.")
        os.system(f"tar cvf {data_tar} {data_path}")
        upload_files.append(data_tar)
    class_data_tar_list = []
    for class_data_path in class_data_path_list:
        class_data_tar = f'class_data_{class_data_path.replace("/", "-")}.tar'
        class_data_tar_list.append(class_data_tar)
        upload_files.append(class_data_tar)
        print("Pack the class data file.")
        os.system(f"tar cvf {class_data_tar} {class_data_path}")
    payload = {
        "train_type": "Stable-diffusion",
        "model_id": model_id,
        "filenames": upload_files,
        "params": {
            "training_params": {
                "s3_model_path": s3_model_path,
                "model_name": model_name,
                "data_tar_list": data_tar_list,
                "class_data_tar_list": class_data_tar_list,
            }
        }
    }
    print("Post request for upload s3 presign url.")
    response = requests.post(url=url, json=payload, headers={'x-api-key': api_key})
    json_response = response.json()
    for local_tar_path, s3_presigned_url in response.json()["s3PresignUrl"].items():
        upload_file_to_s3_by_presign_url(local_tar_path, s3_presigned_url)
    return json_response

def wrap_load_model_params(model_name):
    origin_model_path = dreambooth_shared.dreambooth_models_path
    setattr(dreambooth_shared, 'dreambooth_models_path', base_model_folder)
    resp = load_model_params(model_name)
    setattr(dreambooth_shared, 'dreambooth_models_path', origin_model_path)
    return resp

def wrap_get_local_config(model_name):
    config = from_file(model_name)
    return config

def wrap_get_cloud_config(model_name):
    origin_model_path = dreambooth_shared.dreambooth_models_path
    setattr(dreambooth_shared, 'dreambooth_models_path', base_model_folder)
    config = from_file(model_name)
    setattr(dreambooth_shared, 'dreambooth_models_path', origin_model_path)
    return config

def wrap_save_config(model_name):
    origin_model_path = dreambooth_shared.dreambooth_models_path
    setattr(dreambooth_shared, 'dreambooth_models_path', base_model_folder)
    save_config(model_name)
    setattr(dreambooth_shared, 'dreambooth_models_path', origin_model_path)

def async_create_model_on_sagemaker(
        new_model_name: str,
        ckpt_path: str,
        from_hub=False,
        new_model_url="",
        new_model_token="",
        extract_ema=False,
        train_unfrozen=False,
        is_512=True,
):
    params = copy.deepcopy(locals())
    if len(params["ckpt_path"]) == 0 or len(params["new_model_name"]) == 0:
        logging.error("ckpt_path or model_name is not setting.")
        return
    if re.match("^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,30}$", params["new_model_name"]) is None:
        logging.error("model_name is not match pattern.")
        return
    ckpt_key = ckpt_path
    url = get_variable_from_json('api_gateway_url')
    api_key = get_variable_from_json('api_token')
    if url is None or api_key is None:
        logging.error("Url or API-Key is not setting.")
        return
    url += "model"
    if params["ckpt_path"].startswith("cloud-"):
        if params["ckpt_path"] not in ckpt_dict:
            logging.error("Cloud checkpoint is not exist.")
            return
        ckpt_name_list = ckpt_dict[ckpt_key]["name"]
        if len(ckpt_name_list) == 0:
            logging.error("Checkpoint name error.")
            return
        params["ckpt_path"] = ckpt_name_list[0].rstrip(".tar")
        payload = {
            "model_type": "Stable-diffusion",
            "name": new_model_name,
            "filenames": [],
            "params": {
                "ckpt_from_cloud": True,
                "s3_ckpt_path": ckpt_dict[ckpt_key]["s3Location"],
                "create_model_params": params
            }
        }
        print("Post request for upload s3 presign url.")
        response = requests.post(url=url, json=payload, headers={'x-api-key': api_key})
        json_response = response.json()
        model_id = json_response["job"]["id"]
        payload = {
            "model_id": model_id,
            "status": "Creating",
            "multi_parts_tags": {}
        }
    elif params["ckpt_path"].startswith("local-"):
        # The ckpt path has a hash suffix?
        params["ckpt_path"] = " ".join(params["ckpt_path"].split(" ")[:1])
        params["ckpt_path"] = params["ckpt_path"].lstrip("local-")
        # Prepare for creating model on cloud.
        local_model_path = f'models/Stable-diffusion/{params["ckpt_path"]}'
        local_tar_path = f'{params["ckpt_path"]}.tar'

        part_size = 1000 * 1024 * 1024
        file_size = os.stat(local_model_path)
        parts_number = math.ceil(file_size.st_size/part_size)

        payload = {
            "model_type": "Stable-diffusion",
            "name": new_model_name,
            "filenames": [{
                "filename": local_tar_path,
                "parts_number": parts_number
            }],
            "params": {"create_model_params": params}
        }
        print("Post request for upload s3 presign url.")
        response = requests.post(url=url, json=payload, headers={'x-api-key': api_key})
        json_response = response.json()
        model_id = json_response["job"]["id"]
        multiparts_tags=[]
        if not from_hub:
            print("Pack the model file.")
            os.system(f"tar cvf {local_tar_path} {local_model_path}")
            s3_base = json_response["job"]["s3_base"]
            print(f"Upload to S3 {s3_base}")
            print(f"Model ID: {model_id}")
            # Upload src model to S3.
            s3_signed_urls_resp = response.json()["s3PresignUrl"][local_tar_path]
            multiparts_tags = upload_multipart_files_to_s3_by_signed_url(
                local_tar_path,
                s3_signed_urls_resp,
                part_size
            )
            payload = {
                "model_id": model_id,
                "status": "Creating",
                "multi_parts_tags": {local_tar_path: multiparts_tags}
            }
    else:
        logging.error("Create model params error.")
        return
    # Start creating model on cloud.
    response = requests.put(url=url, json=payload, headers={'x-api-key': api_key})
    print(response)

def cloud_create_model(
        new_model_name: str,
        ckpt_path: str,
        from_hub=False,
        new_model_url="",
        new_model_token="",
        extract_ema=False,
        train_unfrozen=False,
        is_512=True,
):
    upload_thread = threading.Thread(target=async_create_model_on_sagemaker,
                                     args=(new_model_name, ckpt_path, from_hub, new_model_url, new_model_token, extract_ema, train_unfrozen, is_512))
    upload_thread.start()

def cloud_train(
        train_model_name: str,
        local_model_name=False
):
    # Get data path and class data path.
    print(f"Start cloud training {train_model_name}")
    config = wrap_get_local_config("dummy_local_model")
    data_path_list = []
    class_data_path_list = []
    for concept in config.concepts():
        if concept.instance_data_dir:
            data_path_list.append(concept.instance_data_dir)
        if concept.class_data_dir:
            class_data_path_list.append(concept.class_data_dir)
    model_list = get_cloud_db_models()
    db_config_path = "models/dreambooth/dummy_local_model/db_config.json"
    # db_config_path = f"models/dreambooth/{model_name}/db_config.json"
    # os.makedirs(os.path.dirname(db_config_path), exist_ok=True)
    # os.system(f"cp {dummy_db_config_path} {db_config_path}")
    for model in model_list:
        model_id = model["id"]
        model_name = model["model_name"]
        model_s3_path = model["output_s3_location"]
        if model_name == train_model_name:
            # upload_thread = threading.Thread(target=async_prepare_for_training_on_sagemaker,
            #                                 args=(model_id, model_name, s3_model_path,data_path, class_data_path))
            # upload_thread.start()
            response = async_prepare_for_training_on_sagemaker(
                model_id, model_name, model_s3_path, data_path_list, class_data_path_list, db_config_path)
            job_id = response["job"]["id"]
            url = get_variable_from_json('api_gateway_url')
            api_key = get_variable_from_json('api_token')
            if url is None or api_key is None:
                logging.error("Url or API-Key is not setting.")
                break
            url += "train"
            payload = {
                "train_job_id": job_id,
                "status": "Training"
            }
            # Start creating model on cloud.
            url = get_variable_from_json('api_gateway_url')
            api_key = get_variable_from_json('api_token')
            if url is None or api_key is None:
                logging.error("Url or API-Key is not setting.")
                break
            url += "train"
            payload = {
                "train_job_id": job_id,
                "status": "Training"
            }
            # Start creating model on cloud.
            response = requests.put(url=url, json=payload, headers={'x-api-key': api_key}).json()
            print(f"Start training response:\n{response}")


def get_train_job_list():
    # Start creating model on cloud.
    url = get_variable_from_json('api_gateway_url')
    api_key = get_variable_from_json('api_token')
    if url is None or api_key is None:
        logging.error("Url or API-Key is not setting.")
        return []

    table = []
    try:
        url += "trains?types=Stable-diffusion"
        response = requests.get(url=url, headers={'x-api-key': api_key}).json()
        response['trainJobs'].reverse()
        for trainJob in response['trainJobs']:
            table.append([trainJob['id'][:6], trainJob['modelName'], trainJob["status"], trainJob['sagemakerTrainName']])
    except requests.exceptions.RequestException as e:
        print(f"exception {e}")

    return table


def get_create_model_job_list():
    # Start creating model on cloud.
    url = get_variable_from_json('api_gateway_url')
    api_key = get_variable_from_json('api_token')
    if url is None or api_key is None:
        logging.error("Url or API-Key is not setting.")
        return []

    table = []
    try:
        url += "models?types=Stable-diffusion"
        response = requests.get(url=url, headers={'x-api-key': api_key}).json()
        response['models'].reverse()
        for model in response['models']:
            table.append([model['id'][:6], model['model_name'], model["status"]])
    except requests.exceptions.RequestException as e:
        print(f"exception {e}")

    return table
