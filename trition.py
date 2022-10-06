import tritonclient.grpc as grpclient
from tritonclient.utils import InferenceServerException
import sys

def init_triton(url):
    try:
        triton_client = grpclient.InferenceServerClient(
            url=url,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None
        )
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)
    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)
    if not triton_client.is_model_ready('yolov7'):
        print("FAILED : is_model_ready")
        sys.exit(1)
    return triton_client
    # if FLAGS.model_info:
    #     # Model metadata
    #     try:
    #         metadata = triton_client.get_model_metadata(FLAGS.model)
    #         print(metadata)
    #     except InferenceServerException as ex:
    #         if "Request for unknown model" not in ex.message():
    #             print("FAILED : get_model_metadata")
    #             print("Got: {}".format(ex.message()))
    #             sys.exit(1)
    #         else:
    #             print("FAILED : get_model_metadata")
    #             sys.exit(1)

    #     # Model configuration
    #     try:
    #         config = triton_client.get_model_config(FLAGS.model)
    #         if not (config.config.name == FLAGS.model):
    #             print("FAILED: get_model_config")
    #             sys.exit(1)
    #         print(config)
    #     except InferenceServerException as ex:
    #         print("FAILED : get_model_config")
    #         print("Got: {}".format(ex.message()))
    #         sys.exit(1)