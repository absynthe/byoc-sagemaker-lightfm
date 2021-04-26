from sagemaker_inference import model_server

HANDLER_SERVICE = '/home/model-server/handler_service.py:handle'

def main():
    model_server.start_model_server(handler_service=HANDLER_SERVICE)