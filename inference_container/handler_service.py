from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer

import numpy as np
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
)

class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    
    Determines specific default inference handlers to use based on model being used.
    
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
    """
    
    class DefaultLightfmInferenceHandler(default_inference_handler.DefaultInferenceHandler):
        VALID_CONTENT_TYPES = (content_types.NPY)

        @staticmethod
        def default_model_fn(self, model_dir):
            import pickle

            logger.info('Loading LightFM model...')
            return pickle.load(open( model_dir + "/model.pickle", "rb" ))

        @staticmethod
        def default_input_fn(self, input_data, content_type):
            """A default input_fn that can handle JSON, CSV and NPZ formats.

                Args:
                    input_data: the request payload serialized in the content_type format
                    content_type: the request content_type

                Returns: JSON
            """
            return decoder.decode(input_data, content_type)

        @staticmethod
        def default_predict_fn(self, data, model):
            """A default predict_fn for. Calls a model on data deserialized in input_fn.

                Args:
                    data: input data (numpy array) for prediction deserialized by input_fn
                    model: LightFM model loaded in memory by model_fn

                Returns: a prediction
            """

            f: lambda x: model.predict(x)
            return f(data)

        @staticmethod
        def default_output_fn(self, prediction, accept):
            """A default output_fn. Serializes predictions from predict_fn to JSON, CSV or NPY format.

                Args:
                    prediction: a prediction result from predict_fn
                    accept: type which the output data needs to be serialized

                Returns: output data serialized
            """
            return encoder.encode(prediction, accept)    
    
    def __init__(self):
        transformer = Transformer(default_inference_handler=self.DefaultLightfmInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)        