FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3

# Install sagemaker-inference toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip install multi-model-server sagemaker-inference

# Install LightFM using conda to avoid compile issues
RUN conda install -yc conda-forge lightfm

# Create model folder
RUN mkdir -p /opt/ml/model

ENV SM_MODEL_DIR /opt/ml/model

# Copy entrypoint script to the image
COPY serve.py /usr/local/bin/serve.py
RUN chmod +x /usr/local/bin/serve.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY handler_service.py /home/model-server/handler_service.py

# Define an entrypoint script for the docker image
EXPOSE 8080
ENTRYPOINT ["python", "/usr/local/bin/serve.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]