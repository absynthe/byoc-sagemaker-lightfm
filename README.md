# Overview

In this example we show how to package a LightFM container, extending the SageMaker scikit-learn container, with a Python example which works with the MovieLens dataset. By extending the SageMaker scikit-learn container we already have some of the necessary training and inference toolkits already preinstalled on the container to make it work with SageMaker. We will still have to write code to make sure that the LightFM library can nicely interact with the SageMaker entry points.

# Overview
Please use the ['Extending a container' notebook](./extending_a_container.ipynb) as a starting point