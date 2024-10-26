# start from this pythorch image as a base as it supports data serialization
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime 
RUN pip install fluke-fl

# Install git
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get install python3-git -y

WORKDIR /fluke

# Mount volume for datasets
ENV EXP_CONF=""
ENV ALG_CONF=""
ENV MODE=""

ENTRYPOINT [ "sh", "-c", "wandb login ${API}; fluke-get config ${EXP_CONF}; fluke-get config ${ALG_CONF}; fluke --config=config/${EXP_CONF}.yaml ${MODE} config/${ALG_CONF}.yaml" ]