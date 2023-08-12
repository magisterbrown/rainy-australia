
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev
COPY tests/  tests
COPY inference_instance/ inference_instance
RUN ls
RUN pip install -r inference_instance/requirements.txt
ENTRYPOINT python3 -m unittest discover tests
