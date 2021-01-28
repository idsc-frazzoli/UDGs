
FROM python:3.8

RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc libomp-dev \
  && rm -rf /var/lib/apt/lists/*

#VOLUME ["/forces-pro"]
#ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages/forces-pro":${PYTHONPATH}

WORKDIR /gokart-mpc
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN echo $PYTHONPATH
COPY . .
#RUN find .
#ENV DISABLE_CONTRACTS=1

#RUN pipdeptree
RUN python setup.py develop --no-deps
#RUN dg-demo --help
#CMD ["dg-demo"]