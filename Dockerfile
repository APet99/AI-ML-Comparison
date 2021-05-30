FROM python:3.7

COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD streamlit run home.py --server.enableCORS false
#ENTRYPOINT ["streamlit","run"]
#CMD ["home.py"]

#FROM python:3.7
#EXPOSE 8501
#
#COPY requirements.txt ./requirements.txt
#RUN pip3 install -r requirements.txt
#COPY . .
#CMD streamlit run home.py

#FROM ubuntu:18.04
#
#RUN apt-get update && apt-get install python3.7 -y && apt-get install python3-pip -y
#
#EXPOSE 8501
#
#WORKDIR /streamlit-docker
#COPY requirements.txt ./requirements.txt
#
#RUN pip3 install -r requirements.txt
#COPY . .
#
#CMD streamlit run home.py
#
#ENV LC_ALL=C.UTF-8
#ENV LANG=C.UTF-8
#RUN mkdir -p /root/.streamlit
#RUN bash -c 'echo -e "\
#[general]\n\
#email = \"\"\n\
#" > /root/.streamlit/credentials.toml'
#
#RUN bash -c 'echo -e "\
#[server]\n\
#enableCORS = false\n\
#" > /root/.streamlit/config.toml'

