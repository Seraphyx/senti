# Use an official Python runtime as a parent image
# FROM python:3.6-slim
# FROM tensorflow/tensorflow
FROM tiangolo/uwsgi-nginx-flask

LABEL maintainer="seraphyx@github.com"

# Add git
RUN apt-get update && apt-get install -y --no-install-recommends git

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Ser Flask app
ENV FLASK_APP=$PWD/app/server/app.py

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80 5000

# Install custom keras_text
RUN pip install git+git://github.com/Seraphyx/keras-text.git@master

# Install Spacy
# RUN conda install -y -c conda-forge spacy

# Install all English Models
RUN python -m spacy download en
# RUN python -m spacy download en_core_web_sm
# RUN python -m spacy download en_core_web_md
# RUN python -m spacy download en_core_web_lg
# RUN python -m spacy download en_vectors_web_lg

# Install NLTK
# RUN conda install -c anaconda nltk



# Run app.py when the container launches
# CMD ["flask", "run"]
# CMD ["jupyter", "notebook"]

# Build with name
# docker build -t senti .
#
# View images
# docker images
#
# Run the image in detach mode
# docker run -it --rm -p 5000:5000 senti
#
# For proper use replace "/home/ec2-user/ds-docker" with the current directory
# docker run -it --rm -p 8888:8888 -v /home/ec2-user/ds-docker:/home/jovyan/work senti
#
# View Image ID
# docker ps -a
# docker container ls
#
# Enter the running docker container
# docker exec -it [container-id] bash
#
# In order to leave the running container press 'Ctrl + p' and 'Ctrl + q'
#