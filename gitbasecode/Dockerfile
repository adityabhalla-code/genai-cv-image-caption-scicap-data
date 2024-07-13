# pull python base image
FROM python:3.10

ENV PYTHONUNBUFFERED=1

# copy application files
ADD . .

# specify working directory
WORKDIR /

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pwd
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "-u", "../main.py"]
