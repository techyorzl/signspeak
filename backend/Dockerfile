FROM pytorch/pytorch:latest  

RUN apt update && apt install -y curl net-tools iputils-ping ffmpeg libsm6 libxext6 libgl1 supervisor

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

COPY supervisord.conf /etc/supervisord.conf

EXPOSE 8001 8002

CMD ["supervisord", "-c", "/etc/supervisord.conf"]