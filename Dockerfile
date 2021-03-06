FROM python: 3.6.13

WORKDIR ./app

COPY requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt

COPY ./ ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
#CMD ["python", "app/main.py"]
