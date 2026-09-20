FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app.py .

COPY model/superkart_sales_model.pkl \
     ./model/superkart_sales_model.pkl

EXPOSE 5000

ENV MODEL_PATH=model/superkart_sales_model.pkl

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "app:app"]