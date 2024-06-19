# 使用官方 Python 运行时作为父镜像
FROM python:3.10.6-slim

# 安裝所需的包和依賴項
RUN apt-get update && \
    apt-get install -y wget gcc make libsqlite3-dev \
    zlib1g-dev libffi-dev libssl-dev curl

# 從源碼安裝 SQLite
RUN wget https://www.sqlite.org/2021/sqlite-autoconf-3350500.tar.gz && \
    tar xvfz sqlite-autoconf-3350500.tar.gz && \
    cd sqlite-autoconf-3350500 && \
    ./configure && \
    make && \
    make install

# 確保使用新安裝的 sqlite3
RUN ldconfig

# 從源碼編譯 Python
ENV PYTHON_VERSION=3.10.6
RUN curl -O https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make altinstall

WORKDIR /app

COPY . /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

ENV NAME World

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
