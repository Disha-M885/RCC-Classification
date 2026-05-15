{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNtJLTfx6clixGYpcy4CYy8",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Disha-M885/RCC-Classification/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from fastapi import FastAPI, File, UploadFile\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import load_model\n",
        "from PIL import Image\n",
        "import numpy as np\n",
        "import io\n",
        "import os\n",
        "import gdown\n",
        "\n",
        "app = FastAPI()\n",
        "#11XxB4c0rpvRJQ0w5yI6VBnr3f4MyOZLe\n",
        "MODEL_PATH = \"RCC.keras\"\n",
        "\n",
        "if not os.path.exists(MODEL_PATH):\n",
        "\n",
        "    url = \"https://drive.google.com/uc?id=11XxB4c0rpvRJQ0w5yI6VBnr3f4MyOZLe\"\n",
        "\n",
        "    gdown.download(url, MODEL_PATH, quiet=False)\n",
        "\n",
        "model = load_model(\n",
        "    MODEL_PATH,\n",
        "    compile=False\n",
        ")\n",
        "\n",
        "classes = [\n",
        "    \"Grade-0\",\n",
        "    \"Grade-1\",\n",
        "    \"Grade-2\",\n",
        "    \"Grade-3\",\n",
        "    \"Grade-4\"\n",
        "]\n",
        "\n",
        "def preprocess(image):\n",
        "    image = image.resize((224,224))\n",
        "    image = np.array(image) / 255.0\n",
        "    image = np.expand_dims(image, axis=0)\n",
        "    return image\n",
        "\n",
        "@app.post(\"/predict\")\n",
        "async def predict(file: UploadFile = File(...)):\n",
        "\n",
        "    image = Image.open(io.BytesIO(await file.read())).convert(\"RGB\")\n",
        "\n",
        "    img = preprocess(image)\n",
        "\n",
        "    pred = model.predict(img)\n",
        "\n",
        "    class_index = np.argmax(pred)\n",
        "\n",
        "    return {\n",
        "        \"prediction\": classes[class_index],\n",
        "        \"confidence\": float(np.max(pred))\n",
        "    }"
      ],
      "metadata": {
        "id": "jSBM-2vAoFcE"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9WB-zF2moBVq",
        "outputId": "844be309-c46c-4e8e-a004-90cd610eaf68"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Public URL: NgrokTunnel: \"https://amends-tweezers-asleep.ngrok-free.dev\" -> \"http://localhost:8000\"\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "INFO:     Started server process [7408]\n",
            "INFO:     Waiting for application startup.\n",
            "INFO:     Application startup complete.\n",
            "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)\n",
            "INFO:     Shutting down\n",
            "INFO:     Waiting for application shutdown.\n",
            "INFO:     Application shutdown complete.\n",
            "INFO:     Finished server process [7408]\n"
          ]
        }
      ],
      "source": [
        "from pyngrok import ngrok\n",
        "import nest_asyncio\n",
        "import uvicorn\n",
        "\n",
        "nest_asyncio.apply()\n",
        "from pyngrok import ngrok\n",
        "\n",
        "ngrok.set_auth_token(\"3DXbzTtkmsQqfB6YX5BeLBVr3wf_2EiVn6Xsagb2WdEnnM7pt\")\n",
        "\n",
        "public_url = ngrok.connect(8000)\n",
        "\n",
        "print(\"Public URL:\", public_url)\n",
        "\n",
        "config = uvicorn.Config(app, host=\"0.0.0.0\", port=8000)\n",
        "server = uvicorn.Server(config)\n",
        "\n",
        "await server.serve()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install fastapi uvicorn pyngrok python-multipart nest-asyncio"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PJUtaqiBoKaI",
        "outputId": "5f7770a7-8045-49d3-9c01-2320f86ffabe"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: fastapi in /usr/local/lib/python3.12/dist-packages (0.136.1)\n",
            "Requirement already satisfied: uvicorn in /usr/local/lib/python3.12/dist-packages (0.46.0)\n",
            "Collecting pyngrok\n",
            "  Downloading pyngrok-8.1.2-py3-none-any.whl.metadata (8.6 kB)\n",
            "Requirement already satisfied: python-multipart in /usr/local/lib/python3.12/dist-packages (0.0.26)\n",
            "Requirement already satisfied: nest-asyncio in /usr/local/lib/python3.12/dist-packages (1.6.0)\n",
            "Requirement already satisfied: starlette>=0.46.0 in /usr/local/lib/python3.12/dist-packages (from fastapi) (0.52.1)\n",
            "Requirement already satisfied: pydantic>=2.9.0 in /usr/local/lib/python3.12/dist-packages (from fastapi) (2.12.3)\n",
            "Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.12/dist-packages (from fastapi) (4.15.0)\n",
            "Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from fastapi) (0.4.2)\n",
            "Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from fastapi) (0.0.4)\n",
            "Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.12/dist-packages (from uvicorn) (8.3.3)\n",
            "Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.12/dist-packages (from uvicorn) (0.16.0)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.12/dist-packages (from pyngrok) (6.0.3)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.9.0->fastapi) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.9.0->fastapi) (2.41.4)\n",
            "Requirement already satisfied: anyio<5,>=3.6.2 in /usr/local/lib/python3.12/dist-packages (from starlette>=0.46.0->fastapi) (4.13.0)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.12/dist-packages (from anyio<5,>=3.6.2->starlette>=0.46.0->fastapi) (3.13)\n",
            "Downloading pyngrok-8.1.2-py3-none-any.whl (25 kB)\n",
            "Installing collected packages: pyngrok\n",
            "Successfully installed pyngrok-8.1.2\n"
          ]
        }
      ]
    }
  ]
}