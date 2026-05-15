{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOn2/W3jKQpV+Qb4348x3NM",
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
        "\n",
        "app = FastAPI()\n",
        "\n",
        "model = load_model(\n",
        "    \"/content/drive/MyDrive/RCC/RCC.keras\",\n",
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
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9WB-zF2moBVq"
      },
      "outputs": [],
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
      "source": [],
      "metadata": {
        "id": "PJUtaqiBoKaI"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}