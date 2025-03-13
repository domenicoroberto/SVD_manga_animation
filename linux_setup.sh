#!/bin/bash

# Aggiorna pip
echo "🔄 Aggiornamento di pip..."
pip install --upgrade pip

# Installa Torch con il comando specifico
echo "🔥 Installazione di Torch e dipendenze CUDA..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
pip install torchao --index-url https://download.pytorch.org/whl/nightly/cu124

# Installa le dipendenze da due file diversi
echo "📦 Installazione delle dipendenze da requirements.txt"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements1.txt non trovato. Saltando..."
fi

pip install diffusers
pip install moviepy==1.0.3
pip install gradio
pip install transformers
pip install av==13.1.0
pip install matplotlib
pip install accelerate
pip install lpips
pip install opencv-python
pip install einops
pip install wandb
pip install bitsandbytes
pip install peft

# Installa una dipendenza extra senza dipendenze aggiuntive
echo "📦 Installazione di facenet_pytorch senza dipendenze aggiuntive..."
pip install --no-deps facenet_pytorch==2.6.0

# Esegui lo script Python
echo "🚀 Download del modello da huggingface..."
python models.py

echo "✅ Setup completato con successo!"
