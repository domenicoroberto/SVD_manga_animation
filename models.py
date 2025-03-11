from huggingface_hub import snapshot_download

# Percorso della cartella dove vuoi salvare il modello
local_save_path = "models"

# Scarica solo la cartella "unet" e salva in una cartella specifica
snapshot_download(
    repo_id="stabilityai/stable-video-diffusion-img2vid",

    local_dir=local_save_path  # Cartella di destinazione
)

print(f"Modello salvato in {local_save_path}")


