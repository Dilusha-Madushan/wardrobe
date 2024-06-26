import os

from dotenv import load_dotenv
load_dotenv()

# Disable Hugging Face Hub symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run()

