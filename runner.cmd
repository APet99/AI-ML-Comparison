if "%~1"=="" (
    echo "No arguments supplied"
    echo "Supply name of virtual environment directory"
    exit
)

echo "Switching to Models branch make sure all changes are saved"
git switch Models
echo "Activating python virtual environment"
.\%1\Script\activate
echo "Installing all requirements"
pip install -r requirements.txt --force
pip install matplotlib~=3.3.3 --force
echo "Building models"
python build_train_models.py
echo "Gathering data"
python runner.py
