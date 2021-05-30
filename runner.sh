if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo "Supply name of virtual environment directory"
    exit
fi

echo "Activating python virtual environment"
source $1/bin/activate
echo "Installing all requirements"
pip install -r requirements.txt --force
pip install matplotlib~=3.3.3 --force
echo "Building models"
python build_train_models.py
echo "Gathering data"
python runner.py
