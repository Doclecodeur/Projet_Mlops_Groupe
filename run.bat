@echo off
echo ================================
echo   Projet MLOps - Credit Risk
echo   Wilguy, Lydia, Soraya
echo ================================
echo.

echo [1/3] Installation des dependances...
pip install -r requirements.txt
echo.

echo [2/3] Entrainement du modele...
python train.py
echo.

echo [3/3] Lancement de l'API Flask...
echo Application disponible sur : http://localhost:5000
echo.
python app.py

pause
