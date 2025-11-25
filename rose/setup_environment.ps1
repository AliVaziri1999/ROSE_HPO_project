Write-Host "=== ROSE HPO Setup (Windows) ===" -ForegroundColor Cyan

# ----------------------------------------
# 1. Check for WSL
# ----------------------------------------
Write-Host "[1] Checking for WSL..."

$wslStatus = wsl.exe --status 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL not installed. Installing..." -ForegroundColor Yellow
    wsl --install
    Write-Host "WSL installation started. Please REBOOT your computer and run this script again." -ForegroundColor Green
    exit
} else {
    Write-Host "WSL is installed." -ForegroundColor Green
}

# ----------------------------------------
# 2. Ensure Ubuntu exists
# ----------------------------------------
Write-Host "[2] Checking Ubuntu distribution..."

$distros = wsl.exe --list --quiet

if ($distros -notcontains "Ubuntu") {
    Write-Host "Ubuntu not found. Installing Ubuntu..." -ForegroundColor Yellow
    wsl --install -d Ubuntu
    Write-Host "Ubuntu installation started. Please REBOOT and run this script again." -ForegroundColor Green
    exit
} else {
    Write-Host "Ubuntu is installed." -ForegroundColor Green
}

# ----------------------------------------
# 3. Create project directory inside WSL
# ----------------------------------------
Write-Host "[3] Creating project directory inside WSL..."

wsl.exe mkdir -p /home/$env:USERNAME/rose_hpo_project

Write-Host "Copying project files into WSL..." -ForegroundColor Yellow

wsl.exe rm -rf /home/$env:USERNAME/rose_hpo_project/*
wsl.exe rm -rf /home/$env:USERNAME/rose_hpo_project/.git

# Copy all files from Windows → WSL
wsl.exe bash -c "cp -r /mnt/$(echo $PWD[0] | tr '[:upper:]' '[:lower:]')${PWD.Substring(1).Replace('\', '/')}/ /home/$env:USERNAME/rose_hpo_project/"

Write-Host "Project copied to WSL." -ForegroundColor Green

# ----------------------------------------
# 4. Create virtual environment in WSL
# ----------------------------------------
Write-Host "[4] Creating Python virtual environment in WSL..."

wsl.exe bash -c "
cd ~/rose_hpo_project &&
python3 -m venv rose_venv &&
source rose_venv/bin/activate &&
pip install --upgrade pip &&
pip install -r requirements.txt
"

Write-Host "Virtual environment created + requirements installed." -ForegroundColor Green

# ----------------------------------------
# DONE
# ----------------------------------------
Write-Host ""
Write-Host "==============================="
Write-Host " INSTALLATION COMPLETE!"
Write-Host "==============================="
Write-Host ""
Write-Host "To start working:"
Write-Host "1. Open WSL Ubuntu"
Write-Host "2. Run:"
Write-Host "     cd ~/rose_hpo_project"
Write-Host "     source rose_venv/bin/activate"
Write-Host "     python -m examples.hpo.runtime.00-runtime-grid ..."
Write-Host ""

pause
