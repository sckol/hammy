# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Add current folder/hammy-lib to PYTHONPATH
$env:PYTHONPATH = "$($pwd.Path)\hammy-lib"

# Display confirmation message
Write-Host "Virtual environment activated and hammy-lib added to PYTHONPATH"
Write-Host "Current PYTHONPATH: $env:PYTHONPATH"