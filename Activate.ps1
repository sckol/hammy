# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Add current folder/hammy_lib to PYTHONPATH
$env:PYTHONPATH = "$($pwd.Path)\hammy_lib"

# Display confirmation message
Write-Host "Virtual environment activated and hammy_lib added to PYTHONPATH"
Write-Host "Current PYTHONPATH: $env:PYTHONPATH"