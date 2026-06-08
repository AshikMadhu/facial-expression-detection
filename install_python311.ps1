# ==============================================================================
# EmotionSense AI: Python 3.11.9 Silent Installer Script
# Installs Python 3.11 to support TensorFlow dependencies natively.
# ==============================================================================

$installerUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$installerPath = Join-Path $PSScriptRoot "python-installer.exe"
$targetPythonPath = "C:\Users\ASHIK\Desktop\emotion\python311\python.exe"

if (Test-Path $targetPythonPath) {
    Write-Host "Python 3.11 is already installed at: $targetPythonPath" -ForegroundColor Green
    Exit 0
}

Write-Host "Downloading Python 3.11.9 from official repository..." -ForegroundColor Cyan
try {
    # Ensure TLS 1.2 is enabled for secure downloading
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "Download complete." -ForegroundColor Green
} catch {
    Write-Error "Failed to download installer: $_"
    Exit 1
}

Write-Host "Starting silent installation of Python 3.11 (User-only, no admin privileges required)..." -ForegroundColor Cyan
try {
    # Install options: Quiet mode, User-level only, do not append to system path, install locally in workspace
    $args = "/quiet InstallAllUsers=0 PrependPath=0 TargetDir=C:\Users\ASHIK\Desktop\emotion\python311"
    $process = Start-Process -FilePath $installerPath -ArgumentList $args -PassThru -Wait
    
    if ($process.ExitCode -ne 0) {
        Write-Error "Installer exited with non-zero status code: $($process.ExitCode)"
        Exit 1
    }
} catch {
    Write-Error "Failed to execute installer process: $_"
    Exit 1
} finally {
    # Delete temporary installer
    if (Test-Path $installerPath) {
        Remove-Item $installerPath -Force
    }
}

# Verify success
if (Test-Path $targetPythonPath) {
    Write-Host "Python 3.11 successfully installed and verified at: $targetPythonPath" -ForegroundColor Green
} else {
    Write-Error "Failed to locate Python 3.11 after installation. Path check failed: $targetPythonPath"
    Exit 1
}
