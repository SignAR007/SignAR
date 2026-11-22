param(
    [string]$Dataset = "fsl-dataset.json",
    [string]$OutputDir = "",
    [int]$Epochs = 80,
    [int]$BatchSize = 32,
    [switch]$TestMode,
    [switch]$SkipInstall,
    [switch]$Incremental
)

$ErrorActionPreference = "Stop"
$toolkitDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $toolkitDir

if (-not $OutputDir) {
    $OutputDir = if ($TestMode) { "test-model" } else { "trained-model" }
}
if ($Epochs -lt 1) {
    throw "Epochs must be at least 1."
}
if ($BatchSize -lt 1) {
    throw "BatchSize must be at least 1."
}
if (-not (Test-Path -LiteralPath $Dataset -PathType Leaf)) {
    throw "Dataset not found: $Dataset. Pass the exported JSON with -Dataset."
}

Write-Output "SIGNAR_STAGE environment"
$python = Join-Path $toolkitDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    if ($SkipInstall) {
        throw "Virtual environment not found. Run without -SkipInstall to create it and install dependencies."
    }

    $basePython = $null
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.CommandType -eq "Application") {
        $versionOutput = & $pythonCommand.Source --version 2>$null
        if ($LASTEXITCODE -eq 0 -and "$versionOutput" -match "Python\s+(\d+\.\d+)") {
            $candidateVersion = $Matches[1]
            if ($candidateVersion -in @("3.10", "3.11", "3.12")) {
                $basePython = $pythonCommand.Source
            }
        }
    }

    if (-not $basePython) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            $runtimeLines = @(& $launcher.Source --list-paths 2>$null)
            if ($LASTEXITCODE -eq 0) {
                foreach ($preferredVersion in @("3.12", "3.11", "3.10")) {
                    foreach ($line in $runtimeLines) {
                        $listedVersion = $null
                        if ($line -match "-V:(?<version>3\.\d+)") {
                            $listedVersion = $Matches.version
                        } elseif ($line -match "CPython(?<version>3\.\d+)") {
                            $listedVersion = $Matches.version
                        }
                        if ($listedVersion -eq $preferredVersion -and $line -match "(?<path>[A-Za-z]:\\.*python\.exe)\s*$") {
                            $candidatePath = $Matches.path
                            if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
                                $basePython = $candidatePath
                                break
                            }
                        }
                    }
                    if ($basePython) { break }
                }
            }
        }
    }

    if (-not $basePython) {
        throw "Python 3.10, 3.11, or 3.12 was not found. Install one, or add its python.exe to PATH, then retry."
    }

    & $basePython -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv using $basePython. Check that this Python installation includes the venv module."
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The Python environment was created but .venv\Scripts\python.exe is missing."
    }
}

$pythonVersion = & $python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($LASTEXITCODE -ne 0) {
    throw "Could not determine the Python version in .venv."
}
if ($pythonVersion.Trim() -notin @("3.10", "3.11", "3.12")) {
    throw "The existing .venv uses Python $($pythonVersion.Trim()). Install Python 3.10-3.12 and recreate .venv before training."
}

$report = Join-Path $OutputDir "dataset-report.json"
if (-not $TestMode) {
    Write-Output "SIGNAR_STAGE dataset"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    if (Test-Path -LiteralPath $report -PathType Leaf) {
        Copy-Item -LiteralPath $report -Destination (Join-Path $OutputDir "dataset-report.previous.json") -Force
    }
    & $python training\validate_dataset.py --dataset $Dataset --report $report
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the dataset. Check the dataset path and JSON format."
    }
    Write-Host "Readiness report saved to $report. Training can continue without held-out evaluation when every label has at least one valid sample; that candidate will not be deployment eligible."
} else {
    Write-Host "TEST MODE: output is for pipeline testing only and must not be deployed." -ForegroundColor Yellow
}

if (-not $SkipInstall) {
    Write-Output "SIGNAR_STAGE dependencies"
    $dependencyCheck = @'
import importlib.metadata as metadata
import importlib.util
import sys

required = {
    "tensorflow": "2.17.1",
    "tf-keras": "2.17.0",
    "scikit-learn": "1.4.2",
    "jax": "0.4.38",
    "jaxlib": "0.4.38",
    "tensorflow-hub": "0.16.1",
    "importlib-resources": None,
}
try:
    numpy_version = metadata.version("numpy")
    if int(numpy_version.split(".", 1)[0]) >= 2:
        raise RuntimeError("numpy must be below 2")
    required["tensorflowjs"] = "4.22.0"
    for package, expected in required.items():
        version = metadata.version(package)
        if expected is not None and version != expected:
            raise RuntimeError(f"{package} needs {expected}")
        if package == "importlib-resources" and tuple(int(part) for part in version.split(".")[:2]) < (5, 9):
            raise RuntimeError("importlib-resources needs to be at least 5.9")
    setuptools_version = metadata.version("setuptools")
    if int(setuptools_version.split(".", 1)[0]) >= 81:
        raise RuntimeError("setuptools must be below 81")
    for module in ("numpy", "tensorflow", "tf_keras", "sklearn", "jax", "tensorflow_hub", "tensorflowjs"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"{module} cannot be imported")
except (metadata.PackageNotFoundError, RuntimeError, ValueError):
    sys.exit(1)
'@
    function Test-TrainingDependencies {
        $checkPath = Join-Path $env:TEMP ("signar-dependency-check-" + [guid]::NewGuid().ToString("N") + ".py")
        $utf8NoBom = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false
        [System.IO.File]::WriteAllText($checkPath, $dependencyCheck, $utf8NoBom)
        try {
            & $python $checkPath
            $checkExitCode = $LASTEXITCODE
            return ($checkExitCode -eq 0)
        } finally {
            Remove-Item -LiteralPath $checkPath -ErrorAction SilentlyContinue
        }
    }

    $dependenciesReady = Test-TrainingDependencies
    if ($dependenciesReady) {
        Write-Output "Python dependencies are ready; skipped package installation."
    } else {
        Write-Output "Installing missing or mismatched Python dependencies."
        & $python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Dependency installation failed."
        }
        if ($env:OS -eq "Windows_NT") {
            & $python -m pip install --no-deps tensorflowjs==4.22.0
            if ($LASTEXITCODE -ne 0) {
                throw "TensorFlow.js converter installation failed."
            }
        }
    }
}

$arguments = @(
    "training\train_hand_model.py",
    "--dataset", $Dataset,
    "--output-dir", $OutputDir,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize"
)
if ($TestMode) {
    $arguments += "--test-mode"
} else {
    $arguments += @("--readiness-report", $report)
}
if ($Incremental) {
    $arguments += "--incremental"
}
& $python @arguments
exit $LASTEXITCODE
