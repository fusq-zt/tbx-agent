[CmdletBinding()]
param(
    [string]$Python = '',
    [switch]$ApiOnly,
    [switch]$SkipPreflight,
    [switch]$Demo,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    foreach ($RawLine in [System.IO.File]::ReadAllLines($Path)) {
        $Line = $RawLine.Trim()
        if ($Line.Length -eq 0 -or $Line.StartsWith('#')) { continue }
        $Pair = $Line.Split('=', 2)
        if ($Pair.Count -ne 2 -or $Pair[0] -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            throw "Invalid .env line (shell expansion is intentionally unsupported): $RawLine"
        }
        $Value = $Pair[1].Trim()
        if ($Value.Length -ge 2 -and (
            ($Value.StartsWith('"') -and $Value.EndsWith('"')) -or
            ($Value.StartsWith("'") -and $Value.EndsWith("'"))
        )) {
            $Value = $Value.Substring(1, $Value.Length - 2)
        }
        $Existing = [Environment]::GetEnvironmentVariable($Pair[0], 'Process')
        if (-not [string]::IsNullOrWhiteSpace($Existing)) { continue }
        [Environment]::SetEnvironmentVariable($Pair[0], $Value, 'Process')
    }
}

function Test-Truthy {
    param([string]$Value)
    return $Value -match '^(?i:1|true|yes|on)$'
}

function Get-PlatformArtifactRoot {
    $LocalCache = if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        [Environment]::GetFolderPath('LocalApplicationData')
    } else {
        $env:LOCALAPPDATA
    }
    if ([string]::IsNullOrWhiteSpace($LocalCache)) {
        $LocalCache = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'AppData\Local'
    }
    return [System.IO.Path]::GetFullPath((Join-Path $LocalCache 'TBX-Agent\artifacts'))
}

function Test-HttpEndpoint {
    param([string]$Url, [int]$TimeoutSeconds = 5)
    try {
        $Response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec $TimeoutSeconds -UseBasicParsing
        return $Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Get-VerifiedLoopbackListenerProcess {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)][string]$ExpectedExecutable
    )
    try {
        $ExpectedPath = [System.IO.Path]::GetFullPath($ExpectedExecutable)
        $Connections = @(
            Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop |
                Where-Object { $_.LocalAddress -in @('127.0.0.1', '::1') }
        )
        foreach ($Connection in $Connections) {
            $Candidate = Get-Process -Id $Connection.OwningProcess -ErrorAction Stop
            if (-not [string]::IsNullOrWhiteSpace($Candidate.Path) -and
                [System.IO.Path]::GetFullPath($Candidate.Path) -eq $ExpectedPath) {
                return $Candidate
            }
        }
    } catch {
        return $null
    }
    return $null
}

Import-DotEnv -Path (Join-Path $ProjectRoot '.env')
$env:TBX_AGENT_PROJECT_ROOT = $ProjectRoot

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Candidate = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        $Python = $Candidate
    } else {
        $Python = 'python'
    }
}

if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_DATA_ROOT)) {
    if (-not [string]::IsNullOrWhiteSpace($env:TBX_RUNTIME_ROOT)) {
        $env:TBX_AGENT_DATA_ROOT = [System.IO.Path]::GetFullPath($env:TBX_RUNTIME_ROOT)
    } else {
        $Base = if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            [Environment]::GetFolderPath('LocalApplicationData')
        } else {
            $env:LOCALAPPDATA
        }
        if ([string]::IsNullOrWhiteSpace($Base)) {
            $Base = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'AppData\Local'
        }
        $env:TBX_AGENT_DATA_ROOT = Join-Path $Base 'TBX-Agent\runtime'
    }
}
$LlmEnvFile = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_LLM_ENV_FILE)) {
    Join-Path $env:TBX_AGENT_DATA_ROOT 'config\llm.env'
} else {
    [System.IO.Path]::GetFullPath($env:TBX_AGENT_LLM_ENV_FILE)
}
Import-DotEnv -Path $LlmEnvFile
if ($Demo) {
    # Explicitly labelled control-flow demo. Apply after both dotenv sources so
    # a real-runtime profile cannot accidentally leak into a mock demonstration.
    $env:TBX_AGENT_VISION_BACKEND = 'mock'
    $env:TBX_AGENT_NARRATOR_BACKEND = 'none'
    $env:LLM_PROVIDER = 'none'
    $env:TBX_AGENT_RETRIEVAL_CONFIG = Join-Path $ProjectRoot 'configs\retrieval.yaml'
    $env:TBX_AGENT_REQUIRE_REAL_INFERENCE = 'false'
    $env:TBX_AGENT_REQUIRE_LLM_INFERENCE = 'false'
    $env:TBX_AGENT_OPENAI_ENABLED = 'false'
    $env:TBX_AGENT_ANATOMY_BACKEND = 'none'
    $env:TBX_AGENT_REQUIRE_ANATOMY_INFERENCE = 'false'
    $env:TBX_AGENT_CONTOUR_REFINEMENT_BACKEND = 'none'
    $SkipPreflight = $true
}
if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_DB_PATH)) {
    $env:TBX_AGENT_DB_PATH = Join-Path $env:TBX_AGENT_DATA_ROOT 'tbx_agent.sqlite3'
}
if ([string]::IsNullOrWhiteSpace($env:TBX_ARTIFACT_ROOT)) {
    # Match bootstrap.ps1/default_artifact_root. An explicit process variable
    # or .env entry always wins; model artifacts are never silently redirected
    # into mutable case/runtime storage.
    $env:TBX_ARTIFACT_ROOT = Get-PlatformArtifactRoot
}

$BindHost = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_BIND_HOST)) {
    '127.0.0.1'
} else { $env:TBX_AGENT_BIND_HOST }
$BindPort = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_BIND_PORT)) {
    8000
} else { [int]$env:TBX_AGENT_BIND_PORT }
$UiPort = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_UI_PORT)) {
    8501
} else { [int]$env:TBX_AGENT_UI_PORT }
$ApiUrl = "http://127.0.0.1:$BindPort"
$env:TBX_AGENT_API_URL = $ApiUrl

$RequireVisionValue = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_REQUIRE_REAL_INFERENCE)) {
    'true'
} else { $env:TBX_AGENT_REQUIRE_REAL_INFERENCE }
$RequireLlmValue = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_REQUIRE_LLM_INFERENCE)) {
    'true'
} else { $env:TBX_AGENT_REQUIRE_LLM_INFERENCE }
$RequireVision = Test-Truthy $RequireVisionValue
$RequireLlm = Test-Truthy $RequireLlmValue
$VisionBackend = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_VISION_BACKEND)) {
    'rank03'
} else { $env:TBX_AGENT_VISION_BACKEND }
$NarratorBackend = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_NARRATOR_BACKEND)) {
    'llama_cpp'
} else { $env:TBX_AGENT_NARRATOR_BACKEND }
$LlmConfig = if ([string]::IsNullOrWhiteSpace($env:TBX_AGENT_LLM_RUNTIME_CONFIG)) {
    Join-Path $ProjectRoot 'configs\llm_runtime.yaml'
} else {
    [System.IO.Path]::GetFullPath($env:TBX_AGENT_LLM_RUNTIME_CONFIG)
}

if ($RequireVision -and $VisionBackend -ne 'rank03') {
    throw "Real inference is required, but TBX_AGENT_VISION_BACKEND='$VisionBackend'. Configure rank03 assets first."
}
if ($RequireLlm -and $NarratorBackend -ne 'llama_cpp') {
    throw "LLM inference is required, but narrator '$NarratorBackend' is not the pinned llama.cpp contract."
}

$ApiArguments = @(
    '-m', 'uvicorn', 'tbx_agent.api.main:app', '--host', $BindHost, '--port', "$BindPort"
)
$UiArguments = @(
    '-m', 'streamlit', 'run', 'ui\streamlit_app.py',
    '--server.address=127.0.0.1', "--server.port=$UiPort"
)

if ($DryRun) {
    Write-Host "Project: $ProjectRoot"
    Write-Host "Runtime data: $env:TBX_AGENT_DATA_ROOT"
    Write-Host "API: $Python $($ApiArguments -join ' ')"
    if (-not $ApiOnly) { Write-Host "UI:  $Python $($UiArguments -join ' ')" }
    Write-Host "Required contract: vision=$RequireVision/$VisionBackend, llm=$RequireLlm/$NarratorBackend"
    if ($Demo) { Write-Host 'Mode: DEMO / MOCK (not real model inference)' }
    if ($RequireLlm) { Write-Host "LLM supervisor: $Python -m tbx_agent.llm serve --config $LlmConfig" }
    Write-Host 'Dry run complete; no process or runtime file was created.'
    exit 0
}

if (-not $SkipPreflight) {
    Write-Host 'Running fail-closed asset/configuration preflight...'
    & $Python -m tbx_agent.preflight
    if ($LASTEXITCODE -ne 0) {
        throw 'Preflight failed. Install the downloaded inference bundle with scripts/install_vision_bundle.py --bundle <zip> --artifact-root <path> --runtime-config <path>, verify the configured LLM, and review .env.'
    }
}

$LogRoot = Join-Path $env:TBX_AGENT_DATA_ROOT 'logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$LlmStdout = Join-Path $LogRoot "llama-$Stamp.stdout.log"
$LlmStderr = Join-Path $LogRoot "llama-$Stamp.stderr.log"
$ApiStdout = Join-Path $LogRoot "api-$Stamp.stdout.log"
$ApiStderr = Join-Path $LogRoot "api-$Stamp.stderr.log"
$UiStdout = Join-Path $LogRoot "ui-$Stamp.stdout.log"
$UiStderr = Join-Path $LogRoot "ui-$Stamp.stderr.log"
$ApiProcess = $null
$UiProcess = $null
$LlmProcess = $null

try {
    if ($RequireLlm -and $NarratorBackend -eq 'llama_cpp') {
        if ($SkipPreflight) {
            & $Python -m tbx_agent.llm verify --config $LlmConfig | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw 'Pinned MedGemma/llama.cpp runtime verification failed.'
            }
        }
        & $Python -m tbx_agent.llm probe --skip-assets --timeout 3 --config $LlmConfig *> $null
        if ($LASTEXITCODE -ne 0) {
            $LaunchSpec = (& $Python -m tbx_agent.llm argv --config $LlmConfig) |
                ConvertFrom-Json
            if ($LASTEXITCODE -ne 0 -or $null -eq $LaunchSpec.argv -or $LaunchSpec.argv.Count -lt 2) {
                throw 'Unable to resolve the verified llama.cpp launch contract.'
            }
            $ExpectedLlmExecutable = [string]$LaunchSpec.argv[0]
            $PortIndex = [Array]::IndexOf([object[]]$LaunchSpec.argv, '--port')
            if ($PortIndex -lt 0 -or $PortIndex + 1 -ge $LaunchSpec.argv.Count) {
                throw 'Verified llama.cpp launch contract omitted its local port.'
            }
            $LlmPort = [int]$LaunchSpec.argv[$PortIndex + 1]
            $QuotedLlmConfig = '"' + ($LlmConfig -replace '"', '\"') + '"'
            $LlmArguments = @('-m', 'tbx_agent.llm', 'serve', '--config', $QuotedLlmConfig)
            $LlmProcess = Start-Process -FilePath $Python -ArgumentList $LlmArguments `
                -WorkingDirectory $ProjectRoot -RedirectStandardOutput $LlmStdout `
                -RedirectStandardError $LlmStderr -WindowStyle Hidden -PassThru
            $Ready = $false
            $WrapperExitedCleanly = $false
            for ($Attempt = 0; $Attempt -lt 180; $Attempt++) {
                if ($LlmProcess.HasExited) {
                    if ($LlmProcess.ExitCode -ne 0) {
                        throw "llama.cpp exited during startup. See $LlmStderr"
                    }
                    # On Windows, Python's os.execv starts llama-server and the
                    # short-lived Python wrapper exits with code 0. Continue
                    # probing, then attach lifecycle management to the verified
                    # loopback listener instead of reporting a false crash.
                    $WrapperExitedCleanly = $true
                }
                & $Python -m tbx_agent.llm probe --skip-assets --timeout 2 --config $LlmConfig *> $null
                if ($LASTEXITCODE -eq 0) { $Ready = $true; break }
                Start-Sleep -Seconds 1
            }
            if (-not $Ready) { throw "llama.cpp did not become ready. See $LlmStderr" }
            if ($WrapperExitedCleanly) {
                $ListenerProcess = Get-VerifiedLoopbackListenerProcess `
                    -Port $LlmPort -ExpectedExecutable $ExpectedLlmExecutable
                if ($null -eq $ListenerProcess) {
                    throw 'Ready llama.cpp listener could not be bound to the verified process.'
                }
                $LlmProcess = $ListenerProcess
            }
        }
    }

    $ApiProcess = Start-Process -FilePath $Python -ArgumentList $ApiArguments -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $ApiStdout -RedirectStandardError $ApiStderr `
        -WindowStyle Hidden -PassThru
    $ApiReady = $false
    for ($Attempt = 0; $Attempt -lt 90; $Attempt++) {
        if ($ApiProcess.HasExited) {
            throw "API exited with code $($ApiProcess.ExitCode). See $ApiStderr"
        }
        if (Test-HttpEndpoint -Url "$ApiUrl/readyz" -TimeoutSeconds 2) {
            $ApiReady = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ApiReady) {
        throw "API did not become ready within 90 seconds. See $ApiStderr"
    }

    if (-not $ApiOnly) {
        $UiProcess = Start-Process -FilePath $Python -ArgumentList $UiArguments -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $UiStdout -RedirectStandardError $UiStderr `
            -WindowStyle Hidden -PassThru
    }

    Write-Host "API: $ApiUrl (readiness: $ApiUrl/readyz)"
    if (-not $ApiOnly) { Write-Host "UI:  http://127.0.0.1:$UiPort" }
    Write-Host "Logs: $LogRoot"
    Write-Host 'Press Ctrl+C to stop both processes.'

    while ($true) {
        if ($ApiProcess.HasExited) {
            throw "API exited with code $($ApiProcess.ExitCode). See $ApiStderr"
        }
        if ($null -ne $UiProcess -and $UiProcess.HasExited) {
            throw "UI exited with code $($UiProcess.ExitCode). See $UiStderr"
        }
        Start-Sleep -Seconds 2
    }
} finally {
    foreach ($Process in @($UiProcess, $ApiProcess, $LlmProcess)) {
        if ($null -ne $Process -and -not $Process.HasExited) {
            Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue
            $Process.WaitForExit(5000) | Out-Null
        }
    }
}
