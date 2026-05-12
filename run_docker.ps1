[CmdletBinding()]
param(
    [switch]$NoBot,
    [switch]$NoBuild,
    [switch]$Logs,
    [string]$ModelId = "ddcolor",
    [string]$Device = "cpu",
    [string]$EnabledModels = "cgan_reference,colorcomic_auto,ddcolor,deoldify",
    [string]$ExtraUvGroups = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }

        $parts = $trimmed.Split("=", 2)
        if ($parts[0].Trim() -eq $Name) {
            return $parts[1].Trim().Trim('"').Trim("'")
        }
    }

    return $null
}

function Normalize-ModelId {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $normalized = $Value.Trim().ToLowerInvariant()
    switch ($normalized) {
        "cgan" { return "cgan_reference" }
        default { return $normalized }
    }
}

function Split-ModelList {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $models = @()
    foreach ($item in $Value.Split(",")) {
        $trimmed = $item.Trim()
        if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
            $models += Normalize-ModelId -Value $trimmed
        }
    }
    return $models
}

function Get-ModelGroups {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Models
    )

    $groups = [ordered]@{}
    foreach ($model in $Models) {
        switch ($model) {
            "passthrough" {}
            "cgan_reference" { $groups["--group model-cgan"] = $true }
            "colorcomic_auto" { $groups["--group model-colorcomic"] = $true }
            "colorcomic_reference" { $groups["--group model-colorcomic"] = $true }
            "ddcolor" { $groups["--group model-ddcolor"] = $true }
            "deoldify" { $groups["--group model-deoldify"] = $true }
            "cobra" { $groups["--group model-cobra"] = $true }
            default {
                throw (
                    "Unsupported model '$model'. Supported: passthrough, cgan, " +
                    "cgan_reference, colorcomic_auto, colorcomic_reference, " +
                    "ddcolor, deoldify, cobra."
                )
            }
        }
    }
    return @($groups.Keys)
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install Docker Desktop and try again."
}

& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker daemon is not available. Start Docker Desktop and try again."
}

$ModelId = Normalize-ModelId -Value $ModelId
if ([string]::IsNullOrWhiteSpace($EnabledModels)) {
    $EnabledModels = $ModelId
}
$enabledModelList = Split-ModelList -Value $EnabledModels
$EnabledModels = $enabledModelList -join ","
$modelsForGroups = @($ModelId) + $enabledModelList
$modelGroups = Get-ModelGroups -Models $modelsForGroups
$resolvedExtraGroups = @($modelGroups)
if (-not [string]::IsNullOrWhiteSpace($ExtraUvGroups)) {
    $resolvedExtraGroups += $ExtraUvGroups
}

$env:COLORIZATION_MODEL_ID = $ModelId
$env:COLORIZATION_DEVICE = $Device
$env:ENABLED_MODELS = $EnabledModels
$env:API_EXTRA_UV_GROUPS = ""
$env:WORKER_EXTRA_UV_GROUPS = ($resolvedExtraGroups -join " ").Trim()
# Kept for manual commands that still read the old variable name.
$env:EXTRA_UV_GROUPS = $env:WORKER_EXTRA_UV_GROUPS

New-Item -ItemType Directory -Force -Path "data", "outputs/service" | Out-Null

$services = @("redis", "api", "worker")
$buildServices = @("api", "worker")
if (-not $NoBot) {
    $token = $env:TELEGRAM_BOT_TOKEN
    if ([string]::IsNullOrWhiteSpace($token)) {
        $token = Get-DotEnvValue -Path ".env" -Name "TELEGRAM_BOT_TOKEN"
    }

    if (
        [string]::IsNullOrWhiteSpace($token) -or
        $token -eq "your-telegram-token"
    ) {
        throw "Set TELEGRAM_BOT_TOKEN in .env or run ./run_docker.ps1 -NoBot."
    }

    $services = @("redis", "postgres", "api", "worker", "bot")
    $buildServices = @("api", "worker", "bot")
}

& docker compose config *> $null
if ($LASTEXITCODE -ne 0) {
    throw "docker-compose.yml is invalid."
}

Write-Host "Starting services: $($services -join ', ')"
Write-Host "Model: $ModelId; device: $Device; enabled models: $EnabledModels"
if (-not [string]::IsNullOrWhiteSpace($env:API_EXTRA_UV_GROUPS)) {
    Write-Host "API uv groups: $env:API_EXTRA_UV_GROUPS"
}
if (-not [string]::IsNullOrWhiteSpace($env:WORKER_EXTRA_UV_GROUPS)) {
    Write-Host "Worker uv groups: $env:WORKER_EXTRA_UV_GROUPS"
}

if (-not $NoBuild) {
    Write-Host ""
    Write-Host "Building images with plain Docker progress..."
    Write-Host "This can take a long time for heavy model groups because torch/CUDA wheels are large."
    foreach ($service in $buildServices) {
        Write-Host ""
        Write-Host "Building $service..."
        & docker compose --progress plain build $service
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose build failed for $service."
        }
    }
}

Write-Host ""
Write-Host "Starting containers..."
$upArgs = @("compose", "up", "-d")
if (-not $NoBuild) {
    $upArgs += "--no-build"
}
$upArgs += $services

& docker @upArgs
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose failed."
}

Write-Host ""
Write-Host "API: http://localhost:8000"
Write-Host "Health: http://localhost:8000/health"
Write-Host "Status: docker compose ps"
Write-Host "Stop: docker compose down"

if ($Logs) {
    & docker compose logs -f @services
}
