# Qwen3-TTS Docker Build & Push Script for RunPod
# Usage: .\build.ps1 -Username <dockerhub_username> [-Tag <tag>] [-Push]

param(
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [string]$Tag = "latest",
    
    [switch]$Push,
    
    [switch]$NoBuildCache
)

$ErrorActionPreference = "Stop"

# Config
$ImageName = "qwen3-tts"
$FullImageName = "$Username/${ImageName}:$Tag"
$QwenTTSSource = $env:QWEN_TTS_SOURCE
if (-not $QwenTTSSource) {
    Write-Error "Set QWEN_TTS_SOURCE env var to the path of your qwen_tts package"
    exit 1
}

Write-Host "🐳 Building Qwen3-TTS Docker Image" -ForegroundColor Cyan
Write-Host "   Image: $FullImageName" -ForegroundColor Gray

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH"
    exit 1
}

# Check source exists
if (-not (Test-Path $QwenTTSSource)) {
    Write-Error "qwen_tts source not found at: $QwenTTSSource"
    exit 1
}

# Copy qwen_tts to build context
$BuildContext = Split-Path -Parent $MyInvocation.MyCommand.Path
$QwenTTSDest = Join-Path $BuildContext "qwen_tts"

Write-Host "📦 Copying qwen_tts package..." -ForegroundColor Yellow
if (Test-Path $QwenTTSDest) {
    Remove-Item -Recurse -Force $QwenTTSDest
}
Copy-Item -Recurse $QwenTTSSource $QwenTTSDest

# Build args
$BuildArgs = @("build", "-t", $FullImageName, "-f", "Dockerfile")
if ($NoBuildCache) {
    $BuildArgs += "--no-cache"
}
$BuildArgs += "."

# Build
Write-Host "🔨 Building image..." -ForegroundColor Yellow
Set-Location $BuildContext
docker @BuildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed!"
    exit 1
}

Write-Host "✅ Build successful: $FullImageName" -ForegroundColor Green

# Push if requested
if ($Push) {
    Write-Host "🚀 Pushing to Docker Hub..." -ForegroundColor Yellow
    docker push $FullImageName
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker push failed! Make sure you're logged in: docker login"
        exit 1
    }
    
    Write-Host "✅ Pushed: $FullImageName" -ForegroundColor Green
}

# Cleanup
Write-Host "🧹 Cleaning up..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $QwenTTSDest

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Image: $FullImageName"
Write-Host ""
Write-Host "To push manually:" -ForegroundColor Gray
Write-Host "  docker login"
Write-Host "  docker push $FullImageName"
Write-Host ""
Write-Host "To deploy on RunPod:" -ForegroundColor Gray
Write-Host "  1. Go to https://console.runpod.io/serverless"
Write-Host "  2. Click 'New Endpoint'"
Write-Host "  3. Image: docker.io/$FullImageName"
Write-Host "  4. GPU: 24GB+ recommended (RTX 4090, A5000, etc.)"
Write-Host "  5. Environment: MODEL_SIZE=0.6B (or 1.7B for better quality)"
