# Build and push Qwen3-TTS Docker image to Docker Hub
# Then create RunPod endpoint

param(
    [string]$DockerUser = "your_dockerhub_username",
    [string]$ImageName = "qwen3-tts-runpod",
    [string]$Tag = "latest"
)

$FullImage = "${DockerUser}/${ImageName}:${Tag}"

Write-Host "Building Docker image: $FullImage" -ForegroundColor Cyan

# Build
docker build -t $FullImage .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Pushing to Docker Hub..." -ForegroundColor Cyan

# Push
docker push $FullImage

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker push failed! Make sure you're logged in: docker login" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Image pushed successfully: $FullImage" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Go to https://console.runpod.io/serverless/new-endpoint" -ForegroundColor Yellow
Write-Host "2. Enter image: $FullImage" -ForegroundColor Yellow
Write-Host "3. Select GPU (24GB recommended for 0.6B, 48GB for 1.7B)" -ForegroundColor Yellow
Write-Host "4. Create endpoint and copy the ID" -ForegroundColor Yellow
Write-Host "5. Set RUNPOD_TTS_ENDPOINT_ID in your .env file" -ForegroundColor Yellow
