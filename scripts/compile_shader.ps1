# PowerShell script to compile GLSL shaders to SPIR-V
# Requires Vulkan SDK with glslc compiler

param(
    [Parameter(Mandatory=$true)]
    [string]$ShaderName
)

$ShaderDir = Join-Path $PSScriptRoot ".." "shaders"
$SpvDir = Join-Path $ShaderDir "spv"
$GlslFile = Join-Path $ShaderDir "$ShaderName.glsl"
$SpvFile = Join-Path $SpvDir "$ShaderName.spv"

# Check if glslc is available
if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: glslc not found. Install Vulkan SDK:" -ForegroundColor Red
    Write-Host "  https://vulkan.lunarg.com/sdk/home" -ForegroundColor Yellow
    exit 1
}

# Check if shader file exists
if (-not (Test-Path $GlslFile)) {
    Write-Host "ERROR: Shader file not found: $GlslFile" -ForegroundColor Red
    exit 1
}

# Create spv directory if it doesn't exist
if (-not (Test-Path $SpvDir)) {
    New-Item -ItemType Directory -Path $SpvDir | Out-Null
    Write-Host "Created directory: $SpvDir"
}

# Compile shader (compute shader)
Write-Host "Compiling $ShaderName.glsl -> spv/$ShaderName.spv..."
glslc -fshader-stage=compute "$GlslFile" -o "$SpvFile"

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Compiled $ShaderName.spv" -ForegroundColor Green
} else {
    Write-Host "ERROR: Compilation failed" -ForegroundColor Red
    exit 1
}
