# Compile all GLSL shaders to SPIR-V
# Requires Vulkan SDK with glslc compiler

$ShaderDir = Join-Path $PSScriptRoot ".." "shaders"
$SpvDir = Join-Path $ShaderDir "spv"

# Check if glslc is available
if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: glslc not found. Install Vulkan SDK:" -ForegroundColor Red
    Write-Host "  https://vulkan.lunarg.com/sdk/home" -ForegroundColor Yellow
    exit 1
}

# Create spv directory if it doesn't exist
if (-not (Test-Path $SpvDir)) {
    New-Item -ItemType Directory -Path $SpvDir | Out-Null
    Write-Host "Created directory: $SpvDir"
}

# Find all GLSL files
$GlslFiles = Get-ChildItem -Path $ShaderDir -Filter "*.glsl" -File

if ($GlslFiles.Count -eq 0) {
    Write-Host "No GLSL files found in $ShaderDir" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($GlslFiles.Count) shader(s) to compile..."
Write-Host ""

$SuccessCount = 0
$FailCount = 0

foreach ($file in $GlslFiles) {
    $ShaderName = $file.BaseName
    $SpvFile = Join-Path $SpvDir "$ShaderName.spv"
    
    # Skip if already compiled and newer
    if ((Test-Path $SpvFile) -and ($file.LastWriteTime -le (Get-Item $SpvFile).LastWriteTime)) {
        Write-Host "  [SKIP] $ShaderName.spv (already up to date)" -ForegroundColor Gray
        continue
    }
    
    Write-Host "  Compiling $ShaderName.glsl..."
    glslc -fshader-stage=compute "$($file.FullName)" -o "$SpvFile"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    [OK] $ShaderName.spv" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "    [FAIL] $ShaderName.glsl" -ForegroundColor Red
        $FailCount++
    }
}

Write-Host ""
Write-Host "Compilation complete: $SuccessCount succeeded, $FailCount failed" -ForegroundColor $(if ($FailCount -eq 0) { "Green" } else { "Yellow" })
