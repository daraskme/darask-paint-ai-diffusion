# Builds the darask-paint plugin zip: drop the result into darask-paint's
# `plugins` folder (next to darask-paint.exe, or the folder configured in
# Settings) and the app extracts + launches it on first use.
#
#   pwsh ./scripts/package_darask_plugin.ps1 [-Version plugin-v1.0.0] [-OutDir dist]
#   -> dist/darask-paint-ai-diffusion-plugin-v1.0.0.zip
#
# The zip carries only the launcher, the headless server, its manifest and
# docs. ComfyUI / PyTorch / the checkpoint are still installed by
# darask-plugin.bat on first run (pinned versions, %LOCALAPPDATA%\DaraskAIDiffusion),
# exactly as when run from a clone. darask_server.py does not import the
# ai_diffusion package, so nothing else from the repository is needed.
param(
    [string]$Version = "plugin-dev",
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$files = @{
    "darask-plugin.bat"  = "darask-plugin.bat"
    "darask-plugin.json" = "darask-plugin.json"
    "darask_server.py"   = "darask_server.py"
    "README_darask.md"   = "README.md"
    "LICENSE"            = "LICENSE"
}
foreach ($file in $files.Keys) {
    if (-not (Test-Path (Join-Path $root $file))) { throw "missing $file" }
}

$manifest = Get-Content (Join-Path $root "darask-plugin.json") -Raw | ConvertFrom-Json
if ($manifest.name -ne "ai-diffusion") { throw "manifest name must be 'ai-diffusion'" }
if ($manifest.launcher -ne "darask-plugin.bat") { throw "manifest launcher must be darask-plugin.bat" }

$name = "darask-paint-ai-diffusion-$Version"
$stage = Join-Path ([IO.Path]::GetTempPath()) ("darask-plugin-pkg-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path (Join-Path $stage $name) | Out-Null
foreach ($file in $files.Keys) {
    Copy-Item (Join-Path $root $file) (Join-Path (Join-Path $stage $name) $files[$file])
}

New-Item -ItemType Directory -Path (Join-Path $root $OutDir) -Force | Out-Null
$zip = Join-Path (Resolve-Path (Join-Path $root $OutDir)).Path "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $stage $name) -DestinationPath $zip
Remove-Item -Recurse -Force $stage
Write-Output $zip
