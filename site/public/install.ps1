$ErrorActionPreference = 'Stop'

$Repo = if ($env:PLSHELP_GITHUB_REPO) { $env:PLSHELP_GITHUB_REPO } else { 'HariharPrasadd/plshelp' }
$Version = if ($env:PLSHELP_VERSION) { $env:PLSHELP_VERSION } else { 'latest' }
$InstallDir = if ($env:PLSHELP_INSTALL_DIR) { $env:PLSHELP_INSTALL_DIR } else { Join-Path $HOME '.local\bin' }

function Fail($Message) {
    Write-Error $Message
    exit 1
}

function Get-LatestVersion {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
    if (-not $release.tag_name) {
        Fail 'Failed to resolve latest release version.'
    }
    return $release.tag_name
}

function Get-Sha256($Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

if ($Version -eq 'latest') {
    $Version = Get-LatestVersion
}

$archRaw = $env:PROCESSOR_ARCHITECTURE
switch ($archRaw.ToUpperInvariant()) {
    'AMD64' { $Arch = 'x86_64' }
    'X86' { Fail 'x86 Windows is not supported.' }
    'ARM64' { Fail 'windows arm64 release artifact is not configured yet.' }
    default { Fail "Unsupported Windows architecture: $archRaw" }
}

$Asset = "plshelp-$Version-windows-$Arch.zip"
$Checksums = "plshelp-$Version-checksums.txt"
$BaseUrl = "https://github.com/$Repo/releases/download/$Version"
$AssetUrl = "$BaseUrl/$Asset"
$ChecksumsUrl = "$BaseUrl/$Checksums"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("plshelp-install-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$ArchivePath = Join-Path $TempDir $Asset
$ChecksumsPath = Join-Path $TempDir $Checksums

try {
    Write-Host "Downloading $Asset"
    Invoke-WebRequest -Uri $AssetUrl -OutFile $ArchivePath
    Write-Host 'Downloading checksums'
    Invoke-WebRequest -Uri $ChecksumsUrl -OutFile $ChecksumsPath

    $expected = Select-String -Path $ChecksumsPath -Pattern ([regex]::Escape($Asset)) | ForEach-Object {
        ($_ -split '\s+')[0]
    } | Select-Object -First 1
    if (-not $expected) {
        Fail "Checksum entry not found for $Asset"
    }

    $actual = Get-Sha256 $ArchivePath
    if ($expected.ToLowerInvariant() -ne $actual) {
        Fail 'Checksum verification failed.'
    }

    Expand-Archive -Path $ArchivePath -DestinationPath $TempDir -Force
    $ExtractedBinary = Join-Path $TempDir 'plshelp.exe'
    if (-not (Test-Path $ExtractedBinary)) {
        Fail 'Extracted archive does not contain plshelp.exe'
    }

    Copy-Item $ExtractedBinary (Join-Path $InstallDir 'plshelp.exe') -Force
    Write-Host "Installed plshelp to $(Join-Path $InstallDir 'plshelp.exe')"
    $pathDirs = $env:PATH -split ';'
    if ($InstallDir -notin $pathDirs) {
        Write-Host "Add $InstallDir to your PATH if it is not already there."
    }
    Write-Host 'Run: plshelp help'
}
finally {
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
