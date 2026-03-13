$ErrorActionPreference = 'Stop'

$Repo = if ($env:PLSHELP_GITHUB_REPO) { $env:PLSHELP_GITHUB_REPO } else { 'HariharPrasadd/plshelp' }
$Version = if ($env:PLSHELP_VERSION) { $env:PLSHELP_VERSION } else { 'latest' }
$InstallDir = if ($env:PLSHELP_INSTALL_DIR) { $env:PLSHELP_INSTALL_DIR } else { Join-Path $HOME '.local\bin' }

function Fail($Message) {
    Write-Error $Message
    exit 1
}

function Get-LatestVersion {
    try {
        $response = Invoke-WebRequest -Uri "https://github.com/$Repo/releases/latest"
        $finalUri = $response.BaseResponse.ResponseUri.AbsoluteUri
    } catch {
        $response = $_.Exception.Response
        if ($response -and $response.Headers['Location']) {
            $finalUri = $response.Headers['Location']
        }
    }

    if (-not $finalUri) {
        Fail 'Failed to resolve latest release version.'
    }

    if ($finalUri -match '/tag/([^/?]+)') {
        return $matches[1]
    }

    Fail 'Failed to resolve latest release version.'
}

function Get-Sha256($Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

function Test-ZipSignature($Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -lt 4) {
            return $false
        }

        $buffer = New-Object byte[] 4
        $bytesRead = $stream.Read($buffer, 0, 4)
        return $bytesRead -eq 4 -and $buffer[0] -eq 0x50 -and $buffer[1] -eq 0x4B
    }
    finally {
        $stream.Dispose()
    }
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

    if (-not (Test-ZipSignature $ArchivePath)) {
        Fail "Downloaded asset is not a valid zip archive: $AssetUrl"
    }

    $expected = Select-String -Path $ChecksumsPath -Pattern ([regex]::Escape($Asset)) | ForEach-Object {
        ($_.Line -split '\s+')[0].Trim()
    } | Select-Object -First 1
    if (-not $expected) {
        Fail "Checksum entry not found for $Asset"
    }

    $actual = Get-Sha256 $ArchivePath
    if ($expected.ToLowerInvariant() -ne $actual) {
        Fail "Checksum verification failed for $Asset. Expected $expected but got $actual."
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
