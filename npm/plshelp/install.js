const fs = require("fs");
const os = require("os");
const path = require("path");
const { execSync } = require("child_process");

const VERSION = process.env.npm_package_version || "0.1.0";
const REPO = "HariharPrasadd/plshelp";

const platformMap = {
  "linux-x64": {
    pkg: "@generalinteraction/plshelp-linux-x64",
    assetSuffix: "linux-x86_64",
    archiveExt: "tar.gz",
    binName: "plshelp",
  },
  "darwin-arm64": {
    pkg: "@generalinteraction/plshelp-darwin-arm64",
    assetSuffix: "darwin-arm64",
    archiveExt: "tar.gz",
    binName: "plshelp",
  },
  "win32-x64": {
    pkg: "@generalinteraction/plshelp-win32-x64",
    assetSuffix: "windows-x86_64",
    archiveExt: "zip",
    binName: "plshelp.exe",
  },
};

function getTagName(version) {
  return version.startsWith("v") ? version : `v${version}`;
}

function shellQuote(value) {
  return `"${String(value).replace(/(["\\$`])/g, "\\$1")}"`;
}

const key = `${os.platform()}-${os.arch()}`;
const entry = platformMap[key];
if (!entry) {
  console.warn(`plshelp: no prebuilt binary for ${key}, skipping`);
  process.exit(0);
}

try {
  require.resolve(`${entry.pkg}/bin/${entry.binName}`);
  process.exit(0);
} catch (_) {}

const tag = getTagName(VERSION);
const archiveName = `plshelp-${tag}-${entry.assetSuffix}.${entry.archiveExt}`;
const url = `https://github.com/${REPO}/releases/download/${tag}/${archiveName}`;
const binDir = path.join(__dirname, "bin");
const dest = path.join(binDir, entry.binName);
const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "plshelp-npm-"));
const archivePath = path.join(tempRoot, archiveName);

fs.mkdirSync(binDir, { recursive: true });

try {
  console.log(`plshelp: downloading ${archiveName}`);

  if (process.platform === "win32") {
    execSync(
      `powershell -NoProfile -Command ${shellQuote(
        `$ErrorActionPreference = 'Stop'; Invoke-WebRequest -Uri '${url}' -OutFile '${archivePath}'; Expand-Archive -Path '${archivePath}' -DestinationPath '${tempRoot}' -Force`
      )}`,
      { stdio: "inherit" }
    );
  } else {
    execSync(
      `curl -fsSL ${shellQuote(url)} -o ${shellQuote(archivePath)}`,
      { stdio: "inherit" }
    );
    execSync(
      `tar -xzf ${shellQuote(archivePath)} -C ${shellQuote(tempRoot)}`,
      { stdio: "inherit" }
    );
  }

  const extracted = path.join(tempRoot, entry.binName);
  if (!fs.existsSync(extracted)) {
    throw new Error(`Extracted archive does not contain ${entry.binName}`);
  }

  fs.copyFileSync(extracted, dest);
  if (process.platform !== "win32") {
    fs.chmodSync(dest, 0o755);
  }
} catch (err) {
  console.warn(`plshelp: fallback install failed: ${err.message}`);
} finally {
  fs.rmSync(tempRoot, { recursive: true, force: true });
}
