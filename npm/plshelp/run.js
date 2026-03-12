#!/usr/bin/env node
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

function getPlatformInfo() {
  const key = `${os.platform()}-${os.arch()}`;
  const map = {
    "linux-x64": { pkg: "plshelp-linux-x64", bin: "plshelp" },
    "darwin-arm64": { pkg: "plshelp-darwin-arm64", bin: "plshelp" },
    "win32-x64": { pkg: "plshelp-win32-x64", bin: "plshelp.exe" },
  };

  const info = map[key];
  if (!info) {
    throw new Error(`Unsupported platform: ${key}`);
  }
  return info;
}

function getBinaryPath() {
  const info = getPlatformInfo();

  try {
    return require.resolve(`${info.pkg}/bin/${info.bin}`);
  } catch (_) {
    const fallback = path.join(__dirname, "bin", info.bin);
    if (fs.existsSync(fallback)) {
      return fallback;
    }
    throw new Error(
      `plshelp binary not found for ${os.platform()}-${os.arch()}. Reinstall the package or publish the matching platform package.`
    );
  }
}

try {
  const bin = getBinaryPath();
  const result = spawnSync(bin, process.argv.slice(2), { stdio: "inherit" });
  if (result.error) {
    throw result.error;
  }
  if (result.signal) {
    process.kill(process.pid, result.signal);
  }
  process.exit(result.status == null ? 1 : result.status);
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
