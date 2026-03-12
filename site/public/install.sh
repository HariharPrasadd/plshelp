#!/usr/bin/env sh
set -eu

REPO="${PLSHELP_GITHUB_REPO:-HariharPrasadd/plshelp}"
VERSION="${PLSHELP_VERSION:-latest}"
INSTALL_DIR="${PLSHELP_INSTALL_DIR:-$HOME/.local/bin}"
TMP_DIR=""

log() {
  printf '%s\n' "$*"
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
  fi
}

trap cleanup EXIT INT TERM

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

latest_version() {
  need_cmd curl
  if command -v jq >/dev/null 2>&1; then
    curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
      | jq -r '.tag_name // empty'
    return
  fi
  curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n 1
}

sha256_file() {
  file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return
  fi
  fail "missing checksum tool (need sha256sum or shasum)"
}

uname_s=$(uname -s)
uname_m=$(uname -m)

case "$uname_s" in
  Darwin) os="darwin" ;;
  Linux) os="linux" ;;
  *) fail "unsupported operating system: $uname_s" ;;
esac

case "$uname_m" in
  arm64|aarch64)
    if [ "$os" = "darwin" ]; then
      arch="arm64"
    else
      fail "linux arm64 release artifact is not configured yet"
    fi
    ;;
  x86_64|amd64) arch="x86_64" ;;
  *) fail "unsupported architecture: $uname_m" ;;
esac

if [ "$VERSION" = "latest" ]; then
  VERSION=$(latest_version)
  [ -n "$VERSION" ] || fail "failed to resolve latest release version"
fi

ASSET="plshelp-${VERSION}-${os}-${arch}.tar.gz"
CHECKSUMS="plshelp-${VERSION}-checksums.txt"
BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
ASSET_URL="${BASE_URL}/${ASSET}"
CHECKSUMS_URL="${BASE_URL}/${CHECKSUMS}"

need_cmd curl
need_cmd tar
mkdir -p "$INSTALL_DIR"
TMP_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t plshelp-install)
ARCHIVE_PATH="$TMP_DIR/$ASSET"
CHECKSUMS_PATH="$TMP_DIR/$CHECKSUMS"

log "Downloading $ASSET"
curl -fsSL "$ASSET_URL" -o "$ARCHIVE_PATH"
log "Downloading checksums"
curl -fsSL "$CHECKSUMS_URL" -o "$CHECKSUMS_PATH"

expected=$(
  awk -v asset="$ASSET" '
    {
      gsub(/.*\//, "", $2)
      if ($2 == asset) {
        print $1
        exit
      }
    }
  ' "$CHECKSUMS_PATH"
)
[ -n "$expected" ] || fail "checksum entry not found for $ASSET"
actual=$(sha256_file "$ARCHIVE_PATH")
[ "$expected" = "$actual" ] || fail "checksum verification failed"

tar -xzf "$ARCHIVE_PATH" -C "$TMP_DIR"
[ -f "$TMP_DIR/plshelp" ] || fail "extracted archive does not contain plshelp binary"
install -m 0755 "$TMP_DIR/plshelp" "$INSTALL_DIR/plshelp"

log "Installed plshelp to $INSTALL_DIR/plshelp"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    log "Add $INSTALL_DIR to your PATH if it is not already there."
    ;;
esac
log "Run: plshelp help"
