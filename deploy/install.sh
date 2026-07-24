#!/usr/bin/env bash
#
# Install MoneyClush Terminal on a Debian/Ubuntu VPS behind nginx + TLS.
# Run as root on the server:  sudo bash install.sh
#
set -euo pipefail

DOMAIN="${DOMAIN:-trd.asto.work}"
APP_DIR="/opt/moneyclush"
APP_USER="moneyclush"
REPO_SRC="${REPO_SRC:-}"   # optional: git URL or local path already uploaded

say() { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo bash install.sh" >&2
  exit 1
fi

say "Installing system packages"
apt-get update -qq
apt-get install -y python3 python3-venv python3-pip nginx certbot \
                   python3-certbot-nginx apache2-utils curl

say "Creating service user and directories"
id -u "$APP_USER" &>/dev/null || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
mkdir -p "$APP_DIR/data"

if [[ -n "$REPO_SRC" ]]; then
  say "Fetching application from $REPO_SRC"
  if [[ "$REPO_SRC" == http*  || "$REPO_SRC" == git@* ]]; then
    apt-get install -y git
    rm -rf "$APP_DIR/src" "$APP_DIR/dashboard" "$APP_DIR/config"
    git clone --depth 1 "$REPO_SRC" /tmp/moneyclush-src
    cp -r /tmp/moneyclush-src/* "$APP_DIR/"
    rm -rf /tmp/moneyclush-src
  else
    cp -r "$REPO_SRC"/* "$APP_DIR/"
  fi
else
  say "REPO_SRC not set — assuming files are already in $APP_DIR"
fi

if [[ ! -f "$APP_DIR/dashboard/server.py" ]]; then
  echo "ERROR: $APP_DIR/dashboard/server.py not found. Upload the project first." >&2
  exit 1
fi

say "Creating virtualenv and installing dependencies"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet \
    fastapi uvicorn httpx pydantic structlog python-dotenv websockets

chown -R "$APP_USER:$APP_USER" "$APP_DIR"

say "Installing systemd unit"
cp "$APP_DIR/deploy/moneyclush.service" /etc/systemd/system/moneyclush.service
systemctl daemon-reload
systemctl enable --now moneyclush

sleep 3
if ! curl -fsS http://127.0.0.1:8642/api/state >/dev/null; then
  echo "WARNING: service is not responding on 127.0.0.1:8642"
  journalctl -u moneyclush -n 30 --no-pager || true
fi

say "Configuring basic auth"
if [[ ! -f /etc/nginx/.htpasswd-moneyclush ]]; then
  read -rp "Username for the dashboard: " BASIC_USER
  htpasswd -c /etc/nginx/.htpasswd-moneyclush "$BASIC_USER"
  chmod 640 /etc/nginx/.htpasswd-moneyclush
  chown root:www-data /etc/nginx/.htpasswd-moneyclush
else
  echo "  /etc/nginx/.htpasswd-moneyclush already exists — leaving it alone"
fi

say "Configuring nginx for $DOMAIN"
sed "s/trd\.asto\.work/$DOMAIN/g" "$APP_DIR/deploy/nginx-trd.asto.work.conf" \
    > "/etc/nginx/sites-available/$DOMAIN"
ln -sf "/etc/nginx/sites-available/$DOMAIN" "/etc/nginx/sites-enabled/$DOMAIN"

# certbot needs a working port-80 vhost first; comment the TLS server block
# until the certificate exists.
if [[ ! -d "/etc/letsencrypt/live/$DOMAIN" ]]; then
  say "Requesting TLS certificate for $DOMAIN"
  echo "  DNS for $DOMAIN must already point at this server."
  awk '/^server \{$/{n++} n==2{print "#" $0; next} {print}' \
      "/etc/nginx/sites-available/$DOMAIN" > /tmp/vhost.tmp
  mv /tmp/vhost.tmp "/etc/nginx/sites-available/$DOMAIN"
  nginx -t && systemctl reload nginx
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
          --register-unsafely-without-email --redirect || {
    echo "certbot failed — check DNS and firewall (ports 80/443)"; exit 1; }
else
  echo "  Certificate already present"
fi

nginx -t && systemctl reload nginx

say "Done"
echo "  Dashboard : https://$DOMAIN"
echo "  Service   : systemctl status moneyclush"
echo "  Logs      : journalctl -u moneyclush -f"
echo "  Metrics   : $APP_DIR/data/paper_metrics.jsonl"
echo
echo "  Running in PAPER mode. It places no real orders and holds no keys."
